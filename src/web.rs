//! Browser entry options and a loading status for the shared map application.
use bevy::prelude::*;
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    future::Future,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll, Waker},
};
use wasm_bindgen::{JsCast, closure::Closure};

thread_local! { static SLICE_MAX: Cell<f64> = const { Cell::new(0.) }; }

pub fn record_slice(ms: f64) {
    SLICE_MAX.with(|value| value.set(value.get().max(ms)));
}

// A microtask yield (Promise.resolve / yield_now) cannot give rendering/input
// an event-loop turn. Schedule a real browser timer instead.
pub async fn yield_to_browser() {
    BrowserYield {
        ready: Rc::new(Cell::new(false)),
        wake: Rc::new(RefCell::new(None)),
        scheduled: false,
    }
    .await;
}
struct BrowserYield {
    ready: Rc<Cell<bool>>,
    wake: Rc<RefCell<Option<Waker>>>,
    scheduled: bool,
}
impl Future for BrowserYield {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.ready.get() {
            return Poll::Ready(());
        }
        *self.wake.borrow_mut() = Some(cx.waker().clone());
        if !self.scheduled {
            self.scheduled = true;
            let ready = self.ready.clone();
            let wake = self.wake.clone();
            let callback = Closure::once_into_js(move || {
                ready.set(true);
                if let Some(waker) = wake.borrow_mut().take() {
                    waker.wake();
                }
            });
            web_sys::window()
                .expect("Browser window")
                .set_timeout_with_callback_and_timeout_and_arguments_0(callback.unchecked_ref(), 0)
                .expect("Schedule generation continuation");
        }
        Poll::Pending
    }
}

#[derive(Resource)]
pub struct RenderSurface(Handle<Image>);

// Render 3D at CSS-pixel resolution (at most 1280x720), then draw the shared
// image beneath the native-resolution HUD. Winit still owns the canvas size,
// so resize events and pointer coordinates retain their normal DPI behavior.
pub fn setup_render(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut cameras: Query<(Entity, &mut Camera), With<Camera3d>>,
) {
    use bevy::{
        camera::{RenderTarget, visibility::RenderLayers},
        render::render_resource::TextureFormat,
    };
    let size = render_size();
    let image = images.add(Image::new_target_texture(
        size.x,
        size.y,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    ));
    for (entity, mut camera) in &mut cameras {
        camera.order = -1;
        commands.entity(entity).insert((
            RenderTarget::Image(image.clone().into()),
            bevy::core_pipeline::prepass::DepthPrepass,
        ));
    }
    let display = commands
        .spawn((Camera2d, IsDefaultUiCamera, RenderLayers::layer(31)))
        .id();
    commands.spawn((
        ImageNode::new(image.clone()),
        UiTargetCamera(display),
        GlobalZIndex(-1),
        Node {
            position_type: PositionType::Absolute,
            left: px(0),
            top: px(0),
            width: percent(100),
            height: percent(100),
            ..default()
        },
    ));
    commands.insert_resource(RenderSurface(image));
}
fn render_size() -> UVec2 {
    let viewport = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("viewport"));
    let (width, height) = viewport.map_or((1280, 720), |v| {
        (
            v.client_width().max(1) as u32,
            v.client_height().max(1) as u32,
        )
    });
    crate::work_budget::render_size(width, height)
}
pub fn resize_render(surface: Res<RenderSurface>, mut images: ResMut<Assets<Image>>) {
    let size = render_size();
    if let Some(mut image) = images.get_mut(&surface.0)
        && image.size() != size
    {
        image.resize(bevy::render::render_resource::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        });
    }
}

pub fn args() -> Vec<String> {
    let seed = web_sys::window()
        .and_then(|w| w.location().search().ok())
        .and_then(|query| {
            query.trim_start_matches('?').split('&').find_map(|part| {
                part.strip_prefix("seed=")
                    .and_then(|s| s.parse::<u32>().ok())
            })
        })
        .unwrap_or(42);
    vec![
        "map_generation_lab".into(),
        "--streaming".into(),
        "--seed".into(),
        seed.to_string(),
    ]
}

#[allow(clippy::too_many_arguments)]
pub fn ready(
    lab: Res<crate::LabState>,
    rig: Res<crate::player::CameraRig>,
    world: Res<crate::terrain::Meadow>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    stream: Option<Res<crate::streaming::StreamWorld>>,
    mut notified: Local<bool>,
    mut last: Local<f64>,
    time: Res<Time<Real>>,
    surface: Res<RenderSurface>,
    images: Res<Assets<Image>>,
    mut history: Local<VecDeque<(f64, f64)>>,
) {
    let now = time.elapsed_secs_f64();
    history.push_back((now, time.delta_secs_f64() * 1000.));
    while history.front().is_some_and(|(t, _)| now - t > 2.) || history.len() > 512 {
        history.pop_front();
    }
    // Keep sampling while a moving horse waits for its next ground chunk.
    // Otherwise the dashboard would omit precisely the streaming stalls.
    if !lab.ready && !*notified {
        return;
    }
    if *notified && time.elapsed_secs_f64() - *last < 0.5 {
        return;
    }
    *last = time.elapsed_secs_f64();
    if let Some(document) = web_sys::window().and_then(|w| w.document()) {
        if let Some(status) = document.get_element_by_id("status") {
            let fps = diagnostics
                .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
                .map(|fps| format!("{fps:.0} FPS"))
                .unwrap_or_else(|| "Ready".into());
            let memory = stream
                .map(|s| format!(" · {:.1} MiB mesh estimate", s.mesh_mib()))
                .unwrap_or_default();
            let mut frames: Vec<_> = history.iter().map(|(_, ms)| *ms).collect();
            frames.sort_by(f64::total_cmp);
            let p95 = frames.get(frames.len() * 95 / 100).copied().unwrap_or(0.);
            let slice = SLICE_MAX.with(|value| value.replace(0.));
            let size = images.get(&surface.0).map_or(UVec2::ZERO, Image::size);
            let _ = status.set_attribute("data-render-size", &format!("{}x{}", size.x, size.y));
            let _ = status.set_attribute("data-frame-p95-ms", &format!("{p95:.2}"));
            let _ = status.set_attribute("data-generation-slice-ms", &format!("{slice:.2}"));
            let interaction = if rig.captured {
                "Exploring · Esc releases the mouse."
            } else {
                "Paused preview · Click map to explore."
            };
            status.set_text_content(Some(&format!(
                "Seed {} · {fps}{memory} · Frame p95 {p95:.1} ms · Generation slice {slice:.1} ms · 3D {}×{} · {interaction}",
                world.seed, size.x, size.y
            )));
        }
        if lab.ready && !*notified {
            if let Some(loading) = document.get_element_by_id("loading") {
                let _ = loading.set_attribute("hidden", "");
            }
            *notified = true;
        }
    }
}

// Keep the existing render-target image while a paused view is settled. The
// native-resolution HUD stays live; input/resize/stream changes refresh 3D.
#[derive(Default)]
pub struct PausedPreview {
    frame: Option<PreviewFrame>,
    changed_at: f64,
}
#[derive(PartialEq)]
struct PreviewFrame {
    pose: Mat4,
    projection: Mat4,
    size: UVec2,
    revision: [u64; 3],
    layer: crate::environment::MapLayer,
}
#[allow(clippy::too_many_arguments)]
pub fn pace_render(
    lab: Res<crate::LabState>,
    rig: Res<crate::player::CameraRig>,
    mut settings: ResMut<bevy::winit::WinitSettings>,
    mut loaded: Local<bool>,
    stream: Option<Res<crate::streaming::StreamWorld>>,
    surface: Res<RenderSurface>,
    images: Res<Assets<Image>>,
    time: Res<Time<Real>>,
    layer: Res<crate::environment::MapLayer>,
    camera: Single<(&mut Camera, &Transform, &Projection), With<Camera3d>>,
    mut preview: Local<PausedPreview>,
) {
    *loaded |= lab.ready;
    let foreground = web_sys::window()
        .and_then(|w| w.document())
        .is_some_and(|d| !d.hidden() && d.has_focus().unwrap_or(false));
    let mode = crate::work_budget::browser_frame_wait(*loaded, rig.captured, foreground).map_or(
        bevy::winit::UpdateMode::Continuous,
        bevy::winit::UpdateMode::reactive_low_power,
    );
    if settings.focused_mode != mode || settings.unfocused_mode != mode {
        settings.focused_mode = mode;
        settings.unfocused_mode = mode;
    }
    let (mut camera, transform, projection) = camera.into_inner();
    let current = PreviewFrame {
        pose: transform.to_matrix(),
        projection: projection.get_clip_from_view(),
        size: images.get(&surface.0).map_or(UVec2::ZERO, Image::size),
        revision: stream
            .as_ref()
            .map_or([0; 3], |s| [s.generated, s.lod_rebuilt, s.terrain_rebuilt]),
        layer: *layer,
    };
    let unchanged = stream.as_ref().is_none_or(|s| s.pending_count() == 0)
        && preview.frame.as_ref() == Some(&current);
    let now = time.elapsed_secs_f64();
    if !unchanged {
        preview.changed_at = now;
    }
    let draw = crate::work_budget::browser_draw_scene(
        *loaded,
        rig.captured && foreground,
        unchanged,
        now - preview.changed_at,
    );
    if draw {
        preview.frame = Some(current);
    }
    if camera.is_active != draw {
        camera.is_active = draw;
    }
}
