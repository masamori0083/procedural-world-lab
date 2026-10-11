//! Held real-map rendering comparisons. Serialized offscreen frame intervals
//! include GPU completion, but are not GPU timestamps or interactive FPS.
use crate::{
    LabState,
    player::{CameraRig, FollowCamera, HorseController},
    streaming::{ChunkKey, MAX_CHUNKS, StreamWorld},
    terrain::Meadow,
    trees::{Species, TreeSettings, TreeShape},
    wind::{Materials, VegetationMaterial},
    world::Grass,
};
use bevy::platform::time::Instant;
use bevy::{
    camera::{ImageRenderTarget, RenderTarget},
    light::NotShadowReceiver,
    prelude::*,
    render::{
        render_resource::TextureFormat,
        view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    },
};

const WARMUP: f64 = 4.;
const MEASURE: f64 = 5.;

#[derive(Resource)]
struct Profile {
    pose: String,
    case: String,
    label: String,
    half_resolution: bool,
    rotate: bool,
    image: Option<Handle<Image>>,
    transform: Option<Transform>,
    ready_since: Option<Instant>,
    captured: bool,
    capture_ok: bool,
    capture_failed: bool,
    begin: Instant,
    frame_begin: Instant,
    rows: Vec<Row>,
}
struct Row {
    frame_ms: f64,
    cpu_ms: f64,
    triangles: usize,
    tree_counts: [usize; 3],
    tree_triangles: usize,
    tree_entities: usize,
    mesh_mib: f64,
    material_updates: usize,
    camera: Vec3,
    lod_rebuilt: u64,
    tree_lod_rebuilt: u64,
    pending: usize,
}
fn argument(args: &[String], flag: &str, fallback: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map_or_else(|| fallback.into(), |w| w[1].clone())
}
pub fn enable(app: &mut App, args: &[String]) {
    let pose = argument(args, "--profile-pose", "grass");
    let case = argument(args, "--profile-case", "all");
    assert!(["grass", "leaves", "far", "grass-shadow"].contains(&pose.as_str()));
    assert!(
        [
            "all",
            "no-grass",
            "no-grass-cast",
            "unlit-grass",
            "no-grass-transmission",
            "pbr-grass",
            "no-leaves",
            "no-receiving",
            "no-leaf-receiving",
            "no-shadows",
            "depth-prepass",
            "half-shadow-resolution"
        ]
        .contains(&case.as_str())
    );
    let label = argument(args, "--profile-label", "run")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    app.insert_resource(bevy::winit::WinitSettings::continuous());
    // Fix wind phase, wildlife and model pose across fresh processes. Real
    // time and the render/GPU workload keep advancing; normal play is unaffected.
    app.world_mut().resource_mut::<Time<Virtual>>().pause();
    if case == "half-shadow-resolution" {
        app.insert_resource(bevy::light::DirectionalLightShadowMap { size: 1024 });
    }
    app.insert_resource(Profile {
        pose,
        case,
        label,
        half_resolution: args.iter().any(|a| a == "--profile-half-resolution"),
        rotate: args.iter().any(|a| a == "--profile-rotate"),
        image: None,
        transform: None,
        ready_since: None,
        captured: false,
        capture_ok: false,
        capture_failed: false,
        begin: Instant::now(),
        frame_begin: Instant::now(),
        rows: Vec::with_capacity(4096),
    })
    .add_systems(First, start_frame)
    .add_systems(
        Update,
        hold_controls
            .before(crate::streaming::update)
            .before(crate::player::controls),
    )
    .add_systems(
        Update,
        hold_camera
            .after(crate::player::follow_camera)
            .before(crate::streaming::schedule),
    )
    .add_systems(Last, finish_frame);
    app.sub_app_mut(bevy::render::RenderApp).add_systems(
        bevy::render::Render,
        wait_for_gpu.in_set(bevy::render::RenderSystems::Cleanup),
    );
}
fn wait_for_gpu(device: Res<bevy::render::renderer::RenderDevice>) {
    device
        .poll(bevy::render::render_resource::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .expect("map profile GPU completion");
}
fn start_frame(mut profile: ResMut<Profile>) {
    profile.frame_begin = Instant::now();
}
fn hold_controls(
    mut rig: ResMut<CameraRig>,
    world: Res<Meadow>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut horse: Single<(&mut Transform, &mut HorseController)>,
) {
    // Same anchor and loaded ring for every comparison; disable live input.
    keys.reset_all();
    rig.captured = false;
    let p = Vec2::new(32., 48.);
    horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
    horse.0.rotation = Quat::IDENTITY;
    horse.1.speed = 0.;
    horse.1.yaw = 0.;
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn hold_camera(
    mut commands: Commands,
    mut profile: ResMut<Profile>,
    world: Res<Meadow>,
    settings: Res<TreeSettings>,
    window: Single<&Window>,
    mut images: ResMut<Assets<Image>>,
    shared: Res<Materials>,
    mut materials: ResMut<Assets<VegetationMaterial>>,
    mut camera: Single<(Entity, &mut Transform), (With<FollowCamera>, Without<HorseController>)>,
    mut sun: Single<&mut DirectionalLight>,
    mut plants: Query<(
        Entity,
        &MeshMaterial3d<VegetationMaterial>,
        &mut Visibility,
        Option<&Grass>,
        Option<&NotShadowReceiver>,
    )>,
) {
    if profile.image.is_none()
        && ["unlit-grass", "no-grass-transmission", "pbr-grass"].contains(&profile.case.as_str())
        && let Some(mut material) = materials.get_mut(&shared.grass)
    {
        if profile.case == "unlit-grass" {
            material.base.unlit = true;
        } else if profile.case == "no-grass-transmission" {
            material.base.diffuse_transmission = 0.;
        } else if profile.case == "pbr-grass" {
            material.extension.full_lighting = true;
        }
    }
    if profile.image.is_none() {
        if profile.case == "depth-prepass" {
            commands
                .entity(camera.0)
                .insert(bevy::core_pipeline::prepass::DepthPrepass);
        }
        let divisor = if profile.half_resolution { 2 } else { 1 };
        let image = images.add(Image::new_target_texture(
            window.physical_width() / divisor,
            window.physical_height() / divisor,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        // Preserve logical screen size, so half-resolution comparisons do not
        // also change screen-size LOD thresholds or the selected geometry.
        commands
            .entity(camera.0)
            .insert(RenderTarget::Image(ImageRenderTarget {
                handle: image.clone(),
                scale_factor: window.scale_factor() / divisor as f32,
            }));
        profile.image = Some(image);
    }
    let pose_name = profile.pose.clone();
    let pose = *profile.transform.get_or_insert_with(|| {
        let p = Vec2::new(32., 48.);
        let h = world.ground(p);
        let (eye, target) = match pose_name.as_str() {
            // All grass below this overhead view is beyond the 10m proxy fade.
            // all/no-grass-cast captures must match: no flattened proxy shadows.
            "grass-shadow" => (Vec3::new(32., h + 14., 48.), Vec3::new(32., h, 47.9)),
            "far" => (Vec3::new(32., h + 14., 90.), Vec3::new(32., h + 1., 40.)),
            "leaves" => {
                let g = world.stream_generator().unwrap();
                let mut candidates = g.tree_neighbours(ChunkKey::at(p));
                candidates
                    .sort_by(|a, b| a.0.distance_squared(p).total_cmp(&b.0.distance_squared(p)));
                let (p, shape) = candidates
                    .into_iter()
                    .filter_map(|(p, size)| {
                        let shape = TreeShape::at(&world, p, size, *settings);
                        (shape.species != Species::Conifer).then_some((p, shape))
                    })
                    .next()
                    .expect("nearby broadleaf tree");
                let target = Vec3::new(p.x, world.ground(p), p.y) + shape.crown_pivot;
                (target + Vec3::new(0., 0., 3.5), target)
            }
            _ => (Vec3::new(32., h + 0.9, 54.), Vec3::new(32., h + 0.65, 40.)),
        };
        Transform::from_translation(eye).looking_at(target, Vec3::Y)
    });
    *camera.1 = pose;
    if profile.rotate
        && let Some(start) = profile.ready_since
    {
        let angle = ((start.elapsed().as_secs_f64() - WARMUP) / MEASURE).clamp(0., 1.)
            * std::f64::consts::TAU;
        camera.1.rotation = Quat::from_rotation_y(angle as f32) * pose.rotation;
    }
    sun.shadow_maps_enabled = profile.case != "no-shadows";
    for (entity, material, mut visibility, grass, no_receiver) in &mut plants {
        let visual_grass = grass.is_some() && material.0 == shared.grass;
        let hidden = (grass.is_some() && profile.case == "no-grass")
            || (material.0 == shared.shadow_grass && profile.case == "no-grass-cast")
            || (material.0 == shared.leaves && profile.case == "no-leaves");
        let desired = if hidden {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if *visibility != desired {
            *visibility = desired;
        }
        let disable_receiving = (visual_grass && profile.case == "no-receiving")
            || (material.0 == shared.leaves && profile.case == "no-leaf-receiving");
        if disable_receiving && no_receiver.is_none() {
            commands.entity(entity).insert(NotShadowReceiver);
        }
    }
}
fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() as f64 * fraction).ceil() as usize).saturating_sub(1)]
}
#[allow(clippy::too_many_arguments)]
fn finish_frame(
    mut commands: Commands,
    mut profile: ResMut<Profile>,
    lab: Res<LabState>,
    stream: Res<StreamWorld>,
    time: Res<Time<Real>>,
    window: Single<&Window>,
    mut events: MessageReader<AssetEvent<VegetationMaterial>>,
    mut exit: MessageWriter<AppExit>,
    camera: Single<(&Camera, &RenderTarget, &GlobalTransform), With<FollowCamera>>,
    config: Res<crate::wind::Config>,
    shadow_map: Res<bevy::light::DirectionalLightShadowMap>,
) {
    assert!(
        profile.begin.elapsed().as_secs() < 90,
        "map profile timed out"
    );
    let material_updates = events
        .read()
        .filter(|e| matches!(e, AssetEvent::Modified { .. }))
        .count();
    if profile.capture_failed {
        eprintln!("INVALID MAP PROFILE: warmup capture was black; no timings accepted");
        exit.write(AppExit::error());
        return;
    }
    let settled = lab.ready
        && stream.loaded_count() == MAX_CHUNKS
        && stream.pending_count() == 0
        && stream.queued == 0;
    let rotating = profile.rotate
        && profile
            .ready_since
            .is_some_and(|s| s.elapsed().as_secs_f64() >= WARMUP);
    if !settled && !rotating {
        assert!(
            profile.rows.is_empty(),
            "map jobs changed during held measurement"
        );
        profile.ready_since = None;
        return;
    }
    let seconds = profile
        .ready_since
        .get_or_insert_with(Instant::now)
        .elapsed()
        .as_secs_f64();
    if seconds >= 2. && !profile.captured {
        std::fs::create_dir_all("captures/map-profile").unwrap();
        let path = format!(
            "captures/map-profile/{}-{}-{}.png",
            profile.label, profile.pose, profile.case
        );
        commands
            // Match the exact target, including Retina scale. Screenshot::image
            // uses scale=1 and would intercept an unused, black image target.
            .spawn(profile_screenshot(camera.1))
            .observe(save_to_disk(path))
            .observe(
                |capture: On<ScreenshotCaptured>, mut profile: ResMut<Profile>| {
                    let ok =
                        capture.image.data.as_deref().is_some_and(|d| {
                            d.chunks_exact(4).any(|p| p[..3].iter().any(|v| *v != 0))
                        });
                    profile.capture_ok = ok;
                    profile.capture_failed = !ok;
                },
            );
        profile.captured = true;
    }
    if (WARMUP..WARMUP + MEASURE).contains(&seconds) {
        assert!(
            profile.capture_ok,
            "profile image not validated before measurement"
        );
        assert!(
            camera
                .2
                .translation()
                .distance(profile.transform.unwrap().translation)
                < 0.001,
            "held profile camera drifted"
        );
        let (tree_counts, tree_vertices, tree_entities) = stream.tree_stats(camera.2.translation());
        let row = Row {
            frame_ms: time.delta_secs_f64() * 1000.,
            cpu_ms: profile.frame_begin.elapsed().as_secs_f64() * 1000.,
            triangles: stream.vegetation_vertices().0 / 3,
            tree_counts,
            tree_triangles: tree_vertices / 3,
            tree_entities,
            mesh_mib: stream.mesh_mib(),
            material_updates,
            camera: camera.2.translation(),
            lod_rebuilt: stream.lod_rebuilt,
            tree_lod_rebuilt: stream.tree_lod_rebuilt,
            pending: stream.pending_count(),
        };
        profile.rows.push(row);
    }
    if seconds < WARMUP + MEASURE {
        return;
    }
    assert!(!profile.rows.is_empty());
    let mut frames: Vec<_> = profile.rows.iter().map(|r| r.frame_ms).collect();
    let mut cpu: Vec<_> = profile.rows.iter().map(|r| r.cpu_ms).collect();
    let row = &profile.rows[0];
    let divisor = if profile.half_resolution { 2 } else { 1 };
    let summary = format!(
        "pose,case,samples,frame_median_ms,frame_p95_ms,frame_p99_ms,main_cpu_median_ms,resident_vegetation_triangles,material_updates,physical_width,physical_height,logical_width,logical_height,camera_x,camera_y,camera_z,fast_grass_shadows,shadow_map_size,rotating,lod_installs,max_pending,frozen_animation_clock,serialized_offscreen\n{},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{},{:.4},{:.4},{:.4},{},{},{},{},{},true,true\n",
        profile.pose,
        profile.case,
        profile.rows.len(),
        percentile(&mut frames, 0.5),
        percentile(&mut frames, 0.95),
        percentile(&mut frames, 0.99),
        percentile(&mut cpu, 0.5),
        row.triangles,
        profile
            .rows
            .iter()
            .map(|r| r.material_updates)
            .sum::<usize>(),
        window.physical_width() / divisor,
        window.physical_height() / divisor,
        window.width(),
        window.height(),
        row.camera.x,
        row.camera.y,
        row.camera.z,
        config.fast_grass_shadows,
        shadow_map.size,
        profile.rotate,
        profile
            .rows
            .last()
            .unwrap()
            .lod_rebuilt
            .saturating_sub(row.lod_rebuilt),
        profile.rows.iter().map(|r| r.pending).max().unwrap()
    );
    let mut lines = summary.lines();
    let summary = format!(
        "{},tree_candidates_near,tree_candidates_middle,tree_candidates_far,candidate_tree_triangles,resident_tree_entities,cpu_mesh_payload_mib,tree_lod_installs\n{},{},{},{},{},{},{:.3},{}\n",
        lines.next().unwrap(),
        lines.next().unwrap(),
        row.tree_counts[0],
        row.tree_counts[1],
        row.tree_counts[2],
        row.tree_triangles,
        row.tree_entities,
        row.mesh_mib,
        profile
            .rows
            .last()
            .unwrap()
            .tree_lod_rebuilt
            .saturating_sub(row.tree_lod_rebuilt)
    );
    let mut raw =
        "frame_ms,main_cpu_ms,resident_vegetation_triangles,material_updates,lod_installs_total,pending\n".to_string();
    for row in &profile.rows {
        raw.push_str(&format!(
            "{:.6},{:.6},{},{},{},{}\n",
            row.frame_ms,
            row.cpu_ms,
            row.triangles,
            row.material_updates,
            row.lod_rebuilt,
            row.pending
        ));
    }
    std::fs::create_dir_all("reports").unwrap();
    let prefix = format!(
        "reports/map-profile-{}-{}-{}",
        profile.label, profile.pose, profile.case
    );
    std::fs::write(format!("{prefix}.csv"), raw).unwrap();
    std::fs::write(format!("{prefix}-summary.csv"), &summary).unwrap();
    println!("MAP PROFILE\n{summary}");
    exit.write(AppExit::Success);
}
fn profile_screenshot(target: &RenderTarget) -> Screenshot {
    Screenshot(target.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retina_screenshot_matches_the_camera_target_exactly() {
        let target = RenderTarget::Image(ImageRenderTarget {
            handle: Handle::default(),
            scale_factor: 2.,
        });
        let screenshot = profile_screenshot(&target);
        let RenderTarget::Image(image) = screenshot.0 else {
            panic!("image target required")
        };
        assert_eq!(image.scale_factor, 2.);
        assert_eq!(image.handle, Handle::default());
    }
}
