//! Browser entry options and a loading status for the shared map application.
use bevy::prelude::*;

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

pub fn ready(
    lab: Res<crate::LabState>,
    world: Res<crate::terrain::Meadow>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    stream: Option<Res<crate::streaming::StreamWorld>>,
    mut notified: Local<bool>,
    mut last: Local<f64>,
    time: Res<Time<Real>>,
) {
    if !lab.ready {
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
            status.set_text_content(Some(&format!(
                "Seed {} · {fps}{memory} · Click map to explore; Esc releases the mouse.",
                world.seed
            )));
        }
        if !*notified {
            if let Some(loading) = document.get_element_by_id("loading") {
                let _ = loading.set_attribute("hidden", "");
            }
            *notified = true;
        }
    }
}
