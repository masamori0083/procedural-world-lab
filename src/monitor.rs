//! Lightweight, process-local resource readings for the map screen.
use crate::{LabState, terrain::Meadow};
use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};
use std::time::{Duration, Instant};
#[cfg(not(target_os = "macos"))]
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

const MIB: f64 = 1024.0 * 1024.0;
#[cfg(target_os = "macos")]
const GPU_NOTE: &str = "N/A (Metal)";
#[cfg(not(target_os = "macos"))]
const GPU_NOTE: &str = "N/A (unavailable)";

#[derive(Clone, Copy, Default)]
pub struct ProcessSnapshot {
    cpu_ms: u64,
    rss_bytes: u64,
}

pub struct ProcessSampler {
    #[cfg(not(target_os = "macos"))]
    system: System,
    #[cfg(not(target_os = "macos"))]
    pid: Option<Pid>,
}

impl ProcessSampler {
    pub fn new() -> Self {
        Self {
            #[cfg(not(target_os = "macos"))]
            system: System::new(),
            #[cfg(not(target_os = "macos"))]
            pid: sysinfo::get_current_pid().ok(),
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn refresh(&mut self) {
        if let Some(pid) = self.pid {
            self.system.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[pid]),
                false,
                ProcessRefreshKind::nothing().with_cpu().with_memory(),
            );
        }
    }

    pub fn snapshot(&mut self) -> Option<ProcessSnapshot> {
        #[cfg(target_os = "macos")]
        {
            macos_snapshot()
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.refresh();
            self.pid
                .and_then(|pid| self.system.process(pid))
                .map(|process| ProcessSnapshot {
                    cpu_ms: process.accumulated_cpu_time(),
                    rss_bytes: process.memory(),
                })
        }
    }
}

#[cfg(target_os = "macos")]
fn macos_snapshot() -> Option<ProcessSnapshot> {
    // These calls inspect this process itself, so they also work when macOS
    // denies process-list enumeration to a sandboxed app.
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let mut task = std::mem::MaybeUninit::<libc::mach_task_basic_info>::uninit();
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    // SAFETY: Both pointers address writable, correctly sized native structs.
    // getrusage/task_info initialize them only on their success return codes.
    let (usage, task) = unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0
            || libc::task_info(
                mach2::traps::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                task.as_mut_ptr().cast(),
                &mut count,
            ) != 0
        {
            return None;
        }
        (usage.assume_init(), task.assume_init())
    };
    let cpu_ms = ((usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) as u64) * 1000
        + ((usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) as u64) / 1000;
    Some(ProcessSnapshot {
        cpu_ms,
        rss_bytes: task.resident_size,
    })
}

#[derive(Clone, Copy, Default)]
pub struct GenerationUsage {
    pub cpu_ms: Option<f64>,
    pub rss_delta_mib: Option<f64>,
}

impl GenerationUsage {
    pub fn between(before: Option<ProcessSnapshot>, after: Option<ProcessSnapshot>) -> Self {
        match (before, after) {
            (Some(before), Some(after)) => Self {
                cpu_ms: Some(after.cpu_ms.saturating_sub(before.cpu_ms) as f64),
                rss_delta_mib: Some((after.rss_bytes as f64 - before.rss_bytes as f64) / MIB),
            },
            _ => Self::default(),
        }
    }
}

#[derive(Resource)]
pub struct ResourceMonitor {
    sampler: ProcessSampler,
    generation: GenerationUsage,
    latest: Option<ProcessSnapshot>,
    last_sample: Instant,
    cpu_percent: Option<f64>,
    visible: bool,
}

impl ResourceMonitor {
    pub fn new(
        sampler: ProcessSampler,
        generation: GenerationUsage,
        latest: Option<ProcessSnapshot>,
    ) -> Self {
        Self {
            sampler,
            generation,
            latest,
            last_sample: Instant::now(),
            cpu_percent: None,
            visible: true,
        }
    }
}

#[derive(Component)]
pub struct ResourcePanel;

pub fn setup(mut commands: Commands) {
    commands.spawn((
        ResourcePanel,
        Text::new("RESOURCE MONITOR\nSampling process CPU..."),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
        TextColor(Color::srgb(0.93, 0.96, 0.90)),
        Node {
            position_type: PositionType::Absolute,
            right: px(18),
            top: px(18),
            padding: UiRect::all(px(13)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.055, 0.095, 0.065, 0.88)),
    ));
}

pub fn update(
    keys: Res<ButtonInput<KeyCode>>,
    mut monitor: ResMut<ResourceMonitor>,
    lab: Res<LabState>,
    map: Res<Meadow>,
    diagnostics: Res<DiagnosticsStore>,
    mut panel: Single<(&mut Text, &mut Visibility), With<ResourcePanel>>,
) {
    if keys.just_pressed(KeyCode::F3) {
        monitor.visible = !monitor.visible;
        *panel.1 = if monitor.visible {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    let elapsed = monitor.last_sample.elapsed();
    if elapsed < Duration::from_secs(1) {
        return;
    }

    let now = monitor.sampler.snapshot();
    monitor.cpu_percent = match (monitor.latest, now) {
        (Some(before), Some(after)) => {
            Some(after.cpu_ms.saturating_sub(before.cpu_ms) as f64 / elapsed.as_secs_f64() / 10.0)
        }
        _ => None,
    };
    monitor.latest = now;
    monitor.last_sample = Instant::now();

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|value| value.smoothed());
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|value| value.smoothed());
    let stages = map.timings;
    let build_label = if map.is_streaming() {
        "INITIAL SETUP (chunks below left)"
    } else {
        "MAP BUILD (one time)"
    };
    panel.0.0 = format!(
        "RESOURCE MONITOR  [F3 hide]\n\
         {build_label}\n\
         Terrain wall       {:>7.1} ms\n\
           Height           {:>7.1} ms\n\
           Water            {:>7.1} ms\n\
           Environment      {:>7.1} ms\n\
           Vegetation       {:>7.1} ms\n\
           Navigation       {:>7.1} ms\n\
         Process CPU        {:>7}\n\
         RAM change         {:>7}\n\
         Scene setup wall   {:>7.1} ms\n\
         LIVE (this app, 1 s)\n\
         CPU                 {:>6}\n\
         RAM                 {:>6}\n\
         FPS / frame     {:>5} / {:>6}\n\
         GPU time / use  {GPU_NOTE}\n\
         CPU 100% = one core; RAM = RSS",
        lab.terrain_ms,
        stages.height_ms,
        stages.water_ms,
        stages.environment_ms,
        stages.vegetation_ms,
        stages.navigation_ms,
        monitor
            .generation
            .cpu_ms
            .map(|n| format!("{n:.0} ms"))
            .unwrap_or_else(|| "N/A".into()),
        monitor
            .generation
            .rss_delta_mib
            .map(|n| format!("{n:+.1} MiB"))
            .unwrap_or_else(|| "N/A".into()),
        lab.scene_setup_ms,
        monitor
            .cpu_percent
            .map(|n| format!("{n:.1}%"))
            .unwrap_or_else(|| "N/A".into()),
        monitor
            .latest
            .map(|s| format!("{:.1} MiB", s.rss_bytes as f64 / MIB))
            .unwrap_or_else(|| "N/A".into()),
        fps.map(|n| format!("{n:.0}"))
            .unwrap_or_else(|| "N/A".into()),
        frame_ms
            .map(|n| format!("{n:.1} ms"))
            .unwrap_or_else(|| "N/A".into()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_usage_handles_missing_sample_and_multicore_cpu_time() {
        assert!(GenerationUsage::between(None, None).cpu_ms.is_none());
        let before = ProcessSnapshot {
            cpu_ms: 10,
            rss_bytes: 2 * 1024 * 1024,
        };
        let after = ProcessSnapshot {
            cpu_ms: 160,
            rss_bytes: 3 * 1024 * 1024,
        };
        let usage = GenerationUsage::between(Some(before), Some(after));
        assert_eq!(usage.cpu_ms, Some(150.0));
        assert_eq!(usage.rss_delta_mib, Some(1.0));
    }
}
