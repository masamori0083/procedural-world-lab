mod animals;
mod environment;
mod monitor;
mod player;
mod river;
mod smoke;
mod stream_smoke;
mod streaming;
mod terrain;
mod terrain_lod;
mod trees;
mod vegetation;
mod water;
mod watershed;
mod world;

use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, prelude::*};
use std::time::Instant;

#[derive(Resource, Default)]
pub struct LabState {
    pub ready: bool,
    pub terrain_ms: f64,
    pub scene_setup_ms: f64,
    pub height_min: f32,
    pub height_max: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed = args
        .windows(2)
        .find(|p| p[0] == "--seed")
        .and_then(|p| p[1].parse::<u32>().ok())
        .unwrap_or(20261003);
    let stream_mode = args
        .iter()
        .any(|a| a == "--streaming" || a == "--stream-smoke-test");
    let stream_smoke = args.iter().any(|a| a == "--stream-smoke-test");
    let smoke_test = args.iter().any(|a| a == "--smoke-test");
    let mut settings = trees::TreeSettings::default();
    for (flag, value) in [
        ("--tree-variation", &mut settings.variation),
        ("--forest-uniformity", &mut settings.forest_uniformity),
        ("--landmark-strength", &mut settings.landmark_strength),
    ] {
        if let Some(v) = args
            .windows(2)
            .find(|p| p[0] == flag)
            .and_then(|p| p[1].parse::<f32>().ok())
            .filter(|v| v.is_finite())
        {
            *value = v.clamp(0.0, 1.0);
        }
    }
    if stream_mode && args.iter().any(|a| a == "--generation-report") {
        streaming::generation_report(
            streaming::Generator::new(
                seed,
                terrain::TerrainSettings {
                    mountain_strength: args
                        .windows(2)
                        .find(|p| p[0] == "--mountain-strength")
                        .and_then(|p| p[1].parse::<f32>().ok())
                        .filter(|v| v.is_finite())
                        .unwrap_or(0.65)
                        .clamp(0., 1.),
                },
            ),
            settings,
        );
        return;
    }
    let mut process = monitor::ProcessSampler::new();
    let before = process.snapshot();
    let begin = Instant::now();
    let mountain = args
        .windows(2)
        .find(|p| p[0] == "--mountain-strength")
        .and_then(|p| p[1].parse::<f32>().ok());
    let mut meadow = if stream_mode {
        terrain::Meadow::streamed(
            seed,
            terrain::TerrainSettings {
                mountain_strength: mountain.unwrap_or(0.65),
            },
        )
    } else if let Some(mountain_strength) = mountain {
        terrain::Meadow::with_settings(seed, terrain::TerrainSettings { mountain_strength })
    } else {
        terrain::Meadow::new(seed)
    };
    if let Some(generator) = meadow.stream_generator() {
        meadow.trees = [
            streaming::ChunkKey(-1, -1),
            streaming::ChunkKey(-1, 0),
            streaming::ChunkKey(0, -1),
            streaming::ChunkKey(0, 0),
        ]
        .into_iter()
        .flat_map(|key| generator.trees(key))
        .collect();
    }
    let terrain_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let after = process.snapshot();
    let generation = monitor::GenerationUsage::between(before, after);
    let heights = if stream_mode {
        terrain::HeightStats {
            minimum: 0.0,
            maximum: 0.0,
        }
    } else {
        meadow.height_stats()
    };
    if !stream_mode && args.iter().any(|arg| arg == "--generation-report") {
        std::fs::create_dir_all("reports").expect("Cannot create reports directory");
        let path = format!("reports/seed-{seed}.csv");
        let river = meadow.river.as_ref().unwrap();
        let csv = format!(
            "seed,generator_version,mountain_strength,min_height_m,max_height_m,trees,total_generation_ms,height_ms,water_ms,vegetation_ms,river_length_m,river_source_level_m,river_outlet_level_m,generation_cpu_ms,generation_rss_delta_mib,environment_ms,river_sources,river_network_length_m,map_width_m,grid_side,cell_size_m,navigation_ms,reachable_walkable_fraction\n{seed},{},{},{},{},{},{terrain_ms:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{:.3},{},{:.3},{},{},{},{:.3},{:.4}\n",
            terrain::GENERATOR_VERSION,
            meadow.settings.mountain_strength,
            heights.minimum,
            heights.maximum,
            meadow.trees.len(),
            meadow.timings.height_ms,
            meadow.timings.water_ms,
            meadow.timings.vegetation_ms,
            river.length(),
            river.points[0].level,
            river.points.last().unwrap().level,
            generation
                .cpu_ms
                .map(|n| format!("{n:.3}"))
                .unwrap_or_default(),
            generation
                .rss_delta_mib
                .map(|n| format!("{n:.3}"))
                .unwrap_or_default(),
            meadow.timings.environment_ms,
            river.source_count,
            river.total_length(),
            terrain::EXTENT * 2.0,
            terrain::SIDE,
            terrain::CELL,
            meadow.timings.navigation_ms,
            meadow.reachable_fraction,
        );
        std::fs::write(&path, csv).expect("Cannot save generation report");
        println!(
            "Saved {path}: elevation {:.1}..{:.1} m, generation {terrain_ms:.2} ms (height {:.2}, water {:.2}, environment {:.2}, vegetation {:.2}, navigation {:.2}), {} trees",
            heights.minimum,
            heights.maximum,
            meadow.timings.height_ms,
            meadow.timings.water_ms,
            meadow.timings.environment_ms,
            meadow.timings.vegetation_ms,
            meadow.timings.navigation_ms,
            meadow.trees.len()
        );
        return;
    }
    let mut app = App::new();
    app.insert_resource(meadow)
        .insert_resource(monitor::ResourceMonitor::new(process, generation, after))
        .insert_resource(settings)
        .insert_resource(LabState {
            terrain_ms,
            height_min: heights.minimum,
            height_max: heights.maximum,
            ..default()
        })
        .insert_resource(ClearColor(Color::srgb(0.64, 0.78, 0.85)))
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.85, 0.91, 1.0),
            brightness: 350.0,
            ..default()
        })
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: format!("{}/assets", env!("CARGO_MANIFEST_DIR")),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Map Generation Lab".into(),
                        resolution: (1280, 800).into(),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .insert_resource(vegetation::LodConfig {
            enabled: !args.iter().any(|a| a == "--no-vegetation-lod"),
        })
        .insert_resource(terrain_lod::Config {
            enabled: !args.iter().any(|a| a == "--no-terrain-lod"),
        })
        .init_resource::<player::CameraRig>()
        .init_resource::<environment::MapLayer>()
        .add_systems(
            Startup,
            (
                setup_scene,
                water::setup,
                player::setup,
                animals::setup,
                monitor::setup,
                streaming::setup,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                smoke::drive,
                animals::prepare,
                stream_smoke::drive,
                streaming::update,
                player::controls,
                animals::wander,
                animals::animate,
                player::follow_camera,
                world::update_layer,
                streaming::update_terrain_lod,
                streaming::update_lod,
                player::update_hud,
                monitor::update,
                world::sway_trees,
                world::drift_clouds,
                water::animate,
            )
                .chain(),
        );
    if stream_smoke {
        std::fs::create_dir_all(format!("captures/stream-seed-{seed}"))
            .expect("Cannot create captures directory");
        app.init_resource::<stream_smoke::StreamSmoke>()
            .add_systems(Update, stream_smoke::verify.after(player::update_hud));
    }
    if smoke_test && !stream_mode {
        std::fs::create_dir_all(format!("captures/seed-{seed}"))
            .expect("Cannot create captures directory");
        app.init_resource::<smoke::SmokeTest>().add_systems(
            Update,
            smoke::verify
                .after(player::update_hud)
                .after(monitor::update),
        );
    }
    app.run();
}

fn setup_scene(
    commands: Commands,
    world: Res<terrain::Meadow>,
    settings: Res<trees::TreeSettings>,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<StandardMaterial>>,
    mut lab: ResMut<LabState>,
) {
    let start = Instant::now();
    let timings = world.timings;
    world::setup(commands, world, settings, meshes, materials);
    lab.scene_setup_ms = start.elapsed().as_secs_f64() * 1000.0;
    info!(
        "Map generation: terrain {:.2} ms, CPU scene setup {:.2} ms (excludes GPU upload / model loading)",
        lab.terrain_ms, lab.scene_setup_ms
    );
    info!(
        "Generation stages: height {:.2} ms, water {:.2} ms, environment {:.2} ms, vegetation {:.2} ms",
        timings.height_ms, timings.water_ms, timings.environment_ms, timings.vegetation_ms
    );
}
