//! Native streaming checks: riding across a seam, eviction, distant travel and revisit.
use crate::{
    LabState,
    environment::MapLayer,
    player::{CameraRig, HorseController, ViewMode},
    streaming::{ChunkKey, MAX_CHUNKS, StreamWorld},
    terrain::Meadow,
};
use bevy::{
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    window::PrimaryWindow,
};
use std::time::Instant;
#[derive(Resource)]
pub struct StreamSmoke {
    start: Instant,
    stage: usize,
    fired: bool,
    settled_spawn: bool,
    elapsed: f32,
    saved: usize,
    initial_meshes: usize,
    initial_height: f32,
    river_phase: Vec2,
    headwater_source: Vec2,
}
impl Default for StreamSmoke {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            stage: 0,
            fired: false,
            settled_spawn: false,
            elapsed: 0.,
            saved: 0,
            initial_meshes: 0,
            initial_height: 0.,
            river_phase: Vec2::ZERO,
            headwater_source: Vec2::ZERO,
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub fn drive(
    smoke: Option<ResMut<StreamSmoke>>,
    stream: Option<Res<StreamWorld>>,
    time: Res<Time>,
    lab: Res<LabState>,
    world: Res<Meadow>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut rig: ResMut<CameraRig>,
    mut horse: Single<(&mut Transform, &mut HorseController)>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
) {
    let Some(mut smoke) = smoke else {
        return;
    };
    let Some(stream) = stream else {
        return;
    };
    assert!(
        smoke.start.elapsed().as_secs() < 120,
        "Streaming smoke timed out"
    );
    window.focused = true;
    rig.captured = true;
    keys.reset_all();
    if !smoke.fired {
        match smoke.stage {
            1 => {
                let p = (-24..=24)
                    .map(|z| Vec2::new(95., z as f32 * 2.))
                    .find(|p| {
                        (0..=36).all(|step| {
                            let q = *p + Vec2::X * step as f32 * 0.5;
                            world.walkable(q) && !stream.tree_blocks(q, None)
                        })
                    })
                    .expect("Boundary needs a clear riding segment");
                assert!(world.walkable(p));
                horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
                horse.1.yaw = -std::f32::consts::FRAC_PI_2;
                horse.0.rotation = Quat::from_rotation_y(horse.1.yaw);
            }
            2 | 3 => {
                let target = if smoke.stage == 2 {
                    Vec2::new(1260., -880.)
                } else {
                    Vec2::new(-1260., -1040.)
                };
                let p = crate::animals::safe_spawn(&world, target);
                assert!(p.distance(target) < 40.);
                horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
                horse.1.speed = 0.;
                horse.1.yaw = 0.;
                horse.0.rotation = Quat::IDENTITY;
            }
            4 => keys.press(KeyCode::KeyR),
            5 => keys.press(KeyCode::KeyB),
            6..=9 => keys.press(KeyCode::F4),
            10 => keys.press(KeyCode::KeyL),
            12 => {
                let (p, yaw, source) = headwater_view(&world);
                smoke.headwater_source = source;
                horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
                horse.1.speed = 0.;
                horse.1.yaw = yaw;
                horse.0.rotation = Quat::from_rotation_y(yaw);
                rig.mode = ViewMode::ThirdPerson;
            }
            13 => keys.press(KeyCode::KeyB),
            14 => {
                let g = world.stream_generator().unwrap();
                let plan = g.hydrology.get((0, 0)).unwrap();
                let p = (0..24)
                    .map(|n| {
                        plan.lake + Vec2::from_angle(n as f32 * std::f32::consts::TAU / 24.) * 30.
                    })
                    .find(|p| world.walkable(*p))
                    .expect("Lake needs a dry shore view");
                let toward = plan.lake - p;
                horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
                horse.1.speed = 0.;
                horse.1.yaw = (-toward.x).atan2(-toward.y);
                horse.0.rotation = Quat::from_rotation_y(horse.1.yaw);
                rig.mode = ViewMode::ThirdPerson;
            }
            15 => {
                let (p, yaw) = landmark_view(&world, &stream);
                horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
                horse.1.speed = 0.;
                horse.1.yaw = yaw;
                horse.0.rotation = Quat::from_rotation_y(yaw);
                rig.frame_tree();
            }
            16 => keys.press(KeyCode::KeyB),
            _ => {}
        }
        smoke.fired = true;
    }
    if smoke.stage == 1 {
        keys.press(KeyCode::KeyW);
        keys.press(KeyCode::ShiftLeft);
    }
    if lab.ready
        && stream.has_ground(Vec2::new(horse.0.translation.x, horse.0.translation.z))
        && stream.loaded_count() == MAX_CHUNKS
        && stream.pending_count() == 0
    {
        // A test-only teleport requests a new area before its watershed exists.
        // Select the final dry position AFTER workers have prepared that area.
        if matches!(smoke.stage, 2 | 3) && !smoke.settled_spawn {
            let target = if smoke.stage == 2 {
                Vec2::new(1260., -880.)
            } else {
                Vec2::new(-1260., -1040.)
            };
            let p = crate::animals::safe_spawn(&world, target);
            assert!(world.walkable(p) && p.distance(target) < 40.);
            horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
            smoke.settled_spawn = true;
            smoke.elapsed = 0.;
        } else {
            smoke.elapsed += time.delta_secs().min(0.1);
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub fn verify(
    mut commands: Commands,
    mut smoke: ResMut<StreamSmoke>,
    stream: Res<StreamWorld>,
    world: Res<Meadow>,
    rig: Res<CameraRig>,
    layer: Res<MapLayer>,
    horse: Single<(&Transform, &HorseController)>,
    meshes: Res<Assets<Mesh>>,
    river: Res<crate::water::RiverMaterial>,
    materials: Res<Assets<StandardMaterial>>,
    mut exit: MessageWriter<AppExit>,
) {
    if smoke.elapsed < if smoke.stage == 1 { 2.0 } else { 1.0 } {
        return;
    }
    let p = Vec2::new(horse.0.translation.x, horse.0.translation.z);
    assert!(stream.loaded_count() <= MAX_CHUNKS);
    assert!(stream.pending_count() <= 2);
    assert!(stream.has_ground(p));
    assert!(
        (horse.0.translation.y - world.ground(p) - 0.025).abs() < 0.001,
        "Horse grounding at stage {}: {p:?}",
        smoke.stage
    );
    match smoke.stage {
        0 => {
            smoke.initial_meshes = meshes.len() - stream.mesh_asset_count();
            smoke.initial_height = world.ground(Vec2::new(27., -23.));
        }
        1 => {
            assert!(
                p.x > 104. && horse.1.speed > 8.,
                "Horse must ride across the chunk seam: {p:?}"
            );
            assert!(ChunkKey::at(p).0 >= 1);
        }
        2 | 3 => {
            assert!(p.length() > 1400.);
            assert!(stream.evicted >= MAX_CHUNKS as u64);
            assert!(
                meshes.len() == smoke.initial_meshes + stream.mesh_asset_count(),
                "Chunk mesh assets leaked"
            );
        }
        4 => {
            assert!(p.length() < 0.001);
            assert_eq!(world.ground(Vec2::new(27., -23.)), smoke.initial_height);
            assert!(stream.generated >= 3 * MAX_CHUNKS as u64);
            assert!(meshes.len() == smoke.initial_meshes + stream.mesh_asset_count());
        }
        5 => assert_eq!(rig.mode, ViewMode::Overview),
        6..=9 => assert_eq!(
            *layer,
            [
                MapLayer::Moisture,
                MapLayer::ForestDensity,
                MapLayer::Rockiness,
                MapLayer::Natural
            ][smoke.stage - 6]
        ),
        10 | 11 => {
            assert!(
                stream.water_vertices() > 0,
                "Streamed river surface is missing"
            );
            assert!(stream.watershed_count() <= 13);
            assert!(world.walkable(p));
            assert!(world.water_distance(p) < 45.);
            assert_eq!(rig.mode, ViewMode::ThirdPerson);
            let phase = materials.get(&river.0).unwrap().uv_transform.translation;
            if smoke.stage == 10 {
                smoke.river_phase = phase;
            } else {
                assert!(
                    phase.distance(smoke.river_phase) > 0.01,
                    "Water texture must keep flowing"
                );
            }
        }
        12 | 13 => {
            let generator = world.stream_generator().unwrap();
            assert_eq!(
                generator.landform(smoke.headwater_source),
                crate::terrain::Landform::Mountains
            );
            assert!(world.walkable(p));
            assert!(stream.water_vertices() > 0);
            if smoke.stage == 13 {
                assert_eq!(rig.mode, ViewMode::Overview);
            }
        }
        14 => {
            assert!(world.walkable(p));
            assert!(world.water_distance(p) < 20.);
        }
        15 | 16 => {
            assert!(world.walkable(p) && !stream.tree_blocks(p, None));
            assert!(stream.landmark_trees().count() > 0);
            if smoke.stage == 16 {
                assert_eq!(rig.mode, ViewMode::Overview);
            }
        }
        _ => {
            if smoke.saved < 17 {
                return;
            }
            println!(
                "STREAM SMOKE PASS: seam crossing, +/- distant coordinates, bounded 49 chunks / 2 tasks, asset eviction, deterministic revisit, cameras, map layers, shared rivers, bank view and mountain headwaters and calm lake, isolated landmark trees; 17 captures"
            );
            exit.write(AppExit::Success);
            return;
        }
    }
    let name = [
        "start",
        "chunk-boundary",
        "far-east",
        "far-negative",
        "return",
        "overview",
        "moisture",
        "forest",
        "rockiness",
        "natural",
        "river-bank",
        "river-flow",
        "headwater",
        "headwater-overview",
        "lake",
        "landmark-tree",
        "landmark-overview",
    ][smoke.stage];
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(format!(
            "captures/stream-seed-{}/{name}.png",
            world.seed
        )))
        .observe(|_: On<ScreenshotCaptured>, mut smoke: ResMut<StreamSmoke>| smoke.saved += 1);
    smoke.stage += 1;
    smoke.elapsed = 0.;
    smoke.fired = false;
    smoke.settled_spawn = false;
}

fn headwater_view(world: &Meadow) -> (Vec2, f32, Vec2) {
    let g = world.stream_generator().unwrap();
    let plan = g.hydrology.get((0, 0)).unwrap();
    let source = plan
        .segments
        .iter()
        .filter(|s| !plan.segments.iter().any(|r| r.b.p == s.a.p))
        .max_by(|a, b| g.base_height(a.a.p).total_cmp(&g.base_height(b.a.p)))
        .unwrap();
    let normal = (source.b.p - source.a.p).normalize().perp();
    for offset in [10., 16., 24., 32., 40.] {
        for side in [-1., 1.] {
            let p = source.a.p + normal * offset * side;
            if world.walkable(p) {
                let toward = source.a.p - p;
                return (p, (-toward.x).atan2(-toward.y), source.a.p);
            }
        }
    }
    panic!("Mountain headwater requires a dry observation point")
}

fn landmark_view(world: &Meadow, stream: &StreamWorld) -> (Vec2, f32) {
    let mut trees: Vec<_> = stream.landmark_trees().collect();
    trees.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.x.total_cmp(&b.0.x)));
    for (tree, height) in trees {
        for i in 0..24 {
            let p = tree
                + Vec2::from_angle(i as f32 * std::f32::consts::TAU / 24.) * (height * 2. + 30.);
            if stream.has_ground(p)
                && world.walkable(p)
                && !stream.tree_blocks(p, None)
                && world.ground(p) < world.ground(tree) + 4.
            {
                let d = tree - p;
                return (p, (-d.x).atan2(-d.y));
            }
        }
    }
    panic!("Needs an isolated landmark and a dry observation point")
}
