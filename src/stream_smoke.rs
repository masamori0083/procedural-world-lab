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
    frames_ms: Vec<f64>,
    transition_frames_ms: Vec<f64>,
    transition_done: bool,
    report_rows: Vec<String>,
    river_phase: Vec2,
    headwater_source: Vec2,
    sand_start: Vec2,
    ford: Option<crate::watershed::Ford>,
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
            frames_ms: Vec::new(),
            transition_frames_ms: Vec::new(),
            transition_done: false,
            report_rows: Vec::new(),
            river_phase: Vec2::ZERO,
            headwater_source: Vec2::ZERO,
            sand_start: Vec2::ZERO,
            ford: None,
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub fn drive(
    smoke: Option<ResMut<StreamSmoke>>,
    stream: Option<Res<StreamWorld>>,
    time: Res<Time>,
    real_time: Res<Time<Real>>,
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
                        plan.lake
                            + Vec2::from_angle(n as f32 * std::f32::consts::TAU / 24.)
                                * (crate::watershed::LAKE_RADIUS + 13.)
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
            17 | 18 => keys.press(KeyCode::KeyV),
            19 => {
                let (p, yaw) = regional_river_view(&world);
                horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
                horse.1.speed = 0.;
                horse.1.yaw = yaw;
                horse.0.rotation = Quat::from_rotation_y(yaw);
                rig.mode = ViewMode::ThirdPerson;
            }
            20 => keys.press(KeyCode::KeyB),
            21 => {
                let (p, yaw) = lake_inlet_view(&world);
                horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
                horse.1.speed = 0.;
                horse.1.yaw = yaw;
                horse.0.rotation = Quat::from_rotation_y(yaw);
                *rig = CameraRig::default();
                rig.captured = true;
            }
            22 => keys.press(KeyCode::KeyB),
            23 => {
                let (p, yaw) = confluence_view(&world);
                horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
                horse.1.speed = 0.;
                horse.1.yaw = yaw;
                horse.0.rotation = Quat::from_rotation_y(yaw);
                *rig = CameraRig::default();
                rig.captured = true;
            }
            24 => keys.press(KeyCode::KeyB),
            25 => {
                let (p, yaw) = sand_bank_view(&world);
                smoke.sand_start = p;
                horse.0.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
                horse.1.speed = 0.;
                horse.1.yaw = yaw;
                horse.0.rotation = Quat::from_rotation_y(yaw);
                *rig = CameraRig::default();
                rig.captured = true;
            }
            27 => {
                let lake = world
                    .stream_generator()
                    .unwrap()
                    .hydrology
                    .get((0, 0))
                    .unwrap()
                    .lake;
                let p = Vec2::new(horse.0.translation.x, horse.0.translation.z);
                let forward = (lake - p).normalize();
                horse.1.yaw = (-forward.x).atan2(-forward.y);
                horse.1.speed = 0.;
            }
            28 => {
                let p = Vec2::new(horse.0.translation.x, horse.0.translation.z);
                smoke.ford = world.stream_generator().unwrap().nearest_ford(p);
                assert!(smoke.ford.is_some(), "No ford available for crossing");
                keys.press(KeyCode::KeyG);
            }
            31 => keys.press(KeyCode::KeyB),
            _ => {}
        }
        smoke.fired = true;
    }
    if matches!(smoke.stage, 1 | 26 | 27) {
        keys.press(KeyCode::KeyW);
        if smoke.stage == 1 {
            keys.press(KeyCode::ShiftLeft);
        }
    }
    if matches!(smoke.stage, 29 | 30) {
        let ford = smoke.ford.unwrap();
        let p = Vec2::new(horse.0.translation.x, horse.0.translation.z);
        let across = (p - ford.center).dot(ford.across);
        let target = if smoke.stage == 29 {
            0.
        } else {
            ford.width + 15.
        };
        if across < target {
            keys.press(KeyCode::KeyW);
        } else {
            horse.1.speed = 0.;
        }
    }
    if (!smoke.transition_done || stream.pending_count() > 0 || stream.loaded_count() < MAX_CHUNKS)
        && real_time.delta_secs_f64() > 0.
    {
        smoke
            .transition_frames_ms
            .push(real_time.delta_secs_f64() * 1000.);
    }
    if lab.ready
        && stream.has_ground(Vec2::new(horse.0.translation.x, horse.0.translation.z))
        && stream.loaded_count() == MAX_CHUNKS
        && stream.pending_count() == 0
    {
        smoke.transition_done = true;
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
            smoke.frames_ms.push(real_time.delta_secs_f64() * 1000.);
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
    config: Res<crate::vegetation::LodConfig>,
    terrain_config: Res<crate::terrain_lod::Config>,
    mut exit: MessageWriter<AppExit>,
) {
    if smoke.elapsed
        < if matches!(smoke.stage, 1 | 27) {
            2.0
        } else {
            1.0
        }
    {
        return;
    }
    let p = Vec2::new(horse.0.translation.x, horse.0.translation.z);
    if matches!(smoke.stage, 29 | 30) {
        let ford = smoke.ford.unwrap();
        let target = if smoke.stage == 29 {
            0.
        } else {
            ford.width + 15.
        };
        if (p - ford.center).dot(ford.across) < target {
            return;
        }
    }
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
            let (actual, full) = stream.terrain_triangles();
            if terrain_config.enabled {
                assert!(actual < full, "Terrain LOD must reduce geometry at spawn");
                let counts = stream.terrain_lod_counts();
                assert!(
                    counts.iter().all(|n| *n > 0),
                    "Three terrain levels must be exercised"
                );
            } else {
                assert_eq!(actual, full);
            }
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
        17 => assert_eq!(rig.mode, ViewMode::FirstPerson),
        18 => assert_eq!(rig.mode, ViewMode::ThirdPerson),
        19..=24 => {
            assert!(world.walkable(p));
            assert!(stream.water_vertices() > 0);
            assert!(world.water_distance(p) < 45.);
            if matches!(smoke.stage, 20 | 22 | 24) {
                assert_eq!(rig.mode, ViewMode::Overview);
            }
        }
        25..=27 => {
            assert!(world.walkable(p) && !stream.tree_blocks(p, None));
            let g = world.stream_generator().unwrap();
            assert!(g.water_depth(p) < -0.015);
            assert!(p.distance(smoke.sand_start) < 6.);
            if smoke.stage == 26 {
                assert!(
                    p.distance(smoke.sand_start) > 1.5,
                    "Horse must walk along dry sand"
                );
            } else if smoke.stage == 27 {
                let lake = g.hydrology.get((0, 0)).unwrap().lake;
                let inward = (lake - p).normalize();
                assert_eq!(horse.1.speed, 0.);
                assert!(
                    !world.walkable(p + inward * 0.4),
                    "Horse must stop at actual shoreline"
                );
            }
        }
        28..=31 => {
            let ford = smoke.ford.unwrap();
            assert!(world.walkable(p) && !stream.tree_blocks(p, None));
            if smoke.stage == 28 {
                assert!(p.distance(ford.entry()) < 0.01);
            }
            if smoke.stage == 29 {
                let depth = world.stream_generator().unwrap().water_depth(p);
                assert!(
                    depth > 0.015 && depth <= crate::watershed::MAX_WADING_DEPTH,
                    "Horse must be standing in shallow flowing water: {depth}"
                );
                assert!((p - ford.center).dot(ford.across).abs() < 0.2);
            }
            if matches!(smoke.stage, 30 | 31) {
                assert!((p - ford.center).dot(ford.across) >= ford.width + 15.);
                assert!(world.stream_generator().unwrap().water_depth(p) < 0.);
            }
            if smoke.stage == 31 {
                assert_eq!(rig.mode, ViewMode::Overview);
            }
        }
        _ => {
            if smoke.saved < 32 {
                return;
            }
            std::fs::create_dir_all("reports").expect("Cannot create smoke reports");
            let mode = format!(
                "{}{}{}",
                if config.enabled { "lod" } else { "full" },
                if terrain_config.enabled {
                    ""
                } else {
                    "-terrain-full"
                },
                if stream.scheduler_name() == "DISTANCE" {
                    "-distance"
                } else {
                    ""
                }
            );
            std::fs::write(format!("reports/stream-smoke-seed-{}-{mode}.csv", world.seed),
                format!("view,settled_frames,mean_frame_ms,max_frame_ms,resident_cpu_mib,vegetation_triangles,full_vegetation_triangles,tree_near,tree_middle,tree_far,grass_near,grass_middle,grass_far,lod_rebuilds,last_lod_worker_ms,last_lod_install_ms,terrain_near,terrain_middle,terrain_far,terrain_triangles,full_terrain_triangles,terrain_rebuilds,last_terrain_worker_ms,scheduler,queued,predicted_x,predicted_z,ground_wait_ms,max_ground_wait_ms,ground_wait_events,near_wait_ms,last_near_wait_ms,max_near_wait_ms,transition_frames,max_transition_frame_ms\n{}\n", smoke.report_rows.join("\n"))).expect("Cannot save smoke metrics");
            println!(
                "STREAM SMOKE PASS: seam crossing, +/- distant coordinates, bounded 49 chunks / 2 tasks, asset eviction, deterministic revisit, cameras, map layers, shared rivers, bank view and mountain headwaters and calm lake, isolated landmark trees, vegetation LOD and FPS/follow switching, regional river connection, lake inlet, curved confluence, dry sand walking, water stop and walking across a shallow ford; 32 captures"
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
        "lod-first-person",
        "lod-follow-return",
        "regional-river",
        "regional-river-overview",
        "lake-inlet",
        "lake-inlet-overview",
        "confluence",
        "confluence-overview",
        "sand-bank",
        "sand-bank-walk",
        "shore-stop",
        "ford-entry",
        "ford-wading",
        "ford-crossed",
        "ford-overview",
    ][smoke.stage];
    let (actual, full) = stream.vegetation_vertices();
    let (t, g) = stream.lod_counts();
    let mean = smoke.frames_ms.iter().sum::<f64>() / smoke.frames_ms.len().max(1) as f64;
    let maximum = smoke.frames_ms.iter().copied().fold(0., f64::max);
    let mut row = format!(
        "{name},{},{mean:.3},{maximum:.3},{:.3},{},{},{},{},{},{},{},{},{},{:.3},{:.3}",
        smoke.frames_ms.len(),
        stream.mesh_mib(),
        actual / 3,
        full / 3,
        t[0],
        t[1],
        t[2],
        g[0],
        g[1],
        g[2],
        stream.lod_rebuilt,
        stream.last_lod_ms,
        stream.last_lod_install_ms
    );
    let terrain = stream.terrain_lod_counts();
    let (triangles, full_triangles) = stream.terrain_triangles();
    row.push_str(&format!(
        ",{},{},{},{},{},{},{:.3}",
        terrain[0],
        terrain[1],
        terrain[2],
        triangles,
        full_triangles,
        stream.terrain_rebuilt,
        stream.last_terrain_ms
    ));
    row.push_str(&format!(
        ",{},{},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{},{:.3}",
        stream.scheduler_name(),
        stream.queued,
        stream.predicted.x,
        stream.predicted.y,
        stream.ground_wait_ms,
        stream.max_ground_wait_ms,
        stream.ground_wait_events,
        stream.near_wait_ms,
        stream.last_near_wait_ms,
        stream.max_near_wait_ms,
        smoke.transition_frames_ms.len(),
        smoke
            .transition_frames_ms
            .iter()
            .copied()
            .fold(0., f64::max)
    ));
    smoke.report_rows.push(row);
    smoke.frames_ms.clear();
    smoke.transition_frames_ms.clear();
    smoke.transition_done = false;
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

fn sand_bank_view(world: &Meadow) -> (Vec2, f32) {
    let g = world.stream_generator().unwrap();
    let lake = g.hydrology.get((0, 0)).unwrap().lake;
    for i in 0..64 {
        let direction = Vec2::from_angle(i as f32 * std::f32::consts::TAU / 64.);
        let p = lake + direction * 28.8;
        let forward = direction.perp();
        let neighbours = g.tree_neighbours(ChunkKey::at(p));
        let clear = |q: Vec2| neighbours.iter().all(|(t, _)| t.distance(q) > 2.5);
        if (0..=20).all(|step| {
            let q = p + forward * step as f32 * 0.2;
            world.walkable(q) && clear(q)
        }) && (0..=16).all(|step| clear(p - direction * step as f32 * 0.2))
        {
            return (p, (-forward.x).atan2(-forward.y));
        }
    }
    panic!("No dry sandy lake bank for native walk test");
}

fn confluence_view(world: &Meadow) -> (Vec2, f32) {
    let g = world.stream_generator().unwrap();
    let plan = g.hydrology.get((0, 0)).unwrap();
    for segment in &plan.segments {
        let joint = segment.a.p;
        if joint.distance(plan.lake) < crate::watershed::LAKE_RADIUS + 80.
            || plan.segments.iter().filter(|s| s.b.p == joint).count() < 2
        {
            continue;
        }
        let normal = (segment.b.p - joint).normalize().perp();
        for offset in [16., 24., 32., 40.] {
            for side in [-1., 1.] {
                let p = joint + normal * offset * side;
                if world.walkable(p) && world.water_distance(p) < 45. {
                    let toward = joint - p;
                    return (p, (-toward.x).atan2(-toward.y));
                }
            }
        }
    }
    panic!("Confluence requires a dry observation point")
}

fn lake_inlet_view(world: &Meadow) -> (Vec2, f32) {
    let g = world.stream_generator().unwrap();
    let plan = g.hydrology.get((0, 0)).unwrap();
    for segment in &plan.segments {
        if segment.a.p.distance(plan.lake) <= crate::watershed::LAKE_RADIUS
            || segment.b.p.distance(plan.lake) > crate::watershed::LAKE_RADIUS
        {
            continue;
        }
        let mut lo = 0.;
        let mut hi = 1.;
        for _ in 0..16 {
            let t = (lo + hi) * 0.5;
            if segment.a.p.lerp(segment.b.p, t).distance(plan.lake) > crate::watershed::LAKE_RADIUS
            {
                lo = t;
            } else {
                hi = t;
            }
        }
        let inlet = segment.a.p.lerp(segment.b.p, (lo + hi) * 0.5);
        let normal = (segment.b.p - segment.a.p).normalize().perp();
        for offset in [16., 24., 32., 40.] {
            for side in [-1., 1.] {
                let p = inlet + normal * offset * side;
                if world.walkable(p) {
                    let toward = inlet - p;
                    return (p, (-toward.x).atan2(-toward.y));
                }
            }
        }
    }
    panic!("Lake inlet requires a dry observation point")
}

fn regional_river_view(world: &Meadow) -> (Vec2, f32) {
    let g = world.stream_generator().unwrap();
    let plan = g.hydrology.get((0, 0)).unwrap();
    let mut crossings: Vec<_> = plan
        .segments
        .iter()
        .filter(|s| crate::watershed::key(s.a.p) != crate::watershed::key(s.b.p))
        .collect();
    crossings.sort_by(|a, b| a.a.p.length_squared().total_cmp(&b.a.p.length_squared()));
    for segment in crossings {
        let middle = segment.a.p.lerp(segment.b.p, 0.5);
        let normal = (segment.b.p - segment.a.p).normalize().perp();
        for offset in [16., 24., 32., 40.] {
            for side in [-1., 1.] {
                let p = middle + normal * offset * side;
                if world.walkable(p) {
                    let toward = middle - p;
                    return (p, (-toward.x).atan2(-toward.y));
                }
            }
        }
    }
    panic!("Regional river requires a dry observation point")
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
