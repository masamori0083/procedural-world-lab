//! Native checks through the same controls and animation systems as the lab.
use crate::{
    LabState,
    animals::{ACTOR_COUNT, BoundAnimation, HorseVisual, Wanderer},
    environment::MapLayer,
    monitor::ResourcePanel,
    player::{CameraRig, FollowCamera, HorseController, ViewMode},
    terrain::Meadow,
    world::{Grass, TerrainSurface},
};
use bevy::{
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    window::PrimaryWindow,
};
use std::time::Instant;

type CameraEntities = (
    With<FollowCamera>,
    Without<HorseController>,
    Without<Wanderer>,
);

type EnvironmentChecks<'w, 's> = (
    Res<'w, MapLayer>,
    Res<'w, Assets<Mesh>>,
    Single<'w, 's, &'static Visibility, With<Grass>>,
);

#[derive(Resource)]
pub struct SmokeTest {
    start: Instant,
    elapsed: f32,
    stage: usize,
    fired: bool,
    saved: usize,
    origin: Vec3,
    paused_animals: Vec<(Entity, Vec3)>,
    flow_start: Option<Vec2>,
    expanded_start: Vec3,
}
impl Default for SmokeTest {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            elapsed: 0.0,
            stage: 0,
            fired: false,
            saved: 0,
            origin: Vec3::ZERO,
            paused_animals: vec![],
            flow_start: None,
            expanded_start: Vec3::ZERO,
        }
    }
}

pub fn drive(
    smoke: Option<ResMut<SmokeTest>>,
    time: Res<Time>,
    lab: Res<LabState>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut rig: ResMut<CameraRig>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
) {
    let Some(mut smoke) = smoke else {
        return;
    };
    assert!(smoke.start.elapsed().as_secs() < 120, "Lab smoke timed out");
    if !lab.ready {
        return;
    }
    window.focused = true;
    rig.captured = smoke.stage != 7;
    keys.reset_all();
    smoke.elapsed += time.delta_secs().min(0.1);
    match smoke.stage {
        1 => {
            keys.press(KeyCode::KeyW);
            keys.press(KeyCode::ShiftLeft);
        }
        2 | 3 if !smoke.fired => {
            keys.press(KeyCode::KeyV);
            smoke.fired = true;
        }
        4 => {
            keys.press(KeyCode::KeyS);
            keys.press(KeyCode::KeyD);
        }
        5 if !smoke.fired => {
            keys.press(KeyCode::KeyR);
            smoke.fired = true;
        }
        6 if !smoke.fired => {
            keys.press(KeyCode::KeyB);
            smoke.fired = true;
        }
        7 => {
            for key in [KeyCode::Space, KeyCode::KeyF, KeyCode::KeyC, KeyCode::KeyZ] {
                keys.press(key);
            }
        }
        10 if !smoke.fired => {
            keys.press(KeyCode::KeyL);
            smoke.fired = true;
        }
        12 if !smoke.fired => {
            rig.mode = ViewMode::Overview;
            smoke.fired = true;
        }
        13..=16 if !smoke.fired => {
            keys.press(KeyCode::F4);
            smoke.fired = true;
        }
        18 if !smoke.fired => {
            keys.press(KeyCode::KeyN);
            smoke.fired = true;
        }
        20..=22 if !smoke.fired => {
            keys.press(KeyCode::KeyN);
            smoke.fired = true;
        }
        19 => {
            keys.press(KeyCode::KeyW);
            keys.press(KeyCode::ShiftLeft);
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify(
    mut commands: Commands,
    mut smoke: ResMut<SmokeTest>,
    lab: Res<LabState>,
    rig: Res<CameraRig>,
    world: Res<Meadow>,
    horse: Single<(Entity, &Transform, &HorseController)>,
    mut camera: Single<&mut Transform, CameraEntities>,
    bindings: Query<&BoundAnimation>,
    visuals: Query<&Visibility, With<HorseVisual>>,
    animals: Query<(Entity, &Transform), With<Wanderer>>,
    water: Res<crate::water::RiverMaterial>,
    materials: Res<Assets<StandardMaterial>>,
    resource_panel: Single<&Text, With<ResourcePanel>>,
    terrain_mesh: Single<(&Mesh3d, &MeshMaterial3d<StandardMaterial>), With<TerrainSurface>>,
    environment: EnvironmentChecks<'_, '_>,
    mut exit: MessageWriter<AppExit>,
) {
    if !lab.ready {
        return;
    }
    let (layer, meshes, grass) = environment;
    assert_eq!(bindings.iter().count(), ACTOR_COUNT);
    assert_eq!(animals.iter().count(), ACTOR_COUNT - 1);
    assert!(bindings.iter().all(|a| a.current < 4));
    let (entity, transform, controller) = horse.into_inner();
    let animation = bindings.iter().find(|a| a.owner == entity).unwrap();
    assert!(camera.translation.is_finite());
    assert!(
        camera.translation.y
            >= world.ground(Vec2::new(camera.translation.x, camera.translation.z)) + 0.25
    );
    for (_, transform) in &animals {
        let p = Vec2::new(transform.translation.x, transform.translation.z);
        assert!(world.walkable(p));
        assert!((transform.translation.y - world.ground(p) - 0.02).abs() < 0.001);
    }
    if smoke.stage == 7 && smoke.paused_animals.is_empty() {
        smoke.paused_animals = animals.iter().map(|(e, t)| (e, t.translation)).collect();
    }
    let duration = match smoke.stage {
        0 => 2.0,
        1 => 3.0,
        2 | 3 | 4 | 6 | 8 | 9 | 18 => 1.5,
        19 => 2.0,
        5 => 0.7,
        _ => 1.0,
    };
    if smoke.elapsed < duration {
        return;
    }
    match smoke.stage {
        0 => {
            assert!(resource_panel.0.contains("RESOURCE MONITOR"));
            assert!(resource_panel.0.contains("MAP BUILD (one time)"));
            assert!(resource_panel.0.contains("LIVE (this app, 1 s)"));
            assert!(resource_panel.0.contains("GPU time / use"));
            smoke.origin = transform.translation;
            assert_eq!(animation.current, 0);
            capture(&mut commands, world.seed, "follow-camera.png");
        }
        1 => {
            assert!(transform.translation.distance(smoke.origin) > 12.0);
            assert!(controller.speed > 8.0);
            assert_eq!(animation.current, 2);
            capture(&mut commands, world.seed, "gallop.png");
        }
        2 => {
            assert_eq!(rig.mode, ViewMode::FirstPerson);
            assert!(visuals.iter().all(|v| *v == Visibility::Hidden));
            capture(&mut commands, world.seed, "first-person.png");
        }
        3 => {
            assert_eq!(rig.mode, ViewMode::ThirdPerson);
            assert!(visuals.iter().all(|v| *v == Visibility::Inherited));
        }
        4 => {
            assert!(controller.speed < -1.0);
            assert!(controller.yaw < -1.0);
            assert_eq!(animation.current, 1);
        }
        5 => {
            assert!(transform.translation.distance(smoke.origin) < 0.01);
            assert!(controller.speed.abs() < 0.01);
        }
        6 => {
            assert_eq!(rig.mode, ViewMode::Overview);
            assert!(camera.translation.y - transform.translation.y > 80.0);
            capture(&mut commands, world.seed, "map-overview.png");
        }
        7 => {
            assert!(!rig.captured);
            assert_eq!(animation.current, 0);
            for (entity, position) in &smoke.paused_animals {
                assert_eq!(animals.get(*entity).unwrap().1.translation, *position);
            }
        }
        8 | 9 => {
            let pond = &world.ponds[smoke.stage - 8];
            // Observe from the downhill bank at horse-eye height; this angle
            // exposed the old floating water disc that overhead views missed.
            let eye = (0..96)
                .map(|step| {
                    let direction = Vec2::from_angle(step as f32 * std::f32::consts::TAU / 96.0);
                    let p = pond.center + direction * pond.radii * 1.5;
                    Vec3::new(p.x, world.ground(p) + 2.5, p.y)
                })
                .min_by(|a, b| a.y.total_cmp(&b.y))
                .unwrap();
            **camera = Transform::from_translation(eye)
                .looking_at(Vec3::new(pond.center.x, pond.level, pond.center.y), Vec3::Y);
            capture(
                &mut commands,
                world.seed,
                if smoke.stage == 8 {
                    "pond-1-shore.png"
                } else {
                    "pond-2-shore.png"
                },
            );
        }
        10 => {
            let (bank, _) = world.river_bank();
            assert!(
                Vec2::new(transform.translation.x, transform.translation.z).distance(bank) < 0.01
            );
            assert!(world.walkable(bank));
            assert_eq!(rig.mode, ViewMode::ThirdPerson);
            smoke.flow_start = Some(materials.get(&water.0).unwrap().uv_transform.translation);
            capture(&mut commands, world.seed, "river-flow-1.png");
        }
        11 => {
            let offset = materials.get(&water.0).unwrap().uv_transform.translation;
            assert!(
                offset.distance(smoke.flow_start.unwrap()) > 0.01,
                "River texture must advance"
            );
            capture(&mut commands, world.seed, "river-flow-2.png");
        }
        12 => {
            let river = world.river.as_ref().unwrap();
            let point = river.points[river.points.len() / 2];
            let target = Vec3::new(point.position.x, point.level, point.position.y);
            **camera = Transform::from_translation(
                target
                    + Vec3::new(
                        0.0,
                        crate::terrain::EXTENT * 1.5,
                        crate::terrain::EXTENT * 0.3,
                    ),
            )
            .looking_at(target, Vec3::Y);
            capture(&mut commands, world.seed, "river-overview.png");
        }
        13..=16 => {
            use bevy::mesh::VertexAttributeValues;
            let expected = [
                MapLayer::Moisture,
                MapLayer::ForestDensity,
                MapLayer::Rockiness,
                MapLayer::Natural,
            ][smoke.stage - 13];
            assert_eq!(*layer, expected);
            assert_eq!(
                **grass,
                if expected == MapLayer::Natural {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                }
            );
            assert_eq!(
                materials.get(&terrain_mesh.1.0).unwrap().unlit,
                expected != MapLayer::Natural
            );
            assert_eq!(
                materials.get(&terrain_mesh.1.0).unwrap().fog_enabled,
                expected == MapLayer::Natural
            );
            let mesh = meshes.get(&terrain_mesh.0.0).unwrap();
            let Some(VertexAttributeValues::Float32x4(colors)) =
                mesh.attribute(Mesh::ATTRIBUTE_COLOR)
            else {
                panic!("Missing terrain colors")
            };
            assert_eq!(*colors, world.surface_colors(expected));
            let Some(VertexAttributeValues::Float32x3(positions)) =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION)
            else {
                panic!("Missing terrain positions")
            };
            for p in positions {
                assert!((world.ground(Vec2::new(p[0], p[2])) - p[1]).abs() < 0.001);
            }
            capture(
                &mut commands,
                world.seed,
                [
                    "moisture.png",
                    "tree-density.png",
                    "rockiness.png",
                    "environment-restored.png",
                ][smoke.stage - 13],
            );
        }
        17 => {
            let river = world.river.as_ref().unwrap();
            assert!(river.source_count >= 2);
            let join = river
                .reaches
                .iter()
                .map(|r| *r.last().unwrap())
                .filter(|end| river.reaches.iter().any(|r| r[0].position == end.position))
                .min_by(|a, b| {
                    a.position
                        .length_squared()
                        .total_cmp(&b.position.length_squared())
                })
                .expect("River network must have a confluence");
            let target = Vec3::new(join.position.x, join.level, join.position.y);
            **camera = Transform::from_translation(target + Vec3::new(0.0, 35.0, 22.0))
                .looking_at(target, Vec3::Y);
            capture(&mut commands, world.seed, "river-confluence.png");
        }
        18 => {
            smoke.expanded_start = transform.translation;
            assert!(Vec2::new(transform.translation.x, transform.translation.z).length() > 160.0);
            assert!(
                (transform.translation.y
                    - world.ground(Vec2::new(transform.translation.x, transform.translation.z))
                    - 0.025)
                    .abs()
                    < 0.001
            );
            capture(&mut commands, world.seed, "region-east.png");
        }
        19 => {
            assert!(transform.translation.distance(smoke.expanded_start) > 8.0);
            assert!(controller.speed > 8.0);
            assert_eq!(animation.current, 2);
            capture(&mut commands, world.seed, "region-east-gallop.png");
        }
        20..=22 => {
            let view = smoke.stage - 19;
            let (p, _) = world.exploration_view(view);
            assert!(Vec2::new(transform.translation.x, transform.translation.z).distance(p) < 0.01);
            capture(
                &mut commands,
                world.seed,
                ["region-north.png", "region-west.png", "region-south.png"][view - 1],
            );
        }
        _ => {
            if smoke.saved < 19 {
                return;
            }
            println!(
                "MAP LAB SMOKE PASS: eight animated actors, peaceful wildlife, movement/cameras/pause, pond and river banks, moving river texture, reversible environment layers, river confluence, four-region exploration and nineteen captures"
            );
            exit.write(AppExit::Success);
            return;
        }
    }
    smoke.stage += 1;
    smoke.elapsed = 0.0;
    smoke.fired = false;
}

fn capture(commands: &mut Commands, seed: u32, name: &'static str) {
    let path = format!("captures/seed-{seed}/{name}");
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path))
        .observe(|_: On<ScreenshotCaptured>, mut smoke: ResMut<SmokeTest>| smoke.saved += 1);
}
