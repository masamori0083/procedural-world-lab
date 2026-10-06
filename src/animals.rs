use crate::{
    LabState,
    player::{CameraRig, HorseController},
    terrain::Meadow,
};
use bevy::{prelude::*, world_serialization::WorldInstanceReady};
use std::time::Duration;

pub const ACTOR_COUNT: usize = 8;

#[derive(Component)]
struct ModelAnimations {
    gltf: Handle<Gltf>,
    owner: Entity,
}

#[derive(Component)]
pub struct HorseVisual;

#[derive(Component)]
pub struct BoundAnimation {
    pub owner: Entity,
    nodes: [AnimationNodeIndex; 4],
    pub current: usize,
}

#[derive(Component)]
pub struct Wanderer {
    pub speed: f32,
    yaw: f32,
    timer: f32,
    phase: u32,
    eating: bool,
    home: Vec2,
}

pub fn spawn_model(
    commands: &mut Commands,
    assets: &AssetServer,
    owner: Entity,
    path: &'static str,
    scale: f32,
    is_player: bool,
) -> Entity {
    let mut model = commands.spawn((
        WorldAssetRoot(assets.load(GltfAssetLabel::Scene(0).from_asset(path))),
        Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::PI))
            .with_scale(Vec3::splat(scale)),
        ModelAnimations {
            gltf: assets.load(path),
            owner,
        },
    ));
    if is_player {
        model.insert(HorseVisual);
    }
    model.observe(on_model_ready);
    let child = model.id();
    commands.entity(owner).add_child(child);
    child
}

fn on_model_ready(
    ready: On<WorldInstanceReady>,
    mut commands: Commands,
    models: Query<&ModelAnimations>,
    children: Query<&Children>,
    gltfs: Res<Assets<Gltf>>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let Ok(model) = models.get(ready.entity) else {
        return;
    };
    let Some(gltf) = gltfs.get(&model.gltf) else {
        error!("Missing glTF metadata");
        return;
    };
    // The files contain combat clips, but the lab binds only peaceful actions.
    let mut clips = Vec::new();
    for name in ["Idle", "Walk", "Gallop", "Eating"] {
        let Some(clip) = gltf.named_animations.get(name) else {
            error!("Missing {name}");
            return;
        };
        clips.push(clip.clone());
    }
    let (graph, indices) = AnimationGraph::from_clips(clips);
    let graph = graphs.add(graph);
    for entity in children.iter_descendants(ready.entity) {
        if let Ok(mut player) = players.get_mut(entity) {
            let mut transitions = AnimationTransitions::new();
            transitions
                .play(&mut player, indices[0], Duration::ZERO)
                .repeat();
            commands.entity(entity).insert((
                AnimationGraphHandle(graph.clone()),
                transitions,
                BoundAnimation {
                    owner: model.owner,
                    nodes: indices.clone().try_into().expect("Four clips"),
                    current: 0,
                },
            ));
        }
    }
}

pub fn prepare(
    mut lab: ResMut<LabState>,
    bindings: Query<&BoundAnimation>,
    stream: Option<Res<crate::streaming::StreamWorld>>,
    horse: Single<&Transform, With<HorseController>>,
) {
    lab.ready = bindings.iter().count() == ACTOR_COUNT
        && stream.as_ref().is_none_or(|stream| {
            stream.has_ground(Vec2::new(horse.translation.x, horse.translation.z))
        });
}

pub fn safe_spawn(world: &Meadow, preferred: Vec2) -> Vec2 {
    if world.walkable(preferred) {
        return preferred;
    }
    for radius in 1..40 {
        for turn in 0..32 {
            let p = preferred
                + Vec2::from_angle(turn as f32 * std::f32::consts::TAU / 32.0) * radius as f32;
            if world.walkable(p) {
                return p;
            }
        }
    }
    Vec2::ZERO
}

pub fn setup(mut commands: Commands, assets: Res<AssetServer>, world: Res<Meadow>) {
    for (i, (path, point, scale)) in [
        ("models/Deer.gltf", Vec2::new(-10.0, -20.0), 0.46),
        ("models/Deer.gltf", Vec2::new(-16.0, -25.0), 0.42),
        ("models/Alpaca.gltf", Vec2::new(14.0, -13.0), 0.42),
        ("models/Alpaca.gltf", Vec2::new(20.0, -18.0), 0.39),
        ("models/Horse_White.gltf", Vec2::new(8.0, 6.0), 0.45),
        ("models/Wolf.gltf", Vec2::new(-8.0, -35.0), 0.52),
        ("models/Wolf.gltf", Vec2::new(7.0, -44.0), 0.52),
    ]
    .into_iter()
    .enumerate()
    {
        let p = safe_spawn(&world, point);
        let owner = commands
            .spawn((
                Name::new(path),
                Transform::from_xyz(p.x, world.ground(p) + 0.02, p.y),
                Visibility::default(),
                Wanderer {
                    speed: 0.0,
                    yaw: i as f32 * 1.3,
                    timer: 1.5 + i as f32,
                    phase: world.seed.wrapping_add(i as u32 * 997),
                    eating: i % 2 == 0,
                    home: p,
                },
            ))
            .id();
        spawn_model(&mut commands, &assets, owner, path, scale, false);
    }
}

pub fn wander(
    time: Res<Time>,
    rig: Res<CameraRig>,
    lab: Res<LabState>,
    world: Res<Meadow>,
    horse: Single<&Transform, (With<HorseController>, Without<Wanderer>)>,
    mut animals: Query<(&mut Transform, &mut Wanderer), Without<HorseController>>,
) {
    if !rig.captured || !lab.ready {
        return;
    }
    let dt = time.delta_secs().min(0.1);
    for (mut transform, mut animal) in &mut animals {
        animal.timer -= dt;
        if animal.timer <= 0.0 {
            animal.phase = animal.phase.wrapping_add(1);
            animal.eating = !animal.phase.is_multiple_of(3);
            animal.yaw += 0.8 + (animal.phase as f32 * 1.7).sin();
            animal.timer = if animal.eating { 5.0 } else { 8.0 };
        }
        let speed = if animal.eating { 0.0 } else { 1.1 };
        let next =
            transform.translation + Quat::from_rotation_y(animal.yaw) * Vec3::NEG_Z * speed * dt;
        let p = Vec2::new(next.x, next.z);
        if speed > 0.0
            && (!world.walkable(p)
                || p.distance(animal.home) > 20.0
                || next.distance(horse.translation) < 2.5)
        {
            animal.yaw += std::f32::consts::PI * 0.7;
            animal.speed = 0.0;
        } else {
            transform.translation = Vec3::new(p.x, world.ground(p) + 0.02, p.y);
            animal.speed = speed;
        }
        transform.rotation = Quat::from_rotation_y(animal.yaw);
    }
}

pub fn animate(
    rig: Res<CameraRig>,
    horses: Query<&HorseController>,
    animals: Query<&Wanderer>,
    mut players: Query<(
        &mut AnimationPlayer,
        &mut AnimationTransitions,
        &mut BoundAnimation,
    )>,
) {
    for (mut player, mut transitions, mut binding) in &mut players {
        let (clip, rate) = if let Ok(horse) = horses.get(binding.owner) {
            let speed = horse.speed.abs();
            if speed > 4.8 {
                (2, (speed / 8.0).clamp(0.7, 1.5))
            } else if speed > 0.08 {
                (1, (speed / 3.0).clamp(0.3, 1.5))
            } else {
                (0, 1.0)
            }
        } else if let Ok(animal) = animals.get(binding.owner) {
            if animal.speed > 0.1 {
                (1, 0.6)
            } else if animal.eating {
                (3, 1.0)
            } else {
                (0, 1.0)
            }
        } else {
            continue;
        };
        if binding.current != clip {
            transitions
                .play(&mut player, binding.nodes[clip], Duration::from_millis(180))
                .repeat();
            binding.current = clip;
        }
        if let Some(active) = player.animation_mut(binding.nodes[clip]) {
            active.set_speed(if rig.captured { rate } else { 0.0 });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::time::TimeUpdateStrategy;

    #[test]
    fn animal_spawn_is_dry_for_multiple_seeds() {
        for seed in [20261003, 42, 314159] {
            let map = Meadow::new(seed);
            for p in [
                Vec2::new(7.0, -44.0),
                Vec2::new(-16.0, -25.0),
                Vec2::new(20.0, -18.0),
            ] {
                assert!(map.walkable(safe_spawn(&map, p)));
            }
        }
    }

    #[test]
    fn wandering_stays_grounded_and_pauses_without_chasing_horse() {
        let mut app = App::new();
        let world = Meadow::new(20261003);
        let home = Vec2::new(-8.0, -35.0);
        let ground = world.ground(home);
        let mut rig = CameraRig::default();
        rig.captured = true;
        app.add_plugins(MinimalPlugins)
            .insert_resource(world)
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                1.0 / 60.0,
            )))
            .insert_resource(LabState {
                ready: true,
                ..default()
            })
            .insert_resource(rig)
            .add_systems(Update, wander);
        let horse = app
            .world_mut()
            .spawn((
                HorseController {
                    speed: 0.0,
                    yaw: 0.0,
                },
                Transform::from_xyz(0.0, ground, 0.0),
            ))
            .id();
        let animal = app
            .world_mut()
            .spawn((
                Transform::from_xyz(home.x, ground + 0.02, home.y),
                Wanderer {
                    speed: 0.0,
                    yaw: 0.0,
                    timer: 8.0,
                    phase: 0,
                    eating: false,
                    home,
                },
            ))
            .id();
        let origin = app.world().get::<Transform>(animal).unwrap().translation;
        for _ in 0..3600 {
            app.update();
        }
        let position = app.world().get::<Transform>(animal).unwrap().translation;
        let p = Vec2::new(position.x, position.z);
        let map = app.world().resource::<Meadow>();
        assert!(position.distance(origin) > 0.1);
        assert!(p.distance(home) <= 20.0);
        assert!(map.walkable(p));
        assert!((position.y - map.ground(p) - 0.02).abs() < 0.001);
        assert_eq!(
            app.world().get::<Transform>(horse).unwrap().translation,
            Vec3::new(0.0, ground, 0.0)
        );
        app.world_mut().resource_mut::<CameraRig>().captured = false;
        for _ in 0..120 {
            app.update();
        }
        assert_eq!(
            app.world().get::<Transform>(animal).unwrap().translation,
            position
        );
    }
}
