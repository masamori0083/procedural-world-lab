use crate::{
    LabState,
    animals::{self, HorseVisual},
    terrain::{EXTENT, Meadow, PLAY_RADIUS},
};
use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};

#[derive(Component)]
pub struct HorseController {
    pub speed: f32,
    pub yaw: f32,
}

#[derive(Component)]
pub struct FollowCamera;

type FollowView<'w, 's> = Single<
    'w,
    's,
    (&'static mut Transform, &'static mut DistanceFog),
    (With<FollowCamera>, Without<HorseController>),
>;

#[derive(Component)]
pub(crate) struct Hud;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    FirstPerson,
    ThirdPerson,
    Overview,
}

#[derive(Resource)]
pub struct CameraRig {
    pub mode: ViewMode,
    pub captured: bool,
    first_pitch: f32,
    third_pitch: f32,
    distance: f32,
    snap: bool,
    overview_height: f32,
    region_index: usize,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self {
            mode: ViewMode::ThirdPerson,
            captured: false,
            first_pitch: 0.0,
            third_pitch: 0.27,
            distance: 7.5,
            snap: true,
            overview_height: EXTENT * 1.5,
            region_index: 0,
        }
    }
}

impl CameraRig {
    pub(crate) fn frame_tree(&mut self) {
        self.mode = ViewMode::ThirdPerson;
        self.third_pitch = 0.;
        self.distance = 14.;
        self.snap = true;
    }

    pub fn toggle(&mut self) {
        self.mode = match self.mode {
            ViewMode::FirstPerson => ViewMode::ThirdPerson,
            ViewMode::ThirdPerson | ViewMode::Overview => ViewMode::FirstPerson,
        };
        self.snap = true;
    }
    fn pitch(&self) -> f32 {
        match self.mode {
            ViewMode::FirstPerson => self.first_pitch,
            ViewMode::ThirdPerson | ViewMode::Overview => self.third_pitch,
        }
    }
}

pub fn setup(mut commands: Commands, assets: Res<AssetServer>, world: Res<Meadow>) {
    let owner = commands
        .spawn((
            HorseController {
                speed: 0.0,
                yaw: 0.0,
            },
            Transform::from_xyz(0.0, world.ground(Vec2::ZERO) + 0.025, 0.0),
            Visibility::default(),
        ))
        .id();
    animals::spawn_model(
        &mut commands,
        &assets,
        owner,
        "models/Horse.gltf",
        0.45,
        true,
    );
    commands.spawn((
        FollowCamera,
        Camera3d::default(),
        Projection::from(PerspectiveProjection {
            fov: 72.0_f32.to_radians(),
            near: 0.08,
            far: EXTENT * 5.0,
            ..default()
        }),
        Transform::from_xyz(0.0, 5.0, 8.0),
        DistanceFog {
            color: Color::srgb(0.64, 0.78, 0.85),
            falloff: FogFalloff::Linear {
                start: 75.0,
                end: EXTENT * 1.5,
            },
            ..default()
        },
    ));
    commands.spawn((
        Hud,
        Text::new("Loading animals..."),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..default()
        },
        TextColor(Color::srgb(0.93, 0.96, 0.90)),
        Node {
            position_type: PositionType::Absolute,
            left: px(20),
            bottom: px(18),
            padding: UiRect::all(px(14)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.055, 0.095, 0.065, 0.88)),
    ));
}

#[allow(clippy::too_many_arguments)]
pub fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mouse: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
    world: Res<Meadow>,
    stream: Option<Res<crate::streaming::StreamWorld>>,
    lab: Res<LabState>,
    mut rig: ResMut<CameraRig>,
    window: Single<(&Window, &mut CursorOptions), With<PrimaryWindow>>,
    horse: Single<(&mut Transform, &mut HorseController)>,
) {
    let (window, mut cursor) = window.into_inner();
    let (mut transform, mut horse) = horse.into_inner();
    if !window.focused || keys.just_pressed(KeyCode::Escape) {
        rig.captured = false;
    } else if buttons.just_pressed(MouseButton::Left) {
        rig.captured = true;
    }
    cursor.visible = !rig.captured;
    cursor.grab_mode = if rig.captured {
        CursorGrabMode::Locked
    } else {
        CursorGrabMode::None
    };
    if !window.focused {
        horse.speed = 0.0;
        return;
    }
    if keys.just_pressed(KeyCode::KeyV) {
        rig.toggle();
    }
    if keys.just_pressed(KeyCode::KeyB) {
        rig.mode = if rig.mode == ViewMode::Overview {
            ViewMode::ThirdPerson
        } else {
            ViewMode::Overview
        };
        rig.snap = true;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        transform.translation = Vec3::new(0.0, world.ground(Vec2::ZERO) + 0.025, 0.0);
        horse.speed = 0.0;
        horse.yaw = 0.0;
        transform.rotation = Quat::IDENTITY;
        rig.snap = true;
    }
    if keys.just_pressed(KeyCode::KeyG) && lab.ready {
        let p = Vec2::new(transform.translation.x, transform.translation.z);
        if let Some(ford) = world.stream_generator().and_then(|g| g.nearest_ford(p)) {
            let p = ford.entry();
            transform.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
            horse.speed = 0.;
            horse.yaw = (-ford.across.x).atan2(-ford.across.y);
            transform.rotation = Quat::from_rotation_y(horse.yaw);
            rig.mode = ViewMode::ThirdPerson;
            rig.snap = true;
        }
    }
    if keys.just_pressed(KeyCode::KeyL) && lab.ready {
        let bank = if let Some(s) = &stream {
            s.river_bank(
                &world,
                Vec2::new(transform.translation.x, transform.translation.z),
            )
        } else {
            Some(world.river_bank())
        };
        let Some((p, yaw)) = bank else {
            return;
        };
        transform.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
        horse.speed = 0.0;
        horse.yaw = yaw;
        transform.rotation = Quat::from_rotation_y(yaw);
        rig.mode = ViewMode::ThirdPerson;
        rig.snap = true;
    }
    if keys.just_pressed(KeyCode::KeyN) && lab.ready && !world.is_streaming() {
        let (p, yaw) = world.exploration_view(rig.region_index);
        rig.region_index = (rig.region_index + 1) % 4;
        transform.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
        horse.yaw = yaw;
        horse.speed = 0.0;
        transform.rotation = Quat::from_rotation_y(yaw);
        rig.mode = ViewMode::ThirdPerson;
        rig.snap = true;
    }
    if !rig.captured || !lab.ready {
        horse.speed = 0.0;
        return;
    }
    let dt = time.delta_secs().min(0.1);
    let turn = f32::from(keys.pressed(KeyCode::KeyA)) - f32::from(keys.pressed(KeyCode::KeyD));
    horse.yaw += turn * dt * 1.6 - mouse.delta.x * 0.0022;
    match rig.mode {
        ViewMode::FirstPerson => {
            rig.first_pitch = (rig.first_pitch + mouse.delta.y * 0.0018).clamp(-1.1, 1.1)
        }
        ViewMode::ThirdPerson => {
            rig.third_pitch = (rig.third_pitch + mouse.delta.y * 0.0018).clamp(-0.08, 0.85)
        }
        ViewMode::Overview => {
            rig.overview_height =
                (rig.overview_height - scroll.delta.y * 4.0).clamp(30.0, EXTENT * 1.8)
        }
    }
    if rig.mode != ViewMode::Overview {
        rig.distance = (rig.distance - scroll.delta.y * 0.6).clamp(4.0, 14.0);
    }
    let input = f32::from(keys.pressed(KeyCode::KeyW)) - f32::from(keys.pressed(KeyCode::KeyS));
    let running = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let target = input
        * if input < 0.0 {
            2.0
        } else if running {
            9.0
        } else {
            3.5
        };
    horse.speed = move_towards(
        horse.speed,
        target,
        dt * if input == 0.0 { 12.0 } else { 7.0 },
    );
    transform.rotation = Quat::from_rotation_y(horse.yaw);
    let displacement = transform.rotation * Vec3::NEG_Z * horse.speed * dt;
    let steps = (displacement.length() / 0.2).ceil().max(1.0) as usize;
    for _ in 0..steps {
        let next = transform.translation + displacement / steps as f32;
        let p = Vec2::new(next.x, next.z);
        if world.walkable(p)
            && stream
                .as_ref()
                .is_none_or(|stream| stream.has_ground(p) && !stream.tree_blocks(p, None))
        {
            transform.translation = Vec3::new(p.x, world.ground(p) + 0.025, p.y);
        } else {
            horse.speed = 0.0;
            break;
        }
    }
}

fn move_towards(current: f32, target: f32, amount: f32) -> f32 {
    current + (target - current).clamp(-amount, amount)
}

fn first_person_position(horse: &Transform) -> Vec3 {
    horse.translation + horse.rotation * Vec3::new(0.0, 1.95, -1.35)
}

#[allow(clippy::too_many_arguments)]
pub fn follow_camera(
    time: Res<Time>,
    world: Res<Meadow>,
    stream: Option<Res<crate::streaming::StreamWorld>>,
    mut rig: ResMut<CameraRig>,
    horse: Single<&Transform, (With<HorseController>, Without<FollowCamera>)>,
    camera: FollowView<'_, '_>,
    mut visuals: Query<&mut Visibility, With<HorseVisual>>,
) {
    let (mut camera, mut fog) = camera.into_inner();
    // Preserve aerial map readability while keeping depth haze on the ground.
    if world.is_streaming() {
        rig.overview_height = rig.overview_height.min(140.);
    }
    fog.falloff = if world.is_streaming() {
        FogFalloff::Linear {
            start: if rig.mode == ViewMode::Overview {
                180.
            } else {
                100.
            },
            end: 260.,
        }
    } else if rig.mode == ViewMode::Overview {
        FogFalloff::Linear {
            start: EXTENT * 3.0,
            end: EXTENT * 5.0,
        }
    } else {
        FogFalloff::Linear {
            start: 75.0,
            end: EXTENT * 1.5,
        }
    };
    for mut visibility in &mut visuals {
        *visibility = if rig.mode == ViewMode::FirstPerson {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
    }
    let pitch = rig.pitch();
    match rig.mode {
        ViewMode::FirstPerson => {
            camera.translation = first_person_position(&horse);
            let (yaw, _, _) = horse.rotation.to_euler(EulerRot::YXZ);
            camera.rotation = Quat::from_euler(EulerRot::YXZ, yaw, -pitch, 0.0);
        }
        ViewMode::ThirdPerson => {
            let focus = horse.translation + Vec3::Y * 1.25;
            let offset = horse.rotation
                * Vec3::new(0.0, rig.distance * pitch.sin(), rig.distance * pitch.cos());
            let desired = focus + offset;
            // Shorten the boom at terrain or tree intersections.
            let target = clear_camera_boom(focus, desired, &world, stream.as_deref());
            let blend = if rig.snap {
                1.0
            } else {
                1.0 - (-12.0 * time.delta_secs()).exp()
            };
            camera.translation = camera.translation.lerp(target, blend);
            camera.translation.y = camera
                .translation
                .y
                .max(world.ground(Vec2::new(camera.translation.x, camera.translation.z)) + 0.35);
            camera.look_at(focus + horse.rotation * Vec3::NEG_Z * 1.0, Vec3::Y);
        }
        ViewMode::Overview => {
            let focus = horse.translation;
            camera.translation =
                focus + Vec3::new(0.0, rig.overview_height, rig.overview_height * 0.45);
            camera.look_at(focus, Vec3::Y);
        }
    }
    rig.snap = false;
}

fn clear_camera_boom(
    focus: Vec3,
    desired: Vec3,
    world: &Meadow,
    stream: Option<&crate::streaming::StreamWorld>,
) -> Vec3 {
    let mut safe = focus;
    for step in 1..=48 {
        let p = focus.lerp(desired, step as f32 / 48.0);
        let horizontal = Vec2::new(p.x, p.z);
        let blocked_by_tree = world.trees.iter().any(|(center, size)| {
            horizontal.distance(*center) < size * 0.45 + 0.3
                && p.y < world.ground(*center) + 4.0 * size
        });
        if p.y < world.ground(horizontal) + 0.35
            || blocked_by_tree
            || stream.is_some_and(|stream| stream.tree_blocks(horizontal, Some(p.y)))
        {
            break;
        }
        safe = p;
    }
    safe
}

#[allow(clippy::too_many_arguments)]
pub fn update_hud(
    rig: Res<CameraRig>,
    lab: Res<LabState>,
    world: Res<Meadow>,
    stream: Option<Res<crate::streaming::StreamWorld>>,
    settings: Res<crate::trees::TreeSettings>,
    layer: Res<crate::environment::MapLayer>,
    horse: Single<(&Transform, &HorseController)>,
    mut hud: Single<&mut Text, With<Hud>>,
) {
    let (position, horse) = horse.into_inner();
    let view = match rig.mode {
        ViewMode::FirstPerson => "FIRST PERSON",
        ViewMode::ThirdPerson => "FOLLOW CAMERA",
        ViewMode::Overview => "MAP OVERVIEW",
    };
    let status = if !lab.ready {
        "Loading models..."
    } else if rig.captured {
        "Explore / peaceful wildlife"
    } else {
        "Click to explore / resume"
    };
    if let Some(stream) = stream {
        let (tree_lod, grass_lod) = stream.lod_counts();
        let (vertices, full) = stream.vegetation_vertices();
        let terrain_lod = stream.terrain_lod_counts();
        let (terrain_triangles, full_terrain_triangles) = stream.terrain_triangles();
        hud.0 = format!(
            "MAP GENERATION LAB | STREAMING v{} | {view} | {status}\nSeed {}   Position {:.0}, {:.0} m   {}\nChunks {} / {}   Pending {} / 2   Generated {}   Evicted {}\nResident mesh + LOD data {:.1} MiB (CPU estimate)   Trees {}\nTree LOD N/M/F {}/{}/{}   Grass {}/{}/{}\nVegetation triangles {} / {} full   Rebuilds {} ({:.1} ms)\nTerrain LOD N/M/F {}/{}/{}   Triangles {} / {} full\nTerrain rebuilds {} ({:.1} ms)\nWatersheds {}   River sources {}   Water triangles {}\nLast chunk worker {:.1} ms   Install {:.2} ms\nW/S Move   A/D + Mouse Turn   Shift Run   V FPS / Follow\nB Overview   Wheel Zoom   Esc Pause   R Start   L River bank   G Ford\nF3 Resources   F4 Map layers\n{}",
            crate::streaming::VERSION,
            world.seed,
            position.translation.x,
            position.translation.z,
            world
                .landform(Vec2::new(position.translation.x, position.translation.z))
                .label(),
            stream.loaded_count(),
            crate::streaming::MAX_CHUNKS,
            stream.pending_count(),
            stream.generated,
            stream.evicted,
            stream.mesh_mib(),
            stream.tree_count(),
            tree_lod[0],
            tree_lod[1],
            tree_lod[2],
            grass_lod[0],
            grass_lod[1],
            grass_lod[2],
            vertices / 3,
            full / 3,
            stream.lod_rebuilt,
            stream.last_lod_ms,
            terrain_lod[0],
            terrain_lod[1],
            terrain_lod[2],
            terrain_triangles,
            full_terrain_triangles,
            stream.terrain_rebuilt,
            stream.last_terrain_ms,
            stream.watershed_count(),
            stream.river_sources(),
            stream.water_vertices() / 3,
            stream.last_build_ms,
            stream.last_install_ms,
            layer.legend()
        );
        return;
    }
    let text = format!(
        "MAP GENERATION LAB  |  {view}  |  {status}\nSeed {}   Mountains {:.2}   Generator v{}\nMap {:.0} x {:.0} m   Walk radius {:.0} m   {}\nElevation {:.1} to {:.1} m   Trees {}   Rivers {} sources / {:.0} m\nVariation {:.2}   Forest similarity {:.2}\nTerrain {:.1} ms   Scene setup {:.1} ms   {:.1} m/s\nW/S Move   A/D + Mouse Turn   Shift Run   V FPS / Follow\nB Overview   Wheel Zoom   Esc Pause   R Start   L River bank   N Next region\nF3 Resources   F4 Map layers\n{}",
        world.seed,
        world.settings.mountain_strength,
        crate::terrain::GENERATOR_VERSION,
        EXTENT * 2.0,
        EXTENT * 2.0,
        PLAY_RADIUS,
        world
            .landform(Vec2::new(position.translation.x, position.translation.z))
            .label(),
        lab.height_min,
        lab.height_max,
        world.trees.len(),
        world.river.as_ref().unwrap().source_count,
        world.river.as_ref().unwrap().total_length(),
        settings.variation,
        settings.forest_uniformity,
        lab.terrain_ms,
        lab.scene_setup_ms,
        horse.speed.abs(),
        layer.legend()
    );
    if hud.0 != text {
        hud.0 = text;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::time::TimeUpdateStrategy;
    use std::time::Duration;

    fn scene() -> (App, Entity) {
        scene_with(Meadow::new(20261003))
    }

    fn scene_with(map: Meadow) -> (App, Entity) {
        let mut app = App::new();
        let ground = map.ground(Vec2::ZERO);
        app.add_plugins(MinimalPlugins)
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                1.0 / 60.0,
            )))
            .insert_resource(map)
            .insert_resource(LabState {
                ready: true,
                ..default()
            })
            .insert_resource(CameraRig {
                captured: true,
                ..default()
            })
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<AccumulatedMouseMotion>()
            .init_resource::<AccumulatedMouseScroll>()
            .add_systems(Update, controls);
        app.world_mut().spawn((
            Window {
                focused: true,
                ..default()
            },
            PrimaryWindow,
            CursorOptions::default(),
        ));
        let horse = app
            .world_mut()
            .spawn((
                HorseController {
                    speed: 0.0,
                    yaw: 0.0,
                },
                Transform::from_xyz(0.0, ground + 0.025, 0.0),
            ))
            .id();
        app.update();
        (app, horse)
    }

    fn frames(app: &mut App, count: usize, keys: &[KeyCode]) {
        for frame in 0..count {
            let mut input = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            input.reset_all();
            for key in keys {
                input.press(*key);
            }
            if frame > 0 {
                input.clear();
            }
            app.update();
        }
    }

    #[test]
    fn horse_can_walk_across_a_ford_and_back_without_entering_deep_water() {
        for seed in [42, 20261003, 314159] {
            let map = Meadow::streamed(seed, crate::terrain::TerrainSettings::default());
            let ford = map
                .stream_generator()
                .unwrap()
                .hydrology
                .get((0, 0))
                .unwrap()
                .fords[0];
            let (mut app, horse) = scene_with(map);
            frames(&mut app, 1, &[KeyCode::KeyG]);
            // The shortcut selects the nearest crossing; test a fixed selected crossing.
            let start = ford.entry();
            let h = app.world().resource::<Meadow>().ground(start);
            app.world_mut()
                .get_mut::<Transform>(horse)
                .unwrap()
                .translation = Vec3::new(start.x, h + 0.025, start.y);
            app.world_mut()
                .get_mut::<HorseController>(horse)
                .unwrap()
                .yaw = (-ford.across.x).atan2(-ford.across.y);
            let frames_to_cross = ((2. * (ford.width + 15.) + 1.) / 3.5 * 60.).ceil() as usize + 30;
            frames(&mut app, frames_to_cross, &[KeyCode::KeyW]);
            let p = app.world().get::<Transform>(horse).unwrap().translation;
            let p = Vec2::new(p.x, p.z);
            assert!(
                (p - ford.center).dot(ford.across) > ford.width + 15.,
                "Horse failed to reach other bank: {seed} {p:?}"
            );
            assert!(
                app.world()
                    .resource::<Meadow>()
                    .stream_generator()
                    .unwrap()
                    .water_depth(p)
                    < 0.
            );
            app.world_mut()
                .get_mut::<HorseController>(horse)
                .unwrap()
                .yaw += std::f32::consts::PI;
            frames(&mut app, frames_to_cross, &[KeyCode::KeyW]);
            let t = app.world().get::<Transform>(horse).unwrap();
            let p = Vec2::new(t.translation.x, t.translation.z);
            assert!((p - ford.center).dot(ford.across) < -ford.width - 14.);
            assert!(app.world().resource::<Meadow>().walkable(p));
        }
    }

    #[test]
    fn horse_walks_on_previously_blocked_sand_then_stops_at_water() {
        for seed in [42, 20261003] {
            let map = Meadow::streamed(seed, crate::terrain::TerrainSettings::default());
            let g = map.stream_generator().unwrap();
            let lake = g.hydrology.get((0, 0)).unwrap().lake;
            let direction = (0..64)
                .map(|i| Vec2::from_angle(i as f32 * std::f32::consts::TAU / 64.))
                .find(|d| {
                    (0..=16)
                        .all(|step| map.walkable(lake + *d * 31. + d.perp() * step as f32 * 0.2))
                })
                .unwrap();
            let start = lake + direction * 31.;
            let (mut app, horse) = scene_with(map);
            let aim = |app: &mut App, position: Vec2, forward: Vec2| {
                let ground = app.world().resource::<Meadow>().ground(position);
                let mut t = app.world_mut().get_mut::<Transform>(horse).unwrap();
                t.translation = Vec3::new(position.x, ground + 0.025, position.y);
                let mut c = app.world_mut().get_mut::<HorseController>(horse).unwrap();
                c.yaw = (-forward.x).atan2(-forward.y);
                c.speed = 0.;
            };
            aim(&mut app, start, direction.perp());
            frames(&mut app, 45, &[KeyCode::KeyW]);
            let t = app.world().get::<Transform>(horse).unwrap();
            let p = Vec2::new(t.translation.x, t.translation.z);
            assert!(p.distance(start) > 1.5);
            assert!(
                p.distance(lake) < 33.6,
                "Must remain on previously forbidden sand"
            );
            let inward = (lake - p).normalize();
            aim(&mut app, p, inward);
            frames(&mut app, 180, &[KeyCode::KeyW]);
            let t = app.world().get::<Transform>(horse).unwrap();
            let end = Vec2::new(t.translation.x, t.translation.z);
            assert!(end.distance(p) > 1.);
            assert!(app.world().resource::<Meadow>().walkable(end));
            assert_eq!(app.world().get::<HorseController>(horse).unwrap().speed, 0.);
            let map = app.world().resource::<Meadow>();
            assert!(!map.walkable(end + inward * 0.4));
        }
    }

    #[test]
    fn horse_can_explore_beyond_the_previous_map_boundary() {
        let (mut app, horse) = scene();
        frames(&mut app, 1, &[KeyCode::KeyN]);
        let world = app.world().resource::<Meadow>();
        let (p, _) = world.exploration_view(0);
        let transform = app.world().get::<Transform>(horse).unwrap();
        assert!(Vec2::new(transform.translation.x, transform.translation.z).distance(p) < 0.001);
        assert!(world.walkable(p) && p.length() > 160.0);
        frames(&mut app, 120, &[KeyCode::KeyW, KeyCode::ShiftLeft]);
        let transform = app.world().get::<Transform>(horse).unwrap();
        let end = Vec2::new(transform.translation.x, transform.translation.z);
        assert!(end.distance(p) > 8.0);
        let map = app.world().resource::<Meadow>();
        assert!(map.walkable(end));
        assert!((transform.translation.y - map.ground(end) - 0.025).abs() < 0.001);
    }

    #[test]
    fn river_shortcut_lands_on_a_dry_bank_and_reset_returns_to_spawn() {
        let (mut app, horse) = scene();
        let origin = app.world().get::<Transform>(horse).unwrap().translation;
        frames(&mut app, 1, &[KeyCode::KeyB]);
        frames(&mut app, 1, &[KeyCode::KeyL]);
        let transform = app.world().get::<Transform>(horse).unwrap();
        let p = Vec2::new(transform.translation.x, transform.translation.z);
        let world = app.world().resource::<Meadow>();
        assert!(world.walkable(p));
        assert!(p.distance(world.river_bank().0) < 0.001);
        assert!((transform.translation.y - world.ground(p) - 0.025).abs() < 0.001);
        assert_eq!(
            app.world().resource::<CameraRig>().mode,
            ViewMode::ThirdPerson
        );
        frames(&mut app, 1, &[KeyCode::KeyR]);
        assert_eq!(
            app.world().get::<Transform>(horse).unwrap().translation,
            origin
        );
    }

    #[test]
    fn running_is_unlimited_and_combat_keys_do_not_interrupt_exploration() {
        let (mut app, horse) = scene();
        let start = app.world().get::<Transform>(horse).unwrap().translation;
        frames(
            &mut app,
            180,
            &[
                KeyCode::KeyW,
                KeyCode::ShiftLeft,
                KeyCode::Space,
                KeyCode::KeyF,
                KeyCode::KeyC,
                KeyCode::KeyX,
                KeyCode::KeyZ,
            ],
        );
        let position = app.world().get::<Transform>(horse).unwrap().translation;
        assert!(position.distance(start) > 12.0);
        assert!(app.world().get::<HorseController>(horse).unwrap().speed > 8.0);
        let p = Vec2::new(position.x, position.z);
        assert!(app.world().resource::<Meadow>().walkable(p));
        assert!((position.y - app.world().resource::<Meadow>().ground(p) - 0.025).abs() < 0.001);
        frames(&mut app, 1, &[KeyCode::KeyR]);
        assert_eq!(
            app.world().get::<Transform>(horse).unwrap().translation,
            start
        );
    }

    #[test]
    fn pause_and_unfocused_window_stop_movement_and_overview_can_return_to_follow() {
        let (mut app, horse) = scene();
        frames(&mut app, 1, &[KeyCode::KeyB]);
        assert_eq!(app.world().resource::<CameraRig>().mode, ViewMode::Overview);
        frames(&mut app, 1, &[KeyCode::KeyB]);
        assert_eq!(
            app.world().resource::<CameraRig>().mode,
            ViewMode::ThirdPerson
        );
        frames(&mut app, 1, &[KeyCode::Escape]);
        let start = app.world().get::<Transform>(horse).unwrap().translation;
        frames(&mut app, 60, &[KeyCode::KeyW]);
        assert_eq!(
            app.world().get::<Transform>(horse).unwrap().translation,
            start
        );
        app.world_mut().resource_mut::<CameraRig>().captured = true;
        let mut windows = app.world_mut().query::<&mut Window>();
        windows.single_mut(app.world_mut()).unwrap().focused = false;
        frames(&mut app, 60, &[KeyCode::KeyW]);
        assert_eq!(
            app.world().get::<Transform>(horse).unwrap().translation,
            start
        );
        assert!(!app.world().resource::<CameraRig>().captured);
    }

    #[test]
    fn camera_toggle_preserves_each_views_pitch() {
        let mut rig = CameraRig {
            first_pitch: -0.3,
            ..default()
        };
        rig.toggle();
        assert_eq!(rig.mode, ViewMode::FirstPerson);
        assert_eq!(rig.pitch(), -0.3);
        assert!(rig.snap);
        rig.toggle();
        assert_eq!(rig.mode, ViewMode::ThirdPerson);
        assert_eq!(rig.pitch(), 0.27);
    }
    #[test]
    fn horse_eye_tracks_position_and_heading() {
        let horse = Transform::from_xyz(8.0, 3.0, 4.0)
            .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
        assert!(first_person_position(&horse).distance(Vec3::new(6.65, 4.95, 4.0)) < 0.0001);
    }
    #[test]
    fn chase_camera_stays_above_land() {
        let world = Meadow::new(42);
        let focus = Vec3::new(0.0, world.ground(Vec2::ZERO) + 1.5, 0.0);
        let low = focus + Vec3::new(0.0, -4.0, 15.0);
        let safe = clear_camera_boom(focus, low, &world, None);
        assert!(safe.y >= world.ground(Vec2::new(safe.x, safe.z)) + 0.35);
        assert!(safe.distance(focus) < low.distance(focus));
    }
}
