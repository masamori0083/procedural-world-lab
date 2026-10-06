use crate::environment::MapLayer;
use crate::terrain::{AREA_SCALE, EXTENT, Meadow, PLACEMENT_EXTENT, PLAY_RADIUS, SeedRandom};
use crate::trees::{TreeSettings, shapes};
use bevy::{
    asset::RenderAssetUsages, light::CascadeShadowConfigBuilder, mesh::Indices, prelude::*,
    render::render_resource::PrimitiveTopology,
};

#[derive(Component)]
pub struct Canopy {
    phase: f32,
}

#[derive(Component)]
pub struct Cloud {
    phase: f32,
}

#[derive(Component)]
pub struct TerrainSurface;

#[derive(Component)]
pub struct Grass;

pub fn setup(
    mut commands: Commands,
    world: Res<Meadow>,
    tree_settings: Res<TreeSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !world.is_streaming() {
        commands.spawn((
            TerrainSurface,
            Mesh3d(meshes.add(world.mesh())),
            MeshMaterial3d(materials.add(StandardMaterial {
                perceptual_roughness: 1.0,
                ..default()
            })),
        ));
    }
    commands.spawn((
        DirectionalLight {
            illuminance: 11000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, -0.5, 0.0)),
        CascadeShadowConfigBuilder {
            maximum_distance: 110.0,
            ..default()
        }
        .build(),
    ));

    if !world.is_streaming() {
        let leaf_template = Sphere::new(1.0).mesh().ico(1).unwrap();
        let leaves = meshes.add(leaf_template.clone());
        // Vertex colors allow restrained per-tree hues with one shared material.
        let tree_material = materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            ..default()
        });
        for ((p, _), tree) in world.trees.iter().zip(shapes(&world, *tree_settings)) {
            let base = Vec3::new(p.x, world.ground(*p), p.y);
            commands.spawn((
                Name::new(format!(
                    "Tree {} / {:?} / trunk and branches",
                    tree.seed, tree.species
                )),
                Mesh3d(meshes.add(tree.wood_mesh())),
                MeshMaterial3d(tree_material.clone()),
                Transform::from_translation(base),
            ));
            commands.spawn((
                Name::new(format!("Tree {} / crown", tree.seed)),
                Mesh3d(meshes.add(tree.crown_mesh(&leaf_template))),
                MeshMaterial3d(tree_material.clone()),
                Transform::from_translation(base + tree.crown_pivot),
                Canopy {
                    phase: tree.seed as f32 / u32::MAX as f32 * std::f32::consts::TAU,
                },
            ));
        }

        let water_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.48, 0.61),
            metallic: 0.12,
            perceptual_roughness: 0.24,
            reflectance: 0.45,
            ..default()
        });
        for pond in &world.ponds {
            commands.spawn((
                Mesh3d(meshes.add(world.water_mesh(pond))),
                MeshMaterial3d(water_material.clone()),
                Transform::default(),
            ));
        }

        let mut random = SeedRandom(world.seed ^ 0xd33c_9017);
        let rock_material = materials.add(Color::srgb(0.42, 0.45, 0.41));
        for _ in 0..180 * AREA_SCALE {
            let p = Vec2::new(
                random.range(-PLACEMENT_EXTENT, PLACEMENT_EXTENT),
                random.range(-PLACEMENT_EXTENT, PLACEMENT_EXTENT),
            );
            if p.length() < 22.0 || world.near_water(p, 3.0) {
                continue;
            }
            let habitat = world.environment_at(p);
            if random.unit() > 0.025 + habitat.rockiness * 0.65 {
                continue;
            }
            // Small decorative stones sit below the horse's collision clearance.
            commands.spawn((
                Mesh3d(leaves.clone()),
                MeshMaterial3d(rock_material.clone()),
                Transform::from_xyz(p.x, world.ground(p) + 0.08, p.y)
                    .with_scale(Vec3::new(
                        random.range(0.35, 0.7),
                        0.17,
                        random.range(0.3, 0.6),
                    ))
                    .with_rotation(Quat::from_rotation_y(
                        random.range(0.0, std::f32::consts::TAU),
                    )),
            ));
        }

        commands.spawn((
            Grass,
            Mesh3d(meshes.add(grass_mesh(&world))),
            MeshMaterial3d(materials.add(StandardMaterial {
                cull_mode: None,
                perceptual_roughness: 1.0,
                ..default()
            })),
        ));
    }
    let cloud_mesh = meshes.add(Sphere::new(1.0).mesh().ico(2).unwrap());
    let cloud_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.96, 0.98, 1.0),
        unlit: true,
        fog_enabled: false,
        ..default()
    });
    let mut clouds = SeedRandom(world.seed ^ 0x849aec21);
    let cloud_count = 28;
    for index in 0..cloud_count {
        let angle =
            index as f32 * std::f32::consts::TAU / cloud_count as f32 + clouds.range(-0.1, 0.1);
        let center = Vec2::from_angle(angle) * clouds.range(EXTENT * 0.45, EXTENT * 1.1);
        let height = clouds.range(52.0, 72.0);
        let width = clouds.range(0.8, 1.3);
        commands
            .spawn((
                Cloud { phase: angle },
                Transform::from_xyz(center.x, height, center.y),
                Visibility::default(),
            ))
            .with_children(|root| {
                for offset in [
                    Vec3::ZERO,
                    Vec3::new(-8.0, -1.0, 1.0),
                    Vec3::new(7.0, -0.5, -1.0),
                    Vec3::new(-1.0, 2.0, -3.0),
                ] {
                    root.spawn((
                        Mesh3d(cloud_mesh.clone()),
                        MeshMaterial3d(cloud_material.clone()),
                        bevy::light::NotShadowCaster,
                        Transform::from_translation(offset * width)
                            .with_scale(Vec3::new(11.0, 3.2, 6.0) * width),
                    ));
                }
            });
    }
}

pub fn drift_clouds(
    time: Res<Time>,
    world: Res<Meadow>,
    horse: Single<&Transform, (With<crate::player::HorseController>, Without<Cloud>)>,
    mut clouds: Query<(&Cloud, &mut Transform)>,
) {
    let dt = time.delta_secs().min(0.1);
    for (cloud, mut transform) in &mut clouds {
        transform.translation.x += dt * (0.45 + cloud.phase.sin() * 0.12);
        if world.is_streaming() {
            let center = Vec2::new(horse.translation.x, horse.translation.z);
            transform.translation.x =
                center.x + (transform.translation.x - center.x + 420.).rem_euclid(840.) - 420.;
            transform.translation.z =
                center.y + (transform.translation.z - center.y + 420.).rem_euclid(840.) - 420.;
            transform.translation.y = 75. + cloud.phase.sin() * 8.;
        } else if transform.translation.x > EXTENT + 100.0 {
            transform.translation.x = -EXTENT - 100.0;
        }
    }
}

fn grass_mesh(world: &Meadow) -> Mesh {
    let mut random = SeedRandom(world.seed ^ 0xa15d_2357);
    let mut positions = vec![];
    let mut normals = vec![];
    let mut colors = vec![];
    let mut indices = vec![];
    for _ in 0..6000 * AREA_SCALE {
        let p = Vec2::new(
            random.range(-PLAY_RADIUS, PLAY_RADIUS),
            random.range(-PLAY_RADIUS, PLAY_RADIUS),
        );
        if world.near_water(p, 1.0) {
            continue;
        }
        if world.slope(p) > 0.75 {
            continue;
        }
        let habitat = world.environment_at(p);
        if random.unit() > habitat.grass_density * (1.0 - habitat.tree_density * 0.35) {
            continue;
        }
        let h = world.ground(p);
        let angle = random.range(0.0, std::f32::consts::TAU);
        let blade_height = random.range(0.15, 0.45) * (0.75 + habitat.moisture * 0.5);
        let base_color =
            Vec3::new(0.30, 0.33, 0.09).lerp(Vec3::new(0.14, 0.35, 0.07), habitat.moisture);
        let tip_color =
            Vec3::new(0.52, 0.51, 0.23).lerp(Vec3::new(0.36, 0.58, 0.16), habitat.moisture);
        for turn in [0.0, 1.2, 2.4] {
            let dir = Vec2::new((angle + turn).cos(), (angle + turn).sin()) * 0.06;
            let a = Vec3::new(p.x - dir.x, h, p.y - dir.y);
            let b = Vec3::new(p.x + dir.x, h, p.y + dir.y);
            let c = Vec3::new(p.x + dir.x * 1.5, h + blade_height, p.y + dir.y * 1.5);
            let start = positions.len() as u32;
            let normal = (b - a).cross(c - a).normalize();
            positions.extend([a.to_array(), b.to_array(), c.to_array()]);
            normals.extend([normal.to_array(); 3]);
            colors.extend([
                [base_color.x, base_color.y, base_color.z, 1.0],
                [base_color.x, base_color.y, base_color.z, 1.0],
                [tip_color.x, tip_color.y, tip_color.z, 1.0],
            ]);
            indices.extend([start, start + 1, start + 2]);
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

type TerrainViews<'w, 's> = Query<
    'w,
    's,
    (
        &'static Mesh3d,
        &'static MeshMaterial3d<StandardMaterial>,
        Option<&'static crate::streaming::ChunkSurface>,
    ),
    With<TerrainSurface>,
>;

pub fn update_layer(
    keys: Res<ButtonInput<KeyCode>>,
    world: Res<Meadow>,
    mut layer: ResMut<MapLayer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    terrain: TerrainViews<'_, '_>,
    mut grass: Query<&mut Visibility, With<Grass>>,
) {
    if !keys.just_pressed(KeyCode::F4) {
        return;
    }
    *layer = layer.next();
    for (mesh, material, chunk) in &terrain {
        if let Some(mut mesh) = meshes.get_mut(&mesh.0) {
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_COLOR,
                chunk.map_or_else(
                    || world.surface_colors(*layer),
                    |chunk| chunk.colors(*layer),
                ),
            );
        }
        if let Some(mut material) = materials.get_mut(&material.0) {
            material.unlit = *layer != MapLayer::Natural;
            material.fog_enabled = *layer == MapLayer::Natural;
        }
    }
    for mut visible in &mut grass {
        *visible = if *layer == MapLayer::Natural {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

pub fn sway_trees(time: Res<Time>, mut canopies: Query<(&Canopy, &mut Transform)>) {
    let t = time.elapsed_secs();
    for (canopy, mut transform) in &mut canopies {
        transform.rotation = Quat::from_rotation_z((t * 0.8 + canopy.phase).sin() * 0.025);
    }
}
