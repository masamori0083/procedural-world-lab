//! Stable grove identities, restrained forest variation and exaggerated isolated landmarks.
use crate::terrain::{Meadow, SeedRandom};
use bevy::{asset::RenderAssetUsages, prelude::*, render::render_resource::PrimitiveTopology};

#[derive(Resource, Clone, Copy)]
pub struct TreeSettings {
    pub variation: f32,
    pub forest_uniformity: f32,
    pub landmark_strength: f32,
}
impl Default for TreeSettings {
    fn default() -> Self {
        Self {
            variation: 0.65,
            forest_uniformity: 0.90,
            landmark_strength: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Rounded,
    Slender,
    Spreading,
}
impl Species {
    fn from_seed(seed: u32) -> Self {
        match mix(seed) % 3 {
            0 => Self::Rounded,
            1 => Self::Slender,
            _ => Self::Spreading,
        }
    }
}

// 32-bit mix listed in skeeto/hash-prospector (Unlicense).
// Source: https://github.com/skeeto/hash-prospector.
fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}
fn cell_seed(seed: u32, x: i32, z: i32) -> u32 {
    mix(seed ^ (x as u32).wrapping_mul(0x9e37_79b9) ^ (z as u32).wrapping_mul(0x85eb_ca6b))
}

/// Jittered 48m grove cells keep nearby trees related without a visible grid.
fn grove_species(world: &Meadow, p: Vec2) -> Species {
    let cell = (p / 48.0).floor().as_ivec2();
    let mut nearest = (f32::INFINITY, 0, Vec2::ZERO);
    for z in cell.y - 1..=cell.y + 1 {
        for x in cell.x - 1..=cell.x + 1 {
            let s = cell_seed(world.seed ^ 0xf015_19a3, x, z);
            let mut random = SeedRandom(s);
            let center = (Vec2::new(x as f32, z as f32)
                + Vec2::new(random.range(0.25, 0.75), random.range(0.25, 0.75)))
                * 48.0;
            let distance = center.distance_squared(p);
            if distance < nearest.0 {
                nearest = (distance, s, center);
            }
        }
    }
    let habitat = world.environment_at(nearest.2);
    // Streaming grove identity uses raw terrain/climate, so a neighbouring
    // watershed being cached or evicted cannot change species at an edge.
    let (elevation, moisture) = world.stream_generator().map_or(
        (
            habitat.elevation - world.ground(Vec2::ZERO),
            habitat.moisture,
        ),
        |g| {
            (
                g.base_height(nearest.2) - g.base_height(Vec2::ZERO),
                0.24 + 0.30 * g.noise(nearest.2 * 0.018 + Vec2::new(219., -53.))
                    + 0.15 * g.noise(nearest.2 * 0.035 + Vec2::splat(31.)),
            )
        },
    );
    // Whole groves share a preference; a minority vary to avoid hard biome borders.
    if mix(nearest.1).is_multiple_of(4) {
        Species::from_seed(nearest.1)
    } else if elevation > 16.0 {
        Species::Slender
    } else if moisture > 0.58 {
        Species::Rounded
    } else {
        Species::Spreading
    }
}

/// Weighted actual neighbours, independent of render/load order.
pub fn isolation(p: Vec2, neighbours: &[(Vec2, f32)]) -> f32 {
    let crowd: f32 = neighbours
        .iter()
        .filter_map(|(q, _)| {
            let d = p.distance(*q);
            (d > 0.01 && d < 30.).then(|| {
                let t = 1. - d / 30.;
                t * t * (3. - 2. * t)
            })
        })
        .sum();
    (1. - crowd / 1.5).clamp(0., 1.)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandmarkForm {
    Ancient,
    Wide,
    WindBent,
    Forked,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LeafCluster {
    pub center: Vec3,
    pub radii: Vec3,
    pub rotation: Quat,
    pub shade: f32,
}
#[derive(Clone, Debug, PartialEq)]
pub struct TreeShape {
    pub seed: u32,
    pub species: Species,
    pub trunk: Vec<(Vec3, f32)>,
    pub branches: Vec<Vec<(Vec3, f32)>>,
    pub leaves: Vec<LeafCluster>,
    pub crown_pivot: Vec3,
    pub leaf_color: Vec3,
    pub bark_color: Vec3,
    pub roughness: f32,
    pub landmark: Option<LandmarkForm>,
}
impl TreeShape {
    pub fn at(world: &Meadow, p: Vec2, size: f32, settings: TreeSettings) -> Self {
        // Position-based identity survives changes to traversal/spawn order.
        let seed = mix(world.seed ^ mix(p.x.to_bits()) ^ mix(p.y.to_bits().rotate_left(13)));
        let mut random = SeedRandom(seed);
        let local_species = Species::from_seed(seed ^ 0x45bd_716c);
        let species = if random.unit() < settings.forest_uniformity.clamp(0.0, 1.0) {
            grove_species(world, p)
        } else {
            local_species
        };
        let v = settings.variation.clamp(0.0, 1.0);
        let (height, spread, vertical, color) = match species {
            Species::Rounded => (3.6, 1.65, 1.45, Vec3::new(0.27, 0.45, 0.16)),
            Species::Slender => (4.15, 1.15, 1.85, Vec3::new(0.31, 0.48, 0.20)),
            Species::Spreading => (3.25, 1.95, 1.18, Vec3::new(0.32, 0.46, 0.17)),
        };
        let height = height * (1.0 + random.range(-0.20, 0.20) * v);
        let spread = spread * (1.0 + random.range(-0.22, 0.22) * v);
        let lean = Vec3::new(random.range(-0.22, 0.22), 0.0, random.range(-0.22, 0.22)) * v;
        let thickness = 0.30 * (1.0 + random.range(-0.18, 0.18) * v);
        let trunk: Vec<_> = (0..5)
            .map(|i| {
                let f = i as f32 / 4.0;
                let bend =
                    Vec3::new(random.range(-0.07, 0.07), 0.0, random.range(-0.07, 0.07)) * f * v;
                (
                    (Vec3::Y * height * f + lean * f * f + bend) * size,
                    thickness * (1.0 - 0.65 * f) * size * if i == 0 { 1.14 } else { 1.0 },
                )
            })
            .collect();
        let tip = trunk.last().unwrap().0;
        let crown_pivot = Vec3::Y * height * 0.70 * size;
        let brightness = 1.0 + random.range(-0.09, 0.09) * v;
        let warmth = random.range(-0.018, 0.018) * v;
        let leaf_color = color * brightness * (0.95 + world.environment_at(p).moisture * 0.1)
            + Vec3::new(warmth, 0.0, -warmth * 0.3);
        let bark_color = Vec3::new(0.28, 0.19, 0.11) * (1.0 + random.range(-0.10, 0.10) * v);
        let count = 5 + (random.range(0.0, 4.0) * v) as usize;
        let orientation = random.range(-std::f32::consts::PI, std::f32::consts::PI) * v;
        let mut leaves = vec![LeafCluster {
            center: tip + Vec3::Y * (0.60 * size),
            radii: Vec3::new(spread, vertical, spread * 0.90) * size,
            rotation: Quat::from_rotation_y(orientation),
            shade: 1.04,
        }];
        let mut branches = Vec::new();
        for i in 0..count {
            let angle = orientation
                + i as f32 * std::f32::consts::TAU / count as f32
                + random.range(-0.32, 0.32) * v;
            let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
            let reach = spread * (0.67 + random.range(-0.18, 0.18) * v);
            let center = tip
                + (direction * reach + Vec3::Y * (-0.10 + random.range(-0.50, 0.55) * v)) * size;
            let bulk = 0.78 + random.range(-0.18, 0.18) * v;
            leaves.push(LeafCluster {
                center,
                radii: Vec3::new(
                    spread * bulk,
                    vertical * bulk * (1.0 + random.range(-0.12, 0.12) * v),
                    spread * bulk * (0.85 + random.range(-0.12, 0.12) * v),
                ) * size,
                rotation: Quat::from_euler(
                    EulerRot::XYZ,
                    random.range(-0.20, 0.20) * v,
                    angle,
                    random.range(-0.20, 0.20) * v,
                ),
                shade: 1.0 + random.range(-0.055, 0.055) * v,
            });
            let start = trunk[2 + i % 2].0;
            let end = center - Vec3::Y * vertical * bulk * size * 0.22;
            let elbow = start.lerp(end, 0.58) - Vec3::Y * 0.12 * size;
            branches.push(vec![
                (start, 0.095 * size),
                (elbow, 0.064 * size),
                (end, 0.022 * size),
            ]);
        }
        Self {
            seed,
            species,
            trunk,
            branches,
            leaves,
            crown_pivot,
            leaf_color,
            bark_color,
            roughness: v * 0.09,
            landmark: None,
        }
    }

    /// Streaming landmarks deliberately exaggerate shape, while dense groves
    /// retain the baseline. No change to placement, identity, or natural colors.
    pub fn with_isolation(
        world: &Meadow,
        p: Vec2,
        size: f32,
        settings: TreeSettings,
        isolation: f32,
    ) -> Self {
        let t = ((isolation - 0.25) / 0.75).clamp(0., 1.);
        let amount = t * t * (3. - 2. * t) * settings.landmark_strength.clamp(0., 1.);
        let mut tree = Self::at(
            world,
            p,
            size,
            TreeSettings {
                forest_uniformity: settings.forest_uniformity * (1. - amount * 0.9),
                ..settings
            },
        );
        if amount <= 0. {
            return tree;
        }
        let mut r = SeedRandom(tree.seed ^ 0x81a7_d339);
        let form = match mix(tree.seed ^ 0xf412_88d1) % 4 {
            0 => LandmarkForm::Ancient,
            1 => LandmarkForm::Wide,
            2 => LandmarkForm::WindBent,
            _ => LandmarkForm::Forked,
        };
        let scale = 1. + amount * r.range(2.0, 3.1);
        let (wide, tall, bend, thick) = match form {
            LandmarkForm::Ancient => (1.05, 1.45, 0.5, 2.0),
            LandmarkForm::Wide => (1.8, 0.9, 0.8, 1.4),
            LandmarkForm::WindBent => (1.2, 1.1, 3.0, 1.6),
            LandmarkForm::Forked => (1.15, 1.3, 0.7, 1.5),
        };
        let horizontal = 1. + (wide - 1.) * amount;
        let vertical = 1. + (tall - 1.) * amount;
        let heading = r.range(0., std::f32::consts::TAU);
        let lean = Vec3::new(heading.cos(), 0., heading.sin()) * bend * size * amount;
        let tip_height = tree.trunk.last().unwrap().0.y;
        let transform = |v: Vec3| {
            let f = (v.y / tip_height).clamp(0., 1.);
            (v * Vec3::new(horizontal, vertical, horizontal) + lean * f * f) * scale
        };
        for (point, radius) in &mut tree.trunk {
            *point = transform(*point);
            *radius *= scale * (1. + (thick - 1.) * amount);
        }
        for branch in &mut tree.branches {
            for (point, radius) in branch {
                *point = transform(*point);
                *radius *= scale * (1. + amount * 0.7);
            }
        }
        for leaf in &mut tree.leaves {
            leaf.center = transform(leaf.center);
            let asymmetry = 1. + r.range(-0.28, 0.28) * amount;
            let gaps = if form == LandmarkForm::Forked {
                1. - 0.35 * amount
            } else {
                1.
            };
            leaf.radii *=
                Vec3::new(horizontal * asymmetry, vertical, horizontal / asymmetry) * scale * gaps;
        }
        tree.crown_pivot = transform(tree.crown_pivot);
        if amount > 0.45 {
            tree.landmark = Some(form);
        }
        tree
    }

    pub fn wood_mesh(&self) -> Mesh {
        let mut surface = TreeSurface::default();
        surface.tube(&self.trunk, self.bark_color);
        for branch in &self.branches {
            surface.tube(branch, self.bark_color * 0.96);
        }
        surface.mesh()
    }

    pub fn crown_mesh(&self, unit: &Mesh) -> Mesh {
        use bevy::mesh::VertexAttributeValues;
        let Some(VertexAttributeValues::Float32x3(vertices)) =
            unit.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("Crown template needs 3D positions");
        };
        let indices: Vec<_> = unit.indices().unwrap().iter().collect();
        let mut surface = TreeSurface::default();
        let mut random = SeedRandom(self.seed ^ 0xc419_a733);
        for cluster in &self.leaves {
            // One displacement per shared vertex preserves a closed crown surface.
            let points: Vec<_> = vertices
                .iter()
                .map(|p| {
                    let p = Vec3::from(*p) * (1.0 + random.range(-1.0, 1.0) * self.roughness);
                    cluster.center - self.crown_pivot + cluster.rotation * (p * cluster.radii)
                })
                .collect();
            for tri in indices.chunks_exact(3) {
                surface.triangle(
                    points[tri[0]],
                    points[tri[1]],
                    points[tri[2]],
                    self.leaf_color * cluster.shade,
                );
            }
        }
        surface.mesh()
    }
}

#[derive(Default)]
struct TreeSurface {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
}
impl TreeSurface {
    fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, srgb: Vec3) {
        let normal = (b - a).cross(c - a).normalize();
        assert!(normal.is_finite(), "Degenerate tree face");
        self.positions
            .extend([a.to_array(), b.to_array(), c.to_array()]);
        self.normals.extend([normal.to_array(); 3]);
        let color = Color::srgb(srgb.x, srgb.y, srgb.z).to_linear();
        self.colors
            .extend([[color.red, color.green, color.blue, 1.0]; 3]);
    }
    fn tube(&mut self, path: &[(Vec3, f32)], color: Vec3) {
        const SIDES: usize = 7;
        let axis = (path.last().unwrap().0 - path[0].0).normalize();
        let reference = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
        let u = (reference - axis * reference.dot(axis)).normalize();
        let v = axis.cross(u);
        let rings: Vec<Vec<Vec3>> = path
            .iter()
            .map(|(center, radius)| {
                (0..SIDES)
                    .map(|i| {
                        let angle = i as f32 * std::f32::consts::TAU / SIDES as f32;
                        *center + (u * angle.cos() + v * angle.sin()) * *radius
                    })
                    .collect()
            })
            .collect();
        for pair in rings.windows(2) {
            for i in 0..SIDES {
                let j = (i + 1) % SIDES;
                self.triangle(pair[0][i], pair[0][j], pair[1][i], color);
                self.triangle(pair[0][j], pair[1][j], pair[1][i], color);
            }
        }
        let last = rings.len() - 1;
        for i in 0..SIDES {
            let j = (i + 1) % SIDES;
            self.triangle(path[0].0, rings[0][j], rings[0][i], color);
            self.triangle(path[last].0, rings[last][i], rings[last][j], color);
        }
    }
    fn mesh(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
    }
}

pub fn shapes(world: &Meadow, settings: TreeSettings) -> Vec<TreeShape> {
    world
        .trees
        .iter()
        .map(|(p, size)| TreeShape::at(world, *p, *size, settings))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn isolated_landmarks_are_large_diverse_repeatable_and_opt_out_is_exact() {
        let world = Meadow::streamed(42, crate::terrain::TerrainSettings::default());
        let settings = TreeSettings::default();
        let unit = Sphere::new(1.).mesh().ico(1).unwrap();
        let mut forms = std::collections::HashSet::new();
        for i in 0..64 {
            let p = Vec2::new(i as f32 * 7. + 40., -51.);
            let base = TreeShape::at(&world, p, 1., settings);
            let hero = TreeShape::with_isolation(&world, p, 1., settings, 1.);
            assert_eq!(hero, TreeShape::with_isolation(&world, p, 1., settings, 1.));
            assert_eq!(hero.seed, base.seed);
            assert!(hero.trunk.last().unwrap().0.y > base.trunk.last().unwrap().0.y * 2.0);
            assert!(hero.trunk[0].1 > base.trunk[0].1 * 3.0);
            assert_eq!(hero.trunk[0].0, Vec3::ZERO);
            forms.insert(format!("{:?}", hero.landmark.unwrap()));
            assert!(hero.leaf_color.y > hero.leaf_color.x && hero.leaf_color.x > hero.leaf_color.z);
            for mesh in [hero.wood_mesh(), hero.crown_mesh(&unit)] {
                let Some(bevy::mesh::VertexAttributeValues::Float32x3(points)) =
                    mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                else {
                    panic!()
                };
                assert!(points.iter().all(|p| Vec3::from(*p).is_finite()));
            }
            assert_eq!(base, TreeShape::with_isolation(&world, p, 1., settings, 0.));
            assert_eq!(
                base,
                TreeShape::with_isolation(
                    &world,
                    p,
                    1.,
                    TreeSettings {
                        landmark_strength: 0.,
                        ..settings
                    },
                    1.
                )
            );
        }
        assert_eq!(forms.len(), 4);
    }
    #[test]
    fn isolation_falls_with_actual_neighbours_without_counting_self() {
        let p = Vec2::ZERO;
        assert_eq!(isolation(p, &[(p, 1.)]), 1.);
        let crowded = [
            (p, 1.),
            (Vec2::X * 10., 1.),
            (Vec2::Y * 10., 1.),
            (-Vec2::X * 10., 1.),
        ];
        assert_eq!(isolation(p, &crowded), 0.);
        assert!(isolation(p, &crowded[..2]) < isolation(p, &[(Vec2::X * 23., 1.)]));
        assert_eq!(isolation(p, &[(Vec2::X * 30., 1.)]), 1.);
    }
    #[test]
    fn identities_are_repeatable_order_independent_and_individually_varied() {
        let world = Meadow::new(20261003);
        let other_world = Meadow::new(world.seed + 1);
        let shapes = shapes(&world, TreeSettings::default());
        assert!(shapes.len() > 50);
        for ((p, size), shape) in world.trees.iter().zip(&shapes) {
            assert_eq!(
                *shape,
                TreeShape::at(&world, *p, *size, TreeSettings::default())
            );
            assert_ne!(
                shape.seed,
                TreeShape::at(&other_world, *p, *size, TreeSettings::default()).seed
            );
        }
        let seeds: std::collections::HashSet<_> = shapes.iter().map(|s| s.seed).collect();
        assert_eq!(seeds.len(), shapes.len());
        assert!(shapes.windows(2).all(|p| p[0].leaves != p[1].leaves));
    }
    #[test]
    fn grove_similarity_is_adjustable_without_changing_tree_identity_or_placement() {
        let same = TreeSettings {
            variation: 0.65,
            forest_uniformity: 1.0,
            ..default()
        };
        let mixed = TreeSettings {
            forest_uniformity: 0.0,
            ..same
        };
        let mut related = 0;
        let mut random_related = 0;
        let world = Meadow::new(42);
        for i in 0..200 {
            let p = Vec2::new((i % 20) as f32 * 7.0 - 70.0, (i / 20) as f32 * 11.0 - 55.0);
            let q = p + Vec2::new(2.0, 1.0);
            let a = TreeShape::at(&world, p, 1.0, same);
            let b = TreeShape::at(&world, q, 1.0, same);
            let c = TreeShape::at(&world, p, 1.0, mixed);
            let d = TreeShape::at(&world, q, 1.0, mixed);
            assert_eq!(a.seed, c.seed);
            related += usize::from(a.species == b.species);
            random_related += usize::from(c.species == d.species);
        }
        assert!(related > 175, "Neighbor agreement: {related}/200");
        assert!(random_related < 100);
    }
    #[test]
    fn generated_meshes_are_finite_and_colors_stay_natural_at_variation_extremes() {
        let unit = Sphere::new(1.0).mesh().ico(1).unwrap();
        for seed in [1, 42, 20261003, u32::MAX] {
            let world = Meadow::new(seed);
            for variation in [0.0, 1.0] {
                for tree in shapes(
                    &world,
                    TreeSettings {
                        variation,
                        ..default()
                    },
                ) {
                    assert!(
                        tree.leaf_color.y > tree.leaf_color.x
                            && tree.leaf_color.x > tree.leaf_color.z
                    );
                    assert!(
                        tree.leaf_color.min_element() > 0.10
                            && tree.leaf_color.max_element() < 0.60
                    );
                    assert_eq!(tree.trunk[0].0, Vec3::ZERO);
                    for mesh in [tree.wood_mesh(), tree.crown_mesh(&unit)] {
                        let Some(bevy::mesh::VertexAttributeValues::Float32x3(normals)) =
                            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
                        else {
                            panic!()
                        };
                        assert!(
                            normals
                                .iter()
                                .all(|n| (Vec3::from(*n).length() - 1.0).abs() < 1e-4)
                        );
                    }
                }
            }
        }
    }
}
