//! Coordinate-seeded terrain, bounded asynchronous chunk generation and eviction.
use crate::{
    environment::{EnvironmentSample, MapLayer},
    player::HorseController,
    terrain::{CELL, Landform, Meadow, SeedRandom, TerrainSettings},
    trees::{TreeSettings, TreeShape},
    world::{Grass, TerrainSurface},
};
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::render_resource::PrimitiveTopology,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

pub const VERSION: u32 = 4;
pub const CHUNK_SIZE: f32 = 96.0;
pub const RADIUS: i32 = 3;
pub const MAX_CHUNKS: usize = ((RADIUS * 2 + 1) * (RADIUS * 2 + 1)) as usize;
const MAX_TASKS: usize = 2;
const SIDE: usize = (CHUNK_SIZE / CELL) as usize + 1;
const TREE_CELL: f32 = 12.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChunkKey(pub i32, pub i32);
impl ChunkKey {
    pub fn at(p: Vec2) -> Self {
        let q = (p / CHUNK_SIZE).floor().as_ivec2();
        Self(q.x, q.y)
    }
    pub fn origin(self) -> Vec2 {
        Vec2::new(self.0 as f32, self.1 as f32) * CHUNK_SIZE
    }
    fn distance(self, other: Self) -> i32 {
        (self.0 - other.0).abs().max((self.1 - other.1).abs())
    }
}
// 32-bit mix listed in skeeto/hash-prospector (Unlicense).
// Source: https://github.com/skeeto/hash-prospector.
fn mix(mut n: u32) -> u32 {
    n ^= n >> 16;
    n = n.wrapping_mul(0x7feb_352d);
    n ^= n >> 15;
    n = n.wrapping_mul(0x846c_a68b);
    n ^ (n >> 16)
}
fn coordinate_seed(seed: u32, x: i32, z: i32) -> u32 {
    mix(seed ^ (x as u32).wrapping_mul(0x9e37_79b9) ^ (z as u32).wrapping_mul(0x85eb_ca6b))
}
fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}

#[derive(Clone)]
pub struct Generator {
    pub seed: u32,
    pub settings: TerrainSettings,
    pub hydrology: Arc<crate::watershed::Cache>,
}
impl Generator {
    pub fn new(seed: u32, settings: TerrainSettings) -> Self {
        let g = Self {
            seed,
            settings,
            hydrology: Arc::default(),
        };
        // One coarse spawn-region plan; distant plans run in chunk workers.
        g.hydrology.prepare(&g, crate::watershed::key(Vec2::ZERO));
        g
    }
    pub fn water(&self, p: Vec2) -> crate::river::RiverField {
        self.hydrology.field(p)
    }

    pub fn noise(&self, p: Vec2) -> f32 {
        let q = p.floor().as_ivec2();
        let f = p - p.floor();
        let u = f * f * (Vec2::splat(3.) - f * 2.);
        let h = |x, z| coordinate_seed(self.seed, x, z) as f32 / u32::MAX as f32;
        let a = h(q.x, q.y) + (h(q.x + 1, q.y) - h(q.x, q.y)) * u.x;
        let b = h(q.x, q.y + 1) + (h(q.x + 1, q.y + 1) - h(q.x, q.y + 1)) * u.x;
        a + (b - a) * u.y
    }
    fn weights(&self, p: Vec2) -> (f32, f32, f32) {
        let hills = smooth((self.noise(p * 0.0025 + Vec2::new(31., -89.)) - 0.30) / 0.45);
        let region = (p / 512.).floor().as_ivec2();
        let mut mountains: f32 = 0.;
        for z in region.y - 1..=region.y + 1 {
            for x in region.x - 1..=region.x + 1 {
                let mut random = SeedRandom(coordinate_seed(self.seed ^ 0x9146_aa71, x, z));
                if random.unit() < 0.32 {
                    continue;
                }
                let center = (Vec2::new(x as f32, z as f32)
                    + Vec2::new(random.range(0.15, 0.85), random.range(0.15, 0.85)))
                    * 512.;
                let radius = random.range(65., 135.);
                let warp = (self.noise(p * 0.011 + Vec2::splat(72.)) - 0.5) * 16.;
                mountains = mountains.max(1. - smooth((p.distance(center) + warp) / radius));
            }
        }
        let basin = smooth((self.noise(p * 0.003 + Vec2::new(-147., 27.)) - 0.64) / 0.25);
        (hills, mountains, basin)
    }
    fn raw_height(&self, p: Vec2) -> f32 {
        let (hills, mountains, basin) = self.weights(p);
        let fine = (self.noise(p * 0.025) - 0.5) * 1.2
            + (self.noise(p * 0.008 + Vec2::splat(19.)) - 0.5) * 3.;
        let ridge = 1. - (2. * self.noise(p * 0.014 + Vec2::splat(63.)) - 1.).abs();
        fine + hills
            * (6. + self.settings.mountain_strength * 14.)
            * (0.3 + self.noise(p * 0.009) * 0.7)
            + mountains * self.settings.mountain_strength * 75. * (0.45 + 0.55 * ridge)
            - basin * 6.
    }
    pub fn base_height(&self, p: Vec2) -> f32 {
        let h = self.raw_height(p);
        let origin = self.raw_height(Vec2::ZERO);
        origin + (h - origin) * smooth(p.length() / 22.)
    }
    pub fn height(&self, p: Vec2) -> f32 {
        crate::watershed::carve(self.base_height(p), self.water(p))
    }
    pub fn ground(&self, p: Vec2) -> f32 {
        let cell = (p / CELL).floor() * CELL;
        let f = (p - cell) / CELL;
        let a = self.height(cell);
        let b = self.height(cell + Vec2::X * CELL);
        let c = self.height(cell + Vec2::Y * CELL);
        let d = self.height(cell + Vec2::splat(CELL));
        if f.x + f.y <= 1. {
            a + (b - a) * f.x + (c - a) * f.y
        } else {
            d + (c - d) * (1. - f.x) + (b - d) * (1. - f.y)
        }
    }
    pub fn slope(&self, p: Vec2) -> f32 {
        Vec2::new(
            self.ground(p + Vec2::X * 0.5) - self.ground(p - Vec2::X * 0.5),
            self.ground(p + Vec2::Y * 0.5) - self.ground(p - Vec2::Y * 0.5),
        )
        .length()
    }
    pub fn landform(&self, p: Vec2) -> Landform {
        let (h, m, b) = self.weights(p);
        if m * self.settings.mountain_strength > 0.2 {
            Landform::Mountains
        } else if b > 0.35 {
            Landform::Basin
        } else if h > 0.45 {
            Landform::Hills
        } else {
            Landform::Plains
        }
    }
    pub fn habitat(&self, p: Vec2) -> EnvironmentSample {
        let height = self.ground(p);
        let water = self.water(p);
        let mut sample = EnvironmentSample::evaluate(
            height,
            height - self.ground(Vec2::ZERO),
            self.slope(p),
            (water.distance - water.width).max(0.),
            self.noise(p * 0.018 + Vec2::new(219., -53.)),
            self.noise(p * 0.035 + Vec2::splat(31.)),
        );
        sample.tree_density *= smooth((p.length() - 22.) / 10.);
        sample
    }
    pub fn trees(&self, key: ChunkKey) -> Vec<(Vec2, f32)> {
        self.tree_candidates(key, 0)
    }
    /// Three cells of context cover the 30m isolation radius on every edge.
    pub fn tree_neighbours(&self, key: ChunkKey) -> Vec<(Vec2, f32)> {
        for region in tree_regions(key) {
            self.hydrology.prepare(self, region);
        }
        self.tree_candidates(key, 3)
    }
    fn tree_candidates(&self, key: ChunkKey, halo: i32) -> Vec<(Vec2, f32)> {
        let origin = (key.origin() / TREE_CELL).as_ivec2();
        let side = (CHUNK_SIZE / TREE_CELL) as i32;
        let mut trees = Vec::new();
        for z in origin.y - halo..origin.y + side + halo {
            for x in origin.x - halo..origin.x + side + halo {
                let mut random = SeedRandom(coordinate_seed(self.seed ^ 0x221b_a701, x, z));
                let p = (Vec2::new(x as f32, z as f32)
                    + Vec2::new(random.range(0.30, 0.70), random.range(0.30, 0.70)))
                    * TREE_CELL;
                let chance = random.unit();
                let size = random.range(0.8, 1.4);
                let sample = self.habitat(p);
                if chance < sample.tree_density * 0.95 && sample.slope < 0.65 {
                    trees.push((p, size));
                }
            }
        }
        trees
    }
}

fn tree_regions(key: ChunkKey) -> BTreeSet<crate::watershed::Key> {
    let lo = key.origin() - Vec2::splat(36.);
    let hi = key.origin() + Vec2::splat(CHUNK_SIZE + 36.);
    [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
        .map(crate::watershed::key)
        .into_iter()
        .collect()
}
#[derive(Clone, Debug, PartialEq)]
struct TreeObstacle {
    p: Vec2,
    radius: f32,
    height: f32,
    landmark: bool,
}

#[derive(Component)]
pub struct ChunkSurface {
    samples: Vec<EnvironmentSample>,
    patches: Vec<f32>,
}
impl ChunkSurface {
    pub fn colors(&self, layer: MapLayer) -> Vec<[f32; 4]> {
        self.samples
            .iter()
            .zip(&self.patches)
            .map(|(s, p)| s.color(layer, *p))
            .collect()
    }
}
struct ChunkData {
    surface: Mesh,
    water: Mesh,
    lake: Mesh,
    water_vertices: usize,
    trees_mesh: Mesh,
    grass: Mesh,
    colors: ChunkSurface,
    trees: Vec<(Vec2, f32)>,
    obstacles: Vec<TreeObstacle>,
    elapsed_ms: f64,
    bytes: usize,
}
#[derive(Default)]
struct Batch {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
}
impl Batch {
    fn append(&mut self, mesh: Mesh, offset: Vec3) {
        let Some(VertexAttributeValues::Float32x3(p)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("Mesh positions missing")
        };
        let Some(VertexAttributeValues::Float32x3(n)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("Mesh normals missing")
        };
        let Some(VertexAttributeValues::Float32x4(c)) = mesh.attribute(Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("Mesh colors missing")
        };
        self.positions
            .extend(p.iter().map(|p| (Vec3::from(*p) + offset).to_array()));
        self.normals.extend(n);
        self.colors.extend(c);
    }
    fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]) {
        self.positions
            .extend([a.to_array(), b.to_array(), c.to_array()]);
        self.normals
            .extend([(b - a).cross(c - a).normalize().to_array(); 3]);
        self.colors.extend([color; 3]);
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
fn generate(generator: Generator, key: ChunkKey, settings: TreeSettings) -> ChunkData {
    let start = Instant::now();
    let origin = key.origin();
    generator.hydrology.prepare(
        &generator,
        crate::watershed::key(origin + Vec2::splat(CHUNK_SIZE * 0.5)),
    );
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut samples = Vec::new();
    let mut patches = Vec::new();
    let mut indices = Vec::new();
    for z in 0..SIDE {
        for x in 0..SIDE {
            let p = origin + Vec2::new(x as f32, z as f32) * CELL;
            let height = generator.height(p);
            positions.push([x as f32 * CELL, height, z as f32 * CELL]);
            // World-coordinate central differences keep edge normals identical.
            let dx = generator.height(p + Vec2::X * CELL) - generator.height(p - Vec2::X * CELL);
            let dz = generator.height(p + Vec2::Y * CELL) - generator.height(p - Vec2::Y * CELL);
            normals.push(Vec3::new(-dx, 2. * CELL, -dz).normalize().to_array());
            samples.push(generator.habitat(p));
            patches.push(generator.noise(p * 0.08));
            if x < SIDE - 1 && z < SIDE - 1 {
                let a = (z * SIDE + x) as u32;
                let c = a + SIDE as u32;
                indices.extend([a, c, a + 1, a + 1, c, c + 1]);
            }
        }
    }
    let (water, lake) = water_meshes(&generator, origin, &positions);
    let water_vertices = water.count_vertices() + lake.count_vertices();
    let colors = ChunkSurface { samples, patches };
    let surface = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors.colors(MapLayer::Natural))
    .with_inserted_indices(Indices::U32(indices));
    let neighbours = generator.tree_neighbours(key);
    let trees: Vec<_> = neighbours
        .iter()
        .copied()
        .filter(|(p, _)| ChunkKey::at(*p) == key)
        .collect();
    let world = Meadow::from_stream_generator(generator.clone());
    let template = Sphere::new(1.).mesh().ico(1).unwrap();
    let mut batch = Batch::default();
    let mut obstacles = Vec::new();
    for (p, size) in &trees {
        let tree = TreeShape::with_isolation(
            &world,
            *p,
            *size,
            settings,
            crate::trees::isolation(*p, &neighbours),
        );
        obstacles.push(TreeObstacle {
            p: *p,
            radius: tree.trunk[0].1,
            height: tree.trunk.last().unwrap().0.y,
            landmark: tree.landmark.is_some(),
        });
        let q = *p - origin;
        let offset = Vec3::new(q.x, generator.ground(*p), q.y);
        batch.append(tree.wood_mesh(), offset);
        batch.append(tree.crown_mesh(&template), offset + tree.crown_pivot);
    }
    let mut grass = Batch::default();
    let mut random = SeedRandom(coordinate_seed(generator.seed ^ 0x581c_aa42, key.0, key.1));
    for _ in 0..480 {
        let q = Vec2::new(random.range(0., CHUNK_SIZE), random.range(0., CHUNK_SIZE));
        let p = origin + q;
        let sample = generator.habitat(p);
        if random.unit() > sample.grass_density || sample.slope > 0.75 {
            continue;
        }
        let height = generator.ground(p);
        let direction = Vec2::from_angle(random.range(0., std::f32::consts::TAU)) * 0.09;
        let a = Vec3::new(q.x - direction.x, height, q.y - direction.y);
        let b = Vec3::new(q.x + direction.x, height, q.y + direction.y);
        let c = Vec3::new(q.x, height + random.range(0.18, 0.38), q.y);
        grass.triangle(a, b, c, [0.19, 0.31, 0.08, 1.]);
    }
    let bytes = SIDE * SIDE * 40
        + (SIDE - 1) * (SIDE - 1) * 6 * 4
        + (batch.positions.len() + grass.positions.len()) * 40
        + water_vertices * 32;
    ChunkData {
        surface,
        water,
        lake,
        water_vertices,
        trees_mesh: batch.mesh(),
        grass: grass.mesh(),
        colors,
        trees,
        obstacles,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.,
        bytes,
    }
}
/// Clip exactly the same global 2 m ground triangles used by terrain/collision.
/// Boundary vertices and texture coordinates therefore agree across chunks.
fn water_meshes(g: &Generator, origin: Vec2, land: &[[f32; 3]]) -> (Mesh, Mesh) {
    let fields: Vec<_> = land
        .iter()
        .map(|p| g.water(origin + Vec2::new(p[0], p[2])))
        .collect();
    let mut vertices = Vec::new();
    let mut lake_vertices = Vec::new();
    for z in 0..SIDE - 1 {
        for x in 0..SIDE - 1 {
            let a = z * SIDE + x;
            let c = a + SIDE;
            for ids in [[a, c, a + 1], [a + 1, c, c + 1]] {
                if !ids
                    .iter()
                    .any(|&i| fields[i].distance < fields[i].width + 1.)
                {
                    continue;
                }
                let v = |i: usize| {
                    let f = fields[i];
                    (
                        Vec3::new(land[i][0], f.level, land[i][2]),
                        Vec2::new(0.5 + f.lateral / (2. * f.width), f.along / 6.),
                    )
                };
                let mut poly = Vec::new();
                for edge in 0..3 {
                    let i = ids[edge];
                    let j = ids[(edge + 1) % 3];
                    let di = land[i][1] - fields[i].level;
                    let dj = land[j][1] - fields[j].level;
                    let (a, uv_a) = v(i);
                    let (b, uv_b) = v(j);
                    if di < 0. {
                        poly.push((a, uv_a));
                    }
                    if (di < 0.) != (dj < 0.) {
                        let t = di / (di - dj);
                        poly.push((a.lerp(b, t), uv_a.lerp(uv_b, t)));
                    }
                }
                for i in 1..poly.len().saturating_sub(1) {
                    let tri = [poly[0], poly[i], poly[i + 1]];
                    if (tri[1].0 - tri[0].0)
                        .cross(tri[2].0 - tri[0].0)
                        .length_squared()
                        > 1e-10
                    {
                        let middle = (tri[0].0 + tri[1].0 + tri[2].0) / 3.;
                        let field = g.water(origin + Vec2::new(middle.x, middle.z));
                        if field.along == 0. {
                            lake_vertices.extend(tri);
                        } else {
                            vertices.extend(tri);
                        }
                    }
                }
            }
        }
    }
    (make_water_mesh(vertices), make_water_mesh(lake_vertices))
}
fn make_water_mesh(vertices: Vec<(Vec3, Vec2)>) -> Mesh {
    let positions: Vec<_> = vertices.iter().map(|v| v.0.to_array()).collect();
    let uvs: Vec<_> = vertices.iter().map(|v| v.1.to_array()).collect();
    let mut normals = Vec::new();
    for tri in vertices.chunks_exact(3) {
        normals.extend(
            [(tri[1].0 - tri[0].0)
                .cross(tri[2].0 - tri[0].0)
                .normalize()
                .to_array(); 3],
        );
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
}

struct Resident {
    root: Entity,
    meshes: Vec<AssetId<Mesh>>,
    trees: Vec<(Vec2, f32)>,
    obstacles: Vec<TreeObstacle>,
    bytes: usize,
    water_vertices: usize,
}
#[derive(Resource)]
pub struct StreamWorld {
    generator: Generator,
    settings: TreeSettings,
    center: ChunkKey,
    loaded: BTreeMap<ChunkKey, Resident>,
    pending: BTreeMap<ChunkKey, Task<ChunkData>>,
    terrain_material: Handle<StandardMaterial>,
    trees_material: Handle<StandardMaterial>,
    grass_material: Handle<StandardMaterial>,
    water_material: Handle<StandardMaterial>,
    lake_material: Handle<StandardMaterial>,
    pub generated: u64,
    pub evicted: u64,
    pub last_build_ms: f64,
    pub last_install_ms: f64,
}
impl StreamWorld {
    pub fn river_sources(&self) -> usize {
        self.generator.hydrology.source_count()
    }
    pub fn watershed_count(&self) -> usize {
        self.generator.hydrology.len()
    }
    pub fn mesh_asset_count(&self) -> usize {
        self.loaded.values().map(|r| r.meshes.len()).sum()
    }
    pub fn water_vertices(&self) -> usize {
        self.loaded.values().map(|r| r.water_vertices).sum()
    }
    pub fn river_bank(&self, world: &Meadow, from: Vec2) -> Option<(Vec2, f32)> {
        let mut best: Option<(f32, Vec2, f32)> = None;
        let regions: BTreeSet<_> = self
            .loaded
            .keys()
            .map(|k| crate::watershed::key(k.origin() + Vec2::splat(CHUNK_SIZE * 0.5)))
            .collect();
        for k in regions {
            let Some(plan) = self.generator.hydrology.get(k) else {
                continue;
            };
            for segment in &plan.segments {
                let middle = segment.a.p.lerp(segment.b.p, 0.5);
                let normal = (segment.b.p - segment.a.p).normalize().perp();
                for side in [-1., 1.] {
                    for offset in [10., 16., 24., 32.] {
                        let p = middle + normal * (segment.a.width + offset) * side;
                        if !self.has_ground(p) || !world.walkable(p) || self.tree_blocks(p, None) {
                            continue;
                        }
                        let score = from.distance_squared(p);
                        let toward = middle - p;
                        let yaw = (-toward.x).atan2(-toward.y);
                        if best.is_none_or(|b| score < b.0) {
                            best = Some((score, p, yaw));
                        }
                    }
                }
            }
        }
        best.map(|(_, p, yaw)| (p, yaw))
    }
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    pub fn mesh_mib(&self) -> f64 {
        self.loaded.values().map(|c| c.bytes).sum::<usize>() as f64 / (1024. * 1024.)
    }
    pub fn tree_count(&self) -> usize {
        self.loaded.values().map(|c| c.trees.len()).sum()
    }
    pub fn has_ground(&self, p: Vec2) -> bool {
        self.loaded.contains_key(&ChunkKey::at(p))
    }
    pub fn landmark_trees(&self) -> impl Iterator<Item = (Vec2, f32)> + '_ {
        self.loaded
            .values()
            .flat_map(|c| c.obstacles.iter())
            .filter(|t| t.landmark)
            .map(|t| (t.p, t.height))
    }
    pub fn tree_blocks(&self, p: Vec2, height: Option<f32>) -> bool {
        let key = ChunkKey::at(p);
        for z in key.1 - 1..=key.1 + 1 {
            for x in key.0 - 1..=key.0 + 1 {
                if let Some(chunk) = self.loaded.get(&ChunkKey(x, z))
                    && chunk.obstacles.iter().any(|tree| {
                        p.distance(tree.p) < 0.65 + tree.radius
                            && height
                                .is_none_or(|h| h < self.generator.ground(tree.p) + tree.height)
                    })
                {
                    return true;
                }
            }
        }
        false
    }
}

pub fn setup(
    mut commands: Commands,
    world: Res<Meadow>,
    settings: Res<TreeSettings>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    river: Option<Res<crate::water::RiverMaterial>>,
) {
    let Some(generator) = world.stream_generator() else {
        return;
    };
    commands.insert_resource(StreamWorld {
        generator,
        settings: *settings,
        center: ChunkKey::at(Vec2::ZERO),
        loaded: BTreeMap::new(),
        pending: BTreeMap::new(),
        terrain_material: materials.add(StandardMaterial {
            perceptual_roughness: 1.,
            ..default()
        }),
        trees_material: materials.add(StandardMaterial {
            perceptual_roughness: 1.,
            ..default()
        }),
        grass_material: materials.add(StandardMaterial {
            perceptual_roughness: 1.,
            cull_mode: None,
            ..default()
        }),
        lake_material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.22, 0.48, 0.57),
            perceptual_roughness: 0.32,
            reflectance: 0.4,
            ..default()
        }),
        water_material: river
            .map(|r| r.0.clone())
            .unwrap_or_else(|| materials.add(StandardMaterial::default())),
        generated: 0,
        evicted: 0,
        last_build_ms: 0.,
        last_install_ms: 0.,
    });
}

#[allow(clippy::too_many_arguments)]
pub fn update(
    mut commands: Commands,
    stream: Option<ResMut<StreamWorld>>,
    horse: Single<&Transform, With<HorseController>>,
    layer: Res<MapLayer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut lab: ResMut<crate::LabState>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(mut stream) = stream else {
        return;
    };
    let p = Vec2::new(horse.translation.x, horse.translation.z);
    stream.center = ChunkKey::at(p);
    let center = stream.center;
    let unlit = *layer != MapLayer::Natural;
    if materials
        .get(&stream.terrain_material)
        .is_some_and(|m| m.unlit != unlit || m.fog_enabled == unlit)
        && let Some(mut material) = materials.get_mut(&stream.terrain_material)
    {
        material.unlit = unlit;
        material.fog_enabled = !unlit;
    }
    let remove: Vec<_> = stream
        .loaded
        .keys()
        .copied()
        .filter(|key| key.distance(center) > RADIUS)
        .collect();
    for key in remove {
        let chunk = stream.loaded.remove(&key).unwrap();
        commands.entity(chunk.root).despawn();
        for id in chunk.meshes {
            meshes.remove(id);
        }
        stream.evicted += 1;
    }
    // At most one finished chunk is installed per frame; CPU work never blocks input.
    let keys: Vec<_> = stream.pending.keys().copied().collect();
    for key in keys {
        if let Some(mut data) = block_on(poll_once(stream.pending.get_mut(&key).unwrap())) {
            stream.pending.remove(&key);
            if key.distance(center) > RADIUS {
                continue;
            }
            let start = Instant::now();
            data.surface
                .insert_attribute(Mesh::ATTRIBUTE_COLOR, data.colors.colors(*layer));
            let surface = meshes.add(data.surface);
            let trees = (data.trees_mesh.count_vertices() > 0).then(|| meshes.add(data.trees_mesh));
            let grass = (data.grass.count_vertices() > 0).then(|| meshes.add(data.grass));
            let water = (data.water.count_vertices() > 0).then(|| meshes.add(data.water));
            let lake = (data.lake.count_vertices() > 0).then(|| meshes.add(data.lake));
            // Bevy's GPU slab allocator does not allocate zero-length buffers.
            let mut mesh_ids = vec![surface.id()];
            mesh_ids.extend(
                [
                    trees.as_ref(),
                    grass.as_ref(),
                    water.as_ref(),
                    lake.as_ref(),
                ]
                .into_iter()
                .flatten()
                .map(|m| m.id()),
            );
            let origin = key.origin();
            let root = commands
                .spawn((
                    Name::new(format!("Chunk {},{}", key.0, key.1)),
                    Transform::from_xyz(origin.x, 0., origin.y),
                    Visibility::default(),
                ))
                .with_children(|root| {
                    root.spawn((
                        TerrainSurface,
                        data.colors,
                        Mesh3d(surface.clone()),
                        MeshMaterial3d(stream.terrain_material.clone()),
                    ));
                    if let Some(trees) = trees {
                        root.spawn((Mesh3d(trees), MeshMaterial3d(stream.trees_material.clone())));
                    }
                    if let Some(water) = water {
                        root.spawn((
                            Mesh3d(water),
                            MeshMaterial3d(stream.water_material.clone()),
                            bevy::light::NotShadowCaster,
                        ));
                    }
                    if let Some(lake) = lake {
                        root.spawn((
                            Mesh3d(lake),
                            MeshMaterial3d(stream.lake_material.clone()),
                            bevy::light::NotShadowCaster,
                        ));
                    }
                    if let Some(grass) = grass {
                        root.spawn((
                            Grass,
                            Mesh3d(grass),
                            MeshMaterial3d(stream.grass_material.clone()),
                            if *layer == MapLayer::Natural {
                                Visibility::Inherited
                            } else {
                                Visibility::Hidden
                            },
                        ));
                    }
                })
                .id();
            stream.loaded.insert(
                key,
                Resident {
                    root,
                    meshes: mesh_ids,
                    water_vertices: data.water_vertices,
                    trees: data.trees,
                    obstacles: data.obstacles,
                    bytes: data.bytes,
                },
            );
            stream.generated += 1;
            stream.last_build_ms = data.elapsed_ms;
            stream.last_install_ms = start.elapsed().as_secs_f64() * 1000.;
            lab.scene_setup_ms += stream.last_install_ms;
            break;
        }
    }
    let mut keep = BTreeSet::from([crate::watershed::key(Vec2::ZERO)]);
    keep.extend(stream.pending.keys().flat_map(|k| tree_regions(*k)));
    let mut wanted = Vec::new();
    for z in center.1 - RADIUS..=center.1 + RADIUS {
        for x in center.0 - RADIUS..=center.0 + RADIUS {
            let key = ChunkKey(x, z);
            keep.extend(tree_regions(key));
            if !stream.loaded.contains_key(&key) && !stream.pending.contains_key(&key) {
                wanted.push(key);
            }
        }
    }
    stream.generator.hydrology.retain(&keep);
    wanted.sort_by_key(|key| {
        (
            (key.origin() + Vec2::splat(CHUNK_SIZE * 0.5)).distance_squared(p) as u64,
            *key,
        )
    });
    for key in wanted.into_iter().take(MAX_TASKS - stream.pending.len()) {
        let generator = stream.generator.clone();
        let settings = stream.settings;
        stream.pending.insert(
            key,
            AsyncComputeTaskPool::get().spawn(async move { generate(generator, key, settings) }),
        );
    }
}

pub fn generation_report(generator: Generator, settings: TreeSettings) {
    let mut sampler = crate::monitor::ProcessSampler::new();
    let before = sampler.snapshot();
    let start = Instant::now();
    let mut chunks = Vec::new();
    for z in -1..=1 {
        for x in -1..=1 {
            chunks.push(generate(generator.clone(), ChunkKey(x, z), settings));
        }
    }
    let wall = start.elapsed().as_secs_f64() * 1000.;
    let usage = crate::monitor::GenerationUsage::between(before, sampler.snapshot());
    let mean = chunks.iter().map(|c| c.elapsed_ms).sum::<f64>() / chunks.len() as f64;
    let maximum = chunks.iter().map(|c| c.elapsed_ms).fold(0.0, f64::max);
    let mib = chunks.iter().map(|c| c.bytes).sum::<usize>() as f64 / (1024. * 1024.);
    let tree_count = chunks.iter().map(|c| c.trees.len()).sum::<usize>();
    let optional = |n: Option<f64>| n.map(|v| format!("{v:.3}")).unwrap_or_default();
    std::fs::create_dir_all("reports").expect("Cannot create reports directory");
    let path = format!("reports/stream-seed-{}.csv", generator.seed);
    std::fs::write(&path,format!("seed,stream_generator_version,mountain_strength,retained_chunks,chunk_width_m,grid_side,cell_size_m,total_wall_ms,mean_chunk_ms,max_chunk_ms,cpu_mesh_mib,trees,process_cpu_ms,rss_delta_mib\n{},{},{},9,{},{},{},{wall:.3},{mean:.3},{maximum:.3},{mib:.3},{tree_count},{},{}\n",generator.seed,VERSION,generator.settings.mountain_strength,CHUNK_SIZE,SIDE,CELL,optional(usage.cpu_ms),optional(usage.rss_delta_mib))).expect("Cannot save streaming report");
    println!(
        "Saved {path}: 9 CPU-built chunks retained, {wall:.1} ms total, {mean:.1} ms mean, {mib:.1} MiB mesh data; excludes GPU/model loading; synchronous benchmark, not runtime frame latency"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn generator() -> Generator {
        Generator::new(42, TerrainSettings::default())
    }
    #[test]
    fn tree_isolation_uses_adjacent_chunks_and_survives_cache_eviction() {
        for seed in [42, 20261003] {
            let g = Generator::new(seed, TerrainSettings::default());
            for key in [
                ChunkKey(3, 3),
                ChunkKey(4, 4),
                ChunkKey(-4, -4),
                ChunkKey(-5, -5),
            ] {
                let context = g.tree_neighbours(key);
                let owned = g.trees(key);
                let mut full = Vec::new();
                for z in key.1 - 1..=key.1 + 1 {
                    for x in key.0 - 1..=key.0 + 1 {
                        g.tree_neighbours(ChunkKey(x, z));
                        full.extend(g.trees(ChunkKey(x, z)));
                    }
                }
                let world = Meadow::from_stream_generator(g.clone());
                let shapes: Vec<_> = owned
                    .iter()
                    .map(|(p, size)| {
                        let amount = crate::trees::isolation(*p, &context);
                        assert!((amount - crate::trees::isolation(*p, &full)).abs() < 1e-5);
                        TreeShape::with_isolation(
                            &world,
                            *p,
                            *size,
                            TreeSettings::default(),
                            amount,
                        )
                    })
                    .collect();
                g.hydrology.retain(&BTreeSet::new());
                let again = g.tree_neighbours(key);
                assert_eq!(context, again);
                for ((p, size), shape) in owned.iter().zip(shapes) {
                    assert_eq!(
                        shape,
                        TreeShape::with_isolation(
                            &world,
                            *p,
                            *size,
                            TreeSettings::default(),
                            crate::trees::isolation(*p, &again)
                        )
                    );
                }
            }
        }
    }
    #[test]
    fn asynchronous_chunks_stay_bounded_and_release_assets_on_travel() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(Meadow::streamed(42, TerrainSettings::default()))
            .init_resource::<TreeSettings>()
            .init_resource::<MapLayer>()
            .init_resource::<crate::LabState>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .add_systems(Startup, setup)
            .add_systems(Update, update);
        let horse = app
            .world_mut()
            .spawn((
                HorseController {
                    speed: 0.0,
                    yaw: 0.0,
                },
                Transform::default(),
            ))
            .id();
        let settle = |app: &mut App| {
            let start = Instant::now();
            loop {
                app.update();
                let stream = app.world().resource::<StreamWorld>();
                assert!(stream.loaded_count() <= MAX_CHUNKS && stream.pending_count() <= MAX_TASKS);
                assert!(stream.watershed_count() <= 13);
                assert!(app.world().resource::<Assets<Mesh>>().len() <= MAX_CHUNKS * 5);
                if stream.loaded_count() == MAX_CHUNKS && stream.pending_count() == 0 {
                    break;
                }
                assert!(
                    start.elapsed().as_secs() < 15,
                    "Chunk generation did not settle"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        settle(&mut app);
        let old = app.world().resource::<StreamWorld>().loaded[&ChunkKey(0, 0)].root;
        let trees = app.world().resource::<StreamWorld>().loaded[&ChunkKey(0, 0)]
            .trees
            .clone();
        for p in [
            Vec2::new(1260., -880.),
            Vec2::new(-1260., -1040.),
            Vec2::ZERO,
        ] {
            app.world_mut()
                .get_mut::<Transform>(horse)
                .unwrap()
                .translation = Vec3::new(p.x, 0., p.y);
            settle(&mut app);
            assert_eq!(
                app.world().resource::<Assets<Mesh>>().len(),
                app.world().resource::<StreamWorld>().mesh_asset_count()
            );
            let stream = app.world().resource::<StreamWorld>();
            assert!(stream.has_ground(p));
            assert!(
                stream
                    .loaded
                    .keys()
                    .all(|key| key.distance(ChunkKey::at(p)) <= RADIUS)
            );
        }
        assert!(app.world().get_entity(old).is_err());
        assert_eq!(
            app.world().resource::<StreamWorld>().loaded[&ChunkKey(0, 0)].trees,
            trees
        );
        assert!(app.world().resource::<StreamWorld>().evicted >= 3 * MAX_CHUNKS as u64);
    }

    #[test]
    fn streamed_water_seams_uvs_shorelines_and_revisit_match() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            for region in [(0, 0), (2, -2), (-2, -2)] {
                g.hydrology.prepare(&g, region);
                let plan = g.hydrology.get(region).unwrap();
                for axis in [0, 1] {
                    let segment = plan
                        .segments
                        .iter()
                        .find(|s| {
                            (s.a.p[axis] / CHUNK_SIZE).floor() != (s.b.p[axis] / CHUNK_SIZE).floor()
                        })
                        .expect("River must span render chunks");
                    let start = segment.a.p;
                    let end = segment.b.p;
                    let lo = start[axis].min(end[axis]);
                    let hi = start[axis].max(end[axis]);
                    let seam = (lo / CHUNK_SIZE).floor().mul_add(CHUNK_SIZE, CHUNK_SIZE);
                    let t = (seam - start[axis]) / (end[axis] - start[axis]);
                    assert!(seam <= hi && (0.0..=1.).contains(&t));
                    let crossing = start.lerp(end, t);
                    let mut left = crossing;
                    left[axis] = seam - 0.1;
                    let mut right = crossing;
                    right[axis] = seam + 0.1;
                    let keys = [ChunkKey::at(left), ChunkKey::at(right)];
                    let chunks = keys.map(|k| generate(g.clone(), k, TreeSettings::default()));
                    let boundary = |chunk: &ChunkData, key: ChunkKey| {
                        let combined = [chunk.water.clone(), chunk.lake.clone()];
                        let mut values = Vec::new();
                        for mesh in &combined {
                            let Some(VertexAttributeValues::Float32x3(p)) =
                                mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                            else {
                                panic!()
                            };
                            let Some(VertexAttributeValues::Float32x2(uv)) =
                                mesh.attribute(Mesh::ATTRIBUTE_UV_0)
                            else {
                                panic!()
                            };
                            for (p, uv) in p.iter().zip(uv) {
                                let p =
                                    Vec3::from(*p) + Vec3::new(key.origin().x, 0., key.origin().y);
                                let q = Vec2::new(p.x, p.z);
                                assert!(
                                    g.ground(q) <= p.y + 0.003,
                                    "Water must cover the carved bed: {p:?}"
                                );
                                if (q[axis] - seam).abs() < 0.0001 {
                                    values.push(
                                        [p.x, p.y, p.z, uv[0], uv[1]]
                                            .map(|n| (n * 1000.).round() as i64),
                                    );
                                }
                            }
                        }
                        values.sort();
                        values.dedup();
                        values
                    };
                    let a = boundary(&chunks[0], keys[0]);
                    let b = boundary(&chunks[1], keys[1]);
                    assert!(a.len() >= 2, "Must check an actual wet seam");
                    assert_eq!(a, b, "Water position/phase mismatch at {crossing:?}");
                    let repeated = generate(g.clone(), keys[0], TreeSettings::default());
                    assert_eq!(
                        chunks[0].water.attribute(Mesh::ATTRIBUTE_POSITION),
                        repeated.water.attribute(Mesh::ATTRIBUTE_POSITION)
                    );
                    assert_eq!(
                        chunks[0].water.attribute(Mesh::ATTRIBUTE_UV_0),
                        repeated.water.attribute(Mesh::ATTRIBUTE_UV_0)
                    );
                }
                for s in &plan.segments {
                    let p = s.a.p.lerp(s.b.p, 0.5);
                    let f = g.water(p);
                    assert!(
                        g.ground(p) < f.level,
                        "Channel needs a submerged bed at {p:?}: water {:?}, ground {}",
                        f,
                        g.ground(p)
                    );
                    assert_eq!(g.habitat(p).tree_density, 0.);
                    assert_eq!(g.habitat(p).grass_density, 0.);
                }
            }
        }
    }

    #[test]
    fn coordinates_edges_and_revisit_are_stable() {
        assert_eq!(ChunkKey::at(Vec2::new(-0.1, -96.1)), ChunkKey(-1, -2));
        let g = generator();
        let a = generate(g.clone(), ChunkKey(-1, 0), TreeSettings::default());
        let b = generate(g.clone(), ChunkKey(0, 0), TreeSettings::default());
        for attribute in [Mesh::ATTRIBUTE_POSITION, Mesh::ATTRIBUTE_NORMAL] {
            let Some(VertexAttributeValues::Float32x3(left)) = a.surface.attribute(attribute)
            else {
                panic!()
            };
            let Some(VertexAttributeValues::Float32x3(right)) = b.surface.attribute(attribute)
            else {
                panic!()
            };
            for z in 0..SIDE {
                if attribute == Mesh::ATTRIBUTE_POSITION {
                    assert_eq!(left[z * SIDE + SIDE - 1][1], right[z * SIDE][1]);
                } else {
                    assert_eq!(left[z * SIDE + SIDE - 1], right[z * SIDE]);
                }
            }
        }
        let again = generate(g.clone(), ChunkKey(-1, 0), TreeSettings::default());
        assert_eq!(a.trees, again.trees);
        assert_eq!(
            a.surface.attribute(Mesh::ATTRIBUTE_POSITION),
            again.surface.attribute(Mesh::ATTRIBUTE_POSITION)
        );
        let other = generate(
            Generator::new(43, g.settings),
            ChunkKey(-1, 0),
            TreeSettings::default(),
        );
        assert_ne!(
            a.surface.attribute(Mesh::ATTRIBUTE_POSITION),
            other.surface.attribute(Mesh::ATTRIBUTE_POSITION)
        );
        for p in [
            Vec2::new(-0.3, 91.7),
            Vec2::new(192.8, -307.2),
            Vec2::new(1536.4, 902.1),
        ] {
            assert!(g.ground(p).is_finite());
            let cell = (p / CELL).floor() * CELL;
            let f = (p - cell) / CELL;
            let vertices = if f.x + f.y <= 1. {
                [
                    (cell, 1. - f.x - f.y),
                    (cell + Vec2::X * CELL, f.x),
                    (cell + Vec2::Y * CELL, f.y),
                ]
            } else {
                [
                    (cell + Vec2::splat(CELL), f.x + f.y - 1.),
                    (cell + Vec2::Y * CELL, 1. - f.x),
                    (cell + Vec2::X * CELL, 1. - f.y),
                ]
            };
            let expected: f32 = vertices.into_iter().map(|(p, w)| g.height(p) * w).sum();
            assert!((g.ground(p) - expected).abs() < 0.0001);
        }
    }
}
