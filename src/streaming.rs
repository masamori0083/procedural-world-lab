//! Coordinate-seeded terrain, bounded asynchronous chunk generation and eviction.
use crate::{
    environment::{EnvironmentSample, MapLayer},
    player::HorseController,
    terrain::{CELL, Landform, Meadow, SeedRandom, TerrainSettings},
    terrain_lod::{self, Detail as TerrainDetail},
    trees::{TreeSettings, TreeShape},
    vegetation::{self, Blueprint, Levels},
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

pub const VERSION: u32 = 14;
pub const CHUNK_SIZE: f32 = 96.0;
pub const RADIUS: i32 = 3;
pub const MAX_CHUNKS: usize = ((RADIUS * 2 + 1) * (RADIUS * 2 + 1)) as usize;
const MAX_TASKS: usize = 2;
const SIDE: usize = (CHUNK_SIZE / CELL) as usize + 1;
const TREE_CELL: f32 = 12.0;
const LOOKAHEAD_SECONDS: f32 = 2.0;

#[derive(Resource)]
pub struct PriorityConfig {
    pub enabled: bool,
}
impl Default for PriorityConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Work {
    Chunk(ChunkKey),
    Terrain(ChunkKey),
    Vegetation(ChunkKey),
}
impl Work {
    fn key(self) -> ChunkKey {
        match self {
            Self::Chunk(key) | Self::Terrain(key) | Self::Vegetation(key) => key,
        }
    }
}

#[derive(Clone, Copy)]
struct Motion {
    position: Vec2,
    predicted: Vec2,
}
impl Motion {
    fn new(transform: &Transform, horse: &HorseController) -> Self {
        let position = Vec2::new(transform.translation.x, transform.translation.z);
        let velocity = transform.rotation * Vec3::NEG_Z * horse.speed;
        let ahead =
            (Vec2::new(velocity.x, velocity.z) * LOOKAHEAD_SECONDS).clamp_length_max(CHUNK_SIZE);
        Self {
            position,
            predicted: position + ahead,
        }
    }
    // First intersection with a chunk's horizontal bounds. Corners and negative
    // coordinates use the same grid as chunk ownership; no extra chunks are kept.
    fn entry(self, key: ChunkKey) -> Option<f32> {
        let delta = self.predicted - self.position;
        if delta.length_squared() < 0.0001 {
            return None;
        }
        let lo = key.origin();
        let hi = lo + Vec2::splat(CHUNK_SIZE);
        let mut enter = 0_f32;
        let mut exit = 1_f32;
        for axis in 0..2 {
            if delta[axis].abs() < 0.00001 {
                if self.position[axis] < lo[axis] || self.position[axis] >= hi[axis] {
                    return None;
                }
            } else {
                let a = (lo[axis] - self.position[axis]) / delta[axis];
                let b = (hi[axis] - self.position[axis]) / delta[axis];
                enter = enter.max(a.min(b));
                exit = exit.min(a.max(b));
                if enter > exit {
                    return None;
                }
            }
        }
        if enter == exit && ChunkKey::at(self.predicted) != key {
            None
        } else {
            Some(enter)
        }
    }
    fn priority(self, work: Work, detail: usize) -> (u8, usize, u64, ChunkKey) {
        let key = work.key();
        let foot = key == ChunkKey::at(self.position);
        let distance = self.position.distance_squared(
            self.position
                .clamp(key.origin(), key.origin() + Vec2::splat(CHUNK_SIZE)),
        ) as u64;
        let entry = self.entry(key);
        let (rank, distance) = match work {
            Work::Chunk(_) if foot => (0, 0),
            Work::Terrain(_) if foot => (1, 0),
            Work::Chunk(_) if entry.is_some() => (2, (entry.unwrap() * 1_000_000.) as u64),
            Work::Chunk(_) => (3, distance),
            Work::Terrain(_) => (4, distance),
            Work::Vegetation(_) => (5, distance),
        };
        (rank, detail, distance, key)
    }
}

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
        let field = self.water(p);
        let height = crate::watershed::carve(self.base_height(p), field);
        self.hydrology.ford_height(p, height, field)
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
    /// Water depth on the SAME triangle plane clipped by water_meshes.
    /// Analytic field levels can differ at a bend or river/lake junction.
    pub fn water_depth(&self, p: Vec2) -> f32 {
        let cell = (p / CELL).floor() * CELL;
        let f = (p - cell) / CELL;
        let (points, weights) = if f.x + f.y <= 1. {
            (
                [cell, cell + Vec2::X * CELL, cell + Vec2::Y * CELL],
                [1. - f.x - f.y, f.x, f.y],
            )
        } else {
            (
                [
                    cell + Vec2::splat(CELL),
                    cell + Vec2::Y * CELL,
                    cell + Vec2::X * CELL,
                ],
                [f.x + f.y - 1., 1. - f.x, 1. - f.y],
            )
        };
        let fields = points.map(|q| self.water(q));
        if !fields.iter().any(|f| f.distance < f.width + 1.) {
            return f32::NEG_INFINITY;
        }
        fields
            .iter()
            .zip(weights)
            .map(|(f, w)| f.level * w)
            .sum::<f32>()
            - self.ground(p)
    }

    /// Distance estimate for vegetation, measured from the wet bed profile.
    /// Wet rendered terrain always has zero vegetation potential.
    fn bank_distance(&self, p: Vec2, ground: f32) -> f32 {
        let f = self.water(p);
        let distance = (f.distance - crate::watershed::wet_radius(f.width)).max(0.);
        if distance < CELL * 2. && (ground <= f.level || self.water_depth(p) >= -0.015) {
            0.
        } else {
            distance
        }
    }

    pub fn water_distance(&self, p: Vec2) -> f32 {
        self.bank_distance(p, self.ground(p))
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
        let mut sample = EnvironmentSample::evaluate(
            height,
            height - self.ground(Vec2::ZERO),
            self.slope(p),
            self.bank_distance(p, height),
            self.noise(p * 0.018 + Vec2::new(219., -53.)),
            self.noise(p * 0.035 + Vec2::splat(31.)),
        );
        sample.tree_density *=
            smooth((p.length() - 22.) / 10.) * (1. - self.hydrology.ford_strength(p));
        if self.hydrology.ford_route(p) {
            sample.tree_density = 0.;
        }
        sample.grass_density *= 1. - 0.7 * self.hydrology.ford_strength(p);
        sample
    }
    pub fn nearest_ford(&self, p: Vec2) -> Option<crate::watershed::Ford> {
        self.hydrology
            .get(crate::watershed::key(p))?
            .fords
            .iter()
            .copied()
            .min_by(|a, b| {
                a.center
                    .distance_squared(p)
                    .total_cmp(&b.center.distance_squared(p))
            })
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
                if chance < sample.tree_density * 0.95
                    && sample.slope < 0.65
                    && sample.water_distance > 3.
                    && self.water_depth(p) < -0.015
                {
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
    surface_detail: TerrainDetail,
    surface_bytes: usize,
    stones: Mesh,
    water: Mesh,
    lake: Mesh,
    water_vertices: usize,
    vegetation: Arc<Blueprint>,
    built: vegetation::Built,
    colors: ChunkSurface,
    trees: Vec<(Vec2, f32)>,
    obstacles: Vec<TreeObstacle>,
    elapsed_ms: f64,
    bytes: usize,
}
#[derive(Default)]
pub(crate) struct Batch {
    pub(crate) positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
}
impl Batch {
    pub(crate) fn append(&mut self, mesh: Mesh, offset: Vec3) {
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
    pub(crate) fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]) {
        self.positions
            .extend([a.to_array(), b.to_array(), c.to_array()]);
        self.normals
            .extend([(b - a).cross(c - a).normalize().to_array(); 3]);
        self.colors.extend([color; 3]);
    }
    pub(crate) fn mesh(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
    }
}
#[cfg(test)]
fn generate(generator: Generator, key: ChunkKey, settings: TreeSettings) -> ChunkData {
    generate_at(generator, key, settings, None, TerrainDetail::Near)
}
fn generate_at(
    generator: Generator,
    key: ChunkKey,
    settings: TreeSettings,
    view: Option<vegetation::View>,
    surface_detail: TerrainDetail,
) -> ChunkData {
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
        }
    }
    let (water, lake) = water_meshes(&generator, origin, &positions);
    let water_vertices = water.count_vertices() + lake.count_vertices();
    let neighbours = generator.tree_neighbours(key);
    let trees: Vec<_> = neighbours
        .iter()
        .copied()
        .filter(|(p, _)| ChunkKey::at(*p) == key)
        .collect();
    let world = Meadow::from_stream_generator(generator.clone());
    let mut placed = Vec::new();
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
        placed.push(vegetation::PlacedTree::new(tree, offset));
    }
    let mut blades = Vec::new();
    let mut random = SeedRandom(coordinate_seed(generator.seed ^ 0x581c_aa42, key.0, key.1));
    for _ in 0..480 {
        let q = Vec2::new(random.range(0., CHUNK_SIZE), random.range(0., CHUNK_SIZE));
        let p = origin + q;
        let sample = generator.habitat(p);
        if random.unit() > sample.grass_density
            || sample.slope > 0.75
            || generator.water_depth(p) >= -0.015
        {
            continue;
        }
        let height = generator.ground(p);
        let direction = Vec2::from_angle(random.range(0., std::f32::consts::TAU)) * 0.09;
        let a = Vec3::new(q.x - direction.x, height, q.y - direction.y);
        let b = Vec3::new(q.x + direction.x, height, q.y + direction.y);
        let c = Vec3::new(q.x, height + random.range(0.18, 0.38), q.y);
        blades.push(vegetation::Blade {
            a,
            b,
            c,
            color: {
                let c =
                    Vec3::new(0.30, 0.33, 0.09).lerp(Vec3::new(0.14, 0.35, 0.07), sample.moisture);
                [c.x, c.y, c.z, 1.]
            },
        });
    }
    // Small riparian bushes share the grass batch/LOD; no extra draw calls or collision.
    let mut shrubs = Vec::new();
    let mut random = SeedRandom(coordinate_seed(generator.seed ^ 0x21b5_724d, key.0, key.1));
    for _ in 0..64 {
        let q = Vec2::new(random.range(0., CHUNK_SIZE), random.range(0., CHUNK_SIZE));
        let p = origin + q;
        let sample = generator.habitat(p);
        if sample.water_distance < 1.
            || sample.water_distance > 18.
            || sample.slope > 0.6
            || random.unit() > sample.grass_density * 0.75
            || generator.water_depth(p) >= -0.015
        {
            continue;
        }
        let radius = random.range(0.35, 0.85);
        shrubs.push(vegetation::Shrub {
            center: Vec3::new(q.x, generator.ground(p), q.y),
            radius,
            color: [0.19, 0.36 + random.range(-0.03, 0.03), 0.10, 1.],
        });
    }
    let vegetation = Arc::new(Blueprint {
        trees: placed,
        grass: blades,
        shrubs,
        origin: Vec3::new(origin.x, 0., origin.y),
    });
    let built = vegetation.build(vegetation.levels(view, None));
    let stones = ford_stone_mesh(&ford_stones(&generator, key), origin);
    let ground = simplify_surface(
        &generator,
        key,
        surface_detail,
        &trees,
        positions,
        normals,
        samples,
        patches,
    );
    let bytes = ground.bytes
        + built.vertices * 40
        + vegetation.bytes()
        + water_vertices * 48
        + stones.count_vertices() * 40;
    ChunkData {
        surface: ground.mesh,
        surface_detail,
        surface_bytes: ground.bytes,
        stones,
        water,
        lake,
        water_vertices,
        vegetation,
        built,
        colors: ground.colors,
        trees,
        obstacles,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.,
        bytes,
    }
}

struct SurfaceBuilt {
    mesh: Mesh,
    colors: ChunkSurface,
    detail: TerrainDetail,
    bytes: usize,
    elapsed_ms: f64,
}

#[allow(clippy::too_many_arguments)]
fn simplify_surface(
    g: &Generator,
    key: ChunkKey,
    detail: TerrainDetail,
    trees: &[(Vec2, f32)],
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    samples: Vec<EnvironmentSample>,
    patches: Vec<f32>,
) -> SurfaceBuilt {
    let start = Instant::now();
    let heights: Vec<_> = positions.iter().map(|p| p[1]).collect();
    let protected: Vec<_> = positions
        .iter()
        .map(|p| {
            if detail == TerrainDetail::Near {
                return false;
            }
            let p = key.origin() + Vec2::new(p[0], p[2]);
            let water = g.water(p);
            water.distance <= water.width + 12.
                || trees
                    .iter()
                    .any(|(tree, _)| (p - *tree).abs().max_element() <= CELL)
        })
        .collect();
    let full_indices = terrain_lod::indices(&heights, &protected, SIDE, detail);
    let mut remap = BTreeMap::new();
    let mut selected = if detail == TerrainDetail::Near {
        (0..positions.len()).collect()
    } else {
        Vec::new()
    };
    let indices: Vec<u32> = full_indices
        .into_iter()
        .map(|index| {
            if detail == TerrainDetail::Near {
                return index;
            }
            *remap.entry(index).or_insert_with(|| {
                let next = selected.len() as u32;
                selected.push(index as usize);
                next
            })
        })
        .collect();
    let colors = ChunkSurface {
        samples: selected.iter().map(|i| samples[*i]).collect(),
        patches: selected.iter().map(|i| patches[*i]).collect(),
    };
    let bytes = selected.len() * (40 + size_of::<EnvironmentSample>() + size_of::<f32>())
        + indices.len() * 4;
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        selected.iter().map(|i| positions[*i]).collect::<Vec<_>>(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        selected.iter().map(|i| normals[*i]).collect::<Vec<_>>(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors.colors(MapLayer::Natural))
    .with_inserted_indices(Indices::U32(indices));
    SurfaceBuilt {
        mesh,
        colors,
        detail,
        bytes,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.,
    }
}

fn rebuild_surface(
    g: &Generator,
    key: ChunkKey,
    detail: TerrainDetail,
    trees: &[(Vec2, f32)],
) -> SurfaceBuilt {
    let start = Instant::now();
    for region in tree_regions(key) {
        g.hydrology.prepare(g, region);
    }
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut samples = Vec::new();
    let mut patches = Vec::new();
    for z in 0..SIDE {
        for x in 0..SIDE {
            let p = key.origin() + Vec2::new(x as f32, z as f32) * CELL;
            positions.push([x as f32 * CELL, g.height(p), z as f32 * CELL]);
            let dx = g.height(p + Vec2::X * CELL) - g.height(p - Vec2::X * CELL);
            let dz = g.height(p + Vec2::Y * CELL) - g.height(p - Vec2::Y * CELL);
            normals.push(Vec3::new(-dx, 2. * CELL, -dz).normalize().to_array());
            samples.push(g.habitat(p));
            patches.push(g.noise(p * 0.08));
        }
    }
    let mut built = simplify_surface(g, key, detail, trees, positions, normals, samples, patches);
    built.elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
    built
}

#[derive(Clone, Debug, PartialEq)]
struct FordStone {
    center: Vec3,
    radii: Vec2,
    height: f32,
    yaw: f32,
    color: [f32; 4],
}

/// Four small stone groups frame the dry approaches, leaving the crossing open.
/// Seed by the shared ford, then assign each stone to one chunk by its center.
fn ford_stones(g: &Generator, key: ChunkKey) -> Vec<FordStone> {
    let mut stones = Vec::new();
    let origin = key.origin();
    let Some(plan) = g.hydrology.get(crate::watershed::key(
        origin + Vec2::splat(CHUNK_SIZE * 0.5),
    )) else {
        return stones;
    };
    for ford in &plan.fords {
        if ford.center.x < origin.x - 30.
            || ford.center.y < origin.y - 30.
            || ford.center.x > origin.x + CHUNK_SIZE + 30.
            || ford.center.y > origin.y + CHUNK_SIZE + 30.
        {
            continue;
        }
        let cell = ford.center.floor().as_ivec2();
        let mut random = SeedRandom(coordinate_seed(g.seed ^ 0x792b_4a61, cell.x, cell.y));
        let along = ford.across.perp();
        for bank in [-1., 1.] {
            for edge in [-1., 1.] {
                for _ in 0..5 {
                    let lateral = edge * random.range(2.8, 4.8);
                    let setback = random.range(0.7, 2.3);
                    let radii = Vec2::new(random.range(0.35, 0.65), random.range(0.30, 0.55));
                    let height = random.range(0.35, 0.60);
                    let yaw = random.range(0., std::f32::consts::TAU);
                    let shade = random.range(0.38, 0.52);
                    // Find the actual dry shoreline, not the nominal river width.
                    let Some(distance) = (0..80).map(|i| 1. + i as f32 * 0.25).find(|d| {
                        let p = ford.center + along * lateral + ford.across * bank * *d;
                        g.water_depth(p) < -0.08
                    }) else {
                        continue;
                    };
                    let p =
                        ford.center + along * lateral + ford.across * bank * (distance + setback);
                    if ChunkKey::at(p) != key {
                        continue;
                    }
                    let radius = radii.max_element();
                    let mut base = g.ground(p);
                    let dry = (0..8).all(|i| {
                        let q =
                            p + Vec2::from_angle(i as f32 * std::f32::consts::TAU / 8.) * radius;
                        base = base.min(g.ground(q));
                        g.water_depth(q) < -0.015
                    });
                    if !dry {
                        continue;
                    }
                    stones.push(FordStone {
                        center: Vec3::new(p.x, base - 0.06, p.y),
                        radii,
                        height,
                        yaw,
                        color: [shade, shade * 0.98, shade * 0.93, 1.],
                    });
                }
            }
        }
    }
    stones
}

fn ford_stone_mesh(stones: &[FordStone], origin: Vec2) -> Mesh {
    let mut batch = Batch::default();
    for stone in stones {
        let center = stone.center - Vec3::new(origin.x, 0., origin.y);
        let ring = |i: usize, scale: f32, y: f32| {
            let v = Vec2::from_angle(stone.yaw + i as f32 * std::f32::consts::TAU / 6.)
                * stone.radii
                * scale;
            center + Vec3::new(v.x, y * stone.height, v.y)
        };
        let top = center + Vec3::new(stone.radii.x * 0.12, stone.height, 0.);
        for i in 0..6 {
            let a = ring(i, 0.75, 0.);
            let b = ring(i + 1, 0.75, 0.);
            let c = ring(i, 1., 0.35);
            let d = ring(i + 1, 1., 0.35);
            batch.triangle(a, c, b, stone.color);
            batch.triangle(b, c, d, stone.color);
            batch.triangle(c, top, d, stone.color);
            batch.triangle(center, a, b, stone.color);
        }
    }
    batch.mesh()
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
    (
        make_water_mesh(g, origin, vertices),
        make_water_mesh(g, origin, lake_vertices),
    )
}
fn make_water_mesh(g: &Generator, origin: Vec2, vertices: Vec<(Vec3, Vec2)>) -> Mesh {
    let positions: Vec<_> = vertices.iter().map(|v| v.0.to_array()).collect();
    let uvs: Vec<_> = vertices.iter().map(|v| v.1.to_array()).collect();
    let colors: Vec<_> = vertices
        .iter()
        .map(|v| {
            let strength = g.hydrology.ford_strength(origin + Vec2::new(v.0.x, v.0.z));
            [
                1. + strength * 0.65,
                1. + strength * 0.35,
                1. + strength * 0.20,
                1.,
            ]
        })
        .collect();
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
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
}

struct Resident {
    root: Entity,
    surface_entity: Entity,
    surface_mesh: AssetId<Mesh>,
    surface_detail: TerrainDetail,
    surface_bytes: usize,
    surface_triangles: usize,
    tree_entity: Entity,
    grass_entity: Entity,
    vegetation: Arc<Blueprint>,
    levels: Levels,
    vegetation_meshes: Vec<AssetId<Mesh>>,
    vegetation_vertices: usize,
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
    pending_lod: BTreeMap<ChunkKey, Task<vegetation::Built>>,
    pending_surface: BTreeMap<ChunkKey, Task<SurfaceBuilt>>,
    priority_enabled: bool,
    pub predicted: Vec2,
    pub queued: usize,
    pub ground_wait_ms: f64,
    pub max_ground_wait_ms: f64,
    pub ground_wait_events: u64,
    ground_blocked: bool,
    ground_wait_run_ms: f64,
    near_wait: Option<(ChunkKey, Instant)>,
    pub near_wait_ms: f64,
    pub last_near_wait_ms: f64,
    pub max_near_wait_ms: f64,
    terrain_lod_enabled: bool,
    pub terrain_rebuilt: u64,
    pub last_terrain_ms: f64,
    lod_enabled: bool,
    pub lod_rebuilt: u64,
    pub last_lod_ms: f64,
    pub last_lod_install_ms: f64,
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
    pub fn scheduler_name(&self) -> &'static str {
        if self.priority_enabled {
            "PRIORITY"
        } else {
            "DISTANCE"
        }
    }
    pub fn record_ground_wait(&mut self, seconds: f64) {
        if seconds <= 0. {
            return;
        }
        if self.ground_wait_run_ms == 0. {
            self.ground_wait_events += 1;
        }
        self.ground_blocked = true;
        self.ground_wait_run_ms += seconds * 1000.;
        self.ground_wait_ms += seconds * 1000.;
        self.max_ground_wait_ms = self.max_ground_wait_ms.max(self.ground_wait_run_ms);
    }
    fn measure_near_wait(&mut self, key: ChunkKey) {
        let detailed = self
            .loaded
            .get(&key)
            .is_some_and(|chunk| chunk.surface_detail == TerrainDetail::Near);
        if detailed {
            if let Some((target, start)) = self.near_wait.take()
                && target == key
            {
                self.last_near_wait_ms = start.elapsed().as_secs_f64() * 1000.;
                self.max_near_wait_ms = self.max_near_wait_ms.max(self.last_near_wait_ms);
            }
            self.near_wait_ms = 0.;
        } else {
            if self.near_wait.is_none_or(|(target, _)| target != key) {
                self.near_wait = Some((key, Instant::now()));
            }
            self.near_wait_ms = self.near_wait.unwrap().1.elapsed().as_secs_f64() * 1000.;
        }
    }
    fn priority(
        &self,
        work: Work,
        motion: Motion,
        view: Option<vegetation::View>,
    ) -> (u8, usize, u64, ChunkKey) {
        let key = work.key();
        let detail = match work {
            Work::Chunk(_) => 0,
            Work::Terrain(_) => self.loaded.get(&key).map_or(3, |chunk| {
                if self.terrain_lod_enabled {
                    TerrainDetail::choose(key.origin(), motion.position, Some(chunk.surface_detail))
                        .index()
                } else {
                    0
                }
            }),
            Work::Vegetation(_) => self.loaded.get(&key).map_or(9, |chunk| {
                let desired = chunk.vegetation.levels(view, Some(chunk.levels));
                desired.trees.index() * 3 + desired.grass.index()
            }),
        };
        if !self.priority_enabled {
            let distance = match work {
                Work::Chunk(_) => (key.origin() + Vec2::splat(CHUNK_SIZE * 0.5))
                    .distance_squared(motion.position) as u64,
                Work::Terrain(_) => key.distance(ChunkKey::at(motion.position)) as u64,
                Work::Vegetation(_) => 0,
            };
            let rank = match work {
                Work::Chunk(_) => 0,
                Work::Terrain(_) => 1,
                Work::Vegetation(_) => 2,
            };
            return (rank, detail, distance, key);
        }
        motion.priority(work, detail)
    }
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
            .map(|k| {
                crate::watershed::system(crate::watershed::key(
                    k.origin() + Vec2::splat(CHUNK_SIZE * 0.5),
                ))
            })
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
        self.pending.len() + self.pending_lod.len() + self.pending_surface.len()
    }
    pub fn terrain_lod_counts(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        for chunk in self.loaded.values() {
            counts[chunk.surface_detail.index()] += 1;
        }
        counts
    }
    pub fn terrain_triangles(&self) -> (usize, usize) {
        (
            self.loaded.values().map(|c| c.surface_triangles).sum(),
            self.loaded.len() * (SIDE - 1) * (SIDE - 1) * 2,
        )
    }
    pub fn lod_counts(&self) -> ([usize; 3], [usize; 3]) {
        let mut trees = [0; 3];
        let mut grass = [0; 3];
        for chunk in self.loaded.values() {
            trees[chunk.levels.trees.index()] += 1;
            grass[chunk.levels.grass.index()] += 1;
        }
        (trees, grass)
    }
    pub fn vegetation_vertices(&self) -> (usize, usize) {
        (
            self.loaded.values().map(|c| c.vegetation_vertices).sum(),
            self.loaded
                .values()
                .map(|c| c.vegetation.full_vertices())
                .sum(),
        )
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

#[allow(clippy::too_many_arguments)]
pub fn setup(
    mut commands: Commands,
    world: Res<Meadow>,
    settings: Res<TreeSettings>,
    config: Res<vegetation::LodConfig>,
    terrain_config: Res<terrain_lod::Config>,
    priority_config: Res<PriorityConfig>,
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
        pending_lod: BTreeMap::new(),
        pending_surface: BTreeMap::new(),
        priority_enabled: priority_config.enabled,
        predicted: Vec2::ZERO,
        queued: 0,
        ground_wait_ms: 0.,
        max_ground_wait_ms: 0.,
        ground_wait_events: 0,
        ground_blocked: false,
        ground_wait_run_ms: 0.,
        near_wait: None,
        near_wait_ms: 0.,
        last_near_wait_ms: 0.,
        max_near_wait_ms: 0.,
        terrain_lod_enabled: terrain_config.enabled,
        terrain_rebuilt: 0,
        last_terrain_ms: 0.,
        lod_enabled: config.enabled,
        lod_rebuilt: 0,
        last_lod_ms: 0.,
        last_lod_install_ms: 0.,
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
    horse: Single<(&Transform, &HorseController)>,
    camera: Query<(&Transform, &Projection, &Camera), With<crate::player::FollowCamera>>,
    layer: Res<MapLayer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut lab: ResMut<crate::LabState>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(mut stream) = stream else {
        return;
    };
    let view = stream
        .lod_enabled
        .then(|| {
            camera
                .iter()
                .next()
                .map(|(t, p, c)| vegetation::View::from_camera(t, p, c))
        })
        .flatten();
    let p = Vec2::new(horse.0.translation.x, horse.0.translation.z);
    stream.ground_blocked = false;
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
        // Keep any running LOD worker until polled, even after eviction, so it
        // still counts against the shared two-worker limit. Its result is discarded.
        commands.entity(chunk.root).despawn();
        for id in chunk.meshes {
            meshes.remove(id);
        }
        stream.evicted += 1;
    }
    let motion = Motion::new(horse.0, horse.1);
    let mut jobs: Vec<_> = stream
        .pending
        .keys()
        .map(|k| Work::Chunk(*k))
        .chain(stream.pending_surface.keys().map(|k| Work::Terrain(*k)))
        .chain(stream.pending_lod.keys().map(|k| Work::Vegetation(*k)))
        .collect();
    jobs.sort_by_key(|job| stream.priority(*job, motion, view));
    for job in jobs {
        let key = job.key();
        match job {
            Work::Chunk(_) => {
                let Some(mut data) = block_on(poll_once(stream.pending.get_mut(&key).unwrap()))
                else {
                    continue;
                };
                stream.pending.remove(&key);
                if key.distance(center) > RADIUS {
                    continue;
                }
                let start = Instant::now();
                data.surface
                    .insert_attribute(Mesh::ATTRIBUTE_COLOR, data.colors.colors(*layer));
                let surface = meshes.add(data.surface);
                let surface_triangles = meshes.get(&surface).unwrap().indices().unwrap().len() / 3;
                let trees =
                    (data.built.trees.count_vertices() > 0).then(|| meshes.add(data.built.trees));
                let grass =
                    (data.built.grass.count_vertices() > 0).then(|| meshes.add(data.built.grass));
                let vegetation_meshes: Vec<_> = [trees.as_ref(), grass.as_ref()]
                    .into_iter()
                    .flatten()
                    .map(|m| m.id())
                    .collect();
                let water = (data.water.count_vertices() > 0).then(|| meshes.add(data.water));
                let lake = (data.lake.count_vertices() > 0).then(|| meshes.add(data.lake));
                let stones = (data.stones.count_vertices() > 0).then(|| meshes.add(data.stones));
                // Bevy's GPU slab allocator does not allocate zero-length buffers.
                let mut mesh_ids = vec![surface.id()];
                mesh_ids.extend(
                    [
                        trees.as_ref(),
                        grass.as_ref(),
                        water.as_ref(),
                        lake.as_ref(),
                        stones.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    .map(|m| m.id()),
                );
                let origin = key.origin();
                let mut tree_entity = Entity::PLACEHOLDER;
                let mut surface_entity = Entity::PLACEHOLDER;
                let mut grass_entity = Entity::PLACEHOLDER;
                let root = commands
                    .spawn((
                        Name::new(format!("Chunk {},{}", key.0, key.1)),
                        Transform::from_xyz(origin.x, 0., origin.y),
                        Visibility::default(),
                    ))
                    .with_children(|root| {
                        if let Some(stones) = stones {
                            root.spawn((
                                Name::new("Ford bank stones"),
                                Mesh3d(stones),
                                MeshMaterial3d(stream.trees_material.clone()),
                            ));
                        }
                        surface_entity = root
                            .spawn((
                                TerrainSurface,
                                data.colors,
                                Mesh3d(surface.clone()),
                                MeshMaterial3d(stream.terrain_material.clone()),
                            ))
                            .id();
                        let mut entity = root.spawn((
                            Transform::default(),
                            Visibility::Inherited,
                            MeshMaterial3d(stream.trees_material.clone()),
                        ));
                        if let Some(trees) = trees {
                            entity.insert(Mesh3d(trees));
                        }
                        tree_entity = entity.id();
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
                        let mut entity = root.spawn((
                            Grass,
                            Transform::default(),
                            MeshMaterial3d(stream.grass_material.clone()),
                            if *layer == MapLayer::Natural {
                                Visibility::Inherited
                            } else {
                                Visibility::Hidden
                            },
                        ));
                        if let Some(grass) = grass {
                            entity.insert(Mesh3d(grass));
                        }
                        grass_entity = entity.id();
                    })
                    .id();
                stream.loaded.insert(
                    key,
                    Resident {
                        root,
                        surface_entity,
                        surface_mesh: surface.id(),
                        surface_detail: data.surface_detail,
                        surface_bytes: data.surface_bytes,
                        surface_triangles,
                        tree_entity,
                        grass_entity,
                        vegetation: data.vegetation,
                        levels: data.built.levels,
                        vegetation_meshes,
                        vegetation_vertices: data.built.vertices,
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
            }
            Work::Terrain(_) => {
                let Some(mut built) =
                    block_on(poll_once(stream.pending_surface.get_mut(&key).unwrap()))
                else {
                    continue;
                };
                stream.pending_surface.remove(&key);
                let terrain_enabled = stream.terrain_lod_enabled;
                let Some(chunk) = stream.loaded.get_mut(&key) else {
                    continue;
                };
                let desired = if terrain_enabled {
                    TerrainDetail::choose(key.origin(), p, Some(chunk.surface_detail))
                } else {
                    TerrainDetail::Near
                };
                if built.detail != desired {
                    continue;
                }
                built
                    .mesh
                    .insert_attribute(Mesh::ATTRIBUTE_COLOR, built.colors.colors(*layer));
                let triangles = built.mesh.indices().unwrap().len() / 3;
                let surface = meshes.add(built.mesh);
                commands
                    .entity(chunk.surface_entity)
                    .insert((Mesh3d(surface.clone()), built.colors));
                meshes.remove(chunk.surface_mesh);
                chunk.meshes.retain(|id| *id != chunk.surface_mesh);
                chunk.surface_mesh = surface.id();
                chunk.meshes.push(surface.id());
                chunk.bytes = chunk.bytes - chunk.surface_bytes + built.bytes;
                chunk.surface_bytes = built.bytes;
                chunk.surface_triangles = triangles;
                chunk.surface_detail = built.detail;
                stream.terrain_rebuilt += 1;
                stream.last_terrain_ms = built.elapsed_ms;
            }
            Work::Vegetation(_) => {
                let Some(built) = block_on(poll_once(stream.pending_lod.get_mut(&key).unwrap()))
                else {
                    continue;
                };
                stream.pending_lod.remove(&key);
                let Some(chunk) = stream.loaded.get_mut(&key) else {
                    continue;
                };
                if chunk.vegetation.levels(view, Some(chunk.levels)) != built.levels {
                    continue;
                }
                let start = Instant::now();
                let trees = (built.trees.count_vertices() > 0).then(|| meshes.add(built.trees));
                let grass = (built.grass.count_vertices() > 0).then(|| meshes.add(built.grass));
                for (entity, mesh) in [
                    (chunk.tree_entity, trees.as_ref()),
                    (chunk.grass_entity, grass.as_ref()),
                ] {
                    let mut entity = commands.entity(entity);
                    if let Some(mesh) = mesh {
                        entity.insert(Mesh3d(mesh.clone()));
                    } else {
                        entity.remove::<Mesh3d>();
                    }
                }
                commands
                    .entity(chunk.grass_entity)
                    .insert(if *layer == MapLayer::Natural {
                        Visibility::Inherited
                    } else {
                        Visibility::Hidden
                    });
                for id in &chunk.vegetation_meshes {
                    meshes.remove(*id);
                }
                chunk
                    .meshes
                    .retain(|id| !chunk.vegetation_meshes.contains(id));
                chunk.vegetation_meshes = [trees.as_ref(), grass.as_ref()]
                    .into_iter()
                    .flatten()
                    .map(|m| m.id())
                    .collect();
                chunk.meshes.extend(chunk.vegetation_meshes.iter().copied());
                chunk.bytes = chunk.bytes - chunk.vegetation_vertices * 40 + built.vertices * 40;
                chunk.vegetation_vertices = built.vertices;
                chunk.levels = built.levels;
                stream.lod_rebuilt += 1;
                stream.last_lod_ms = built.elapsed_ms;
                stream.last_lod_install_ms = start.elapsed().as_secs_f64() * 1000.;
            }
        }
        // Discarded/unfinished jobs do not spend this budget. One valid install does.
        break;
    }
}

/// Recompute one shared queue after controls and camera updates. Existing workers
/// run to completion (including evicted ones), always counting toward the cap.
pub fn schedule(
    stream: Option<ResMut<StreamWorld>>,
    horse: Single<(&Transform, &HorseController)>,
    camera: Query<(&Transform, &Projection, &Camera), With<crate::player::FollowCamera>>,
) {
    let Some(mut stream) = stream else {
        return;
    };
    let motion = Motion::new(horse.0, horse.1);
    let p = motion.position;
    let center = ChunkKey::at(p);
    let view = stream
        .lod_enabled
        .then(|| {
            camera
                .iter()
                .next()
                .map(|(t, p, c)| vegetation::View::from_camera(t, p, c))
        })
        .flatten();
    stream.predicted = if stream.priority_enabled {
        motion.predicted
    } else {
        p
    };
    stream.measure_near_wait(center);
    if !stream.ground_blocked {
        stream.ground_wait_run_ms = 0.;
    }
    let mut keep = BTreeSet::from([crate::watershed::key(Vec2::ZERO)]);
    keep.extend(stream.pending.keys().flat_map(|k| tree_regions(*k)));
    keep.extend(stream.pending_surface.keys().flat_map(|k| tree_regions(*k)));
    let mut jobs = Vec::new();
    for z in center.1 - RADIUS..=center.1 + RADIUS {
        for x in center.0 - RADIUS..=center.0 + RADIUS {
            let key = ChunkKey(x, z);
            keep.extend(tree_regions(key));
            if !stream.loaded.contains_key(&key) && !stream.pending.contains_key(&key) {
                jobs.push(Work::Chunk(key));
            }
        }
    }
    stream.generator.hydrology.retain(&keep);
    if stream.priority_enabled || stream.loaded_count() == MAX_CHUNKS {
        for (key, chunk) in &stream.loaded {
            if key.distance(center) > RADIUS {
                continue;
            }
            let desired = if stream.terrain_lod_enabled {
                TerrainDetail::choose(key.origin(), p, Some(chunk.surface_detail))
            } else {
                TerrainDetail::Near
            };
            if desired != chunk.surface_detail && !stream.pending_surface.contains_key(key) {
                jobs.push(Work::Terrain(*key));
            }
            if chunk.vegetation.levels(view, Some(chunk.levels)) != chunk.levels
                && !stream.pending_lod.contains_key(key)
            {
                jobs.push(Work::Vegetation(*key));
            }
        }
    }
    jobs.sort_by_key(|job| stream.priority(*job, motion, view));
    let available = MAX_TASKS.saturating_sub(stream.pending_count());
    stream.queued = jobs.len().saturating_sub(available);
    for job in jobs.into_iter().take(available) {
        let key = job.key();
        match job {
            Work::Chunk(_) => {
                let generator = stream.generator.clone();
                let settings = stream.settings;
                let detail = if stream.terrain_lod_enabled {
                    TerrainDetail::choose(key.origin(), p, None)
                } else {
                    TerrainDetail::Near
                };
                stream.pending.insert(
                    key,
                    AsyncComputeTaskPool::get()
                        .spawn(async move { generate_at(generator, key, settings, view, detail) }),
                );
            }
            Work::Terrain(_) => {
                let g = stream.generator.clone();
                let chunk = &stream.loaded[&key];
                let detail = if stream.terrain_lod_enabled {
                    TerrainDetail::choose(key.origin(), p, Some(chunk.surface_detail))
                } else {
                    TerrainDetail::Near
                };
                let trees = chunk.trees.clone();
                stream.pending_surface.insert(
                    key,
                    AsyncComputeTaskPool::get()
                        .spawn(async move { rebuild_surface(&g, key, detail, &trees) }),
                );
            }
            Work::Vegetation(_) => {
                let chunk = &stream.loaded[&key];
                let levels = chunk.vegetation.levels(view, Some(chunk.levels));
                let data = chunk.vegetation.clone();
                stream.pending_lod.insert(
                    key,
                    AsyncComputeTaskPool::get().spawn(async move { data.build(levels) }),
                );
            }
        }
    }
}

pub fn generation_report(generator: Generator, settings: TreeSettings) {
    let mut sampler = crate::monitor::ProcessSampler::new();
    let before = sampler.snapshot();
    let start = Instant::now();
    let mut chunks = Vec::new();
    for z in -1..=1 {
        for x in -1..=1 {
            chunks.push(generate_at(
                generator.clone(),
                ChunkKey(x, z),
                settings,
                None,
                TerrainDetail::Near,
            ));
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
    // Compare vegetation-only mesh construction on identical immutable blueprints.
    let mut csv = String::from(
        "detail,chunks,trees,tree_vertices,grass_vertices,vegetation_mesh_mib,blueprint_mib,mesh_build_ms\n",
    );
    for detail in [
        vegetation::Detail::Near,
        vegetation::Detail::Middle,
        vegetation::Detail::Far,
    ] {
        let start = Instant::now();
        let mut tree_vertices = 0;
        let mut grass_vertices = 0;
        for chunk in &chunks {
            let built = chunk.vegetation.build(Levels {
                trees: detail,
                grass: detail,
            });
            tree_vertices += built.trees.count_vertices();
            grass_vertices += built.grass.count_vertices();
        }
        let wall = start.elapsed().as_secs_f64() * 1000.;
        let descriptors =
            chunks.iter().map(|c| c.vegetation.bytes()).sum::<usize>() as f64 / (1024. * 1024.);
        csv.push_str(&format!("{detail:?},9,{tree_count},{tree_vertices},{grass_vertices},{:.3},{descriptors:.3},{wall:.3}\n", (tree_vertices + grass_vertices) as f64 * 40. / (1024. * 1024.)));
    }
    std::fs::write(
        format!("reports/vegetation-lod-seed-{}.csv", generator.seed),
        csv,
    )
    .expect("Cannot save vegetation comparison");
    let mut terrain_csv = String::from(
        "detail,chunks,vertices,triangles,terrain_mesh_mib,terrain_color_data_mib,worker_ms\n",
    );
    for detail in [
        TerrainDetail::Near,
        TerrainDetail::Middle,
        TerrainDetail::Far,
    ] {
        let start = Instant::now();
        let mut vertices = 0;
        let mut triangles = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            let key = ChunkKey(i as i32 % 3 - 1, i as i32 / 3 - 1);
            let built = rebuild_surface(&generator, key, detail, &chunk.trees);
            vertices += built.mesh.count_vertices();
            triangles += built.mesh.indices().unwrap().len() / 3;
        }
        let mesh_bytes = vertices * 40 + triangles * 3 * 4;
        let color_bytes = vertices * (size_of::<EnvironmentSample>() + size_of::<f32>());
        terrain_csv.push_str(&format!(
            "{detail:?},9,{vertices},{triangles},{:.3},{:.3},{:.3}\n",
            mesh_bytes as f64 / (1024. * 1024.),
            color_bytes as f64 / (1024. * 1024.),
            start.elapsed().as_secs_f64() * 1000.
        ));
    }
    std::fs::write(
        format!("reports/terrain-lod-seed-{}.csv", generator.seed),
        terrain_csv,
    )
    .expect("Cannot save terrain comparison");
    println!(
        "Saved {path}: 9 CPU-built chunks retained, {wall:.1} ms total, {mean:.1} ms mean, {mib:.1} MiB mesh data; excludes GPU/model loading; synchronous benchmark, not runtime frame latency"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prediction_prioritizes_feet_then_the_actual_path_and_reverses_with_motion() {
        let t = Transform::from_xyz(90., 0., 48.)
            .with_rotation(Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2));
        let mut horse = HorseController { speed: 9., yaw: 0. };
        let motion = Motion::new(&t, &horse);
        assert!((motion.predicted - Vec2::new(108., 48.)).length() < 0.0001);
        let foot = ChunkKey(0, 0);
        let front = ChunkKey(1, 0);
        let side = ChunkKey(0, 1);
        assert!(motion.entry(front).is_some());
        assert!(motion.entry(side).is_none());
        let mut jobs = [
            Work::Vegetation(foot),
            Work::Chunk(side),
            Work::Terrain(side),
            Work::Chunk(front),
            Work::Terrain(foot),
            Work::Chunk(foot),
        ];
        jobs.sort_by_key(|w| motion.priority(*w, 0));
        assert_eq!(
            jobs,
            [
                Work::Chunk(foot),
                Work::Terrain(foot),
                Work::Chunk(front),
                Work::Chunk(side),
                Work::Terrain(side),
                Work::Vegetation(foot)
            ]
        );
        horse.speed = -2.;
        let t = Transform::from_xyz(2., 0., 0.).with_rotation(t.rotation);
        let reverse = Motion::new(&t, &horse);
        assert!((reverse.predicted - Vec2::new(-2., 0.)).length() < 0.0001);
        assert!(reverse.entry(ChunkKey(-1, 0)).is_some());
        assert!(
            reverse.entry(ChunkKey(-1, -1)).is_none(),
            "Parallel grid border must use its owner"
        );
        assert!(reverse.entry(ChunkKey(1, 0)).is_none());
        horse.speed = 0.;
        let stopped = Motion::new(&t, &horse);
        assert_eq!(stopped.predicted, stopped.position);
        assert!(stopped.entry(front).is_none());
        horse.speed = 1000.;
        assert!(
            (Motion::new(&t, &horse).predicted - stopped.position).length() <= CHUNK_SIZE + 0.0001
        );
        let corner = Motion {
            position: Vec2::splat(90.),
            predicted: Vec2::splat(108.),
        };
        assert!(corner.entry(ChunkKey(1, 1)).is_some());
        assert!(
            corner.entry(ChunkKey(1, 0)).is_none(),
            "A corner-only touch is not a route"
        );
    }
    fn generator() -> Generator {
        Generator::new(42, TerrainSettings::default())
    }
    fn rendered_surface_height(mesh: &Mesh, key: ChunkKey, p: Vec2) -> f32 {
        let Some(VertexAttributeValues::Float32x3(vertices)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!()
        };
        let Indices::U32(indices) = mesh.indices().unwrap() else {
            panic!()
        };
        let q = p - key.origin();
        indices
            .chunks_exact(3)
            .find_map(|t| {
                let a = Vec3::from(vertices[t[0] as usize]);
                let b = Vec3::from(vertices[t[1] as usize]);
                let c = Vec3::from(vertices[t[2] as usize]);
                let pa = Vec2::new(a.x, a.z);
                let pb = Vec2::new(b.x, b.z);
                let pc = Vec2::new(c.x, c.z);
                let area = (pb - pa).perp_dot(pc - pa);
                let u = (q - pa).perp_dot(pc - pa) / area;
                let v = (pb - pa).perp_dot(q - pa) / area;
                (u >= -0.00001 && v >= -0.00001 && u + v <= 1.00001)
                    .then_some(a.y * (1. - u - v) + b.y * u + c.y * v)
            })
            .expect("Terrain mesh hole")
    }
    #[test]
    fn terrain_levels_preserve_water_banks_roots_and_shared_borders_with_bounded_error() {
        let mut reduced = 0;
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            let ford = g.hydrology.get((0, 0)).unwrap().fords[0];
            for key in [ChunkKey(0, 0), ChunkKey(2, -2), ChunkKey::at(ford.center)] {
                let original = generate(g.clone(), key, TreeSettings::default());
                for detail in [TerrainDetail::Middle, TerrainDetail::Far] {
                    let ground = rebuild_surface(&g, key, detail, &original.trees);
                    assert!(ground.bytes <= original.surface_bytes);
                    reduced += usize::from(ground.bytes < original.surface_bytes);
                    for z in 0..24 {
                        for x in 0..24 {
                            let p =
                                key.origin() + Vec2::new(x as f32 * 4. + 0.5, z as f32 * 4. + 1.5);
                            assert!(
                                (rendered_surface_height(&ground.mesh, key, p) - g.ground(p)).abs()
                                    <= terrain_lod::MAX_ERROR + 0.0002
                            );
                        }
                    }
                    for (p, _) in &original.trees {
                        assert!(
                            (rendered_surface_height(&ground.mesh, key, *p) - g.ground(*p)).abs()
                                < 0.0002,
                            "Tree root changed with terrain LOD"
                        );
                    }
                    for mesh in [&original.water, &original.lake] {
                        let Some(VertexAttributeValues::Float32x3(vertices)) =
                            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                        else {
                            panic!()
                        };
                        for triangle in vertices.chunks_exact(3).step_by(8) {
                            let center = (Vec3::from(triangle[0])
                                + Vec3::from(triangle[1])
                                + Vec3::from(triangle[2]))
                                / 3.;
                            let p = key.origin() + Vec2::new(center.x, center.z);
                            assert!(
                                (rendered_surface_height(&ground.mesh, key, p) - g.ground(p)).abs()
                                    < 0.0002,
                                "LOD changed the ground under rendered water"
                            );
                        }
                    }
                    let next = ChunkKey(key.0 + 1, key.1);
                    let adjacent =
                        rebuild_surface(&g, next, TerrainDetail::Near, &g.tree_neighbours(next));
                    for i in 0..SIDE {
                        let p = key.origin() + Vec2::new(CHUNK_SIZE, i as f32 * CELL);
                        assert!(
                            (rendered_surface_height(&ground.mesh, key, p)
                                - rendered_surface_height(&adjacent.mesh, next, p))
                            .abs()
                                < 0.0001,
                            "Mixed LOD chunk seam"
                        );
                    }
                    let again = rebuild_surface(&g, key, detail, &original.trees);
                    assert_eq!(
                        ground.mesh.attribute(Mesh::ATTRIBUTE_POSITION),
                        again.mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                    );
                    assert_eq!(ground.mesh.indices(), again.mesh.indices());
                    assert_eq!(ground.mesh.count_vertices(), ground.colors.samples.len());
                }
            }
        }
        assert!(reduced >= 12);
    }
    #[test]
    fn ford_stones_mark_both_dry_banks_leave_the_path_open_and_reproduce_across_chunks() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            let plan = g.hydrology.get((0, 0)).unwrap();
            for ford in &plan.fords {
                let lo = ChunkKey::at(ford.center - Vec2::splat(30.));
                let hi = ChunkKey::at(ford.center + Vec2::splat(30.));
                let keys: Vec<_> = (lo.1..=hi.1)
                    .flat_map(|z| (lo.0..=hi.0).map(move |x| ChunkKey(x, z)))
                    .collect();
                let stones: Vec<_> = keys.iter().flat_map(|key| ford_stones(&g, *key)).collect();
                assert!((4..=20).contains(&stones.len()));
                for bank in [-1., 1.] {
                    for edge in [-1., 1.] {
                        assert!(
                            stones.iter().any(|s| {
                                let offset = Vec2::new(s.center.x, s.center.z) - ford.center;
                                offset.dot(ford.across) * bank > 0.
                                    && offset.dot(ford.across.perp()) * edge > 0.
                            }),
                            "Each approach needs stones on both sides: seed {seed}"
                        );
                    }
                }
                let mut centers = BTreeSet::new();
                for stone in &stones {
                    let p = Vec2::new(stone.center.x, stone.center.z);
                    assert!(
                        centers.insert((p.x.to_bits(), p.y.to_bits())),
                        "Duplicate chunk marker"
                    );
                    assert!(g.water_depth(p) < -0.015);
                    assert!(
                        (p - ford.center).perp_dot(ford.across).abs() - stone.radii.max_element()
                            > 2.
                    );
                    assert!(stone.center.y < g.ground(p));
                    assert!(stone.center.y + stone.height > g.ground(p));
                    for i in 0..8 {
                        let q = p + Vec2::from_angle(i as f32 * std::f32::consts::TAU / 8.)
                            * stone.radii.max_element();
                        assert!(g.water_depth(q) < -0.015, "Stone footprint must stay dry");
                    }
                }
                let mesh = ford_stone_mesh(&stones, Vec2::ZERO);
                assert_eq!(mesh.count_vertices(), stones.len() * 72);
                let Some(VertexAttributeValues::Float32x3(normals)) =
                    mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
                else {
                    panic!()
                };
                assert!(normals.iter().all(|n| Vec3::from(*n).is_finite()));
                let before: Vec<_> = keys.iter().map(|key| ford_stones(&g, *key)).collect();
                g.hydrology.retain(&BTreeSet::new());
                g.hydrology.prepare(&g, (0, 0));
                for (key, stones) in keys.iter().zip(before) {
                    assert_eq!(stones, ford_stones(&g, *key));
                    let repeated = generate(g.clone(), *key, TreeSettings::default());
                    assert_eq!(
                        repeated.stones.attribute(Mesh::ATTRIBUTE_POSITION),
                        ford_stone_mesh(&stones, key.origin()).attribute(Mesh::ATTRIBUTE_POSITION)
                    );
                }
            }
        }
    }
    #[test]
    fn ford_has_visible_water_and_the_actual_tree_collisions_leave_both_banks_accessible() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            let ford = g.hydrology.get((0, 0)).unwrap().fords[0];
            let steps = ((ford.width + 15.) * 4.).ceil() as i32;
            let keys: BTreeSet<_> = (-steps..=steps)
                .map(|i| ChunkKey::at(ford.center + ford.across * i as f32 * 0.25))
                .collect();
            let chunks: Vec<_> = keys
                .into_iter()
                .map(|key| (key, generate(g.clone(), key, TreeSettings::default())))
                .collect();
            for i in -steps..=steps {
                let p = ford.center + ford.across * i as f32 * 0.25;
                assert!(
                    chunks
                        .iter()
                        .flat_map(|(_, c)| &c.obstacles)
                        .all(|t| t.p.distance(p) >= t.radius + 0.65),
                    "Ford blocked by a generated tree"
                );
            }
            let (key, chunk) = chunks
                .iter()
                .find(|(key, _)| *key == ChunkKey::at(ford.center))
                .unwrap();
            let Some(VertexAttributeValues::Float32x3(points)) =
                chunk.water.attribute(Mesh::ATTRIBUTE_POSITION)
            else {
                panic!()
            };
            assert!(
                points.chunks_exact(3).any(|tri| {
                    let q: Vec<_> = tri
                        .iter()
                        .map(|p| key.origin() + Vec2::new(p[0], p[2]))
                        .collect();
                    let signs =
                        [0, 1, 2].map(|i| (q[(i + 1) % 3] - q[i]).perp_dot(ford.center - q[i]));
                    signs.iter().all(|s| *s >= -0.001) || signs.iter().all(|s| *s <= 0.001)
                }),
                "Missing visible water in the ford"
            );
            let Some(VertexAttributeValues::Float32x4(colors)) =
                chunk.water.attribute(Mesh::ATTRIBUTE_COLOR)
            else {
                panic!()
            };
            assert!(colors.iter().any(|c| c[0] > 1.5));
            let repeated = generate(g.clone(), *key, TreeSettings::default());
            for attr in [
                Mesh::ATTRIBUTE_POSITION,
                Mesh::ATTRIBUTE_UV_0,
                Mesh::ATTRIBUTE_COLOR,
            ] {
                assert_eq!(chunk.water.attribute(attr), repeated.water.attribute(attr));
            }
        }
    }

    #[test]
    fn dry_sandy_lake_banks_are_walkable_and_rendered_water_is_blocked() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            let world = Meadow::from_stream_generator(g.clone());
            let plan = g.hydrology.get((0, 0)).unwrap();
            let mut recovered = 0;
            for i in 0..64 {
                let direction = Vec2::from_angle(i as f32 * std::f32::consts::TAU / 64.);
                for radius in [29., 30., 31., 32.] {
                    let p = plan.lake + direction * radius;
                    let field = g.water(p);
                    if field.distance < field.width + 1.6 && world.walkable(p) {
                        assert!(g.water_depth(p) < -0.015);
                        recovered += 1;
                    }
                }
            }
            assert!(
                recovered > 20,
                "Dry sand was overblocked for seed {seed}: {recovered}"
            );
            assert!(!world.walkable(plan.lake));
            let key = ChunkKey::at(plan.lake);
            let chunk = generate(g.clone(), key, TreeSettings::default());
            for mesh in [&chunk.water, &chunk.lake] {
                let Some(VertexAttributeValues::Float32x3(points)) =
                    mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                else {
                    panic!()
                };
                for tri in points.chunks_exact(3) {
                    for weights in [[1. / 3.; 3], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]] {
                        let p = tri
                            .iter()
                            .zip(weights)
                            .map(|(v, w)| Vec2::new(v[0], v[2]) * w)
                            .sum::<Vec2>()
                            + key.origin();
                        assert!(
                            !world.walkable(p),
                            "Horse entered rendered water: {seed}, {p:?}"
                        );
                    }
                }
            }
            for shrub in &chunk.vegetation.shrubs {
                let p = key.origin() + Vec2::new(shrub.center.x, shrub.center.z);
                assert!(g.water_depth(p) < -0.015);
                assert!((shrub.center.y - g.ground(p)).abs() < 0.001);
            }
        }
    }

    #[test]
    fn vegetation_levels_reduce_geometry_without_changing_placement_or_grounding() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            let chunk = generate(g.clone(), ChunkKey(0, -1), TreeSettings::default());
            let data = &chunk.vegetation;
            assert!(!data.trees.is_empty() && !data.grass.is_empty());
            let full = data.build(Levels::FULL);
            assert_eq!(full.vertices, data.full_vertices());
            let mut last = full.vertices;
            for detail in [vegetation::Detail::Middle, vegetation::Detail::Far] {
                let built = data.build(Levels {
                    trees: detail,
                    grass: detail,
                });
                assert!(built.vertices < last);
                last = built.vertices;
                if detail == vegetation::Detail::Far {
                    assert_eq!(built.grass.count_vertices(), 0);
                }
                let repeated = data.build(built.levels);
                for mesh in [&built.trees, &built.grass] {
                    let Some(VertexAttributeValues::Float32x3(points)) =
                        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                    else {
                        panic!()
                    };
                    let Some(VertexAttributeValues::Float32x3(normals)) =
                        mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
                    else {
                        panic!()
                    };
                    assert!(
                        points
                            .iter()
                            .chain(normals)
                            .flatten()
                            .all(|v| v.is_finite())
                    );
                    assert!(
                        normals
                            .iter()
                            .all(|n| (Vec3::from(*n).length() - 1.).abs() < 0.001)
                    );
                }
                assert_eq!(
                    built.trees.attribute(Mesh::ATTRIBUTE_POSITION),
                    repeated.trees.attribute(Mesh::ATTRIBUTE_POSITION)
                );
                for t in &data.trees {
                    let p = Vec2::new(t.offset.x, t.offset.z) + ChunkKey(0, -1).origin();
                    assert!((t.offset.y - g.ground(p)).abs() < 0.0001);
                    assert!(
                        t.shape
                            .trunk
                            .iter()
                            .all(|(point, _)| point.distance(t.center - t.offset)
                                <= t.diameter * 0.5)
                    );
                }
            }
            assert!(last < full.vertices / 5);
        }
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
    fn test_stream(priority: bool) -> (App, Entity, Entity) {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(Meadow::streamed(42, TerrainSettings::default()))
            .init_resource::<TreeSettings>()
            .init_resource::<vegetation::LodConfig>()
            .init_resource::<terrain_lod::Config>()
            .insert_resource(PriorityConfig { enabled: priority })
            .init_resource::<MapLayer>()
            .init_resource::<crate::LabState>()
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .add_systems(Startup, setup)
            .add_systems(Update, (update, schedule).chain());
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
        let camera = app
            .world_mut()
            .spawn((
                crate::player::FollowCamera,
                Camera::default(),
                Projection::from(PerspectiveProjection {
                    fov: 72_f32.to_radians(),
                    ..default()
                }),
                Transform::from_xyz(0., 5., 8.).looking_at(Vec3::ZERO, Vec3::Y),
            ))
            .id();
        (app, horse, camera)
    }

    #[test]
    fn movement_stall_metrics_count_attempts_and_separate_episodes() {
        use bevy::{
            input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
            time::TimeUpdateStrategy,
            window::{CursorOptions, PrimaryWindow},
        };
        let (mut app, horse, _) = test_stream(true);
        app.insert_resource(TimeUpdateStrategy::ManualDuration(
            std::time::Duration::from_secs_f64(1. / 60.),
        ))
        .init_resource::<crate::player::CameraRig>()
        .init_resource::<ButtonInput<KeyCode>>()
        .init_resource::<ButtonInput<MouseButton>>()
        .init_resource::<AccumulatedMouseMotion>()
        .init_resource::<AccumulatedMouseScroll>()
        .add_systems(
            Update,
            crate::player::controls.after(update).before(schedule),
        );
        app.world_mut().resource_mut::<crate::LabState>().ready = true;
        app.world_mut()
            .resource_mut::<crate::player::CameraRig>()
            .captured = true;
        app.world_mut().spawn((
            Window {
                focused: true,
                ..default()
            },
            PrimaryWindow,
            CursorOptions::default(),
        ));
        app.update();
        let start = app.world().get::<Transform>(horse).unwrap().translation;
        let step = |app: &mut App, moving: bool| {
            // Hold loading at zero to deterministically exercise the real controls
            // against missing dry ground, regardless of worker timing.
            app.world_mut()
                .resource_mut::<StreamWorld>()
                .pending
                .clear();
            let mut input = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            input.reset_all();
            if moving {
                input.press(KeyCode::KeyW);
            }
            app.update();
        };
        step(&mut app, false);
        assert_eq!(app.world().resource::<StreamWorld>().ground_wait_events, 0);
        for _ in 0..3 {
            step(&mut app, true);
        }
        let s = app.world().resource::<StreamWorld>();
        assert_eq!(s.ground_wait_events, 1);
        assert!((s.ground_wait_ms - 50.).abs() < 0.01);
        assert!((s.max_ground_wait_ms - 50.).abs() < 0.01);
        assert_eq!(
            app.world().get::<Transform>(horse).unwrap().translation,
            start
        );
        step(&mut app, false);
        step(&mut app, true);
        let s = app.world().resource::<StreamWorld>();
        assert_eq!(s.ground_wait_events, 2);
        assert!((s.ground_wait_ms - 1000. / 15.).abs() < 0.01);
        assert!((s.max_ground_wait_ms - 50.).abs() < 0.01);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        let wait = Instant::now();
        while {
            app.update();
            let s = app.world().resource::<StreamWorld>();
            s.loaded_count() != MAX_CHUNKS || s.pending_count() != 0
        } {
            assert!(wait.elapsed().as_secs() < 15);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let total = app.world().resource::<StreamWorld>().ground_wait_ms;
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyW);
        app.update();
        assert!(
            app.world()
                .get::<Transform>(horse)
                .unwrap()
                .translation
                .distance(start)
                > 0.
        );
        assert_eq!(app.world().resource::<StreamWorld>().ground_wait_ms, total);
    }

    #[test]
    fn foot_refinement_is_queued_before_the_ring_and_distance_mode_remains_available() {
        let mut placements = Vec::new();
        for priority in [false, true] {
            let (mut app, horse, _) = test_stream(priority);
            let settle = Instant::now();
            loop {
                app.update();
                let s = app.world().resource::<StreamWorld>();
                if s.loaded_count() == MAX_CHUNKS && s.pending_count() == 0 {
                    break;
                }
                assert!(settle.elapsed().as_secs() < 15);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let target = ChunkKey(3, 0);
            assert_eq!(
                app.world().resource::<StreamWorld>().loaded[&target].surface_detail,
                TerrainDetail::Far
            );
            app.world_mut()
                .get_mut::<Transform>(horse)
                .unwrap()
                .translation = Vec3::new(330., 0., 48.);
            app.update();
            let s = app.world().resource::<StreamWorld>();
            assert!(s.loaded_count() < MAX_CHUNKS);
            assert_eq!(
                s.pending_surface.contains_key(&target),
                priority,
                "Only priority mode should queue foot refinement ahead of the unfinished ring"
            );
            let start = Instant::now();
            loop {
                let before = {
                    let s = app.world().resource::<StreamWorld>();
                    s.generated + s.terrain_rebuilt + s.lod_rebuilt
                };
                app.update();
                let s = app.world().resource::<StreamWorld>();
                assert!(s.pending_count() <= MAX_TASKS && s.loaded_count() <= MAX_CHUNKS);
                assert!(s.generated + s.terrain_rebuilt + s.lod_rebuilt - before <= 1);
                assert!(s.watershed_count() <= 13);
                if s.loaded[&target].surface_detail == TerrainDetail::Near {
                    if !priority {
                        assert_eq!(s.loaded_count(), MAX_CHUNKS);
                    }
                    assert!(s.last_near_wait_ms > 0.);
                    placements.push(s.loaded[&target].trees.clone());
                    println!(
                        "SCHEDULING_COMPARISON mode={} foot_wait_ms={:.3} ready_chunks={} resident_mib={:.3}",
                        s.scheduler_name(),
                        s.last_near_wait_ms,
                        s.loaded_count(),
                        s.mesh_mib()
                    );
                    break;
                }
                assert!(start.elapsed().as_secs() < 15);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        assert_eq!(
            placements[0], placements[1],
            "Scheduling must not alter tree placement"
        );
    }

    #[test]
    fn asynchronous_chunks_stay_bounded_and_release_assets_on_travel() {
        let (mut app, horse, camera) = test_stream(true);
        let settle = |app: &mut App| {
            let start = Instant::now();
            loop {
                let before = {
                    let s = app.world().get_resource::<StreamWorld>();
                    s.map_or(0, |s| s.generated + s.terrain_rebuilt + s.lod_rebuilt)
                };
                app.update();
                let stream = app.world().resource::<StreamWorld>();
                assert!(
                    stream.generated + stream.terrain_rebuilt + stream.lod_rebuilt - before <= 1,
                    "All work types must share one install per frame"
                );
                assert!(stream.loaded_count() <= MAX_CHUNKS && stream.pending_count() <= MAX_TASKS);
                assert!(stream.watershed_count() <= 13);
                assert!(app.world().resource::<Assets<Mesh>>().len() <= MAX_CHUNKS * 6);
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
        let (terrain, full_terrain) = app.world().resource::<StreamWorld>().terrain_triangles();
        assert!(terrain < full_terrain * 9 / 10);
        assert_eq!(
            app.world().resource::<StreamWorld>().loaded[&ChunkKey(0, 0)].surface_detail,
            TerrainDetail::Near
        );
        let old_surfaces: Vec<_> = app
            .world()
            .resource::<StreamWorld>()
            .loaded
            .values()
            .filter(|c| c.surface_detail != TerrainDetail::Near)
            .map(|c| c.surface_mesh)
            .collect();
        let reduced_bytes = app.world().resource::<StreamWorld>().mesh_mib();
        app.world_mut()
            .resource_mut::<StreamWorld>()
            .terrain_lod_enabled = false;
        settle(&mut app);
        assert_eq!(
            app.world().resource::<StreamWorld>().terrain_triangles().0,
            full_terrain
        );
        assert!(app.world().resource::<StreamWorld>().mesh_mib() > reduced_bytes);
        assert!(
            old_surfaces
                .iter()
                .all(|id| app.world().resource::<Assets<Mesh>>().get(*id).is_none())
        );
        app.world_mut()
            .resource_mut::<StreamWorld>()
            .terrain_lod_enabled = true;
        settle(&mut app);
        let (actual, full) = app.world().resource::<StreamWorld>().vegetation_vertices();
        assert!(
            actual < full * 3 / 4,
            "Offscreen/distant chunks must reduce vegetation geometry"
        );
        let obstacles = app.world().resource::<StreamWorld>().loaded[&ChunkKey(0, 0)]
            .obstacles
            .clone();
        // Disable simplification for comparison, then switch back at an overhead viewpoint.
        app.world_mut().resource_mut::<StreamWorld>().lod_enabled = false;
        settle(&mut app);
        let (actual, full) = app.world().resource::<StreamWorld>().vegetation_vertices();
        assert_eq!(actual, full);
        let ids: Vec<_> = app.world().resource::<StreamWorld>().loaded[&ChunkKey(0, 0)]
            .vegetation_meshes
            .clone();
        app.world_mut().resource_mut::<StreamWorld>().lod_enabled = true;
        *app.world_mut().get_mut::<Transform>(camera).unwrap() =
            Transform::from_xyz(0., 180., 80.).looking_at(Vec3::ZERO, Vec3::Y);
        settle(&mut app);
        let stream = app.world().resource::<StreamWorld>();
        assert!(stream.lod_rebuilt > 0);
        assert_eq!(stream.loaded[&ChunkKey(0, 0)].obstacles, obstacles);
        assert!(stream.lod_counts().0[1] + stream.lod_counts().0[2] > 0);
        let assets = app.world().resource::<Assets<Mesh>>();
        assert_eq!(assets.len(), stream.mesh_asset_count());
        assert!(
            ids.iter().all(|id| assets.get(*id).is_none()),
            "Replaced LOD assets must be freed"
        );
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
            assert_eq!(
                stream.loaded[&ChunkKey::at(p)].surface_detail,
                TerrainDetail::Near
            );
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
        // Both workers finish before the next update: nearby terrain must install
        // before a new background chunk, while the ring is still incomplete.
        let target = ChunkKey(3, 0);
        assert_eq!(
            app.world().resource::<StreamWorld>().loaded[&target].surface_detail,
            TerrainDetail::Far
        );
        app.world_mut()
            .get_mut::<Transform>(horse)
            .unwrap()
            .translation = Vec3::new(330., 0., 48.);
        app.update();
        let count = app.world().resource::<StreamWorld>().loaded_count();
        assert!(count < MAX_CHUNKS);
        assert!(
            app.world()
                .resource::<StreamWorld>()
                .pending_surface
                .contains_key(&target)
        );
        let wait = Instant::now();
        while {
            let s = app.world().resource::<StreamWorld>();
            !s.pending_surface.values().all(Task::is_finished)
                || !s.pending.values().all(Task::is_finished)
        } {
            assert!(wait.elapsed().as_secs() < 10);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let before = app.world().resource::<StreamWorld>().terrain_rebuilt;
        app.update();
        let s = app.world().resource::<StreamWorld>();
        assert_eq!(s.terrain_rebuilt, before + 1);
        assert_eq!(
            s.loaded_count(),
            count,
            "Terrain install must spend the only frame slot"
        );
        assert_eq!(s.loaded[&target].surface_detail, TerrainDetail::Near);
        assert!(s.last_near_wait_ms > 0.);
        // Inject a completed result from an obsolete distance request. It must
        // be dropped without replacing the current Near mesh or adding assets.
        settle(&mut app);
        let (g, trees, old_surface) = {
            let stream = app.world().resource::<StreamWorld>();
            (
                stream.generator.clone(),
                stream.loaded[&target].trees.clone(),
                stream.loaded[&target].surface_mesh,
            )
        };
        app.world_mut()
            .resource_mut::<StreamWorld>()
            .pending_surface
            .insert(
                target,
                AsyncComputeTaskPool::get()
                    .spawn(async move { rebuild_surface(&g, target, TerrainDetail::Far, &trees) }),
            );
        let wait = Instant::now();
        while !app.world().resource::<StreamWorld>().pending_surface[&target].is_finished() {
            assert!(wait.elapsed().as_secs() < 10);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let before = app.world().resource::<StreamWorld>().terrain_rebuilt;
        app.update();
        let s = app.world().resource::<StreamWorld>();
        assert!(!s.pending_surface.contains_key(&target));
        assert_eq!(s.terrain_rebuilt, before);
        assert_eq!(s.loaded[&target].surface_mesh, old_surface);
        settle(&mut app);
        assert_eq!(
            app.world().resource::<Assets<Mesh>>().len(),
            app.world().resource::<StreamWorld>().mesh_asset_count()
        );
    }

    #[test]
    fn rendered_water_covers_every_lake_inlet_without_a_gap() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            for owner in [(0, 0), (-2, -2), (2, -2)] {
                g.hydrology.prepare(&g, owner);
                let plan = g.hydrology.get(owner).unwrap();
                let samples: Vec<_> = plan
                    .segments
                    .iter()
                    .filter(|s| s.b.p.distance(plan.lake) <= crate::watershed::LAKE_RADIUS + 24.)
                    .flat_map(|s| (0..=64).map(move |i| s.a.p.lerp(s.b.p, i as f32 / 64.)))
                    .collect();
                let keys: BTreeSet<_> = samples.iter().copied().map(ChunkKey::at).collect();
                let mut triangles = Vec::new();
                for key in keys {
                    let origin = key.origin();
                    let land: Vec<_> = (0..SIDE)
                        .flat_map(|z| (0..SIDE).map(move |x| (x, z)))
                        .map(|(x, z)| {
                            let local = Vec2::new(x as f32, z as f32) * CELL;
                            [local.x, g.height(origin + local), local.y]
                        })
                        .collect();
                    let (river, lake) = water_meshes(&g, origin, &land);
                    for mesh in [river, lake] {
                        let Some(VertexAttributeValues::Float32x3(positions)) =
                            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
                        else {
                            panic!()
                        };
                        triangles.extend(positions.chunks_exact(3).map(|tri| {
                            tri.iter()
                                .map(|p| origin + Vec2::new(p[0], p[2]))
                                .collect::<Vec<_>>()
                        }));
                    }
                }
                for p in samples {
                    assert!(
                        triangles.iter().any(|tri| {
                            let signs =
                                [0, 1, 2].map(|i| (tri[(i + 1) % 3] - tri[i]).perp_dot(p - tri[i]));
                            signs.iter().all(|s| *s >= -0.001) || signs.iter().all(|s| *s <= 0.001)
                        }),
                        "Missing rendered water at lake inlet: seed {seed}, system {owner:?}, p {p:?}"
                    );
                }
            }
        }
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
                            let a = crate::watershed::key(s.a.p);
                            let b = crate::watershed::key(s.b.p);
                            if axis == 0 { a.0 != b.0 } else { a.1 != b.1 }
                        })
                        .expect("River must span the internal 768m region boundary");
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
