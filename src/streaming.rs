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
use bevy::platform::time::Instant;
use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::VisibilityRange,
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::render_resource::PrimitiveTopology,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const VERSION: u32 = 19;
pub const CHUNK_SIZE: f32 = 96.0;
#[cfg(not(target_arch = "wasm32"))]
pub const RADIUS: i32 = 3;
#[cfg(target_arch = "wasm32")]
pub const RADIUS: i32 = 1;
pub const MAX_CHUNKS: usize = ((RADIUS * 2 + 1) * (RADIUS * 2 + 1)) as usize;
#[cfg(not(target_arch = "wasm32"))]
const MAX_TASKS: usize = 2;
#[cfg(target_arch = "wasm32")]
const MAX_TASKS: usize = 1;
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
pub(crate) fn coordinate_seed(seed: u32, x: i32, z: i32) -> u32 {
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
    pub refined_grass: bool,
}
impl Generator {
    pub fn new(seed: u32, settings: TerrainSettings) -> Self {
        let g = Self {
            seed,
            settings,
            hydrology: Arc::default(),
            refined_grass: true,
        };
        // One coarse spawn-region plan; distant plans run in chunk workers.
        g.hydrology.prepare(&g, crate::watershed::key(Vec2::ZERO));
        g
    }
    pub fn with_grass_look(mut self, refined: bool) -> Self {
        self.refined_grass = refined;
        self
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
        block_on(self.tree_candidates_incremental(
            key,
            halo,
            &mut crate::work_budget::WorkBudget::unlimited(),
        ))
    }
    async fn tree_candidates_incremental(
        &self,
        key: ChunkKey,
        halo: i32,
        budget: &mut crate::work_budget::WorkBudget,
    ) -> Vec<(Vec2, f32)> {
        let origin = (key.origin() / TREE_CELL).as_ivec2();
        let side = (CHUNK_SIZE / TREE_CELL) as i32;
        let mut trees = Vec::new();
        for z in origin.y - halo..origin.y + side + halo {
            budget.checkpoint().await;
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
    wind: Vec<[f32; 2]>,
    uvs: Vec<[f32; 2]>,
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
        if let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            self.uvs.extend(uvs);
        } else {
            self.uvs.extend(std::iter::repeat_n([0.; 2], p.len()));
        }
    }
    pub(crate) fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]) {
        self.positions
            .extend([a.to_array(), b.to_array(), c.to_array()]);
        self.normals
            .extend([(b - a).cross(c - a).normalize().to_array(); 3]);
        self.colors.extend([color; 3]);
        self.uvs.extend([[0.; 2]; 3]);
    }
    pub(crate) fn append_wind(
        &mut self,
        mesh: Mesh,
        offset: Vec3,
        root: f32,
        height: f32,
        phase: f32,
        amplitude: f32,
    ) {
        let first = self.positions.len();
        self.append(mesh, offset);
        self.wind.extend(self.positions[first..].iter().map(|p| {
            let height = ((p[1] - root) / height.max(0.1)).clamp(0., 1.);
            [amplitude * height * height, phase]
        }));
    }
    pub(crate) fn plant_triangle(
        &mut self,
        points: [Vec3; 3],
        colors: [[f32; 4]; 3],
        wind: [[f32; 2]; 3],
    ) {
        self.triangle(points[0], points[1], points[2], colors[0]);
        let first = self.colors.len() - 3;
        self.colors[first..].copy_from_slice(&colors);
        self.wind.extend(wind);
    }
    pub(crate) fn plant_uvs(&mut self, uvs: [[f32; 2]; 3]) {
        let start = self.uvs.len() - 3;
        self.uvs[start..].copy_from_slice(&uvs);
    }
    pub(crate) fn plant_normals(&mut self, normals: [Vec3; 3]) {
        let start = self.normals.len() - 3;
        for (out, normal) in self.normals[start..].iter_mut().zip(normals) {
            *out = normal.normalize().to_array();
        }
    }
    pub(crate) fn mesh(self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors);
        if !self.wind.is_empty() {
            assert_eq!(self.wind.len(), mesh.count_vertices());
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs);
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, self.wind);
        }
        mesh
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
    block_on(generate_incremental(
        generator,
        key,
        settings,
        view,
        surface_detail,
        &mut crate::work_budget::WorkBudget::unlimited(),
    ))
}

async fn generate_incremental(
    generator: Generator,
    key: ChunkKey,
    settings: TreeSettings,
    view: Option<vegetation::View>,
    surface_detail: TerrainDetail,
    budget: &mut crate::work_budget::WorkBudget,
) -> ChunkData {
    let start = Instant::now();
    let origin = key.origin();
    let fields = crate::grassfield::Fields::new(generator.seed, origin);
    generator
        .hydrology
        .prepare_incremental(
            &generator,
            crate::watershed::key(origin + Vec2::splat(CHUNK_SIZE * 0.5)),
            budget,
        )
        .await;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut samples = Vec::new();
    let mut patches = Vec::new();
    for z in 0..SIDE {
        budget.checkpoint().await;
        for x in 0..SIDE {
            let p = origin + Vec2::new(x as f32, z as f32) * CELL;
            let height = generator.height(p);
            positions.push([x as f32 * CELL, height, z as f32 * CELL]);
            // World-coordinate central differences keep edge normals identical.
            let dx = generator.height(p + Vec2::X * CELL) - generator.height(p - Vec2::X * CELL);
            let dz = generator.height(p + Vec2::Y * CELL) - generator.height(p - Vec2::Y * CELL);
            normals.push(Vec3::new(-dx, 2. * CELL, -dz).normalize().to_array());
            let sample = generator.habitat(p);
            patches.push(crate::grassfield::ground_tint(
                &generator, &fields, p, sample,
            ));
            samples.push(sample);
        }
    }
    let (water, lake) = water_meshes_incremental(&generator, origin, &positions, budget).await;
    let water_vertices = water.count_vertices() + lake.count_vertices();
    for region in tree_regions(key) {
        generator
            .hydrology
            .prepare_incremental(&generator, region, budget)
            .await;
    }
    let neighbours = generator.tree_candidates_incremental(key, 3, budget).await;
    let trees: Vec<_> = neighbours
        .iter()
        .copied()
        .filter(|(p, _)| ChunkKey::at(*p) == key)
        .collect();
    let world = Meadow::from_stream_generator(generator.clone());
    let mut placed = Vec::new();
    let mut obstacles = Vec::new();
    for (p, size) in &trees {
        budget.checkpoint().await;
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
    let blades = crate::grassfield::generate_incremental(
        &generator, key, &fields, &positions, &samples, budget,
    )
    .await;
    // Small riparian bushes share the grass batch/LOD; no extra draw calls or collision.
    let mut shrubs = Vec::new();
    let mut random = SeedRandom(coordinate_seed(generator.seed ^ 0x21b5_724d, key.0, key.1));
    for _ in 0..64 {
        budget.checkpoint().await;
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
        refined_grass: generator.refined_grass,
    });
    let built = vegetation
        .build_streamed_incremental(vegetation.stream_request(view, None), None, budget)
        .await;
    budget.checkpoint().await;
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
    budget.checkpoint().await;
    let bytes = ground.bytes
        + built.vertices * vegetation::VERTEX_BYTES
        + vegetation.bytes()
        + built
            .request
            .as_ref()
            .map_or(0, vegetation::StreamRequest::bytes)
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
    block_on(rebuild_surface_incremental(
        g,
        key,
        detail,
        trees,
        &mut crate::work_budget::WorkBudget::unlimited(),
    ))
}

async fn rebuild_surface_incremental(
    g: &Generator,
    key: ChunkKey,
    detail: TerrainDetail,
    trees: &[(Vec2, f32)],
    budget: &mut crate::work_budget::WorkBudget,
) -> SurfaceBuilt {
    let start = Instant::now();
    let fields = crate::grassfield::Fields::new(g.seed, key.origin());
    for region in tree_regions(key) {
        g.hydrology.prepare_incremental(g, region, budget).await;
    }
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut samples = Vec::new();
    let mut patches = Vec::new();
    for z in 0..SIDE {
        budget.checkpoint().await;
        for x in 0..SIDE {
            let p = key.origin() + Vec2::new(x as f32, z as f32) * CELL;
            positions.push([x as f32 * CELL, g.height(p), z as f32 * CELL]);
            let dx = g.height(p + Vec2::X * CELL) - g.height(p - Vec2::X * CELL);
            let dz = g.height(p + Vec2::Y * CELL) - g.height(p - Vec2::Y * CELL);
            normals.push(Vec3::new(-dx, 2. * CELL, -dz).normalize().to_array());
            let sample = g.habitat(p);
            patches.push(crate::grassfield::ground_tint(g, &fields, p, sample));
            samples.push(sample);
        }
    }
    let mut built = simplify_surface(g, key, detail, trees, positions, normals, samples, patches);
    budget.checkpoint().await;
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
#[cfg(test)]
fn water_meshes(g: &Generator, origin: Vec2, land: &[[f32; 3]]) -> (Mesh, Mesh) {
    block_on(water_meshes_incremental(
        g,
        origin,
        land,
        &mut crate::work_budget::WorkBudget::unlimited(),
    ))
}
async fn water_meshes_incremental(
    g: &Generator,
    origin: Vec2,
    land: &[[f32; 3]],
    budget: &mut crate::work_budget::WorkBudget,
) -> (Mesh, Mesh) {
    let mut fields = Vec::with_capacity(land.len());
    for row in land.chunks(64) {
        budget.checkpoint().await;
        fields.extend(row.iter().map(|p| g.water(origin + Vec2::new(p[0], p[2]))));
    }
    let mut vertices = Vec::new();
    let mut lake_vertices = Vec::new();
    for z in 0..SIDE - 1 {
        budget.checkpoint().await;
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
    leaf_entity: Entity,
    grass_entity: Entity,
    grass_shadow_entity: Entity,
    grass_shadows: bool,
    grass_triangles: usize,
    vegetation: Arc<Blueprint>,
    levels: Levels,
    request: vegetation::StreamRequest,
    tree_lods: Vec<TreeResident>,
    grass_lods: Vec<TreeResident>,
    part_meshes: Vec<AssetId<Mesh>>,
    part_vertices: usize,
    vegetation_meshes: Vec<AssetId<Mesh>>,
    vegetation_vertices: usize,
    meshes: Vec<AssetId<Mesh>>,
    trees: Vec<(Vec2, f32)>,
    obstacles: Vec<TreeObstacle>,
    bytes: usize,
    water_vertices: usize,
}

#[derive(Default)]
struct TreeResident {
    pairs: [Option<TreePair>; 3],
}
struct TreePair {
    entities: Vec<Entity>,
    meshes: Vec<AssetId<Mesh>>,
    vertices: usize,
}
/// Add only newly requested tree levels and release expired levels. Existing
/// levels keep their handles, materials and common crown transform.
#[allow(clippy::too_many_arguments)]
fn install_tree_lods(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    root: Entity,
    blueprint: &Blueprint,
    plan: &vegetation::TreePlan,
    resident: &mut Vec<TreeResident>,
    additions: Vec<vegetation::TreeLodMesh>,
    wood: &Handle<crate::wind::VegetationMaterial>,
    leaves: &Handle<crate::wind::VegetationMaterial>,
) -> (Vec<AssetId<Mesh>>, Vec<AssetId<Mesh>>) {
    resident.resize_with(blueprint.trees.len(), TreeResident::default);
    let mut removed = Vec::new();
    let mut added = Vec::new();
    for (index, tree) in resident.iter_mut().enumerate() {
        for level in 0..3 {
            if plan.masks[index] & (1 << level) == 0
                && let Some(pair) = tree.pairs[level].take()
            {
                for entity in pair.entities {
                    commands.entity(entity).despawn();
                }
                for id in pair.meshes {
                    meshes.remove(id);
                    removed.push(id);
                }
            }
        }
    }
    for data in additions {
        let mut pair = TreePair {
            entities: vec![],
            meshes: vec![],
            vertices: 0,
        };
        for (mesh, material) in [(data.wood, wood), (data.leaves, leaves)] {
            if mesh.count_vertices() == 0 {
                continue;
            }
            let bounds = crate::wind::bounds(&mesh);
            pair.vertices += mesh.count_vertices();
            let mesh = meshes.add(mesh);
            pair.meshes.push(mesh.id());
            added.push(mesh.id());
            let entity = commands
                .spawn((
                    Name::new(format!("Tree {} {:?}", data.index, data.detail)),
                    Transform::from_translation(data.center),
                    Visibility::Inherited,
                    bounds,
                    Mesh3d(mesh),
                    MeshMaterial3d(material.clone()),
                    ChildOf(root),
                ))
                .id();
            pair.entities.push(entity);
        }
        assert!(resident[data.index].pairs[data.detail.index()].is_none());
        resident[data.index].pairs[data.detail.index()] = Some(pair);
    }
    for (index, tree) in resident.iter().enumerate() {
        for detail in [
            vegetation::Detail::Near,
            vegetation::Detail::Middle,
            vegetation::Detail::Far,
        ] {
            if let Some(pair) = &tree.pairs[detail.index()] {
                let range =
                    vegetation::tree_range(blueprint.trees[index].diameter, plan, index, detail);
                for entity in &pair.entities {
                    if let Some(range) = &range {
                        commands.entity(*entity).insert(range.clone());
                    } else {
                        commands.entity(*entity).remove::<VisibilityRange>();
                    }
                }
            }
        }
    }
    (removed, added)
}
fn tree_vertices(resident: &[TreeResident]) -> usize {
    resident
        .iter()
        .flat_map(|t| t.pairs.iter().flatten())
        .map(|p| p.vertices)
        .sum()
}
fn tree_mesh_ids(resident: &[TreeResident]) -> impl Iterator<Item = AssetId<Mesh>> + '_ {
    resident
        .iter()
        .flat_map(|t| t.pairs.iter().flatten())
        .flat_map(|p| p.meshes.iter().copied())
}
// Reuse the same bounded cache ownership as trees, with a visual and optional
// shadow mesh in each pair. Patch geometry keeps the original world roots.
#[allow(clippy::too_many_arguments)]
fn install_grass_lods(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    root: Entity,
    plan: &vegetation::GrassPlan,
    resident: &mut Vec<TreeResident>,
    additions: Vec<vegetation::GrassLodMesh>,
    visual_material: &Handle<crate::wind::VegetationMaterial>,
    shadow_material: &Handle<crate::wind::VegetationMaterial>,
    visible: bool,
    shadows: bool,
) -> (Vec<AssetId<Mesh>>, Vec<AssetId<Mesh>>) {
    resident.resize_with(vegetation::GRASS_PATCH_COUNT, TreeResident::default);
    let mut removed = Vec::new();
    let mut added = Vec::new();
    for (i, patch) in resident.iter_mut().enumerate() {
        for level in 0..2 {
            if plan.masks[i] & (1 << level) == 0
                && let Some(pair) = patch.pairs[level].take()
            {
                for entity in pair.entities {
                    commands.entity(entity).despawn();
                }
                for id in pair.meshes {
                    meshes.remove(id);
                    removed.push(id);
                }
            }
        }
    }
    for data in additions {
        let mut pair = TreePair {
            entities: vec![],
            meshes: vec![],
            vertices: 0,
        };
        for (shadow, mesh, material) in [
            (false, data.visual, visual_material),
            (true, data.shadow, shadow_material),
        ] {
            if mesh.count_vertices() == 0 {
                continue;
            }
            let bounds = crate::wind::bounds(&mesh);
            pair.vertices += mesh.count_vertices();
            let mesh = meshes.add(mesh);
            added.push(mesh.id());
            pair.meshes.push(mesh.id());
            let mut entity = commands.spawn((
                Name::new(format!(
                    "Grass patch {} {:?} {}",
                    data.index, data.detail, shadow
                )),
                Grass,
                Transform::from_translation(data.center),
                bounds,
                Mesh3d(mesh),
                MeshMaterial3d(material.clone()),
                ChildOf(root),
                if visible {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                },
            ));
            if shadow {
                entity.insert(bevy::camera::visibility::RenderLayers::layer(1));
                if !shadows {
                    entity.insert(bevy::light::NotShadowCaster);
                }
            } else {
                entity.insert((
                    bevy::light::NotShadowCaster,
                    bevy::light::TransmittedShadowReceiver,
                ));
            }
            pair.entities.push(entity.id());
        }
        assert!(resident[data.index].pairs[data.detail.index()].is_none());
        resident[data.index].pairs[data.detail.index()] = Some(pair);
    }
    for (i, patch) in resident.iter().enumerate() {
        for detail in [vegetation::Detail::Near, vegetation::Detail::Middle] {
            if let Some(pair) = &patch.pairs[detail.index()] {
                for entity in &pair.entities {
                    if let Some(range) = vegetation::grass_range(plan, i, detail) {
                        commands.entity(*entity).insert(range);
                    } else {
                        commands.entity(*entity).remove::<VisibilityRange>();
                    }
                }
            }
        }
    }
    (removed, added)
}
impl Resident {
    fn grass_detail(&self) -> vegetation::Detail {
        if self.request.grass.is_some() {
            if self.grass_lods.iter().any(|p| p.pairs[0].is_some()) {
                vegetation::Detail::Near
            } else {
                vegetation::Detail::Middle
            }
        } else {
            self.levels.grass
        }
    }
    fn grass_shadow_entities(&self) -> impl Iterator<Item = Entity> + '_ {
        std::iter::once(self.grass_shadow_entity).chain(
            self.grass_lods
                .iter()
                .filter_map(|p| p.pairs[0].as_ref())
                .filter_map(|p| p.entities.get(1).copied()),
        )
    }
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
    lod_eye: Vec3,
    pub lod_rebuilt: u64,
    pub tree_lod_rebuilt: u64,
    pub last_lod_ms: f64,
    pub last_lod_install_ms: f64,
    terrain_material: Handle<StandardMaterial>,
    trees_material: Handle<crate::wind::VegetationMaterial>,
    leaves_material: Handle<crate::wind::VegetationMaterial>,
    grass_material: Handle<crate::wind::VegetationMaterial>,
    grass_shadow_material: Handle<crate::wind::VegetationMaterial>,
    stones_material: Handle<StandardMaterial>,
    water_material: Handle<StandardMaterial>,
    lake_material: Handle<StandardMaterial>,
    pub generated: u64,
    pub evicted: u64,
    pub last_build_ms: f64,
    pub last_install_ms: f64,
}
impl StreamWorld {
    pub fn grass_look(&self) -> &'static str {
        if self.generator.refined_grass {
            "Refined"
        } else {
            "Legacy"
        }
    }
    /// Potential geometry in grass caster batches, before GPU/cascade culling.
    pub fn grass_shadow_stats(&self) -> (usize, usize, usize) {
        let mut stats = (0, 0, 0);
        for c in self.loaded.values().filter(|c| c.grass_triangles > 0) {
            stats.1 += 1;
            if c.grass_shadows {
                stats.0 += 1;
                stats.2 += c.grass_triangles;
            }
        }
        stats
    }
    pub fn tall_grass_count(&self, key: ChunkKey) -> usize {
        self.loaded.get(&key).map_or(0, |chunk| {
            chunk
                .vegetation
                .grass
                .iter()
                .filter(|b| b.c.y - (b.a.y + b.b.y) * 0.5 > 0.75)
                .count()
        })
    }
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
                let desired = chunk
                    .vegetation
                    .stream_request(view, Some(&chunk.request))
                    .levels;
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
    /// Candidate tree draws before frustum/cascade culling. Adjacent levels
    /// both count in a fade band; cached but inactive levels do not count.
    pub fn tree_stats(&self, eye: Vec3) -> ([usize; 3], usize, usize) {
        let mut counts = [0; 3];
        let mut vertices = 0;
        let mut entities = 0;
        for chunk in self.loaded.values() {
            for (index, tree) in chunk.tree_lods.iter().enumerate() {
                let placed = &chunk.vegetation.trees[index];
                let distance = (placed.center + chunk.vegetation.origin).distance(eye);
                for detail in [
                    vegetation::Detail::Near,
                    vegetation::Detail::Middle,
                    vegetation::Detail::Far,
                ] {
                    if let Some(pair) = &tree.pairs[detail.index()] {
                        entities += pair.entities.len();
                        let active = vegetation::tree_range(
                            placed.diameter,
                            &chunk.request.trees,
                            index,
                            detail,
                        )
                        .is_none_or(|r| {
                            distance >= r.start_margin.start && distance <= r.end_margin.end
                        });
                        if active {
                            counts[detail.index()] += 1;
                            vertices += pair.vertices;
                        }
                    }
                }
            }
        }
        (counts, vertices, entities)
    }
    pub fn lod_counts(&self) -> ([usize; 3], [usize; 3]) {
        let mut grass = [0; 3];
        for chunk in self.loaded.values() {
            if let Some(plan) = &chunk.request.grass {
                for (i, occupied) in plan.occupied.iter().enumerate() {
                    if !occupied {
                        continue;
                    }
                    let distance =
                        (plan.centers[i] + chunk.vegetation.origin).distance(self.lod_eye);
                    let mut active = false;
                    for detail in [vegetation::Detail::Near, vegetation::Detail::Middle] {
                        if chunk.grass_lods[i].pairs[detail.index()].is_some()
                            && vegetation::grass_range(plan, i, detail).is_none_or(|r| {
                                distance >= r.start_margin.start && distance <= r.end_margin.end
                            })
                        {
                            grass[detail.index()] += 1;
                            active = true;
                        }
                    }
                    if !active {
                        grass[2] += 1;
                    }
                }
            } else {
                grass[chunk.levels.grass.index()] += 1;
            }
        }
        (self.tree_stats(self.lod_eye).0, grass)
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
    mut plants: ResMut<Assets<crate::wind::VegetationMaterial>>,
    river: Option<Res<crate::water::RiverMaterial>>,
    shared_plants: Res<crate::wind::Materials>,
) {
    let Some(generator) = world.stream_generator() else {
        return;
    };
    let refined = generator.refined_grass;
    commands.insert_resource(StreamWorld {
        generator,
        settings: *settings,
        center: ChunkKey::at(Vec2::ZERO),
        lod_eye: Vec3::ZERO,
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
        tree_lod_rebuilt: 0,
        last_lod_ms: 0.,
        last_lod_install_ms: 0.,
        terrain_material: materials.add(StandardMaterial {
            perceptual_roughness: 1.,
            ..default()
        }),
        stones_material: materials.add(StandardMaterial {
            perceptual_roughness: 1.,
            ..default()
        }),
        trees_material: shared_plants.wood.clone(),
        leaves_material: shared_plants.leaves.clone(),
        grass_shadow_material: shared_plants.shadow_grass.clone(),
        grass_material: if refined {
            shared_plants.grass.clone()
        } else {
            plants.add(crate::wind::VegetationMaterial {
                base: StandardMaterial {
                    cull_mode: None,
                    double_sided: true,
                    perceptual_roughness: 1.,
                    ..default()
                },
                ..default()
            })
        },
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
                let tree_bounds = crate::wind::bounds(&data.built.trees);
                let leaf_bounds = crate::wind::bounds(&data.built.leaves);
                let grass_bounds = crate::wind::bounds(&data.built.grass);
                let shadow_bounds = crate::wind::bounds(&data.built.shadow_grass);
                let grass_triangles = if data.vegetation.refined_grass {
                    data.built.shadow_grass.count_vertices() / 3
                } else {
                    data.built.grass.count_vertices() / 3
                };
                let part_vertices = data.built.trees.count_vertices()
                    + data.built.leaves.count_vertices()
                    + data.built.grass.count_vertices()
                    + data.built.shadow_grass.count_vertices();
                let trees =
                    (data.built.trees.count_vertices() > 0).then(|| meshes.add(data.built.trees));
                let leaves =
                    (data.built.leaves.count_vertices() > 0).then(|| meshes.add(data.built.leaves));
                let grass =
                    (data.built.grass.count_vertices() > 0).then(|| meshes.add(data.built.grass));
                let shadow_grass = (data.built.shadow_grass.count_vertices() > 0)
                    .then(|| meshes.add(data.built.shadow_grass));
                let mut vegetation_meshes: Vec<_> = [
                    trees.as_ref(),
                    leaves.as_ref(),
                    grass.as_ref(),
                    shadow_grass.as_ref(),
                ]
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
                        leaves.as_ref(),
                        grass.as_ref(),
                        shadow_grass.as_ref(),
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
                let mut leaf_entity = Entity::PLACEHOLDER;
                let mut grass_shadow_entity = Entity::PLACEHOLDER;
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
                                MeshMaterial3d(stream.stones_material.clone()),
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
                            tree_bounds,
                            MeshMaterial3d(stream.trees_material.clone()),
                        ));
                        if let Some(trees) = trees {
                            entity.insert(Mesh3d(trees));
                        }
                        tree_entity = entity.id();
                        let mut entity = root.spawn((
                            Transform::default(),
                            Visibility::Inherited,
                            leaf_bounds,
                            MeshMaterial3d(stream.leaves_material.clone()),
                        ));
                        if let Some(leaves) = leaves {
                            entity.insert(Mesh3d(leaves));
                        }
                        leaf_entity = entity.id();
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
                            bevy::light::NotShadowCaster,
                            Transform::default(),
                            grass_bounds,
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
                        if stream.generator.refined_grass {
                            entity.insert(bevy::light::TransmittedShadowReceiver);
                        }
                        grass_entity = entity.id();
                        if data.vegetation.refined_grass {
                            let mut entity = root.spawn((
                                Grass,
                                Transform::default(),
                                Visibility::Inherited,
                                shadow_bounds,
                                bevy::light::NotShadowCaster,
                                bevy::camera::visibility::RenderLayers::layer(1),
                                MeshMaterial3d(stream.grass_shadow_material.clone()),
                            ));
                            if let Some(mesh) = shadow_grass {
                                entity.insert(Mesh3d(mesh));
                            }
                            grass_shadow_entity = entity.id();
                        } else {
                            grass_shadow_entity = grass_entity;
                        }
                    })
                    .id();
                let part_meshes = vegetation_meshes.clone();
                let request = data.built.request.take().unwrap();
                let mut tree_lods = Vec::new();
                let (_, added) = install_tree_lods(
                    &mut commands,
                    &mut meshes,
                    root,
                    &data.vegetation,
                    &request.trees,
                    &mut tree_lods,
                    std::mem::take(&mut data.built.tree_additions),
                    &stream.trees_material,
                    &stream.leaves_material,
                );
                let mut grass_lods = Vec::new();
                let mut grass_triangles = grass_triangles;
                let mut added = added;
                if let Some(plan) = &request.grass {
                    grass_triangles += data
                        .built
                        .grass_additions
                        .iter()
                        .map(|p| p.shadow.count_vertices() / 3)
                        .sum::<usize>();
                    let (_, ids) = install_grass_lods(
                        &mut commands,
                        &mut meshes,
                        root,
                        plan,
                        &mut grass_lods,
                        std::mem::take(&mut data.built.grass_additions),
                        &stream.grass_material,
                        &stream.grass_shadow_material,
                        *layer == MapLayer::Natural,
                        false,
                    );
                    added.extend(ids);
                }
                mesh_ids.extend(added.iter().copied());
                vegetation_meshes.extend(added);
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
                        leaf_entity,
                        grass_entity,
                        grass_shadow_entity,
                        grass_shadows: false,
                        grass_triangles,
                        vegetation: data.vegetation,
                        levels: data.built.levels,
                        request,
                        tree_lods,
                        grass_lods,
                        part_meshes,
                        part_vertices,
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
                let Some(mut built) =
                    block_on(poll_once(stream.pending_lod.get_mut(&key).unwrap()))
                else {
                    continue;
                };
                stream.pending_lod.remove(&key);
                let tree_material = stream.trees_material.clone();
                let leaf_material = stream.leaves_material.clone();
                let grass_material = stream.grass_material.clone();
                let grass_shadow_material = stream.grass_shadow_material.clone();
                let Some(chunk) = stream.loaded.get_mut(&key) else {
                    continue;
                };
                let desired = chunk.vegetation.stream_request(view, Some(&chunk.request));
                if !built.matches_cache(&chunk.request, &desired) {
                    continue;
                }
                let start = Instant::now();
                if built.parts_changed {
                    chunk.part_vertices = built.trees.count_vertices()
                        + built.leaves.count_vertices()
                        + built.grass.count_vertices()
                        + built.shadow_grass.count_vertices();
                    let tree_bounds = crate::wind::bounds(&built.trees);
                    let leaf_bounds = crate::wind::bounds(&built.leaves);
                    let grass_bounds = crate::wind::bounds(&built.grass);
                    let shadow_bounds = crate::wind::bounds(&built.shadow_grass);
                    chunk.grass_triangles = if chunk.vegetation.refined_grass {
                        built.shadow_grass.count_vertices() / 3
                    } else {
                        built.grass.count_vertices() / 3
                    };
                    let trees = (built.trees.count_vertices() > 0).then(|| meshes.add(built.trees));
                    let leaves =
                        (built.leaves.count_vertices() > 0).then(|| meshes.add(built.leaves));
                    let grass = (built.grass.count_vertices() > 0).then(|| meshes.add(built.grass));
                    let shadow_grass = (built.shadow_grass.count_vertices() > 0)
                        .then(|| meshes.add(built.shadow_grass));
                    for (entity, mesh, bounds) in [
                        (chunk.tree_entity, trees.as_ref(), tree_bounds),
                        (chunk.leaf_entity, leaves.as_ref(), leaf_bounds),
                        (chunk.grass_entity, grass.as_ref(), grass_bounds),
                    ] {
                        let mut entity = commands.entity(entity);
                        if let Some(mesh) = mesh {
                            entity.insert((Mesh3d(mesh.clone()), bounds));
                        } else {
                            entity.remove::<Mesh3d>();
                        }
                    }
                    if chunk.grass_shadow_entity != chunk.grass_entity {
                        let mut entity = commands.entity(chunk.grass_shadow_entity);
                        if let Some(mesh) = shadow_grass.as_ref() {
                            entity.insert((Mesh3d(mesh.clone()), shadow_bounds));
                        } else {
                            entity.remove::<Mesh3d>();
                        }
                        entity.insert(if *layer == MapLayer::Natural {
                            Visibility::Inherited
                        } else {
                            Visibility::Hidden
                        });
                    }
                    commands
                        .entity(chunk.grass_entity)
                        .insert(if *layer == MapLayer::Natural {
                            Visibility::Inherited
                        } else {
                            Visibility::Hidden
                        });
                    for id in &chunk.part_meshes {
                        meshes.remove(*id);
                    }
                    chunk.meshes.retain(|id| !chunk.part_meshes.contains(id));
                    chunk.part_meshes = [
                        trees.as_ref(),
                        leaves.as_ref(),
                        grass.as_ref(),
                        shadow_grass.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    .map(|m| m.id())
                    .collect();
                    chunk.meshes.extend(chunk.part_meshes.iter().copied());
                }
                let request = built.request.take().unwrap();
                let tree_changed =
                    request.trees != chunk.request.trees || !built.tree_additions.is_empty();
                if tree_changed {
                    let (removed, added) = install_tree_lods(
                        &mut commands,
                        &mut meshes,
                        chunk.root,
                        &chunk.vegetation,
                        &request.trees,
                        &mut chunk.tree_lods,
                        built.tree_additions,
                        &tree_material,
                        &leaf_material,
                    );
                    chunk.meshes.retain(|id| !removed.contains(id));
                    chunk.meshes.extend(added);
                }
                if let Some(plan) = &request.grass {
                    let (removed, added) = install_grass_lods(
                        &mut commands,
                        &mut meshes,
                        chunk.root,
                        plan,
                        &mut chunk.grass_lods,
                        built.grass_additions,
                        &grass_material,
                        &grass_shadow_material,
                        *layer == MapLayer::Natural,
                        chunk.grass_shadows,
                    );
                    chunk.meshes.retain(|id| !removed.contains(id));
                    chunk.meshes.extend(added);
                    chunk.grass_triangles = chunk
                        .grass_lods
                        .iter()
                        .filter_map(|p| p.pairs[0].as_ref())
                        .filter_map(|p| p.meshes.get(1))
                        .map(|id| meshes.get(*id).unwrap().count_vertices() / 3)
                        .sum();
                }
                chunk.vegetation_meshes = chunk
                    .part_meshes
                    .iter()
                    .copied()
                    .chain(tree_mesh_ids(&chunk.tree_lods))
                    .chain(tree_mesh_ids(&chunk.grass_lods))
                    .collect();
                let vertices = chunk.part_vertices
                    + tree_vertices(&chunk.tree_lods)
                    + tree_vertices(&chunk.grass_lods);
                chunk.bytes = chunk.bytes - chunk.vegetation_vertices * vegetation::VERTEX_BYTES
                    + vertices * vegetation::VERTEX_BYTES;
                chunk.vegetation_vertices = vertices;
                chunk.levels = request.levels;
                chunk.request = request;
                stream.tree_lod_rebuilt += u64::from(tree_changed);
                stream.lod_rebuilt += 1;
                stream.last_lod_ms = built.elapsed_ms;
                stream.last_lod_install_ms = start.elapsed().as_secs_f64() * 1000.;
            }
        }
        // Discarded/unfinished jobs do not spend this budget. One valid install does.
        break;
    }
}

// Chunk-level shadow radius: enter at 40m, exit at 56m to avoid toggling
// while riding near a boundary. Middle/Far never cast refined grass shadows.
fn grass_shadow_policy(
    key: ChunkKey,
    p: Vec2,
    detail: vegetation::Detail,
    previous: bool,
    refined: bool,
) -> bool {
    if !refined {
        return true;
    }
    if detail != vegetation::Detail::Near {
        return false;
    }
    let origin = key.origin();
    let distance = p.distance(p.clamp(origin, origin + Vec2::splat(CHUNK_SIZE)));
    distance <= if previous { 56. } else { 40. }
}
pub fn update_grass_shadows(
    mut commands: Commands,
    stream: Option<ResMut<StreamWorld>>,
    horse: Single<&Transform, With<HorseController>>,
) {
    let Some(mut stream) = stream else {
        return;
    };
    let p = Vec2::new(horse.translation.x, horse.translation.z);
    let refined = stream.generator.refined_grass;
    for (key, chunk) in &mut stream.loaded {
        let enabled = chunk.grass_triangles > 0
            && grass_shadow_policy(*key, p, chunk.grass_detail(), chunk.grass_shadows, refined);
        if enabled != chunk.grass_shadows {
            for entity in chunk.grass_shadow_entities() {
                let mut entity = commands.entity(entity);
                if enabled {
                    entity.remove::<bevy::light::NotShadowCaster>();
                } else {
                    entity.insert(bevy::light::NotShadowCaster);
                }
            }
            chunk.grass_shadows = enabled;
        }
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
    stream.lod_eye = view.map_or(horse.0.translation, |v| v.position);
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
            if chunk.vegetation.stream_request(view, Some(&chunk.request)) != chunk.request
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
                    AsyncComputeTaskPool::get().spawn(async move {
                        generate_incremental(
                            generator,
                            key,
                            settings,
                            view,
                            detail,
                            &mut crate::work_budget::WorkBudget::runtime(),
                        )
                        .await
                    }),
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
                    AsyncComputeTaskPool::get().spawn(async move {
                        rebuild_surface_incremental(
                            &g,
                            key,
                            detail,
                            &trees,
                            &mut crate::work_budget::WorkBudget::runtime(),
                        )
                        .await
                    }),
                );
            }
            Work::Vegetation(_) => {
                let chunk = &stream.loaded[&key];
                let previous = chunk.request.clone();
                let request = chunk.vegetation.stream_request(view, Some(&previous));
                let data = chunk.vegetation.clone();
                stream.pending_lod.insert(
                    key,
                    AsyncComputeTaskPool::get().spawn(async move {
                        data.build_streamed_incremental(
                            request,
                            Some(&previous),
                            &mut crate::work_budget::WorkBudget::runtime(),
                        )
                        .await
                    }),
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
        "detail,chunks,trees,tree_vertices,grass_vertices,shadow_vertices,vegetation_mesh_mib,blueprint_mib,mesh_build_ms\n",
    );
    for detail in [
        vegetation::Detail::Near,
        vegetation::Detail::Middle,
        vegetation::Detail::Far,
    ] {
        let start = Instant::now();
        let mut tree_vertices = 0;
        let mut grass_vertices = 0;
        let mut shadow_vertices = 0;
        for chunk in &chunks {
            let built = chunk.vegetation.build(Levels {
                trees: detail,
                grass: detail,
            });
            tree_vertices += built.trees.count_vertices() + built.leaves.count_vertices();
            grass_vertices += built.grass.count_vertices();
            shadow_vertices += built.shadow_grass.count_vertices();
        }
        let wall = start.elapsed().as_secs_f64() * 1000.;
        let descriptors =
            chunks.iter().map(|c| c.vegetation.bytes()).sum::<usize>() as f64 / (1024. * 1024.);
        csv.push_str(&format!("{detail:?},9,{tree_count},{tree_vertices},{grass_vertices},{shadow_vertices},{:.3},{descriptors:.3},{wall:.3}\n", (tree_vertices + grass_vertices + shadow_vertices) as f64 * vegetation::VERTEX_BYTES as f64 / (1024. * 1024.)));
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
    fn geometry_fingerprint(data: &ChunkData) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        let mut add = |text: String| {
            for byte in text.bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        };
        let meshes = [
            &data.surface,
            &data.water,
            &data.lake,
            &data.stones,
            &data.built.trees,
            &data.built.leaves,
            &data.built.grass,
            &data.built.shadow_grass,
        ];
        for mesh in meshes
            .into_iter()
            .chain(
                data.built
                    .tree_additions
                    .iter()
                    .flat_map(|t| [&t.wood, &t.leaves]),
            )
            .chain(
                data.built
                    .grass_additions
                    .iter()
                    .flat_map(|g| [&g.visual, &g.shadow]),
            )
        {
            for (attribute, values) in mesh.attributes() {
                add(format!("{}:{values:?}", attribute.name));
            }
            add(format!("{:?}", mesh.indices()));
        }
        add(format!(
            "{:?}:{:?}",
            data.trees,
            data.built.request.as_ref().map(|r| &r.trees.masks)
        ));
        hash
    }
    #[test]
    fn incremental_generation_preserves_published_geometry() {
        // Captured from the published synchronous generator before this fix.
        // Includes every position, normal, tint, UV, index and tree/grass LOD mesh.
        for (key, expected) in [
            (ChunkKey(0, 0), 0xa53e470470909e35),
            (ChunkKey(1, -1), 0x0d1843003a1071d6),
        ] {
            let data = generate_at(
                Generator::new(42, TerrainSettings::default()),
                key,
                TreeSettings::default(),
                None,
                TerrainDetail::Near,
            );
            let reference = geometry_fingerprint(&data);
            // These exact snapshots were captured on macOS/arm64. Native
            // trigonometric floats can differ across platforms; on every
            // platform the cooperative result must match its synchronous one.
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                assert_eq!(reference, expected);
            }
            let mut budget = crate::work_budget::WorkBudget::forced();
            let incremental = block_on(generate_incremental(
                Generator::new(42, TerrainSettings::default()),
                key,
                TreeSettings::default(),
                None,
                TerrainDetail::Near,
                &mut budget,
            ));
            assert_eq!(geometry_fingerprint(&incremental), reference);
            // A later terrain/vegetation rebuild must also yield and preserve
            // the same meshes when resumed after arbitrarily many pauses.
            let generator = Generator::new(42, TerrainSettings::default());
            let mut terrain_budget = crate::work_budget::WorkBudget::forced();
            let task = rebuild_surface_incremental(
                &generator,
                key,
                TerrainDetail::Middle,
                &data.trees,
                &mut terrain_budget,
            );
            let mut task = std::pin::pin!(task);
            assert!(block_on(poll_once(task.as_mut())).is_none());
            let _ = block_on(task);
            let request = data.vegetation.stream_request(None, None);
            let task = data
                .vegetation
                .build_streamed_incremental(request, None, &mut budget);
            let mut task = std::pin::pin!(task);
            assert!(block_on(poll_once(task.as_mut())).is_none());
            let built = block_on(task);
            assert_eq!(built.vertices, data.built.vertices);
        }
    }

    #[test]
    fn browser_chunk_job_yields_before_completing() {
        let mut budget = crate::work_budget::WorkBudget::forced();
        let generator = Generator::new(42, TerrainSettings::default());
        let cache = generator.hydrology.clone();
        let cached_before = cache.len();
        let task = generate_incremental(
            generator,
            ChunkKey(20, 20),
            TreeSettings::default(),
            None,
            TerrainDetail::Near,
            &mut budget,
        );
        let mut task = std::pin::pin!(task);
        assert!(
            block_on(poll_once(task.as_mut())).is_none(),
            "A Web chunk must not finish all generation in one event-loop turn"
        );
        assert_eq!(
            cache.len(),
            cached_before,
            "An unfinished river must not enter the shared cache"
        );
    }
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
                for mesh in [
                    &built.trees,
                    &built.leaves,
                    &built.grass,
                    &built.shadow_grass,
                ] {
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
            // Dense Far conifers keep every outer bough, avoiding a cone-to-
            // needles silhouette jump. Still require at least a 3x reduction.
            assert!(last < full.vertices / 3);
        }
    }
    #[test]
    fn grass_meadows_are_dense_grounded_and_repeatable_with_open_land() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            let mut tall = 0;
            let mut short = 0;
            for key in [ChunkKey(0, 0), ChunkKey(-1, -1), ChunkKey(1, 0)] {
                let chunk = generate(g.clone(), key, TreeSettings::default());
                let grass = &chunk.vegetation.grass;
                assert!(grass.len() <= crate::grassfield::MAX_TUFTS);
                let fields = crate::grassfield::Fields::new(seed, key.origin());
                for b in grass {
                    let root = (b.a + b.b) * 0.5;
                    let p = key.origin() + Vec2::new(root.x, root.z);
                    let height = b.c.y - root.y;
                    assert_eq!(ChunkKey::at(p), key);
                    assert!((root.y - g.ground(p)).abs() < 0.001);
                    assert!(g.water_depth(p) < -0.015);
                    assert!((0.349..=1.301).contains(&height));
                    if height > 0.75 {
                        tall += 1;
                        assert!(fields.strength(p) > 0.);
                        assert!(!g.hydrology.ford_route(p));
                    } else if fields.strength(p) == 0. {
                        short += 1;
                        assert!(height <= 0.551);
                    }
                }
                if key == ChunkKey(0, 0) {
                    g.hydrology.retain(&BTreeSet::new());
                    let again = generate(g.clone(), key, TreeSettings::default());
                    assert_eq!(*grass, again.vegetation.grass);
                    for detail in [vegetation::Detail::Middle, vegetation::Detail::Far] {
                        let mesh = chunk.vegetation.build(Levels {
                            trees: detail,
                            grass: detail,
                        });
                        assert!(mesh.grass.count_vertices() < grass.len() * 18);
                    }
                }
            }
            assert!(
                tall > 300 && short > 20,
                "seed {seed}, tall {tall}, short {short}"
            );
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
    #[test]
    fn grass_shadows_use_distance_hysteresis_near_detail_and_legacy_override() {
        let key = ChunkKey(0, 0);
        let near = vegetation::Detail::Near;
        for distance in [0., 39., 40.] {
            assert!(grass_shadow_policy(
                key,
                Vec2::new(CHUNK_SIZE + distance, 48.),
                near,
                false,
                true
            ));
        }
        for distance in [41., 48., 56.] {
            let p = Vec2::new(CHUNK_SIZE + distance, 48.);
            assert!(!grass_shadow_policy(key, p, near, false, true));
            assert!(grass_shadow_policy(key, p, near, true, true));
        }
        assert!(!grass_shadow_policy(
            key,
            Vec2::new(CHUNK_SIZE + 57., 48.),
            near,
            true,
            true
        ));
        for detail in [vegetation::Detail::Middle, vegetation::Detail::Far] {
            assert!(!grass_shadow_policy(key, Vec2::ZERO, detail, true, true));
            assert!(grass_shadow_policy(
                key,
                Vec2::splat(1000.),
                detail,
                false,
                false
            ));
        }
        assert!(grass_shadow_policy(
            ChunkKey(-1, -1),
            Vec2::splat(-48.),
            near,
            false,
            true
        ));
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
            .init_resource::<Assets<crate::wind::VegetationMaterial>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<crate::wind::Config>()
            .add_systems(Startup, (crate::wind::setup, setup).chain())
            .add_systems(Update, (update, schedule, update_grass_shadows).chain());
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
                assert!(
                    app.world().resource::<Assets<Mesh>>().len()
                        <= MAX_CHUNKS * (8 + vegetation::GRASS_PATCH_COUNT * 3)
                            + stream.tree_count() * 6
                );
                assert_eq!(
                    app.world().resource::<Assets<Mesh>>().len(),
                    stream.mesh_asset_count()
                );
                if stream.loaded_count() == MAX_CHUNKS && stream.pending_count() == 0 {
                    let horse = app.world().get::<Transform>(horse).unwrap().translation;
                    let p = Vec2::new(horse.x, horse.z);
                    for (key, chunk) in &stream.loaded {
                        let expected = chunk.grass_triangles > 0
                            && grass_shadow_policy(
                                *key,
                                p,
                                chunk.grass_detail(),
                                chunk.grass_shadows,
                                true,
                            );
                        assert_eq!(chunk.grass_shadows, expected);
                        assert_eq!(
                            app.world()
                                .get::<bevy::light::NotShadowCaster>(chunk.grass_shadow_entity)
                                .is_none(),
                            expected
                        );
                        assert!(
                            app.world()
                                .get::<bevy::light::TransmittedShadowReceiver>(chunk.grass_entity)
                                .is_some()
                        );
                    }
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
            ids.iter().all(|id| assets.get(*id).is_some()
                == stream.loaded[&ChunkKey(0, 0)].meshes.contains(id)),
            "Kept levels retain handles; expired levels must be freed"
        );
        // At 180m elevation all patches correctly cull. Return near ground
        // before checking that real grass assets remain stable on rotation.
        *app.world_mut().get_mut::<Transform>(camera).unwrap() =
            Transform::from_xyz(32., 8., 54.).looking_at(Vec3::new(32., 1., 40.), Vec3::Y);
        settle(&mut app);
        let stream = app.world().resource::<StreamWorld>();
        let assets = app.world().resource::<Assets<Mesh>>();
        let tree_ids: Vec<_> = stream
            .loaded
            .values()
            .flat_map(|c| tree_mesh_ids(&c.tree_lods))
            .collect();
        let grass_ids: Vec<_> = stream
            .loaded
            .values()
            .flat_map(|c| tree_mesh_ids(&c.grass_lods))
            .collect();
        assert!(!grass_ids.is_empty());
        let before_grass_vertices: usize = stream
            .loaded
            .values()
            .map(|c| tree_vertices(&c.grass_lods))
            .sum();
        for chunk in stream.loaded.values() {
            assert_eq!(
                chunk.vegetation_vertices,
                chunk.part_vertices
                    + tree_vertices(&chunk.tree_lods)
                    + tree_vertices(&chunk.grass_lods)
            );
            for pair in chunk
                .grass_lods
                .iter()
                .flat_map(|p| p.pairs.iter().flatten())
            {
                for id in &pair.meshes {
                    assert!(assets.get(*id).unwrap().count_vertices() > 0);
                }
            }
        }
        let before_tree_installs = stream.tree_lod_rebuilt;
        let before_vertices: usize = stream
            .loaded
            .values()
            .map(|c| tree_vertices(&c.tree_lods))
            .sum();
        for angle in [0.4, 1.6, 3.2, 5.8] {
            app.world_mut()
                .get_mut::<Transform>(camera)
                .unwrap()
                .rotation = Quat::from_rotation_y(angle);
            settle(&mut app);
            let stream = app.world().resource::<StreamWorld>();
            assert_eq!(before_tree_installs, stream.tree_lod_rebuilt);
            assert_eq!(
                grass_ids,
                stream
                    .loaded
                    .values()
                    .flat_map(|c| tree_mesh_ids(&c.grass_lods))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                before_grass_vertices,
                stream
                    .loaded
                    .values()
                    .map(|c| tree_vertices(&c.grass_lods))
                    .sum::<usize>()
            );
            assert_eq!(
                tree_ids,
                stream
                    .loaded
                    .values()
                    .flat_map(|c| tree_mesh_ids(&c.tree_lods))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                before_vertices,
                stream
                    .loaded
                    .values()
                    .map(|c| tree_vertices(&c.tree_lods))
                    .sum::<usize>()
            );
        }
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
