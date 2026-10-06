use crate::environment::{EnvironmentSample, MapLayer};
use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use std::time::Instant;

pub const EXTENT: f32 = 320.0;
pub const CELL: f32 = 2.0;
pub const PLAY_RADIUS: f32 = EXTENT - 20.0;
pub const VEGETATION_RADIUS: f32 = PLAY_RADIUS + 8.0;
pub const PLACEMENT_EXTENT: f32 = EXTENT - 15.0;
pub const AREA_SCALE: usize = ((EXTENT / 160.0) * (EXTENT / 160.0)) as usize;
pub(crate) const SIDE: usize = (EXTENT * 2.0 / CELL) as usize + 1;
pub const GENERATOR_VERSION: u32 = 8;

#[derive(Clone, Copy, Debug)]
pub struct TerrainSettings {
    pub mountain_strength: f32,
}
impl Default for TerrainSettings {
    fn default() -> Self {
        Self {
            mountain_strength: 0.65,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Massif {
    center: Vec2,
    radii: Vec2,
    angle: f32,
}
#[derive(Clone, Copy, Debug)]
struct Layout {
    angle: f32,
    mountains: [Massif; 3],
    basin: Vec2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Landform {
    Plains,
    Hills,
    Mountains,
    Basin,
}
impl Landform {
    pub fn label(self) -> &'static str {
        match self {
            Self::Plains => "PLAINS",
            Self::Hills => "HILLS",
            Self::Mountains => "MOUNTAINS",
            Self::Basin => "BASIN",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HeightStats {
    pub minimum: f32,
    pub maximum: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationTimings {
    pub water_ms: f64,
    pub height_ms: f64,
    pub environment_ms: f64,
    pub vegetation_ms: f64,
    pub navigation_ms: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Pond {
    pub center: Vec2,
    pub radii: Vec2,
    pub level: f32,
}

impl Pond {
    pub fn radius_at(&self, p: Vec2) -> f32 {
        ((p - self.center) / self.radii).length()
    }
}

#[derive(Resource)]
pub struct Meadow {
    pub seed: u32,
    streaming: Option<crate::streaming::Generator>,
    pub settings: TerrainSettings,
    pub timings: GenerationTimings,
    pub ponds: Vec<Pond>,
    pub river: Option<crate::river::River>,
    pub trees: Vec<(Vec2, f32)>,
    pub environment: Vec<EnvironmentSample>,
    heights: Vec<f32>,
    layout: Layout,
    observation_bank: (Vec2, f32),
    exploration_views: [(Vec2, f32); 4],
    pub reachable_fraction: f32,
}

impl Meadow {
    pub fn streamed(seed: u32, mut settings: TerrainSettings) -> Self {
        settings.mountain_strength = if settings.mountain_strength.is_finite() {
            settings.mountain_strength.clamp(0.0, 1.0)
        } else {
            0.65
        };
        Self::from_stream_generator(crate::streaming::Generator::new(seed, settings))
    }
    pub fn from_stream_generator(generator: crate::streaming::Generator) -> Self {
        let seed = generator.seed;
        let settings = generator.settings;
        Self {
            seed,
            settings,
            streaming: Some(generator),
            timings: GenerationTimings::default(),
            ponds: vec![],
            river: None,
            trees: vec![],
            environment: vec![],
            heights: vec![],
            layout: Layout {
                angle: 0.0,
                mountains: [Massif {
                    center: Vec2::ZERO,
                    radii: Vec2::ONE,
                    angle: 0.0,
                }; 3],
                basin: Vec2::ZERO,
            },
            observation_bank: (Vec2::ZERO, 0.0),
            exploration_views: [(Vec2::ZERO, 0.0); 4],
            reachable_fraction: 0.0,
        }
    }
    pub fn stream_generator(&self) -> Option<crate::streaming::Generator> {
        self.streaming.clone()
    }
    pub fn is_streaming(&self) -> bool {
        self.streaming.is_some()
    }

    pub fn new(seed: u32) -> Self {
        Self::with_settings(seed, TerrainSettings::default())
    }

    pub fn with_settings(seed: u32, mut settings: TerrainSettings) -> Self {
        settings.mountain_strength = if settings.mountain_strength.is_finite() {
            settings.mountain_strength.clamp(0.0, 1.0)
        } else {
            TerrainSettings::default().mountain_strength
        };
        let mut layout_random = SeedRandom(seed ^ 0x3141_5926);
        let angle = layout_random.range(0.0, std::f32::consts::TAU);
        let mountains = std::array::from_fn(|index| {
            let direction = angle
                + index as f32 * std::f32::consts::TAU / 3.0
                + layout_random.range(-0.28, 0.28);
            Massif {
                center: Vec2::from_angle(direction) * layout_random.range(165.0, 220.0),
                radii: Vec2::new(
                    layout_random.range(65.0, 90.0),
                    layout_random.range(48.0, 72.0),
                ),
                angle: layout_random.range(0.0, std::f32::consts::PI),
            }
        });
        let mut world = Self {
            seed,
            streaming: None,
            settings,
            timings: GenerationTimings::default(),
            ponds: vec![],
            river: None,
            trees: vec![],
            environment: vec![],
            heights: Vec::with_capacity(SIDE * SIDE),
            observation_bank: (Vec2::ZERO, 0.0),
            exploration_views: [(Vec2::ZERO, 0.0); 4],
            reachable_fraction: 0.0,
            layout: Layout {
                angle,
                mountains,
                basin: Vec2::from_angle(angle + 0.85) * layout_random.range(110.0, 145.0),
            },
        };
        let mut random = SeedRandom(seed ^ 0x8db7_1053);
        let water_start = Instant::now();
        for (center, radii) in [
            (Vec2::new(34.0, -33.0), Vec2::new(15.0, 11.0)),
            (Vec2::new(-53.0, 27.0), Vec2::new(20.0, 13.0)),
        ] {
            let center = center + Vec2::new(random.range(-8.0, 8.0), random.range(-8.0, 8.0));
            // A hillside's centre can be much higher than its downhill bank.
            // Keep the water below the lowest surrounding land, then carve a
            // closed bowl. This is a local basin, not a drainage simulation.
            let mut level = f32::INFINITY;
            for ring in 0..8 {
                let radius = 1.0 + ring as f32 * 0.1;
                for step in 0..96 {
                    let direction = Vec2::from_angle(step as f32 * std::f32::consts::TAU / 96.0);
                    level = level.min(world.land_height(center + direction * radii * radius));
                }
            }
            let level = level - 0.4;
            world.ponds.push(Pond {
                center,
                radii,
                level,
            });
        }
        world.timings.water_ms = water_start.elapsed().as_secs_f64() * 1000.0;
        let height_start = Instant::now();
        for z in 0..SIDE {
            for x in 0..SIDE {
                world.heights.push(world.analytic_height(Vec2::new(
                    x as f32 * CELL - EXTENT,
                    z as f32 * CELL - EXTENT,
                )));
            }
        }
        world.timings.height_ms = height_start.elapsed().as_secs_f64() * 1000.0;
        let river_start = Instant::now();
        // Protect the starting meadow and pond bowls, without limiting water
        // to one side of a prescribed valley. Drainage follows the region heights.
        let allowed: Vec<bool> = (0..SIDE * SIDE)
            .map(|index| {
                let p = crate::river::position(index);
                p.length() > 55.0
                    && world.ponds.iter().all(|pond| {
                        let protected = Pond {
                            radii: pond.radii * 1.6 + Vec2::splat(12.0),
                            ..*pond
                        };
                        protected.radius_at(p) > 1.0
                    })
            })
            .collect();
        let target = world.layout.mountains[0].center * 0.75;
        let source = (0..SIDE * SIDE)
            .filter(|i| allowed[*i])
            .min_by(|a, b| {
                crate::river::position(*a)
                    .distance_squared(target)
                    .total_cmp(&crate::river::position(*b).distance_squared(target))
            })
            .unwrap();
        let river = crate::river::River::generate(&world.heights, &allowed, source);
        for (index, height) in world.heights.iter_mut().enumerate() {
            *height = river.carve(index, *height);
        }
        world.river = Some(river);
        world.timings.water_ms += river_start.elapsed().as_secs_f64() * 1000.0;
        let environment_start = Instant::now();
        world.environment = (0..SIDE * SIDE)
            .map(|index| EnvironmentSample::at(&world, crate::river::position(index)))
            .collect();
        world.timings.environment_ms = environment_start.elapsed().as_secs_f64() * 1000.0;
        let vegetation_start = Instant::now();
        // Habitat potential controls density; exclusions keep water and paths clear.
        for _ in 0..2000 * AREA_SCALE {
            let p = Vec2::new(
                random.range(-PLACEMENT_EXTENT, PLACEMENT_EXTENT),
                random.range(-PLACEMENT_EXTENT, PLACEMENT_EXTENT),
            );
            if p.length() < 17.0 || p.length() > VEGETATION_RADIUS || world.near_water(p, 3.0) {
                continue;
            }
            let environment = world.environment_at(p);
            if random.unit() > environment.tree_density * 0.85 {
                continue;
            }
            // Check the actual triangle slope too, between cached grid samples.
            if world.slope(p) > 0.65 {
                continue;
            }
            if world.trees.iter().any(|(q, _)| q.distance(p) < 5.0) {
                continue;
            }
            let vigor = (0.85 + environment.moisture * 0.25) * (1.0 - environment.rockiness * 0.18);
            world.trees.push((p, random.range(0.8, 1.5) * vigor));
            if world.trees.len() >= 240 * AREA_SCALE {
                break;
            }
        }
        world.timings.vegetation_ms = vegetation_start.elapsed().as_secs_f64() * 1000.0;
        let navigation_start = Instant::now();
        world.prepare_navigation();
        world.timings.navigation_ms = navigation_start.elapsed().as_secs_f64() * 1000.0;
        world
    }

    pub(crate) fn noise(&self, p: Vec2) -> f32 {
        let cell = p.floor();
        let fraction = p - cell;
        let blend = fraction * fraction * (Vec2::splat(3.0) - fraction * 2.0);
        let h = |x: i32, z: i32| {
            let n = (x as u32).wrapping_mul(0x9e37_79b9)
                ^ (z as u32).wrapping_mul(0x85eb_ca6b)
                ^ self.seed;
            hash(n) as f32 / u32::MAX as f32
        };
        let x = cell.x as i32;
        let z = cell.y as i32;
        let a = lerp(h(x, z), h(x + 1, z), blend.x);
        let b = lerp(h(x, z + 1), h(x + 1, z + 1), blend.x);
        lerp(a, b, blend.y)
    }

    fn region_weights(&self, p: Vec2) -> (f32, f32, f32) {
        let hills = smooth((self.noise(p * 0.004 + Vec2::new(143.0, -89.0)) - 0.30) / 0.40);
        let mountains = self
            .layout
            .mountains
            .iter()
            .map(|patch| {
                let q = Vec2::from_angle(patch.angle).rotate(p - patch.center) / patch.radii;
                1.0 - smooth((q.length() - 0.25) / 1.0)
            })
            .fold(0.0, f32::max);
        let basin = 1.0 - smooth(p.distance(self.layout.basin) / 85.0);
        (hills, mountains, basin)
    }

    pub fn landform(&self, p: Vec2) -> Landform {
        if let Some(generator) = &self.streaming {
            return generator.landform(p);
        }
        let (hills, mountains, basin) = self.region_weights(p);
        if mountains * self.settings.mountain_strength > 0.20 {
            Landform::Mountains
        } else if basin > 0.35 {
            Landform::Basin
        } else if hills > 0.45 {
            Landform::Hills
        } else {
            Landform::Plains
        }
    }

    fn raw_height(&self, p: Vec2) -> f32 {
        // Broad regional masks blend plains, rolling hills, local mountain
        // clusters and a shallow basin. No paired walls or fixed valley axis.
        let (hills, mountains, basin) = self.region_weights(p);
        let plains = (self.fractal_noise(p * 0.008) - 0.5) * 4.0;
        let rolling = hills
            * (6.0 + self.settings.mountain_strength * 16.0)
            * (0.25 + self.fractal_noise(p * 0.010 + Vec2::splat(72.0)) * 0.75);
        let warp = Vec2::new(
            self.noise(p * 0.009 + Vec2::new(71.0, 19.0)) - 0.5,
            self.noise(p * 0.009 + Vec2::new(-37.0, 53.0)) - 0.5,
        ) * 18.0;
        let ridge = self.ridged_noise((p + warp) * 0.013);
        let peaks =
            mountains * self.settings.mountain_strength * 85.0 * (0.42 + ridge.powf(1.25) * 0.58);
        let grade = Vec2::from_angle(self.layout.angle).dot(p) * 0.008;
        plains + rolling + peaks + grade - basin * 7.0
    }

    fn fractal_noise(&self, p: Vec2) -> f32 {
        let mut sum = 0.0;
        let mut weight = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        for octave in 0..4 {
            let q = Vec2::from_angle(octave as f32 * 0.71).rotate(p) * frequency
                + Vec2::new(octave as f32 * 19.3, octave as f32 * -31.7);
            sum += self.noise(q) * amplitude;
            weight += amplitude;
            frequency *= 2.03;
            amplitude *= 0.45;
        }
        sum / weight
    }

    fn ridged_noise(&self, p: Vec2) -> f32 {
        let mut sum = 0.0;
        let mut weight = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        for octave in 0..3 {
            let q = Vec2::from_angle(octave as f32 * 0.57).rotate(p) * frequency
                + Vec2::new(octave as f32 * 43.1, octave as f32 * 27.9);
            let ridge = 1.0 - (2.0 * self.noise(q) - 1.0).abs();
            sum += ridge * amplitude;
            weight += amplitude;
            frequency *= 2.07;
            amplitude *= 0.4;
        }
        sum / weight
    }

    pub fn exploration_view(&self, index: usize) -> (Vec2, f32) {
        self.exploration_views[index % self.exploration_views.len()]
    }

    pub fn height_stats(&self) -> HeightStats {
        HeightStats {
            minimum: self.heights.iter().copied().fold(f32::INFINITY, f32::min),
            maximum: self
                .heights
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
        }
    }

    pub fn slope(&self, p: Vec2) -> f32 {
        let dx = self.ground(p + Vec2::X * 0.5) - self.ground(p - Vec2::X * 0.5);
        let dz = self.ground(p + Vec2::Y * 0.5) - self.ground(p - Vec2::Y * 0.5);
        Vec2::new(dx, dz).length()
    }

    fn land_height(&self, p: Vec2) -> f32 {
        let mut h = self.raw_height(p);
        // A calm starting clearing smoothly joins the surrounding land.
        let clearing = 1.0 - smooth(p.length() / 18.0);
        h = lerp(h, self.raw_height(Vec2::ZERO), clearing);
        h
    }

    fn analytic_height(&self, p: Vec2) -> f32 {
        let mut h = self.land_height(p);
        for pond in &self.ponds {
            let radius = pond.radius_at(p);
            let bowl = pond.level - 1.8 + 2.6 * smooth(radius / 1.12);
            let influence = 1.0 - smooth((radius - 1.05) / 0.55);
            h = lerp(h, bowl, influence);
        }
        h
    }

    /// Interpolate the actual rendered triangle, avoiding floating feet on hills.
    pub fn ground(&self, p: Vec2) -> f32 {
        if let Some(generator) = &self.streaming {
            return generator.ground(p);
        }
        let grid =
            ((p + Vec2::splat(EXTENT)) / CELL).clamp(Vec2::ZERO, Vec2::splat((SIDE - 1) as f32));
        let x = (grid.x.floor() as usize).min(SIDE - 2);
        let z = (grid.y.floor() as usize).min(SIDE - 2);
        let u = grid.x - x as f32;
        let v = grid.y - z as f32;
        let a = self.heights[z * SIDE + x];
        let b = self.heights[z * SIDE + x + 1];
        let c = self.heights[(z + 1) * SIDE + x];
        let d = self.heights[(z + 1) * SIDE + x + 1];
        if u + v <= 1.0 {
            a + (b - a) * u + (c - a) * v
        } else {
            d + (c - d) * (1.0 - u) + (b - d) * (1.0 - v)
        }
    }

    pub fn near_water(&self, p: Vec2, margin: f32) -> bool {
        if let Some(g) = &self.streaming {
            let f = g.water(p);
            return f.distance < f.width + 1. + margin;
        }
        self.river.as_ref().is_some_and(|river| {
            let field = river.at(p);
            field.distance < field.width + 1.0 + margin
        }) || self.ponds.iter().any(|pond| {
            let expanded = Pond {
                radii: pond.radii * 0.95 + Vec2::splat(margin + 2.0),
                ..*pond
            };
            expanded.radius_at(p) < 1.0
        })
    }

    /// Approximate horizontal distance to a bank, not a hydrological simulation.
    pub fn water_distance(&self, p: Vec2) -> f32 {
        if let Some(g) = &self.streaming {
            let f = g.water(p);
            return (f.distance - f.width).max(0.);
        }
        let river = self.river.as_ref().map_or(f32::INFINITY, |river| {
            let field = river.at(p);
            (field.distance - field.width - 1.0).max(0.0)
        });
        self.ponds.iter().fold(river, |distance, pond| {
            distance.min(((pond.radius_at(p) - 1.0) * pond.radii.min_element()).max(0.0))
        })
    }

    pub fn environment_at(&self, p: Vec2) -> EnvironmentSample {
        if let Some(generator) = &self.streaming {
            return generator.habitat(p);
        }
        let grid =
            ((p + Vec2::splat(EXTENT)) / CELL).clamp(Vec2::ZERO, Vec2::splat((SIDE - 1) as f32));
        let x = (grid.x.floor() as usize).min(SIDE - 2);
        let z = (grid.y.floor() as usize).min(SIDE - 2);
        let u = grid.x - x as f32;
        let v = grid.y - z as f32;
        let indices = [
            z * SIDE + x,
            z * SIDE + x + 1,
            (z + 1) * SIDE + x,
            (z + 1) * SIDE + x + 1,
        ];
        let weights = [(1.0 - u) * (1.0 - v), u * (1.0 - v), (1.0 - u) * v, u * v];
        let interpolate = |field: fn(EnvironmentSample) -> f32| {
            indices
                .into_iter()
                .zip(weights)
                .map(|(i, w)| field(self.environment[i]) * w)
                .sum()
        };
        EnvironmentSample {
            elevation: interpolate(|s| s.elevation),
            slope: interpolate(|s| s.slope),
            water_distance: interpolate(|s| s.water_distance),
            moisture: interpolate(|s| s.moisture),
            tree_density: interpolate(|s| s.tree_density),
            grass_density: interpolate(|s| s.grass_density),
            rockiness: interpolate(|s| s.rockiness),
        }
    }

    pub fn surface_colors(&self, layer: MapLayer) -> Vec<[f32; 4]> {
        self.environment
            .iter()
            .enumerate()
            .map(|(i, sample)| {
                let patch = if layer == MapLayer::Natural {
                    self.noise(crate::river::position(i) * 0.08)
                } else {
                    0.0
                };
                sample.color(layer, patch)
            })
            .collect()
    }

    pub fn walkable(&self, p: Vec2) -> bool {
        if self.streaming.is_some() {
            return !self.near_water(p, 0.6)
                && self.slope(p) < 0.85
                && self
                    .trees
                    .iter()
                    .all(|(q, size)| q.distance(p) >= 0.65 + size * 0.30);
        }
        if p.length() > PLAY_RADIUS || self.near_water(p, 0.6) {
            return false;
        }
        if self
            .trees
            .iter()
            .any(|(q, size)| q.distance(p) < 0.65 + size * 0.30)
        {
            return false;
        }
        self.slope(p) < 0.85
    }

    pub fn river_bank(&self) -> (Vec2, f32) {
        self.observation_bank
    }

    fn prepare_navigation(&mut self) {
        let river = self.river.as_ref().unwrap();
        let mut best = None;
        // A dry point can be on the far side of a tributary. Only offer an
        // observation bank in the same walkable grid component as the start.
        let walkable: Vec<_> = (0..SIDE * SIDE)
            .map(|i| self.walkable(crate::river::position(i)))
            .collect();
        let mut reachable = vec![false; SIDE * SIDE];
        let start = SIDE / 2 * SIDE + SIDE / 2;
        let mut queue = std::collections::VecDeque::from([start]);
        reachable[start] = true;
        while let Some(i) = queue.pop_front() {
            let x = i % SIDE;
            let z = i / SIDE;
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let nx = x as i32 + dx;
                let nz = z as i32 + dz;
                if nx < 0 || nz < 0 || nx >= SIDE as i32 || nz >= SIDE as i32 {
                    continue;
                }
                let next = nz as usize * SIDE + nx as usize;
                if !reachable[next] && walkable[next] {
                    reachable[next] = true;
                    queue.push_back(next);
                }
            }
        }
        for pair in river.reaches.iter().flat_map(|r| r.windows(2).step_by(8)) {
            let point = pair[0];
            if point.position.length() > PLAY_RADIUS - 18.0 {
                continue;
            }
            let direction = (pair[1].position - point.position).normalize();
            for side in [-1.0, 1.0] {
                let p = point.position
                    + Vec2::new(-direction.y, direction.x) * side * (point.width + 5.0);
                let look = (point.position - p).normalize();
                let camera = p - look * 7.5;
                // The shortcut is an observation spot: keep both the horse
                // and its following camera out of surrounding tree crowns.
                if self.trees.iter().any(|(tree, size)| {
                    tree.distance(p) < size * 5.0 + 2.0 || tree.distance(camera) < size * 5.0 + 2.0
                }) {
                    continue;
                }
                let grid = ((p + Vec2::splat(EXTENT)) / CELL).round().as_uvec2();
                if self.walkable(p)
                    && reachable[grid.y as usize * SIDE + grid.x as usize]
                    && best.is_none_or(|(previous, _): (Vec2, f32)| {
                        p.length_squared() < previous.length_squared()
                    })
                {
                    best = Some((p, (-look.x).atan2(-look.y)));
                }
            }
        }
        self.observation_bank = best.expect("River must have an accessible bank");
        self.reachable_fraction = reachable.iter().filter(|&&v| v).count() as f32
            / walkable.iter().filter(|&&v| v).count().max(1) as f32;
        for (view, direction) in [Vec2::X, Vec2::Y, Vec2::NEG_X, Vec2::NEG_Y]
            .into_iter()
            .enumerate()
        {
            let target = direction * PLAY_RADIUS * 0.75;
            let mut candidates: Vec<_> = reachable
                .iter()
                .enumerate()
                .filter_map(|(i, &yes)| yes.then_some(crate::river::position(i)))
                .collect();
            candidates.sort_by(|a, b| {
                a.distance_squared(target)
                    .total_cmp(&b.distance_squared(target))
            });
            self.exploration_views[view] = candidates
                .into_iter()
                .find_map(|p| {
                    let toward_center = (-p).to_angle();
                    for offset in [0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, std::f32::consts::PI] {
                        let heading = Vec2::from_angle(toward_center + offset);
                        let camera = p - heading * 7.5;
                        if self.trees.iter().any(|(tree, size)| {
                            tree.distance(p) < size * 5.0 + 2.0
                                || tree.distance(camera) < size * 5.0 + 2.0
                        }) {
                            continue;
                        }
                        if (1..=36).all(|step| self.walkable(p + heading * step as f32 * 0.5)) {
                            return Some((p, (-heading.x).atan2(-heading.y)));
                        }
                    }
                    None
                })
                .expect("Reachable region must have a walking viewpoint");
        }
    }

    pub fn river_mesh(&self) -> Mesh {
        self.river.as_ref().unwrap().mesh(&self.heights)
    }

    /// Clip the actual ground triangles at the water level. Every exposed
    /// boundary ends on the rendered bank, rather than on a floating ellipse.
    pub fn water_triangles(&self, pond: &Pond) -> Vec<[Vec3; 3]> {
        let mut surface = Vec::new();
        for z in 0..SIDE - 1 {
            for x in 0..SIDE - 1 {
                let vertex = |dx: usize, dz: usize| {
                    Vec3::new(
                        (x + dx) as f32 * CELL - EXTENT,
                        self.heights[(z + dz) * SIDE + x + dx],
                        (z + dz) as f32 * CELL - EXTENT,
                    )
                };
                let [a, b, c, d] = [vertex(0, 0), vertex(1, 0), vertex(0, 1), vertex(1, 1)];
                for triangle in [[a, c, b], [b, c, d]] {
                    if !triangle
                        .iter()
                        .any(|v| pond.radius_at(Vec2::new(v.x, v.z)) < 1.25)
                    {
                        continue;
                    }
                    let mut polygon = Vec::with_capacity(4);
                    for i in 0..3 {
                        let from = triangle[i];
                        let to = triangle[(i + 1) % 3];
                        let inside = from.y < pond.level;
                        if inside {
                            polygon.push(Vec3::new(from.x, pond.level, from.z));
                        }
                        if inside != (to.y < pond.level) {
                            let t = (pond.level - from.y) / (to.y - from.y);
                            let crossing = from.lerp(to, t);
                            polygon.push(Vec3::new(crossing.x, pond.level, crossing.z));
                        }
                    }
                    for i in 1..polygon.len().saturating_sub(1) {
                        let tri = [polygon[0], polygon[i], polygon[i + 1]];
                        if (tri[1] - tri[0]).cross(tri[2] - tri[0]).length_squared() > 1e-10 {
                            surface.push(tri);
                        }
                    }
                }
            }
        }
        surface
    }

    pub fn water_mesh(&self, pond: &Pond) -> Mesh {
        let positions: Vec<[f32; 3]> = self
            .water_triangles(pond)
            .into_iter()
            .flatten()
            .map(|v| v.to_array())
            .collect();
        let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    }

    pub fn mesh(&self) -> Mesh {
        let mut positions = Vec::with_capacity(SIDE * SIDE);
        let mut normals = Vec::with_capacity(SIDE * SIDE);
        let colors = self.surface_colors(MapLayer::Natural);
        let mut indices = Vec::with_capacity((SIDE - 1) * (SIDE - 1) * 6);
        for z in 0..SIDE {
            for x in 0..SIDE {
                let p = Vec2::new(x as f32 * CELL - EXTENT, z as f32 * CELL - EXTENT);
                let h = self.heights[z * SIDE + x];
                positions.push([p.x, h, p.y]);
                let dx = self.ground(p + Vec2::X) - self.ground(p - Vec2::X);
                let dz = self.ground(p + Vec2::Y) - self.ground(p - Vec2::Y);
                normals.push(Vec3::new(-dx, 2.0, -dz).normalize().to_array());
                if x < SIDE - 1 && z < SIDE - 1 {
                    let a = (z * SIDE + x) as u32;
                    let b = a + 1;
                    let c = a + SIDE as u32;
                    let d = c + 1;
                    indices.extend([a, c, b, b, c, d]);
                }
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
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
// 32-bit mix listed in skeeto/hash-prospector (Unlicense).
// Source: https://github.com/skeeto/hash-prospector.
fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

pub struct SeedRandom(pub u32);
impl SeedRandom {
    pub fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9);
        hash(self.0) as f32 / u32::MAX as f32
    }
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        lerp(a, b, self.unit())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn river_drains_to_boundary_and_its_surface_and_banks_fit_the_ground() {
        for seed in [20261003, 42, 314159] {
            for mountain_strength in [0.0, 0.65, 1.0] {
                let world = Meadow::with_settings(seed, TerrainSettings { mountain_strength });
                let river = world.river.as_ref().unwrap();
                let again = Meadow::with_settings(seed, TerrainSettings { mountain_strength });
                assert_eq!(river.points, again.river.as_ref().unwrap().points);
                assert!(
                    river.length() > 70.0,
                    "seed {seed}: short river {}",
                    river.length()
                );
                let end = river.points.last().unwrap();
                assert!(
                    (end.position.x.abs() - EXTENT).abs() < 0.001
                        || (end.position.y.abs() - EXTENT).abs() < 0.001
                );
                for pair in river.points.windows(2) {
                    assert!(pair[1].level < pair[0].level);
                    assert!(pair[1].distance > pair[0].distance);
                    let centre = pair[0].position;
                    assert!(world.ground(centre) < river.at(centre).level);
                    assert!(!world.walkable(centre));
                }
                let (bank, _) = world.river_bank();
                assert!(world.walkable(bank));
                let triangles = river.triangles(&world.heights);
                assert!(!triangles.is_empty());
                let mut edges = Vec::new();
                for tri in triangles {
                    let [a, b, c] = tri.map(|v| v.position);
                    assert!((b - a).cross(c - a).y > 0.0);
                    for v in [a, b, c] {
                        let p = Vec2::new(v.x, v.z);
                        assert!((v.y - river.at(p).level).abs() < 0.003);
                        assert!(world.ground(p) <= v.y + 0.003);
                        assert!(
                            !world.walkable(p),
                            "Unexpected walkable water {p:?}, field {:?}, ground {}",
                            river.at(p),
                            world.ground(p)
                        );
                    }
                    edges.extend([(a, b), (b, c), (c, a)]);
                }
                for (index, &(a, b)) in edges.iter().enumerate() {
                    if edges.iter().enumerate().any(|(other, &(c, d))| {
                        other != index && a.distance(d) < 0.001 && b.distance(c) < 0.001
                    }) {
                        continue;
                    }
                    if (a.x.abs() - EXTENT).abs() < 0.001 && (b.x.abs() - EXTENT).abs() < 0.001
                        || (a.z.abs() - EXTENT).abs() < 0.001 && (b.z.abs() - EXTENT).abs() < 0.001
                    {
                        continue;
                    }
                    for v in [a, b, (a + b) * 0.5] {
                        assert!(
                            (world.ground(Vec2::new(v.x, v.z)) - v.y).abs() < 0.003,
                            "seed {seed}, strength {mountain_strength}: floating river edge {v:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tributaries_join_once_with_consistent_water_and_flow() {
        for seed in [20261003, 42, 314159] {
            for strength in [0.0, 0.65, 1.0] {
                let world = Meadow::with_settings(
                    seed,
                    TerrainSettings {
                        mountain_strength: strength,
                    },
                );
                let again = Meadow::with_settings(
                    seed,
                    TerrainSettings {
                        mountain_strength: strength,
                    },
                );
                let river = world.river.as_ref().unwrap();
                assert_eq!(river.reaches, again.river.as_ref().unwrap().reaches);
                assert!(
                    (2..=4).contains(&river.source_count),
                    "seed {seed}, strength {strength}: {} sources",
                    river.source_count
                );
                assert!(river.total_length() > river.length() + 25.0);
                let outlet = river.points.last().unwrap();
                let mut sources = 0;
                for reach in &river.reaches {
                    let first = reach.first().unwrap();
                    let end = reach.last().unwrap();
                    let upstream = river
                        .reaches
                        .iter()
                        .filter(|r| r.last().unwrap().position == first.position)
                        .count();
                    if upstream == 0 {
                        sources += 1;
                    }
                    let downstream: Vec<_> = river
                        .reaches
                        .iter()
                        .filter(|r| r[0].position == end.position)
                        .collect();
                    if end.position == outlet.position {
                        assert!(downstream.is_empty());
                    } else {
                        assert_eq!(
                            downstream.len(),
                            1,
                            "Every tributary must continue through exactly one downstream reach"
                        );
                        let join = downstream[0][0];
                        assert_eq!(end.level, join.level);
                        assert_eq!(end.width, join.width);
                        assert!((end.flow - join.flow).abs() < 0.001);
                    }
                    for pair in reach.windows(2) {
                        assert!(pair[1].level < pair[0].level);
                        assert!(pair[1].width >= pair[0].width - 0.001);
                        assert!(pair[1].flow > pair[0].flow);
                        // All channels, including bends and confluences, contain
                        // water above the shared carved bed and block walking.
                        for t in [0.0, 0.5, 1.0] {
                            let p = pair[0].position.lerp(pair[1].position, t);
                            assert!(world.ground(p) < river.at(p).level, "Dry channel at {p:?}");
                            assert!(!world.walkable(p));
                        }
                    }
                    let mut current = end.position;
                    for _ in 0..river.reaches.len() {
                        if current == outlet.position {
                            break;
                        }
                        let next = river
                            .reaches
                            .iter()
                            .find(|r| r[0].position == current)
                            .unwrap();
                        current = next.last().unwrap().position;
                    }
                    assert_eq!(
                        current, outlet.position,
                        "River network cycle or isolated branch"
                    );
                }
                assert_eq!(sources, river.source_count);
            }
        }
    }

    #[test]
    fn water_surface_is_contained_and_every_exposed_edge_meets_the_bank() {
        for seed in [20261003, 42, 314159] {
            for mountain_strength in [0.0, 0.65, 1.0] {
                let world = Meadow::with_settings(seed, TerrainSettings { mountain_strength });
                for pond in &world.ponds {
                    let triangles = world.water_triangles(pond);
                    assert!(!triangles.is_empty());
                    let mut edges = Vec::new();
                    for triangle in triangles {
                        let [a, b, c] = triangle;
                        assert!((b - a).cross(c - a).y > 0.0);
                        for v in triangle {
                            assert_eq!(v.y, pond.level);
                            let p = Vec2::new(v.x, v.z);
                            assert!(world.ground(p) <= pond.level + 0.002);
                            assert!(pond.radius_at(p) < 1.1);
                            assert!(world.near_water(p, 0.0));
                            assert!(!world.walkable(p));
                        }
                        edges.extend([(a, b), (b, c), (c, a)]);
                    }
                    let mut boundary_count = 0;
                    for (index, &(a, b)) in edges.iter().enumerate() {
                        if edges.iter().enumerate().any(|(other, &(c, d))| {
                            other != index && a.distance(d) < 0.001 && b.distance(c) < 0.001
                        }) {
                            continue;
                        }
                        boundary_count += 1;
                        // A free mesh edge must lie on the actual ground at
                        // water level, including its midpoint, on every side.
                        for v in [a, b, (a + b) * 0.5] {
                            let ground = world.ground(Vec2::new(v.x, v.z));
                            assert!(
                                (ground - pond.level).abs() < 0.002,
                                "seed {seed}, strength {mountain_strength}, edge {a:?} {b:?}: ground {ground}, water {}",
                                pond.level
                            );
                        }
                    }
                    assert!(boundary_count > 12);
                    for step in 0..96 {
                        let p = pond.center
                            + Vec2::from_angle(step as f32 * std::f32::consts::TAU / 96.0)
                                * pond.radii
                                * 1.2;
                        assert!(world.ground(p) > pond.level);
                    }
                    assert!(world.ground(pond.center) < pond.level - 1.0);
                }
            }
        }
    }

    #[test]
    fn mountain_strength_changes_relief_and_invalid_settings_are_normalized() {
        for seed in [20261003, 42, 314159] {
            let flat = Meadow::with_settings(
                seed,
                TerrainSettings {
                    mountain_strength: 0.0,
                },
            );
            let strong = Meadow::with_settings(
                seed,
                TerrainSettings {
                    mountain_strength: 1.0,
                },
            );
            let standard = Meadow::new(seed);
            let again = Meadow::with_settings(seed, TerrainSettings::default());
            assert_eq!(standard.heights, again.heights);
            assert_eq!(standard.trees, again.trees);
            let flat_stats = flat.height_stats();
            let strong_stats = strong.height_stats();
            let stats = standard.height_stats();
            // The broad regional grade spans the expanded map.
            let flat_relief_limit = EXTENT * 2.0 * 0.018 + 8.0;
            assert!(flat_stats.maximum - flat_stats.minimum < flat_relief_limit);
            assert!(
                stats.maximum - stats.minimum > 25.0,
                "seed {seed}: {stats:?}"
            );
            assert!(strong_stats.maximum > stats.maximum + 8.0);
            assert!(strong.heights.iter().all(|h| h.is_finite()));
            assert!(
                standard
                    .trees
                    .iter()
                    .all(|(p, _)| standard.slope(*p) <= 0.65)
            );
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                Meadow::new(42).heights,
                Meadow::with_settings(
                    42,
                    TerrainSettings {
                        mountain_strength: invalid
                    }
                )
                .heights
            );
        }
        assert_eq!(
            Meadow::with_settings(
                42,
                TerrainSettings {
                    mountain_strength: -1.0
                }
            )
            .settings
            .mountain_strength,
            0.0
        );
        assert_eq!(
            Meadow::with_settings(
                42,
                TerrainSettings {
                    mountain_strength: 2.0
                }
            )
            .settings
            .mountain_strength,
            1.0
        );
    }

    #[test]
    fn seeded_regions_mix_landforms_and_keep_the_start_open() {
        for seed in [20261003, 42, 314159] {
            let world = Meadow::new(seed);
            let mut counts = [0usize; 4];
            for i in 0..SIDE * SIDE {
                let p = crate::river::position(i);
                if p.length() <= PLAY_RADIUS {
                    let kind = match world.landform(p) {
                        Landform::Plains => 0,
                        Landform::Hills => 1,
                        Landform::Mountains => 2,
                        Landform::Basin => 3,
                    };
                    counts[kind] += 1;
                }
            }
            assert!(
                counts.iter().all(|&count| count > 500),
                "seed {seed}: {counts:?}"
            );
            for step in 0..32 {
                let p = Vec2::from_angle(step as f32 * std::f32::consts::TAU / 32.0) * 12.0;
                assert!(world.walkable(p));
            }
        }
    }

    #[test]
    fn multiple_directions_remain_reachable_at_all_mountain_strengths() {
        for seed in [20261003, 42, 314159] {
            for strength in [0.0, 0.65, 1.0] {
                let world = Meadow::with_settings(
                    seed,
                    TerrainSettings {
                        mountain_strength: strength,
                    },
                );
                let mut visited = vec![false; SIDE * SIDE];
                let start = SIDE / 2 * SIDE + SIDE / 2;
                let mut queue = VecDeque::from([start]);
                visited[start] = true;
                while let Some(index) = queue.pop_front() {
                    let x = index % SIDE;
                    let z = index / SIDE;
                    for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx < 0 || nz < 0 || nx >= SIDE as i32 || nz >= SIDE as i32 {
                            continue;
                        }
                        let next = nz as usize * SIDE + nx as usize;
                        let p = Vec2::new(nx as f32 * CELL - EXTENT, nz as f32 * CELL - EXTENT);
                        if !visited[next] && world.walkable(p) {
                            visited[next] = true;
                            queue.push_back(next);
                        }
                    }
                }
                let (bank, _) = world.river_bank();
                let bank_grid = ((bank + Vec2::splat(EXTENT)) / CELL).round().as_uvec2();
                assert!(visited[bank_grid.y as usize * SIDE + bank_grid.x as usize]);
                assert!(
                    world.reachable_fraction > 0.85,
                    "seed {seed}, strength {strength}: reachable {}",
                    world.reachable_fraction
                );
                for (view, direction) in [Vec2::X, Vec2::Y, Vec2::NEG_X, Vec2::NEG_Y]
                    .into_iter()
                    .enumerate()
                {
                    let (p, _) = world.exploration_view(view);
                    assert!(
                        p.distance(direction * PLAY_RADIUS * 0.75) < 50.0,
                        "seed {seed}, strength {strength}: region {view} too distant {p:?}"
                    );
                    let grid = ((p + Vec2::splat(EXTENT)) / CELL).round().as_uvec2();
                    assert!(visited[grid.y as usize * SIDE + grid.x as usize]);
                }
            }
        }
    }

    #[test]
    fn seed_reproduces_terrain_and_forest() {
        let a = Meadow::new(42);
        let b = Meadow::new(42);
        let c = Meadow::new(43);
        assert_eq!(a.heights, b.heights);
        assert_eq!(a.trees, b.trees);
        assert_ne!(a.heights, c.heights);
    }
    #[test]
    fn spawn_is_dry_and_lakes_are_blocked() {
        for seed in [1, 42, 20261003, u32::MAX] {
            let world = Meadow::new(seed);
            assert!(world.walkable(Vec2::ZERO));
            for pond in &world.ponds {
                assert!(!world.walkable(pond.center));
                assert!(world.ground(pond.center) < pond.level);
            }
            assert!(!world.walkable(Vec2::new(PLAY_RADIUS + 1.0, 0.0)));
        }
    }
    #[test]
    fn feet_follow_the_rendered_triangle() {
        let world = Meadow::new(42);
        let x = 83;
        let z = 70;
        let p = Vec2::new(x as f32 * CELL - EXTENT, z as f32 * CELL - EXTENT);
        let a = world.heights[z * SIDE + x];
        let b = world.heights[z * SIDE + x + 1];
        let c = world.heights[(z + 1) * SIDE + x];
        let d = world.heights[(z + 1) * SIDE + x + 1];
        assert!(
            (world.ground(p + Vec2::new(0.4, 0.6)) - (a * 0.5 + b * 0.2 + c * 0.3)).abs() < 0.0001
        );
        assert!(
            (world.ground(p + Vec2::new(1.6, 1.4)) - (d * 0.5 + c * 0.2 + b * 0.3)).abs() < 0.0001
        );
    }
}
