//! Seeded grass meadows, continuous across render chunks. No external assets.
use crate::{
    environment::EnvironmentSample,
    streaming::{CHUNK_SIZE, ChunkKey, Generator, coordinate_seed},
    terrain::{CELL, SeedRandom},
    vegetation::Blade,
};
use bevy::prelude::*;

const FIELD_CELL: f32 = 128.;
const TUFT_CELL: f32 = 0.75;
pub const MAX_TUFTS: usize = 128 * 128 * 2;

struct Patch {
    center: Vec2,
    radii: Vec2,
    rotation: Mat2,
    phase: f32,
}
pub struct Fields {
    patches: Vec<Patch>,
}
fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}
impl Fields {
    pub fn new(seed: u32, origin: Vec2) -> Self {
        // Include every ellipse which can reach this chunk, including its edges.
        let lo = ((origin - Vec2::splat(48.)) / FIELD_CELL)
            .floor()
            .as_ivec2();
        let hi = ((origin + Vec2::splat(CHUNK_SIZE + 48.)) / FIELD_CELL)
            .floor()
            .as_ivec2();
        let mut patches = Vec::new();
        for z in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let mut r = SeedRandom(coordinate_seed(seed ^ 0x733a_52d1, x, z));
                if r.unit() > 0.72 {
                    continue;
                }
                patches.push(Patch {
                    center: (Vec2::new(x as f32, z as f32)
                        + Vec2::new(r.range(0.15, 0.85), r.range(0.15, 0.85)))
                        * FIELD_CELL,
                    radii: Vec2::new(r.range(22., 38.), r.range(18., 30.)),
                    rotation: Mat2::from_angle(r.range(0., std::f32::consts::TAU)),
                    phase: r.range(0., std::f32::consts::TAU),
                });
            }
        }
        Self { patches }
    }
    pub fn strength(&self, p: Vec2) -> f32 {
        self.patches.iter().fold(0_f32, |cover, patch| {
            let q = patch.rotation * (p - patch.center) / patch.radii;
            let edge =
                q.length() + 0.08 * (q.x * 7. + q.y * 4. + patch.phase).sin() * (q.y * 5.).sin();
            cover.max(smooth((1. - edge) * 4.))
        })
    }
}

fn suitability(sample: EnvironmentSample) -> f32 {
    smooth((sample.grass_density - 0.12) / 0.22)
        * smooth((0.65 - sample.slope) / 0.35)
        * (1. - sample.tree_density * 0.4)
}

// Leave a subtle meadow footprint when distant blades are omitted by LOD.
// The same scalar is used by initial ground meshes and all ground rebuilds.
pub fn ground_tint(g: &Generator, fields: &Fields, p: Vec2, sample: EnvironmentSample) -> f32 {
    let cover = if g.hydrology.ford_route(p) {
        0.
    } else {
        fields.strength(p) * suitability(sample)
    };
    (g.noise(p * 0.08) - cover * 0.7).clamp(0., 1.)
}

// Sample the existing 2m grid, rather than reevaluating habitat for every tuft.
// Ground uses the exact same triangle as horse collision, even on slopes.
fn sample_grid(
    q: Vec2,
    positions: &[[f32; 3]],
    samples: &[EnvironmentSample],
) -> (f32, EnvironmentSample) {
    let side = (CHUNK_SIZE / CELL) as usize + 1;
    let cell = (q / CELL).floor();
    let f = q / CELL - cell;
    let i = cell.y as usize * side + cell.x as usize;
    let (indices, weights) = if f.x + f.y <= 1. {
        ([i, i + 1, i + side], [1. - f.x - f.y, f.x, f.y])
    } else {
        (
            [i + side + 1, i + side, i + 1],
            [f.x + f.y - 1., 1. - f.x, 1. - f.y],
        )
    };
    let blend = |get: fn(EnvironmentSample) -> f32| {
        indices
            .iter()
            .zip(weights)
            .map(|(i, w)| get(samples[*i]) * w)
            .sum()
    };
    let height = indices
        .iter()
        .zip(weights)
        .map(|(i, w)| positions[*i][1] * w)
        .sum();
    (
        height,
        EnvironmentSample {
            elevation: height,
            slope: indices
                .iter()
                .map(|i| samples[*i].slope)
                .fold(0_f32, f32::max),
            water_distance: indices
                .iter()
                .map(|i| samples[*i].water_distance)
                .fold(f32::INFINITY, f32::min),
            moisture: blend(|s| s.moisture),
            tree_density: blend(|s| s.tree_density),
            grass_density: blend(|s| s.grass_density),
            rockiness: blend(|s| s.rockiness),
        },
    )
}

pub fn generate(
    g: &Generator,
    key: ChunkKey,
    fields: &Fields,
    positions: &[[f32; 3]],
    samples: &[EnvironmentSample],
) -> Vec<Blade> {
    let origin = key.origin();
    let grid_origin = (origin / TUFT_CELL).as_ivec2();
    let mut tufts = Vec::new();
    for z in 0..128 {
        for x in 0..128 {
            let mut random = SeedRandom(coordinate_seed(
                g.seed ^ 0x581c_aa42,
                grid_origin.x + x,
                grid_origin.y + z,
            ));
            for slot in 0..2 {
                let q = (Vec2::new(x as f32, z as f32)
                    + Vec2::new(random.range(0.08, 0.92), random.range(0.08, 0.92)))
                    * TUFT_CELL;
                let p = origin + q;
                let mask = fields.strength(p);
                let chance = random.unit();
                if (slot == 1 && mask < 0.65) || (mask == 0. && chance > 0.018) {
                    continue;
                }
                let (ground, sample) = sample_grid(q, positions, samples);
                let cover = if g.hydrology.ford_route(p) {
                    0.
                } else {
                    mask * suitability(sample)
                };
                let probability = if slot == 0 {
                    0.018 * sample.grass_density
                } else {
                    0.
                } + cover * 0.9;
                if chance > probability
                    || sample.slope > 0.65
                    || sample.grass_density <= 0.
                    || sample.water_distance < 2.
                    || g.water_depth(p) >= -0.015
                {
                    continue;
                }
                // Keep the marked crossing and its dry approaches readable.
                if slot == 1 && cover == 0. {
                    continue;
                }
                let height = random.range(0.35, 0.55) + cover * random.range(0.50, 0.75);
                let direction = Vec2::from_angle(random.range(0., std::f32::consts::TAU)) * 0.09;
                let tint = Vec3::new(0.29, 0.36, 0.10)
                    .lerp(Vec3::new(0.15, 0.39, 0.08), sample.moisture)
                    * random.range(0.94, 1.06);
                tufts.push(Blade {
                    a: Vec3::new(q.x - direction.x, ground, q.y - direction.y),
                    b: Vec3::new(q.x + direction.x, ground, q.y + direction.y),
                    c: Vec3::new(q.x, ground + height, q.y),
                    color: [tint.x, tint.y, tint.z, 1.],
                });
            }
        }
    }
    debug_assert!(tufts.len() <= MAX_TUFTS);
    tufts
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn field_edges_are_seeded_continuous_and_leave_open_land() {
        for seed in [42, 20261003, 314159] {
            let mut tall = 0;
            let mut open = 0;
            for key in [ChunkKey(0, 0), ChunkKey(-1, -1), ChunkKey(1, 0)] {
                let fields = Fields::new(seed, key.origin());
                let right = Fields::new(seed, key.origin() + Vec2::X * CHUNK_SIZE);
                for z in 0..=96 {
                    let edge = key.origin() + Vec2::new(CHUNK_SIZE, z as f32);
                    assert_eq!(fields.strength(edge), right.strength(edge));
                    assert!(
                        (fields.strength(edge + Vec2::X * 0.001) - fields.strength(edge)).abs()
                            < 0.001
                    );
                }
                for z in 0..24 {
                    for x in 0..24 {
                        let p = key.origin() + Vec2::new(x as f32, z as f32) * 4.;
                        let s = fields.strength(p);
                        assert_eq!(s, Fields::new(seed, key.origin()).strength(p));
                        assert!((0. ..=1.).contains(&s));
                        tall += usize::from(s > 0.7);
                        open += usize::from(s == 0.);
                    }
                }
            }
            assert!(
                tall > 5 && open > tall * 2,
                "seed {seed}, tall {tall}, open {open}"
            );
        }
    }
}
