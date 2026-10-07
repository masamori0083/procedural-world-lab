//! Coarse interiors with exact 2 m chunk borders and protected shoreline cells.
use bevy::prelude::*;

pub const MAX_ERROR: f32 = 0.12;

#[derive(Resource)]
pub struct Config {
    pub enabled: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    Near,
    Middle,
    Far,
}
impl Detail {
    pub fn index(self) -> usize {
        match self {
            Self::Near => 0,
            Self::Middle => 1,
            Self::Far => 2,
        }
    }
    pub fn choose(origin: Vec2, horse: Vec2, previous: Option<Self>) -> Self {
        let distance = (horse - horse.clamp(origin, origin + Vec2::splat(96.))).length();
        let near = if previous == Some(Self::Near) {
            112.
        } else {
            80.
        };
        let middle = if matches!(previous, Some(Self::Near | Self::Middle)) {
            208.
        } else {
            176.
        };
        if distance <= near {
            Self::Near
        } else if distance <= middle {
            Self::Middle
        } else {
            Self::Far
        }
    }
}

fn ring(x: usize, z: usize, step: usize, edges: [bool; 4], side: usize) -> Vec<u32> {
    let mut points = Vec::new();
    for (edge, full) in edges.into_iter().enumerate() {
        for i in (0..step).step_by(if full { 1 } else { step }) {
            let (a, b) = match edge {
                0 => (x, z + i),
                1 => (x + i, z + step),
                2 => (x + step, z + step - i),
                _ => (x + step - i, z),
            };
            points.push((b * side + a) as u32);
        }
    }
    points
}

fn edges(fine: &[bool], bx: usize, bz: usize, blocks: usize) -> [bool; 4] {
    [
        bx == 0 || fine[bz * blocks + bx - 1],
        bz + 1 == blocks || fine[(bz + 1) * blocks + bx],
        bx + 1 == blocks || fine[bz * blocks + bx + 1],
        bz == 0 || fine[(bz - 1) * blocks + bx],
    ]
}

fn acceptable(heights: &[f32], side: usize, x: usize, z: usize, step: usize, ring: &[u32]) -> bool {
    let point = |i: u32| Vec2::new((i as usize % side) as f32, (i as usize / side) as f32);
    let center = ((z + step / 2) * side + x + step / 2) as u32;
    let a = point(center);
    let grid_ok = (z..=z + step).all(|pz| {
        (x..=x + step).all(|px| {
            let p = Vec2::new(px as f32, pz as f32);
            (0..ring.len()).any(|i| {
                let b = point(ring[i]);
                let c = point(ring[(i + 1) % ring.len()]);
                let area = (b - a).perp_dot(c - a);
                let u = (p - a).perp_dot(c - a) / area;
                let v = (b - a).perp_dot(p - a) / area;
                if u < -0.00001 || v < -0.00001 || u + v > 1.00001 {
                    return false;
                }
                let height = heights[center as usize] * (1. - u - v)
                    + heights[ring[i] as usize] * u
                    + heights[ring[(i + 1) % ring.len()] as usize] * v;
                (height - heights[pz * side + px]).abs() <= MAX_ERROR
            })
        })
    });
    if !grid_ok {
        return false;
    }
    // Differences between the two triangulations peak at their vertices or
    // edge intersections. Check fan edges against fine-grid lines/diagonals.
    ring.iter().all(|end| {
        let b = point(*end);
        let d = b - a;
        [Vec2::X, Vec2::Y, Vec2::ONE].into_iter().all(|axis| {
            let from = a.dot(axis);
            let to = b.dot(axis);
            if (to - from).abs() < 0.0001 {
                return true;
            }
            ((from.min(to).ceil() as i32)..=(from.max(to).floor() as i32)).all(|line| {
                let t = (line as f32 - from) / (to - from);
                let p = a + d * t;
                let q = p.floor().as_uvec2().min(UVec2::splat((side - 2) as u32));
                let f = p - q.as_vec2();
                let i = q.y as usize * side + q.x as usize;
                let original = if f.x + f.y <= 1. {
                    heights[i] * (1. - f.x - f.y) + heights[i + 1] * f.x + heights[i + side] * f.y
                } else {
                    heights[i + side + 1] * (f.x + f.y - 1.)
                        + heights[i + side] * (1. - f.x)
                        + heights[i + 1] * (1. - f.y)
                };
                let coarse = heights[center as usize] * (1. - t) + heights[*end as usize] * t;
                (coarse - original).abs() <= MAX_ERROR
            })
        })
    })
}

/// Indices into the original grid. Every outer edge keeps all its fine vertices,
/// so different chunk levels meet exactly without skirts or neighbor requests.
pub fn indices(heights: &[f32], protected: &[bool], side: usize, detail: Detail) -> Vec<u32> {
    let cells = side - 1;
    let step = match detail {
        Detail::Near => 1,
        Detail::Middle => 2,
        Detail::Far => 4,
    };
    assert_eq!(heights.len(), side * side);
    assert_eq!(protected.len(), heights.len());
    assert_eq!(cells % step, 0);
    let blocks = cells / step;
    let mut fine = vec![step == 1; blocks * blocks];
    if step > 1 {
        for bz in 0..blocks {
            for bx in 0..blocks {
                let x = bx * step;
                let z = bz * step;
                fine[bz * blocks + bx] =
                    (z..=z + step).any(|pz| (x..=x + step).any(|px| protected[pz * side + px]));
            }
        }
        // Splitting a neighbor changes the shared edge. Recheck until every
        // remaining coarse patch obeys the error bound with its final edge fan.
        loop {
            let mut split = Vec::new();
            for bz in 0..blocks {
                for bx in 0..blocks {
                    if !fine[bz * blocks + bx]
                        && !acceptable(
                            heights,
                            side,
                            bx * step,
                            bz * step,
                            step,
                            &ring(
                                bx * step,
                                bz * step,
                                step,
                                edges(&fine, bx, bz, blocks),
                                side,
                            ),
                        )
                    {
                        split.push(bz * blocks + bx);
                    }
                }
            }
            if split.is_empty() {
                break;
            }
            for i in split {
                fine[i] = true;
            }
        }
    }
    let mut out = Vec::new();
    for bz in 0..blocks {
        for bx in 0..blocks {
            let x = bx * step;
            let z = bz * step;
            if fine[bz * blocks + bx] {
                for pz in z..z + step {
                    for px in x..x + step {
                        let a = (pz * side + px) as u32;
                        let c = a + side as u32;
                        out.extend([a, c, a + 1, a + 1, c, c + 1]);
                    }
                }
            } else {
                let boundary = ring(x, z, step, edges(&fine, bx, bz, blocks), side);
                let center = ((z + step / 2) * side + x + step / 2) as u32;
                for i in 0..boundary.len() {
                    out.extend([center, boundary[i], boundary[(i + 1) % boundary.len()]]);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    #[test]
    fn levels_reduce_geometry_with_closed_interiors_and_identical_fine_borders() {
        let side = 49;
        let heights: Vec<_> = (0..side * side)
            .map(|i| {
                let x = (i % side) as f32;
                let z = (i / side) as f32;
                x * 0.1 + z * 0.2 + (x * 0.1).sin() * 0.03
            })
            .collect();
        let protected: Vec<_> = (0..side * side)
            .map(|i| (21..=27).contains(&(i % side)))
            .collect();
        let p = |i: u32| Vec2::new((i as usize % side) as f32, (i as usize / side) as f32);
        let mut last = usize::MAX;
        for detail in [Detail::Near, Detail::Middle, Detail::Far] {
            let indices = indices(&heights, &protected, side, detail);
            assert!(indices.len() < last);
            last = indices.len();
            let mut edges = BTreeMap::new();
            let mut area = 0.;
            for t in indices.chunks_exact(3) {
                let cross = (p(t[1]) - p(t[0])).perp_dot(p(t[2]) - p(t[0]));
                assert!(cross < 0., "Upward, nondegenerate faces required");
                area -= cross * 0.5;
                for i in 0..3 {
                    let a = t[i].min(t[(i + 1) % 3]);
                    let b = t[i].max(t[(i + 1) % 3]);
                    *edges.entry((a, b)).or_insert(0) += 1;
                }
            }
            assert_eq!(area, 48. * 48.);
            let mut outer = 0;
            for ((a, b), count) in edges {
                if count == 1 {
                    let a = p(a);
                    let b = p(b);
                    assert!(
                        (a.x == b.x && (a.x == 0. || a.x == 48.))
                            || (a.y == b.y && (a.y == 0. || a.y == 48.)),
                        "Unmatched interior edge / T junction"
                    );
                    assert_eq!(
                        a.distance(b),
                        1.,
                        "Every level needs the same 2m border segments"
                    );
                    outer += 1;
                } else {
                    assert_eq!(count, 2);
                }
            }
            assert_eq!(outer, 48 * 4);
            // Dense sampling compares the coarse triangles to the original
            // piecewise-linear surface, including protected strip and fan edges.
            for z in 0..=96 {
                for x in 0..=96 {
                    let q = Vec2::new(x as f32 * 0.5, z as f32 * 0.5);
                    let cell = q.floor().as_uvec2().min(UVec2::splat(47));
                    let f = q - cell.as_vec2();
                    let i = cell.y as usize * side + cell.x as usize;
                    let exact = if f.x + f.y <= 1. {
                        heights[i] * (1. - f.x - f.y)
                            + heights[i + 1] * f.x
                            + heights[i + side] * f.y
                    } else {
                        heights[i + side + 1] * (f.x + f.y - 1.)
                            + heights[i + side] * (1. - f.x)
                            + heights[i + 1] * (1. - f.y)
                    };
                    let rendered = indices
                        .chunks_exact(3)
                        .find_map(|t| {
                            let a = p(t[0]);
                            let b = p(t[1]);
                            let c = p(t[2]);
                            let area = (b - a).perp_dot(c - a);
                            let u = (q - a).perp_dot(c - a) / area;
                            let v = (b - a).perp_dot(q - a) / area;
                            (u >= -0.00001 && v >= -0.00001 && u + v <= 1.00001).then(|| {
                                heights[t[0] as usize] * (1. - u - v)
                                    + heights[t[1] as usize] * u
                                    + heights[t[2] as usize] * v
                            })
                        })
                        .expect("LOD surface hole");
                    let tolerance = if (22.0..=26.0).contains(&q.x) {
                        0.00002
                    } else {
                        MAX_ERROR + 0.00002
                    };
                    assert!((rendered - exact).abs() <= tolerance);
                }
            }
        }
    }
    #[test]
    fn sharp_relief_is_preserved_and_distance_hysteresis_keeps_the_horse_chunk_detailed() {
        let side = 49;
        let heights: Vec<_> = (0..side * side)
            .map(|i| {
                if (i % side + i / side) % 2 == 0 {
                    0.
                } else {
                    2.
                }
            })
            .collect();
        let faces = |detail| {
            indices(&heights, &vec![false; side * side], side, detail)
                .chunks_exact(3)
                .map(|t| (t[0], t[1], t[2]))
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(faces(Detail::Far), faces(Detail::Near));
        assert_eq!(
            Detail::choose(Vec2::ZERO, Vec2::splat(48.), Some(Detail::Far)),
            Detail::Near
        );
        assert_eq!(
            Detail::choose(Vec2::ZERO, Vec2::new(-100., 48.), Some(Detail::Near)),
            Detail::Near
        );
        assert_eq!(
            Detail::choose(Vec2::ZERO, Vec2::new(-100., 48.), Some(Detail::Middle)),
            Detail::Middle
        );
        assert_eq!(
            Detail::choose(Vec2::ZERO, Vec2::new(-200., 48.), Some(Detail::Middle)),
            Detail::Middle
        );
        assert_eq!(
            Detail::choose(Vec2::ZERO, Vec2::new(-200., 48.), Some(Detail::Far)),
            Detail::Far
        );
    }
}
