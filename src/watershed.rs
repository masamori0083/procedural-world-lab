//! Shared inland drainage plans, independent of render chunk order.
//! Each 768 m catchment terminates in a closed lake; regional connections are future work.
use crate::{river::RiverField, streaming::Generator};
use bevy::prelude::*;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    sync::{Arc, RwLock},
};

pub const SIZE: f32 = 768.;
const STEP: f32 = 24.;
const MARGIN: f32 = 48.;
const SIDE: usize = 29;
const BIN: f32 = 24.;
const BIN_SIDE: usize = (SIZE / BIN) as usize;
const INFLUENCE: f32 = 28.;
pub type Key = (i32, i32);
pub fn key(p: Vec2) -> Key {
    let q = ((p + Vec2::splat(SIZE * 0.5)) / SIZE).floor().as_ivec2();
    (q.x, q.y)
}
fn origin(k: Key) -> Vec2 {
    Vec2::new(k.0 as f32, k.1 as f32) * SIZE - Vec2::splat(SIZE * 0.5)
}

#[derive(Default)]
pub struct Cache(RwLock<BTreeMap<Key, Arc<Plan>>>);
impl Cache {
    pub fn prepare(&self, g: &Generator, k: Key) {
        if self.0.read().unwrap().contains_key(&k) {
            return;
        }
        // Expensive planning is outside the lock, in the chunk worker.
        let plan = Arc::new(Plan::generate(g, k));
        self.0.write().unwrap().entry(k).or_insert(plan);
    }
    pub fn get(&self, k: Key) -> Option<Arc<Plan>> {
        self.0.read().unwrap().get(&k).cloned()
    }
    pub fn retain(&self, keep: &BTreeSet<Key>) {
        self.0.write().unwrap().retain(|k, _| keep.contains(k));
    }
    pub fn source_count(&self) -> usize {
        self.0.read().unwrap().values().map(|p| p.sources).sum()
    }
    pub fn len(&self) -> usize {
        self.0.read().unwrap().len()
    }
    pub fn field(&self, p: Vec2) -> RiverField {
        let k = key(p);
        let local = p - origin(k);
        // No channel influence reaches a catchment boundary. Edge normals do not
        // depend on whether the adjacent catchment has already been planned.
        if local.min_element() < 8. || local.max_element() > SIZE - 8. {
            return dry();
        }
        self.0
            .read()
            .unwrap()
            .get(&k)
            .map_or_else(dry, |plan| plan.field(p))
    }
}
fn dry() -> RiverField {
    RiverField {
        distance: 1_000_000.,
        ..default()
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub p: Vec2,
    pub level: f32,
    pub flow: f32,
    pub width: f32,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub a: Node,
    pub b: Node,
}
pub struct Plan {
    pub segments: Vec<Segment>,
    pub lake: Vec2,
    pub lake_level: f32,
    pub sources: usize,
    origin: Vec2,
    bins: Vec<Vec<usize>>,
}
#[derive(Clone, Copy)]
struct Cell {
    h: f32,
    i: usize,
}
impl PartialEq for Cell {
    fn eq(&self, b: &Self) -> bool {
        self.h == b.h && self.i == b.i
    }
}
impl Eq for Cell {}
impl Ord for Cell {
    fn cmp(&self, b: &Self) -> Ordering {
        b.h.total_cmp(&self.h).then_with(|| b.i.cmp(&self.i))
    }
}
impl PartialOrd for Cell {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Plan {
    fn generate(g: &Generator, k: Key) -> Self {
        let origin = origin(k);
        let pos = |i: usize| {
            origin + Vec2::splat(MARGIN) + Vec2::new((i % SIDE) as f32, (i / SIDE) as f32) * STEP
        };
        let land: Vec<_> = (0..SIDE * SIDE).map(|i| g.base_height(pos(i))).collect();
        let allowed: Vec<_> = (0..land.len()).map(|i| pos(i).length() > 70.).collect();
        let outlet = (0..land.len())
            .filter(|&i| {
                let x = i % SIDE;
                let z = i / SIDE;
                allowed[i] && (7..SIDE - 7).contains(&x) && (7..SIDE - 7).contains(&z)
            })
            .min_by(|&a, &b| land[a].total_cmp(&land[b]).then(a.cmp(&b)))
            .unwrap();
        let mut parents = vec![usize::MAX; land.len()];
        let mut filled = vec![f32::INFINITY; land.len()];
        let mut order = Vec::new();
        let mut heap = BinaryHeap::new();
        parents[outlet] = outlet;
        filled[outlet] = land[outlet];
        heap.push(Cell {
            h: land[outlet],
            i: outlet,
        });
        while let Some(c) = heap.pop() {
            order.push(c.i);
            for (dx, dz) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (-1, 1),
                (1, -1),
                (-1, -1),
            ] {
                let x = (c.i % SIDE) as i32 + dx;
                let z = (c.i / SIDE) as i32 + dz;
                if x < 0 || z < 0 || x >= SIDE as i32 || z >= SIDE as i32 {
                    continue;
                }
                let i = z as usize * SIDE + x as usize;
                if !allowed[i] || parents[i] != usize::MAX {
                    continue;
                }
                parents[i] = c.i;
                filled[i] = land[i].max(c.h + 0.002 * pos(i).distance(pos(c.i)));
                heap.push(Cell { h: filled[i], i });
            }
        }
        let mut area = vec![1u32; land.len()];
        let mut distance = vec![0.; land.len()];
        for &i in &order {
            if i != outlet {
                distance[i] = distance[parents[i]] + pos(i).distance(pos(parents[i]));
            }
        }
        for &i in order.iter().rev() {
            if i != outlet {
                area[parents[i]] += area[i];
            }
        }
        let mut used = vec![false; land.len()];
        used[outlet] = true;
        let mut sources = Vec::new();
        let mut elevations: Vec<_> = order.iter().map(|&i| land[i]).collect();
        elevations.sort_by(f32::total_cmp);
        let upland = elevations[elevations.len() * 3 / 4];
        let mountain: Vec<_> = (0..land.len())
            .map(|i| g.landform(pos(i)) == crate::terrain::Landform::Mountains)
            .collect();
        for _ in 0..5 {
            let mut best: Option<(bool, f32, Vec<usize>)> = None;
            for &i in &order {
                if used[i]
                    || land[i] < upland
                    || land[i] - land[outlet] < 2.
                    || sources.iter().any(|&s| pos(s).distance(pos(i)) < 90.)
                {
                    continue;
                }
                let mut path = vec![i];
                let mut j = i;
                while !used[j] {
                    j = parents[j];
                    path.push(j);
                }
                let length = distance[i] - distance[j];
                if length < if sources.is_empty() { 220. } else { 100. } {
                    continue;
                }
                // A mountain source outranks every flat/rolling candidate.
                // Elevation comes before contributing area so plains cannot win
                // just because they collect more cells over a longer route.
                let score = (land[i] - land[outlet]).powi(2)
                    * length.sqrt()
                    * (1. + (area[i] as f32).ln() * 0.03);
                if best
                    .as_ref()
                    .is_none_or(|(m, s, _)| (mountain[i], score) > (*m, *s))
                {
                    best = Some((mountain[i], score, path));
                }
            }
            let Some((_, _, path)) = best else {
                break;
            };
            sources.push(path[0]);
            for i in path {
                used[i] = true;
            }
        }
        // Limit the surface to below nearby ORIGINAL ground, then propagate low
        // levels downstream. This incises depressions instead of floating water
        // above them. Junctions share one node, level, width and flow phase.
        let mut levels = vec![f32::INFINITY; land.len()];
        for &i in &order {
            if !used[i] {
                continue;
            }
            let radius = if i == outlet { 24. } else { 9. };
            let mut level = land[i];
            for n in 0..8 {
                level = level.min(g.base_height(
                    pos(i) + Vec2::from_angle(n as f32 * std::f32::consts::TAU / 8.) * radius,
                ));
            }
            levels[i] = level - 0.6;
        }
        for &i in order.iter().rev() {
            if used[i] && i != outlet {
                let parent = parents[i];
                levels[parent] =
                    levels[parent].min(levels[i] - 0.002 * pos(i).distance(pos(parent)));
            }
        }
        let lake = pos(outlet);
        let lake_level = levels[outlet];
        // Flatten the mouth and ease back into the original channel over six
        // coarse cells. A hard distance cutoff merely moves the water wall
        // upstream; the smooth ramp keeps that transition gradual too.
        for &i in &order {
            if used[i] {
                let t = ((distance[i] - 36.) / 144.).clamp(0., 1.);
                let blend = t * t * (3. - 2. * t);
                let mouth = lake_level + distance[i] * 0.002;
                levels[i] = mouth + (levels[i] - mouth) * blend;
            }
        }
        let node = |i: usize| Node {
            p: pos(i),
            level: levels[i],
            flow: -distance[i],
            width: 2.6 + 2.0 * (area[i] as f32 / area[outlet] as f32).sqrt(),
        };
        let segments: Vec<_> = order
            .iter()
            .filter(|&&i| used[i] && i != outlet)
            .map(|&i| Segment {
                a: node(i),
                b: node(parents[i]),
            })
            .collect();
        let mut plan = Self {
            segments,
            lake,
            lake_level,
            sources: sources.len(),
            origin,
            bins: vec![vec![]; BIN_SIDE * BIN_SIDE],
        };
        for (i, s) in plan.segments.iter().enumerate() {
            let lo = ((s.a.p.min(s.b.p) - origin - Vec2::splat(68.)) / BIN)
                .floor()
                .as_ivec2()
                .max(IVec2::ZERO);
            let hi = ((s.a.p.max(s.b.p) - origin + Vec2::splat(68.)) / BIN)
                .floor()
                .as_ivec2()
                .min(IVec2::splat(BIN_SIDE as i32 - 1));
            for z in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    plan.bins[z as usize * BIN_SIDE + x as usize].push(i);
                }
            }
        }
        plan
    }
    pub fn field(&self, p: Vec2) -> RiverField {
        let bin = ((p - self.origin) / BIN)
            .floor()
            .as_ivec2()
            .clamp(IVec2::ZERO, IVec2::splat(BIN_SIDE as i32 - 1));
        let mut f = dry();
        let mut best = f32::INFINITY;
        for &i in &self.bins[bin.y as usize * BIN_SIDE + bin.x as usize] {
            let s = self.segments[i];
            let d = s.b.p - s.a.p;
            let t = ((p - s.a.p).dot(d) / d.length_squared()).clamp(0., 1.);
            let offset = p - s.a.p.lerp(s.b.p, t);
            let width = s.a.width + (s.b.width - s.a.width) * t;
            let bank = offset.length() - width;
            if bank < best {
                best = bank;
                f = RiverField {
                    distance: offset.length(),
                    width,
                    level: s.a.level + (s.b.level - s.a.level) * t,
                    along: s.a.flow + (s.b.flow - s.a.flow) * t,
                    lateral: d.normalize().perp_dot(offset),
                };
            }
        }
        let lake_distance = p.distance(self.lake);
        if lake_distance - 17. < best {
            f = RiverField {
                distance: lake_distance,
                width: 17.,
                level: self.lake_level,
                along: 0.,
                lateral: (p - self.lake).x,
            };
        }
        f
    }
}
pub fn carve(land: f32, f: RiverField) -> f32 {
    let bank = f.distance - f.width;
    if bank >= INFLUENCE {
        return land;
    }
    let t = ((bank - 1.) / (INFLUENCE - 1.)).clamp(0., 1.);
    let blend = t * t * (3. - 2. * t);
    let t = ((f.distance - f.width * 0.6) / (f.width * 0.4 + 2.)).clamp(0., 1.);
    let bed = f.level - 0.95 + 1.65 * t * t * (3. - 2. * t);
    land.min(bed + (land - bed) * blend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainSettings;
    #[test]
    fn headwaters_start_in_uplands_and_prefer_available_mountains() {
        for seed in [42, 20261003, 314159] {
            for strength in [0., 0.65, 1.] {
                let g = Generator::new(
                    seed,
                    TerrainSettings {
                        mountain_strength: strength,
                    },
                );
                for k in [(0, 0), (2, -2), (-2, -2)] {
                    let plan = Plan::generate(&g, k);
                    let heads: Vec<_> = plan
                        .segments
                        .iter()
                        .filter(|s| !plan.segments.iter().any(|r| r.b.p == s.a.p))
                        .collect();
                    assert_eq!(heads.len(), plan.sources);
                    let mut elevations: Vec<_> = (0..SIDE * SIDE)
                        .map(|i| {
                            let p = origin(k)
                                + Vec2::splat(MARGIN)
                                + Vec2::new((i % SIDE) as f32, (i / SIDE) as f32) * STEP;
                            g.base_height(p)
                        })
                        .collect();
                    elevations.sort_by(f32::total_cmp);
                    let threshold = elevations[elevations.len() * 3 / 4];
                    for head in &heads {
                        assert!(g.base_height(head.a.p) >= threshold - 0.001);
                        assert!(g.base_height(head.a.p) - g.base_height(plan.lake) >= 2.);
                    }
                    if k == (0, 0) && strength >= 0.65 {
                        assert!(
                            heads
                                .iter()
                                .any(|s| g.landform(s.a.p) == crate::terrain::Landform::Mountains),
                            "Expected mountain headwater: seed {seed}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn regional_drainage_is_repeatable_downhill_and_has_shared_junctions() {
        for seed in [42, 20261003, 314159] {
            for strength in [0., 0.65, 1.] {
                let g = Generator::new(
                    seed,
                    TerrainSettings {
                        mountain_strength: strength,
                    },
                );
                for k in [(0, 0), (2, -2), (-2, -2)] {
                    let a = Plan::generate(&g, k);
                    let b = Plan::generate(&g, k);
                    assert_eq!(a.segments, b.segments);
                    assert!(a.sources >= 1);
                    let mut joins = 0;
                    for s in &a.segments {
                        assert!(s.a.level > s.b.level && s.a.flow < s.b.flow);
                        assert!(s.a.level < g.base_height(s.a.p));
                        let downstream: Vec<_> =
                            a.segments.iter().filter(|r| r.a.p == s.b.p).collect();
                        if s.b.p == a.lake {
                            assert!(downstream.is_empty());
                            assert_eq!(s.b.level, a.lake_level);
                            assert!(
                                s.a.level - s.b.level <= 0.08,
                                "River inlet must meet the lake gently"
                            );
                        } else {
                            assert_eq!(downstream.len(), 1);
                            assert_eq!(s.b, downstream[0].a);
                        }
                        if a.segments.iter().filter(|r| r.b.p == s.b.p).count() > 1 {
                            joins += 1;
                        }
                        for p in [s.a.p, s.b.p] {
                            assert!(p.length() > 70.);
                        }
                    }
                    if a.sources > 1 {
                        assert!(joins > 0);
                    }
                }
            }
        }
    }
}
