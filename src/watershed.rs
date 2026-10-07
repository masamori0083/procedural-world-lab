//! Shared inland drainage plans, independent of render chunk order.
//! Four 768 m regions share one drainage graph and terminal lake.
//! The 1536 m planning footprint is bounded; separate lake systems stay independent.
use crate::{river::RiverField, streaming::Generator};
use bevy::prelude::*;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    sync::{Arc, RwLock},
};

pub const SIZE: f32 = 768.;
const SPAN: f32 = SIZE * 2.;
pub const LAKE_RADIUS: f32 = 32.;
const STEP: f32 = 24.;
const MARGIN: f32 = 48.;
const SIDE: usize = ((SPAN - MARGIN * 2.) / STEP) as usize + 1;
const BIN: f32 = 24.;
const BIN_SIDE: usize = (SPAN / BIN) as usize;
const INFLUENCE: f32 = 28.;
pub type Key = (i32, i32);
pub fn key(p: Vec2) -> Key {
    let q = ((p + Vec2::splat(SIZE * 0.5)) / SIZE).floor().as_ivec2();
    (q.x, q.y)
}
fn origin(k: Key) -> Vec2 {
    Vec2::new(k.0 as f32, k.1 as f32) * SIZE - Vec2::splat(SIZE * 0.5)
}
/// Canonical owner of a 2x2 group. Euclidean division also groups negative keys.
pub fn system(k: Key) -> Key {
    (k.0.div_euclid(2) * 2, k.1.div_euclid(2) * 2)
}

#[derive(Default)]
pub struct Cache(RwLock<BTreeMap<Key, Arc<Plan>>>);
impl Cache {
    pub fn prepare(&self, g: &Generator, k: Key) {
        let k = system(k);
        if self.0.read().unwrap().contains_key(&k) {
            return;
        }
        // Expensive planning is outside the lock, in the chunk worker.
        let plan = Arc::new(Plan::generate(g, k));
        self.0.write().unwrap().entry(k).or_insert(plan);
    }
    pub fn get(&self, k: Key) -> Option<Arc<Plan>> {
        self.0.read().unwrap().get(&system(k)).cloned()
    }
    pub fn retain(&self, keep: &BTreeSet<Key>) {
        let keep: BTreeSet<_> = keep.iter().copied().map(system).collect();
        self.0.write().unwrap().retain(|k, _| keep.contains(k));
    }
    pub fn source_count(&self) -> usize {
        self.0.read().unwrap().values().map(|p| p.sources).sum()
    }
    pub fn len(&self) -> usize {
        self.0.read().unwrap().len()
    }
    pub fn ford_strength(&self, p: Vec2) -> f32 {
        self.get(key(p)).map_or(0., |plan| plan.ford_strength(p))
    }
    pub fn ford_route(&self, p: Vec2) -> bool {
        self.get(key(p)).is_some_and(|plan| {
            plan.fords.iter().any(|f| {
                let offset = p - f.center;
                offset.perp_dot(f.across).abs() < 5.5 && offset.dot(f.across).abs() < f.width + 30.
            })
        })
    }
    pub fn ford_height(&self, p: Vec2, ground: f32, field: RiverField) -> f32 {
        self.get(key(p))
            .map_or(ground, |plan| plan.ford_height(p, ground, field))
    }
    pub fn field(&self, p: Vec2) -> RiverField {
        let k = system(key(p));
        let local = p - origin(k);
        // Only the OUTER system perimeter is dry. Rivers cross the internal
        // 768 m boundaries using the exact same graph, independent of loading.
        if local.min_element() < 8. || local.max_element() > SPAN - 8. {
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
pub const MAX_WADING_DEPTH: f32 = 0.30;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ford {
    pub center: Vec2,
    pub across: Vec2,
    pub width: f32,
    pub flow: f32,
}
impl Ford {
    pub fn strength(self, p: Vec2) -> f32 {
        let offset = p - self.center;
        (1. - smooth((offset.perp_dot(self.across).abs() - 5.) / 9.))
            * (1. - smooth((offset.dot(self.across).abs() - self.width - 6.) / 20.))
    }
    pub fn entry(self) -> Vec2 {
        self.center - self.across * (self.width + 15.)
    }
}
pub struct Plan {
    pub segments: Vec<Segment>,
    pub lake: Vec2,
    pub lake_level: f32,
    pub sources: usize,
    pub fords: Vec<Ford>,
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
        let k = system(k);
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
            let mut best: Option<(bool, bool, f32, Vec<usize>)> = None;
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
                // Within each source region prefer mountains, then elevation
                // over contributing area; a long flat route cannot win on area.
                let score = (land[i] - land[outlet]).powi(2)
                    * length.sqrt()
                    * (1. + (area[i] as f32).ln() * 0.03);
                // Prefer a tributary from an unrepresented region before adding
                // another nearby one. The first source still prefers mountains.
                let new_region = !sources.iter().any(|&s| key(pos(s)) == key(pos(i)));
                if best
                    .as_ref()
                    .is_none_or(|(r, m, s, _)| (new_region, mountain[i], score) > (*r, *m, *s))
                {
                    best = Some((new_region, mountain[i], score, path));
                }
            }
            let Some((_, _, _, path)) = best else {
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
            let radius = if i == outlet { LAKE_RADIUS + 7. } else { 9. };
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
        let mut upstream = vec![0f32; land.len()];
        let mut tributaries = vec![0u32; land.len()];
        for &source in &sources {
            tributaries[source] = 1;
        }
        for &i in order.iter().rev() {
            if used[i] && i != outlet {
                let parent = parents[i];
                upstream[parent] = upstream[parent].max(upstream[i] + pos(i).distance(pos(parent)));
                tributaries[parent] += tributaries[i];
            }
        }
        let node = |i: usize| Node {
            p: pos(i),
            level: levels[i],
            flow: -distance[i],
            width: 1.5
                + 2.5 * (1. - (-upstream[i] / 650.).exp())
                + 0.9 * ((tributaries[i] as f32).sqrt() - 1.)
                + 4. * smooth((120. - distance[i]) / 120.),
        };
        let segments: Vec<_> = order
            .iter()
            .filter(|&&i| used[i] && i != outlet)
            .map(|&i| Segment {
                a: node(i),
                b: node(parents[i]),
            })
            .collect();
        let segments = refine(g, &segments, lake);
        let lake_level = segments.iter().find(|s| s.b.p == lake).unwrap().b.level;
        let mut plan = Self {
            segments,
            lake,
            lake_level,
            sources: sources.len(),
            fords: Vec::new(),
            origin,
            bins: vec![vec![]; BIN_SIDE * BIN_SIDE],
        };
        for (i, s) in plan.segments.iter().enumerate() {
            let padding = Vec2::splat(INFLUENCE + s.a.width.max(s.b.width) + 4.);
            let lo = ((s.a.p.min(s.b.p) - origin - padding) / BIN)
                .floor()
                .as_ivec2()
                .max(IVec2::ZERO);
            let hi = ((s.a.p.max(s.b.p) - origin + padding) / BIN)
                .floor()
                .as_ivec2()
                .min(IVec2::splat(BIN_SIDE as i32 - 1));
            for z in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    plan.bins[z as usize * BIN_SIDE + x as usize].push(i);
                }
            }
        }
        plan.place_fords(g, k);
        plan
    }
    pub fn ford_strength(&self, p: Vec2) -> f32 {
        self.fords
            .iter()
            .map(|ford| ford.strength(p))
            .fold(0., f32::max)
    }
    pub fn ford_height(&self, p: Vec2, ground: f32, field: RiverField) -> f32 {
        if field.along == 0. {
            return ground;
        }
        let Some(ford) = self
            .fords
            .iter()
            .find(|f| f.strength(p) > 0. && (field.along - f.flow).abs() < 45.)
        else {
            return ground;
        };
        let t = (field.distance - field.width * 0.6) / (field.width * 0.4 + 2.);
        let bed = field.level - 0.20 + 0.90 * smooth(t);
        let ramp = field.level + 0.70 + (field.distance - field.width - 2.).max(0.) * 0.035;
        let target = if field.distance <= field.width + 2. {
            bed
        } else {
            ramp
        };
        ground + (target - ground) * ford.strength(p)
    }
    fn ground_with_fords(&self, g: &Generator, p: Vec2) -> f32 {
        let cell = (p / crate::terrain::CELL).floor() * crate::terrain::CELL;
        let f = (p - cell) / crate::terrain::CELL;
        let height = |q| {
            let field = self.field(q);
            self.ford_height(q, carve(g.base_height(q), field), field)
        };
        let a = height(cell);
        let b = height(cell + Vec2::X * crate::terrain::CELL);
        let c = height(cell + Vec2::Y * crate::terrain::CELL);
        let d = height(cell + Vec2::splat(crate::terrain::CELL));
        if f.x + f.y <= 1. {
            a + (b - a) * f.x + (c - a) * f.y
        } else {
            d + (c - d) * (1. - f.x) + (b - d) * (1. - f.y)
        }
    }
    fn water_depth_with_fords(&self, g: &Generator, p: Vec2) -> f32 {
        let cell = (p / crate::terrain::CELL).floor() * crate::terrain::CELL;
        let f = (p - cell) / crate::terrain::CELL;
        let (points, weights) = if f.x + f.y <= 1. {
            (
                [
                    cell,
                    cell + Vec2::X * crate::terrain::CELL,
                    cell + Vec2::Y * crate::terrain::CELL,
                ],
                [1. - f.x - f.y, f.x, f.y],
            )
        } else {
            (
                [
                    cell + Vec2::splat(crate::terrain::CELL),
                    cell + Vec2::Y * crate::terrain::CELL,
                    cell + Vec2::X * crate::terrain::CELL,
                ],
                [f.x + f.y - 1., 1. - f.x, 1. - f.y],
            )
        };
        let fields = points.map(|q| self.field(q));
        if !fields.iter().any(|f| f.distance < f.width + 1.) {
            return f32::NEG_INFINITY;
        }
        fields
            .iter()
            .zip(weights)
            .map(|(f, w)| f.level * w)
            .sum::<f32>()
            - self.ground_with_fords(g, p)
    }
    fn place_fords(&mut self, g: &Generator, owner: Key) {
        let mut random = crate::terrain::SeedRandom(
            g.seed
                ^ (owner.0 as u32).wrapping_mul(0x83ac_142b)
                ^ (owner.1 as u32).wrapping_mul(0x514b_01d3),
        );
        let mut candidates = Vec::new();
        for s in &self.segments {
            let center = s.a.p.lerp(s.b.p, 0.5);
            let width = (s.a.width + s.b.width) * 0.5;
            let direction = (s.b.p - s.a.p).normalize();
            let flow = (s.a.flow + s.b.flow) * 0.5;
            if width > 3.2
                || center.distance(self.lake) < 180.
                || center.length() < 90.
                || (s.a.level - s.b.level).abs() / s.a.p.distance(s.b.p) > 0.045
            {
                continue;
            }
            let across = direction.perp();
            let level = (s.a.level + s.b.level) * 0.5;
            // Reject rugged approaches before the more expensive triangle check.
            if [-24., 24.]
                .iter()
                .any(|d| (g.base_height(center + across * *d) - level).abs() > 7.)
            {
                continue;
            }
            candidates.push((
                random.unit(),
                Ford {
                    center,
                    across,
                    width,
                    flow,
                },
            ));
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut checked = 0;
        for (_, ford) in candidates {
            if self
                .fords
                .iter()
                .any(|f| f.center.distance(ford.center) < 180.)
            {
                continue;
            }
            if self.segments.iter().any(|s| {
                let v = s.b.p - s.a.p;
                let t = ((ford.center - s.a.p).dot(v) / v.length_squared()).clamp(0., 1.);
                ford.center.distance(s.a.p + v * t) < 40.
                    && ((s.a.flow + (s.b.flow - s.a.flow) * t) - ford.flow).abs() > 65.
            }) {
                continue;
            }
            checked += 1;
            if checked > 40 {
                break;
            }
            self.fords.push(ford);
            let steps = ((ford.width + 20.) / 0.25).ceil() as i32;
            // A narrowed ford must retain real water on the rendered terrain grid.
            let valid = self.water_depth_with_fords(g, ford.center) > 0.015
                && self.segments.iter().all(|segment| {
                    (0..=10).all(|i| {
                        let t = i as f32 / 10.;
                        let flow = segment.a.flow + (segment.b.flow - segment.a.flow) * t;
                        (flow - ford.flow).abs() >= 14.
                            || self.water_depth_with_fords(g, segment.a.p.lerp(segment.b.p, t)) > 0.
                    })
                })
                && (-steps..=steps).all(|i| {
                    let p = ford.center + ford.across * i as f32 * 0.25;
                    let slope = Vec2::new(
                        self.ground_with_fords(g, p + Vec2::X * 0.5)
                            - self.ground_with_fords(g, p - Vec2::X * 0.5),
                        self.ground_with_fords(g, p + Vec2::Y * 0.5)
                            - self.ground_with_fords(g, p - Vec2::Y * 0.5),
                    )
                    .length();
                    slope < 0.75
                        && (0..=8).all(|i| {
                            let q = if i == 8 {
                                p
                            } else {
                                p + Vec2::from_angle(i as f32 * std::f32::consts::TAU / 8.) * 0.35
                            };
                            let field = self.field(q);
                            if field.distance >= field.width + 5. {
                                return true;
                            }
                            let depth = self.water_depth_with_fords(g, q);
                            depth < -0.015
                                || (ford.strength(q) > 0.92 && depth < MAX_WADING_DEPTH - 0.01)
                        })
                });
            if !valid {
                self.fords.pop();
            }
            if self.fords.len() == 3 {
                break;
            }
        }
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
            let bank = offset.length() - wet_radius(width);
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
        if lake_distance - wet_radius(LAKE_RADIUS) < best {
            f = RiverField {
                distance: lake_distance,
                width: LAKE_RADIUS,
                level: self.lake_level,
                along: 0.,
                lateral: (p - self.lake).x,
            };
        }
        if f.along != 0. {
            // Bring the two sand banks together at the crossing, then taper
            // smoothly back to the ordinary channel. Keep the downhill graph
            // and water level unchanged; terrain and water use this same width.
            let narrowing = self
                .fords
                .iter()
                .filter(|ford| (f.along - ford.flow).abs() < 45.)
                .map(|ford| ford.strength(p))
                .fold(0., f32::max);
            let narrow_width = (f.width * 0.5).max(1.25).min(f.width);
            f.width += (narrow_width - f.width) * narrowing;
        }
        f
    }
}
fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}
fn interpolate(a: Node, b: Node, t: f32) -> Node {
    Node {
        p: a.p.lerp(b.p, t),
        level: a.level + (b.level - a.level) * t,
        flow: a.flow + (b.flow - a.flow) * t,
        width: a.width + (b.width - a.width) * t,
    }
}
/// Round individual reaches while preserving their source/junction/lake anchors.
/// Rebuild one shared downhill graph after smoothing, including arc-length flow.
fn refine(g: &Generator, coarse: &[Segment], lake: Vec2) -> Vec<Segment> {
    let mut ids = BTreeMap::new();
    let mut nodes = Vec::new();
    for s in coarse {
        for node in [s.a, s.b] {
            ids.entry((node.p.x.to_bits(), node.p.y.to_bits()))
                .or_insert_with(|| {
                    nodes.push(node);
                    nodes.len() - 1
                });
        }
    }
    let id = |p: Vec2| ids[&(p.x.to_bits(), p.y.to_bits())];
    let outlet = id(lake);
    let mut parents = vec![usize::MAX; nodes.len()];
    let mut incoming = vec![0; nodes.len()];
    parents[outlet] = outlet;
    for s in coarse {
        parents[id(s.a.p)] = id(s.b.p);
        incoming[id(s.b.p)] += 1;
    }
    let anchor = |i| incoming[i] != 1 || i == outlet;
    let mut refined = nodes.clone();
    let mut next = vec![usize::MAX; nodes.len()];
    next[outlet] = outlet;
    for start in 0..nodes.len() {
        if !anchor(start) || start == outlet {
            continue;
        }
        let mut reach = vec![nodes[start]];
        let mut end = parents[start];
        reach.push(nodes[end]);
        while !anchor(end) {
            end = parents[end];
            reach.push(nodes[end]);
        }
        // Small coherent bends remove long ruler-straight grid runs. Endpoints
        // remain fixed and the offset is only a few metres, not a new route.
        for i in 1..reach.len() - 1 {
            let direction = (reach[i + 1].p - reach[i - 1].p).normalize_or_zero();
            let offset = (g.noise(reach[i].p * 0.004 + Vec2::new(83., -217.)) - 0.5) * 8.;
            reach[i].p += direction.perp() * offset;
        }
        // Endpoint-preserving Chaikin corner cutting, two bounded passes.
        for _ in 0..2 {
            let mut rounded = vec![reach[0]];
            for pair in reach.windows(2) {
                rounded.push(interpolate(pair[0], pair[1], 0.25));
                rounded.push(interpolate(pair[0], pair[1], 0.75));
            }
            rounded.push(*reach.last().unwrap());
            reach = rounded;
        }
        let mut previous = start;
        let interior = reach.len() - 2;
        for mut node in reach.into_iter().skip(1).take(interior) {
            if node.p.length() <= 70.1 {
                node.p = node.p.normalize() * 70.1;
            }
            let index = refined.len();
            refined.push(node);
            next.push(usize::MAX);
            next[previous] = index;
            previous = index;
        }
        next[previous] = end;
    }
    let mut children = vec![Vec::new(); refined.len()];
    for (i, &parent) in next.iter().enumerate() {
        if parent != usize::MAX && parent != i {
            children[parent].push(i);
        }
    }
    let mut order = vec![outlet];
    let mut cursor = 0;
    while cursor < order.len() {
        let i = order[cursor];
        order.extend(children[i].iter().copied());
        cursor += 1;
    }
    for &i in &order {
        let radius = if i == outlet {
            LAKE_RADIUS + 7.
        } else {
            refined[i].width.max(9.)
        };
        let mut cap = g.base_height(refined[i].p);
        for n in 0..8 {
            cap = cap.min(g.base_height(
                refined[i].p + Vec2::from_angle(n as f32 * std::f32::consts::TAU / 8.) * radius,
            ));
        }
        refined[i].level = refined[i].level.min(cap - 0.6);
        if i != outlet {
            let parent = next[i];
            refined[i].flow = refined[parent].flow - refined[i].p.distance(refined[parent].p);
        } else {
            refined[i].flow = 0.;
        }
    }
    for &i in order.iter().rev() {
        if i != outlet {
            let parent = next[i];
            refined[parent].level = refined[parent]
                .level
                .min(refined[i].level - refined[i].p.distance(refined[parent].p) * 0.002);
        }
    }
    let lake_level = refined[outlet].level;
    for &i in &order {
        let distance = -refined[i].flow;
        let mouth = lake_level + distance * 0.002;
        refined[i].level = mouth + (refined[i].level - mouth) * smooth((distance - 36.) / 144.);
    }
    order
        .into_iter()
        .filter(|&i| i != outlet)
        .map(|i| Segment {
            a: refined[i],
            b: refined[next[i]],
        })
        .collect()
}

/// Actual waterline of the bed profile used by `carve`, not its nominal width.
/// Solve -0.95 + 1.65 * smoothstep(t) = 0: t = 0.5506786.
/// Comparing nominal widths can select a dry lake bank over a wet river bed,
/// leaving a ridge across the inlet even though the drainage graph connects.
pub(crate) fn wet_radius(width: f32) -> f32 {
    width * 0.6 + (width * 0.4 + 2.) * 0.550_678_6
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
    fn fords_have_a_narrow_visible_channel_without_breaking_the_flowing_water() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            for owner in [(0, 0), (-2, -2), (2, -2)] {
                let mut plan = Plan::generate(&g, owner);
                let wet_width = |plan: &Plan, ford: Ford| {
                    (-240..=240)
                        .filter(|i| {
                            let p = ford.center + ford.across * *i as f32 * 0.05;
                            plan.water_depth_with_fords(&g, p) > 0.
                        })
                        .count() as f32
                        * 0.05
                };
                let widths: Vec<_> = plan
                    .fords
                    .iter()
                    .map(|ford| (*ford, wet_width(&plan, *ford)))
                    .collect();
                // Follow the curved centerline through each crossing, checking
                // the same piecewise-linear ground used by the water mesh.
                for (ford, _) in &widths {
                    for segment in &plan.segments {
                        for i in 0..=10 {
                            let t = i as f32 / 10.;
                            let flow = segment.a.flow + (segment.b.flow - segment.a.flow) * t;
                            if (flow - ford.flow).abs() < 14. {
                                let p = segment.a.p.lerp(segment.b.p, t);
                                assert!(
                                    plan.water_depth_with_fords(&g, p) > 0.,
                                    "Narrowing broke the river: seed {seed}, {owner:?}, {p:?}"
                                );
                            }
                        }
                    }
                }
                plan.fords.clear();
                for (ford, width) in widths {
                    let ordinary = wet_width(&plan, ford);
                    assert!(width > 1., "Ford needs a visible wet channel");
                    assert!(
                        width < ordinary * 0.75,
                        "Ford should have visibly protruding banks: seed {seed}, {owner:?}, {width}m vs {ordinary}m"
                    );
                }
            }
        }
    }
    #[test]
    fn fords_are_repeatable_shallow_crossings_with_dry_approaches_and_deep_lakes() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            let world = crate::terrain::Meadow::from_stream_generator(g.clone());
            for owner in [(0, 0), (-2, -2), (2, -2)] {
                g.hydrology.prepare(&g, owner);
                let plan = g.hydrology.get(owner).unwrap();
                assert!(
                    !plan.fords.is_empty(),
                    "No suitable ford: seed {seed}, owner {owner:?}"
                );
                assert!(plan.fords.len() <= 3);
                assert!(!world.walkable(plan.lake));
                let mut submerged = 0;
                for ford in &plan.fords {
                    assert!(ford.width <= 3.2 && ford.center.distance(plan.lake) > 180.);
                    let steps = ((ford.width + 20.) / 0.25).floor() as i32;
                    for i in -steps..=steps {
                        let p = ford.center + ford.across * i as f32 * 0.25;
                        assert!(
                            world.walkable(p),
                            "Impassable ford: seed {seed}, owner {owner:?}, p {p:?}, depth {}, slope {}, strength {}",
                            g.water_depth(p),
                            g.slope(p),
                            ford.strength(p)
                        );
                        if g.water_depth(p) > 0.015 {
                            submerged += 1;
                            assert!(g.water_depth(p) <= MAX_WADING_DEPTH);
                            assert!(
                                world.near_water(p, 0.),
                                "Water must still exclude vegetation"
                            );
                        }
                    }
                    assert!(g.water_depth(ford.entry()) < 0.);
                    assert!(
                        g.water_depth(ford.center) > 0.015,
                        "A ford needs visible flowing water"
                    );
                }
                assert!(submerged > 0);
                let mut deep = 0;
                for segment in &plan.segments {
                    let p = segment.a.p.lerp(segment.b.p, 0.5);
                    if plan.ford_strength(p) == 0. && g.water_depth(p) > 0.5 {
                        assert!(
                            !world.walkable(p),
                            "Ordinary deep river must remain blocked"
                        );
                        deep += 1;
                    }
                }
                assert!(deep > 20);
                let before = plan.fords.clone();
                g.hydrology.retain(&BTreeSet::new());
                g.hydrology.prepare(&g, owner);
                assert_eq!(before, g.hydrology.get(owner).unwrap().fords);
            }
        }
    }

    #[test]
    fn rivers_round_bends_and_widen_from_sources_through_joins_to_the_lake() {
        for seed in [42, 20261003, 314159] {
            for strength in [0., 0.65, 1.] {
                let g = Generator::new(
                    seed,
                    TerrainSettings {
                        mountain_strength: strength,
                    },
                );
                for owner in [(0, 0), (-2, -2), (2, -2)] {
                    let plan = Plan::generate(&g, owner);
                    let ends: BTreeSet<_> = plan
                        .segments
                        .iter()
                        .map(|s| (s.b.p.x.to_bits(), s.b.p.y.to_bits()))
                        .collect();
                    let heads: Vec<_> = plan
                        .segments
                        .iter()
                        .filter(|s| !ends.contains(&(s.a.p.x.to_bits(), s.a.p.y.to_bits())))
                        .collect();
                    assert_eq!(heads.len(), plan.sources);
                    assert!(heads.iter().all(|s| (s.a.width - 1.5).abs() < 0.001));
                    let mut max_turn = 0f32;
                    for s in &plan.segments {
                        assert!(s.b.width >= s.a.width - 0.0001, "Width shrinks downstream");
                        assert!((1.49..=10.).contains(&s.a.width));
                        if s.b.p == plan.lake {
                            assert!(s.b.width > 6.);
                        }
                        let next = plan.segments.iter().find(|r| r.a.p == s.b.p);
                        let count = plan.segments.iter().filter(|r| r.b.p == s.b.p).count();
                        if let Some(next) = next {
                            assert_eq!(s.b, next.a);
                            if count == 1 {
                                let a = (s.b.p - s.a.p).normalize();
                                let b = (next.b.p - next.a.p).normalize();
                                let turn = a.dot(b).clamp(-1., 1.).acos().to_degrees();
                                max_turn = max_turn.max(turn);
                                assert!(
                                    turn < 80.,
                                    "Sharp bend {turn} degrees: seed {seed}, {owner:?}"
                                );
                            } else {
                                assert!(next.a.width > s.a.width, "Junction must grow gradually");
                            }
                        }
                    }
                    assert!(
                        plan.segments.iter().any(|s| {
                            let q = (s.a.p - origin(owner) - Vec2::splat(MARGIN)) / STEP;
                            (q - q.round()).abs().min_element() > 0.01
                        }),
                        "Curve must leave the square grid"
                    );
                    println!(
                        "seed {seed} mountain {strength} system {owner:?}: {} curved segments, max turn {max_turn:.1} degrees, source width 1.5m, mouth {:.2}m",
                        plan.segments.len(),
                        plan.segments
                            .iter()
                            .find(|s| s.b.p == plan.lake)
                            .unwrap()
                            .b
                            .width
                    );
                }
            }
        }
    }
    #[test]
    fn lake_inlets_have_no_dry_ridge_on_the_rendered_ground() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            for owner in [(0, 0), (-2, -2), (2, -2)] {
                g.hydrology.prepare(&g, owner);
                let plan = g.hydrology.get(owner).unwrap();
                for segment in plan
                    .segments
                    .iter()
                    .filter(|s| s.b.p.distance(plan.lake) <= LAKE_RADIUS + STEP)
                {
                    for sample in 0..=128 {
                        let p = segment.a.p.lerp(segment.b.p, sample as f32 / 128.);
                        let water = g.water(p);
                        assert!(
                            g.ground(p) < water.level - 0.01,
                            "Dry lake inlet: seed {seed}, system {owner:?}, p {p:?}, radius {}, ground {}, water {}",
                            p.distance(plan.lake),
                            g.ground(p),
                            water.level
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn neighboring_regions_share_a_lake_and_rivers_cross_their_borders() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            for owner in [(0, 0), (-2, -2), (2, -2)] {
                let members = [
                    owner,
                    (owner.0 + 1, owner.1),
                    (owner.0, owner.1 + 1),
                    (owner.0 + 1, owner.1 + 1),
                ];
                for member in members.into_iter().rev() {
                    g.hydrology.prepare(&g, member);
                }
                let plan = g.hydrology.get(owner).unwrap();
                for member in members {
                    assert!(Arc::ptr_eq(&plan, &g.hydrology.get(member).unwrap()));
                    assert_eq!(plan.segments, Plan::generate(&g, member).segments);
                }
                let heads: Vec<_> = plan
                    .segments
                    .iter()
                    .filter(|s| !plan.segments.iter().any(|r| r.b.p == s.a.p))
                    .collect();
                assert!(
                    heads
                        .iter()
                        .map(|s| key(s.a.p))
                        .collect::<BTreeSet<_>>()
                        .len()
                        >= 2,
                    "Need tributaries from multiple regions: seed {seed}, {owner:?}"
                );
                for head in heads {
                    let mut node = head.a;
                    let mut steps = 0;
                    while node.p != plan.lake {
                        let next = plan.segments.iter().find(|s| s.a == node).unwrap();
                        assert!(next.a.level > next.b.level);
                        node = next.b;
                        steps += 1;
                        assert!(steps <= plan.segments.len(), "Drainage cycle");
                    }
                    assert_eq!(node.level, plan.lake_level);
                }
                let crossings: Vec<_> = plan
                    .segments
                    .iter()
                    .filter(|s| key(s.a.p) != key(s.b.p))
                    .collect();
                assert!(!crossings.is_empty());
                for s in crossings {
                    let a_key = key(s.a.p);
                    let b_key = key(s.b.p);
                    let axis = usize::from(a_key.0 == b_key.0);
                    let boundary = origin(owner)[axis] + SIZE;
                    let t = (boundary - s.a.p[axis]) / (s.b.p[axis] - s.a.p[axis]);
                    let p = s.a.p.lerp(s.b.p, t);
                    let f = g.water(p);
                    assert!(f.distance < 0.01, "River disappears at region boundary");
                    assert!((f.level - (s.a.level + (s.b.level - s.a.level) * t)).abs() < 0.001);
                    let mut left = p;
                    let mut right = p;
                    left[axis] -= 0.01;
                    right[axis] += 0.01;
                    assert!((g.height(left) - g.height(right)).abs() < 0.1);
                    assert!((g.water(left).along - g.water(right).along).abs() < 0.1);
                }
                g.hydrology.retain(&BTreeSet::from([members[3]]));
                assert_eq!(g.hydrology.len(), 1);
                assert!(Arc::ptr_eq(&plan, &g.hydrology.get(owner).unwrap()));
                g.hydrology.retain(&BTreeSet::new());
                g.hydrology.prepare(&g, members[3]);
                let revisit = g.hydrology.get(owner).unwrap();
                assert_eq!(plan.segments, revisit.segments);
                assert_eq!(plan.lake, revisit.lake);
                assert_eq!(plan.lake_level, revisit.lake_level);
            }
        }
    }
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
