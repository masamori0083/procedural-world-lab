//! A bounded, coordinate-owned link between two existing lake systems.
//! The original tributaries and lake elevations stay fixed. A route is accepted
//! only when it can descend without crossing an existing channel away from a lake.
use crate::{
    river::RiverField,
    streaming::Generator,
    watershed::{self, Key, Node, Plan, Segment},
    work_budget::WorkBudget,
};
use bevy::prelude::*;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap},
    sync::Arc,
};
const STEP: f32 = 24.;
const WIDTH: f32 = 2.6;
const SPAN: f32 = watershed::SIZE * 2.;
const NX: usize = 125;
const NZ: usize = 61;
#[derive(Clone, Copy)]
struct Visit {
    cost: f32,
    id: usize,
}
impl PartialEq for Visit {
    fn eq(&self, b: &Self) -> bool {
        self.cost == b.cost && self.id == b.id
    }
}
impl Eq for Visit {}
impl Ord for Visit {
    fn cmp(&self, b: &Self) -> Ordering {
        b.cost
            .total_cmp(&self.cost)
            .then_with(|| b.id.cmp(&self.id))
    }
}
impl PartialOrd for Visit {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
pub struct Link {
    pub plan: Plan,
    pub upstream: Vec2,
    pub downstream: Vec2,
}
impl Link {
    pub fn bank_view(&self, g: &Generator) -> Option<(Vec2, f32)> {
        let seam = watershed::origin(watershed::pair(watershed::key(self.upstream))).x + SPAN;
        let mut best: Option<(f32, Vec2, f32)> = None;
        for s in &self.plan.segments {
            let middle = s.a.p.lerp(s.b.p, 0.5);
            if (middle.x - seam).abs() > 100. {
                continue;
            }
            let normal = (s.b.p - s.a.p).normalize().perp();
            for side in [-1., 1.] {
                for offset in [12., 18., 26., 34.] {
                    let p = middle + normal * offset * side;
                    if g.water_depth(p) >= -0.015 || g.slope(p) > 0.65 {
                        continue;
                    }
                    let score = (p.x - seam).abs() + (1. - g.hydrology.trail(p)) * 100.;
                    let toward = middle - p;
                    if best.is_none_or(|v| score < v.0) {
                        best = Some((score, p, (-toward.x).atan2(-toward.y)));
                    }
                }
            }
        }
        best.map(|(_, p, yaw)| (p, yaw))
    }
    pub async fn generate_incremental(
        g: &Generator,
        owner: Key,
        plans: &[Arc<Plan>; 2],
        budget: &mut WorkBudget,
    ) -> Option<Self> {
        let (upper, lower) = if plans[0].lake_level >= plans[1].lake_level {
            (&plans[0], &plans[1])
        } else {
            (&plans[1], &plans[0])
        };
        let origin = watershed::origin(owner) + Vec2::splat(48.);
        let position = |id: usize| origin + Vec2::new((id % NX) as f32, (id / NX) as f32) * STEP;
        let index = |p: Vec2| {
            let q = ((p - origin) / STEP).round().as_ivec2();
            q.y as usize * NX + q.x as usize
        };
        let start = index(upper.lake);
        let goal = index(lower.lake);
        let permitted = |p: Vec2| {
            if p.distance(upper.lake) < 25. || p.distance(lower.lake) < 25. {
                return true;
            }
            if p.length() < 72. {
                return false;
            }
            let cap = [
                Vec2::ZERO,
                Vec2::X * 9.,
                -Vec2::X * 9.,
                Vec2::Y * 9.,
                -Vec2::Y * 9.,
            ]
            .into_iter()
            .map(|d| g.base_height(p + d))
            .fold(f32::INFINITY, f32::min);
            cap >= lower.lake_level + 0.65
                && plans.iter().all(|plan| {
                    let f = plan.field(p);
                    ((f.along == 0. && p.distance(plan.lake) < 40.)
                        || f.distance - watershed::wet_radius(f.width)
                            > watershed::wet_radius(WIDTH) + 4.)
                        && plan.fords.iter().all(|ford| {
                            let offset = p - ford.center;
                            // Protect the entire shaped approach, plus the
                            // connector's bank-carving/normal-sampling halo.
                            let excess = Vec2::new(
                                (offset.perp_dot(ford.across).abs() - 14.).max(0.),
                                (offset.dot(ford.across).abs() - ford.width - 30.).max(0.),
                            );
                            excess.length() > 36.
                        })
                })
        };
        let mut height = Vec::with_capacity(NX * NZ);
        let mut open = Vec::with_capacity(NX * NZ);
        for z in 0..NZ {
            budget.checkpoint().await;
            for x in 0..NX {
                let p = position(z * NX + x);
                height.push(g.base_height(p));
                open.push(permitted(p));
            }
        }
        open[start] = true;
        open[goal] = true;
        let mut costs = vec![f32::INFINITY; NX * NZ];
        let mut parent = vec![usize::MAX; NX * NZ];
        let mut queue = BinaryHeap::new();
        costs[start] = 0.;
        parent[start] = start;
        queue.push(Visit {
            cost: 0.,
            id: start,
        });
        let mut count = 0;
        while let Some(Visit { cost, id }) = queue.pop() {
            if cost > costs[id] {
                continue;
            }
            if id == goal {
                break;
            }
            count += 1;
            if count % 32 == 0 {
                budget.checkpoint().await;
            }
            for (dx, dz) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ] {
                let x = (id % NX) as i32 + dx;
                let z = (id / NX) as i32 + dz;
                if x < 0 || z < 0 || x >= NX as i32 || z >= NZ as i32 {
                    continue;
                }
                let next = z as usize * NX + x as usize;
                if !open[next] {
                    continue;
                }
                let a = position(id);
                let b = position(next);
                if !(1..4).all(|n| permitted(a.lerp(b, n as f32 / 4.))) {
                    continue;
                }
                let length = a.distance(b);
                let excavation = (height[next] - upper.lake_level).max(0.);
                let step = length * (1. + excavation * excavation * 0.015)
                    + (height[id] - height[next]).abs() * 2.;
                let candidate = cost + step;
                if candidate < costs[next] {
                    costs[next] = candidate;
                    parent[next] = id;
                    queue.push(Visit {
                        cost: candidate,
                        id: next,
                    });
                }
            }
        }
        if parent[goal] == usize::MAX {
            return None;
        }
        let mut points = vec![position(goal)];
        let mut cursor = goal;
        while cursor != start {
            cursor = parent[cursor];
            points.push(position(cursor));
        }
        points.reverse();
        // Corner cutting is accepted only if the rounded route still obeys the
        // same channel clearance and terrain constraints as the search edges.
        let mut rounded = points.clone();
        for _ in 0..2 {
            let mut next = vec![rounded[0]];
            for w in rounded.windows(2) {
                next.push(w[0].lerp(w[1], 0.25));
                next.push(w[0].lerp(w[1], 0.75));
            }
            next.push(*rounded.last().unwrap());
            rounded = next;
        }
        if rounded
            .windows(2)
            .all(|w| (0..=4).all(|n| permitted(w[0].lerp(w[1], n as f32 / 4.))))
        {
            points = rounded;
        }
        let mut distances = vec![0.];
        for w in points.windows(2) {
            distances.push(distances.last().unwrap() + w[0].distance(w[1]));
        }
        let total = *distances.last().unwrap();
        let mut ceiling = upper.lake_level;
        let mut nodes = Vec::new();
        for (i, p) in points.into_iter().enumerate() {
            budget.checkpoint().await;
            let t = ((distances[i] - 40.) / (total - 80.)).clamp(0., 1.);
            let desired = upper.lake_level + (lower.lake_level - upper.lake_level) * t;
            let cap = if p.distance(upper.lake) < 40. || p.distance(lower.lake) < 40. {
                f32::INFINITY
            } else {
                [
                    Vec2::ZERO,
                    Vec2::X * 9.,
                    -Vec2::X * 9.,
                    Vec2::Y * 9.,
                    -Vec2::Y * 9.,
                ]
                .into_iter()
                .map(|d| g.base_height(p + d) - 0.6)
                .fold(f32::INFINITY, f32::min)
            };
            ceiling = ceiling.min(desired).min(cap).max(lower.lake_level);
            nodes.push(Node {
                p,
                level: ceiling,
                flow: distances[i] - total,
                width: WIDTH,
            });
        }
        let segments = nodes
            .windows(2)
            .filter(|w| w[0].p.distance_squared(w[1].p) > 0.0001)
            .map(|w| Segment { a: w[0], b: w[1] })
            .collect();
        let mut plan = Plan::channel_incremental(g, owner, segments, lower.lake, budget).await;
        // A new ford must not put its approaches through either old tributaries
        // or a reservoir. Its own-channel checks alone cannot detect that.
        plan.fords.retain(|ford| {
            ford.center.distance(upper.lake) > 180.
                && (-80..=80).all(|i| {
                    let p = ford.center + ford.across * i as f32 * 0.5;
                    plans.iter().all(|old| {
                        let f = old.field(p);
                        f.distance > f.width + 40.
                    })
                })
        });
        Some(Self {
            plan,
            upstream: upper.lake,
            downstream: lower.lake,
        })
    }
    pub fn field(&self, p: Vec2) -> RiverField {
        // The reservoirs keep their original calm surfaces and shoreline.
        if p.distance(self.upstream).min(p.distance(self.downstream)) < 24. {
            return RiverField {
                distance: 1_000_000.,
                ..default()
            };
        }
        self.plan.field(p)
    }
}

#[derive(Default)]
pub struct Trails {
    lake_road: Vec<Vec2>,
    pub(crate) segments: Vec<(Vec2, Vec2)>,
    bins: BTreeMap<(i32, i32), Vec<usize>>,
}
impl Trails {
    pub fn road_view(&self) -> Option<(Vec2, f32)> {
        self.lake_road
            .windows(2)
            .find(|w| w[0].distance(w[1]) >= 8.)
            .map(|w| {
                let d = w[1] - w[0];
                (w[0], (-d.x).atan2(-d.y))
            })
    }
    pub fn road_stop(&self, fraction: f32) -> Option<(Vec2, f32)> {
        if self.lake_road.len() < 2 {
            return None;
        }
        let id = ((self.lake_road.len() - 2) as f32 * fraction.clamp(0., 1.)) as usize;
        let a = self.lake_road[id];
        let b = self.lake_road[id + 1];
        let d = b - a;
        let p = if fraction >= 1. { b } else { a };
        Some((p, (-d.x).atan2(-d.y)))
    }
    #[cfg(test)]
    pub fn lake_road(&self) -> &[Vec2] {
        &self.lake_road
    }
    pub async fn generate_incremental(
        g: &Generator,
        plans: &[Arc<Plan>; 2],
        link: Option<&Link>,
        budget: &mut WorkBudget,
    ) -> Self {
        let mut result = Self::default();
        let field = |p| {
            let k = watershed::system(watershed::key(p));
            let local = p - watershed::origin(k);
            let f = if local.min_element() < 8. || local.max_element() > SPAN - 8. {
                RiverField {
                    distance: 1_000_000.,
                    ..default()
                }
            } else {
                plans[usize::from(k.0 != watershed::pair(k).0)].field(p)
            };
            link.map_or(f, |l| watershed::closest(f, l.field(p)))
        };
        let ground = |p| {
            let f = field(p);
            let h = watershed::carve(g.base_height(p), f);
            let k = watershed::system(watershed::key(p));
            let h = plans[usize::from(k.0 != watershed::pair(k).0)].ford_height(p, h, f);
            link.map_or(h, |l| l.plan.ford_height(p, h, f))
        };
        // The same grid vertices recur throughout collision validation. Cache
        // them locally so dense path checks still fit the yielding work budget.
        let heights = std::sync::Mutex::new(std::collections::BTreeMap::new());
        let grid_height = |p: Vec2| {
            let cell = (p / crate::terrain::CELL).as_ivec2();
            let key = (cell.x, cell.y);
            if let Some(h) = heights.lock().unwrap().get(&key).copied() {
                return h;
            }
            let h = ground(p);
            heights.lock().unwrap().insert(key, h);
            h
        };
        let rendered_ground = |p| crate::streaming::triangle_ground(p, &grid_height);
        let wet = |p: Vec2, allow_fords: bool| {
            let f = field(p);
            if f.distance >= f.width + 1. + crate::terrain::CELL * 2. {
                return false;
            }
            let cell = (p / crate::terrain::CELL).floor() * crate::terrain::CELL;
            let t = (p - cell) / crate::terrain::CELL;
            let vertices = if t.x + t.y <= 1. {
                [
                    cell,
                    cell + Vec2::X * crate::terrain::CELL,
                    cell + Vec2::Y * crate::terrain::CELL,
                ]
            } else {
                [
                    cell + Vec2::splat(crate::terrain::CELL),
                    cell + Vec2::Y * crate::terrain::CELL,
                    cell + Vec2::X * crate::terrain::CELL,
                ]
            };
            let depth =
                crate::streaming::triangle_ground(p, &|q| field(q).level) - rendered_ground(p);
            let k = watershed::system(watershed::key(p));
            let ford = plans[usize::from(k.0 != watershed::pair(k).0)]
                .ford_strength(p)
                .max(link.map_or(0., |l| l.plan.ford_strength(p)));
            if allow_fords && ford > 0.9 && depth <= watershed::MAX_WADING_DEPTH {
                return false;
            }
            vertices.iter().any(|q| {
                let f = field(*q);
                f.distance < f.width + 1.
            }) && depth >= -0.015
        };
        let pair_origin = watershed::origin(watershed::pair(watershed::key(plans[0].lake)));
        let walk = |p: Vec2, allow_fords: bool| {
            if watershed::pair(watershed::key(p)) != watershed::pair(watershed::key(plans[0].lake))
            {
                return false;
            }
            let local = p - pair_origin;
            if local.min_element() < 14. || local.x > SPAN * 2. - 14. || local.y > SPAN - 14. {
                return false;
            }
            // Match the horse's 0.35 m water clearance, including triangle
            // clipping at the shoreline rather than the analytic channel alone.
            !wet(p, allow_fords)
                && (0..8).all(|i| {
                    !wet(
                        p + Vec2::from_angle(i as f32 * std::f32::consts::TAU / 8.) * 0.35,
                        allow_fords,
                    )
                })
                && Vec2::new(
                    rendered_ground(p + Vec2::X * 0.5) - rendered_ground(p - Vec2::X * 0.5),
                    rendered_ground(p + Vec2::Y * 0.5) - rendered_ground(p - Vec2::Y * 0.5),
                )
                .length()
                    < 0.60
        };
        let dry_walk = |p| walk(p, false);
        let road_walk = |p| walk(p, true);
        let fords: Vec<_> = plans
            .iter()
            .flat_map(|p| p.fords.iter())
            .chain(link.into_iter().flat_map(|l| l.plan.fords.iter()))
            .copied()
            .collect();
        result.lake_road = lake_road(plans, &fords, &road_walk, &rendered_ground, budget).await;
        let road = result.lake_road.clone();
        for edge in road.windows(2) {
            result.add(edge[0], edge[1]);
        }
        // Short bank trails and ford approaches supplement the continuous road.
        for ford in plans
            .iter()
            .flat_map(|p| p.fords.iter())
            .chain(link.into_iter().flat_map(|l| l.plan.fords.iter()))
        {
            for side in [-1., 1.] {
                let start = ford.center + ford.across * (ford.width + 8.) * side;
                let mut last: Option<Vec2> = None;
                for i in 0..=20 {
                    budget.checkpoint().await;
                    let p = start + ford.across * i as f32 * 2. * side;
                    if dry_walk(p) {
                        if let Some(a) = last
                            && path_walkable(a, p, &dry_walk, budget).await
                        {
                            result.add(a, p);
                        }
                        last = Some(p);
                    } else {
                        last = None;
                    }
                }
            }
        }
        if let Some(link) = link {
            // Make an inspectable 160 m trail on a dry bank near the shared edge.
            let seam = watershed::origin(watershed::pair(watershed::key(link.upstream))).x + SPAN;
            let crossing = link
                .plan
                .segments
                .iter()
                .position(|s| (s.a.p.x - seam) * (s.b.p.x - seam) <= 0.);
            if let Some(center) = crossing {
                for side in [-1., 1.] {
                    let mut last: Option<Vec2> = None;
                    let mut length = 0.;
                    for s in link
                        .plan
                        .segments
                        .iter()
                        .skip(center.saturating_sub(30))
                        .take(60)
                    {
                        budget.checkpoint().await;
                        let normal = (s.b.p - s.a.p).normalize().perp();
                        for offset in [12., 18., 26., 34.] {
                            let p = s.a.p + normal * offset * side;
                            if dry_walk(p)
                                && match last {
                                    None => true,
                                    Some(a) => {
                                        a.distance(p) < 20.
                                            && path_walkable(a, p, &dry_walk, budget).await
                                    }
                                }
                            {
                                if let Some(a) = last {
                                    result.add(a, p);
                                    length += a.distance(p);
                                }
                                last = Some(p);
                                break;
                            }
                        }
                        if length >= 160. {
                            break;
                        }
                    }
                }
            }
        }
        result
    }
    fn add(&mut self, a: Vec2, b: Vec2) {
        let id = self.segments.len();
        self.segments.push((a, b));
        let lo = ((a.min(b) - Vec2::splat(7.)) / 24.).floor().as_ivec2();
        let hi = ((a.max(b) + Vec2::splat(7.)) / 24.).floor().as_ivec2();
        for z in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                self.bins.entry((x, z)).or_default().push(id);
            }
        }
    }
    pub fn strength(&self, p: Vec2) -> f32 {
        self.coverage(p, 1.25, 1.5)
    }
    pub fn blocks_marker(&self, p: Vec2, radius: f32) -> bool {
        self.coverage(p, 1.25 + radius + 0.35, 0.01) > 0.
    }
    pub fn tree_clearance(&self, p: Vec2) -> f32 {
        self.coverage(p, 5., 1.)
    }
    fn coverage(&self, p: Vec2, width: f32, fade: f32) -> f32 {
        let bin = (p / 24.).floor().as_ivec2();
        let Some(ids) = self.bins.get(&(bin.x, bin.y)) else {
            return 0.;
        };
        ids.iter()
            .map(|&id| {
                let (a, b) = self.segments[id];
                let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0., 1.);
                let d = p.distance(a.lerp(b, t));
                let t = ((d - width) / fade).clamp(0., 1.);
                1. - t * t * (3. - 2. * t)
            })
            .fold(0., f32::max)
    }
}

/// Bounded A* on a 12 m grid, with exact ford portals and validated full-width
/// edges. Nothing is published until a continuous shore-to-shore route exists.
async fn lake_road(
    plans: &[Arc<Plan>; 2],
    fords: &[watershed::Ford],
    walk: &impl Fn(Vec2) -> bool,
    ground: &impl Fn(Vec2) -> f32,
    budget: &mut WorkBudget,
) -> Vec<Vec2> {
    const ROAD_STEP: f32 = 12.;
    const RX: usize = 253;
    const RZ: usize = 125;
    let origin =
        watershed::origin(watershed::pair(watershed::key(plans[0].lake))) + Vec2::splat(24.);
    let mut points: Vec<_> = (0..RX * RZ)
        .map(|id| origin + Vec2::new((id % RX) as f32, (id / RX) as f32) * ROAD_STEP)
        .collect();
    let mut extra = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    // A coarse grid could miss a narrow ford. Connect each exact crossing and
    // its two approaches to nearby dry grid vertices instead of inventing bridges.
    for ford in fords {
        let first = points.len();
        points.extend([
            ford.center - ford.across * (ford.width + 12.),
            ford.center,
            ford.center + ford.across * (ford.width + 12.),
        ]);
        for (a, b) in [(first, first + 1), (first + 1, first + 2)] {
            extra.entry(a).or_default().push(b);
            extra.entry(b).or_default().push(a);
        }
        for portal in [first, first + 2] {
            let q = ((points[portal] - origin) / ROAD_STEP).round().as_ivec2();
            for z in q.y - 1..=q.y + 1 {
                for x in q.x - 1..=q.x + 1 {
                    if x >= 0 && z >= 0 && x < RX as i32 && z < RZ as i32 {
                        let id = z as usize * RX + x as usize;
                        extra.entry(id).or_default().push(portal);
                        extra.entry(portal).or_default().push(id);
                    }
                }
            }
        }
    }
    let mut starts = Vec::new();
    let mut goals = vec![false; points.len()];
    for (id, &p) in points.iter().enumerate().take(RX * RZ) {
        if p.distance(plans[0].lake) <= 48. && p.distance(plans[0].lake) >= 28. && walk(p) {
            starts.push(id);
        }
        if p.distance(plans[1].lake) <= 48. && p.distance(plans[1].lake) >= 28. && walk(p) {
            goals[id] = true;
        }
        if id % 128 == 0 {
            budget.checkpoint().await;
        }
    }
    starts.sort_by(|&a, &b| {
        let score = |id: usize| {
            points[id].distance(plans[1].lake)
                + 4. * (points[id].distance(plans[0].lake) - 32.).abs()
        };
        score(a).total_cmp(&score(b)).then_with(|| a.cmp(&b))
    });
    if !goals.iter().any(|v| *v) {
        return Vec::new();
    }
    let heuristic = |id: usize| (points[id].distance(plans[1].lake) - 48.).max(0.) * 1.15;
    let mut costs = vec![f32::INFINITY; points.len()];
    let mut parent = vec![usize::MAX; points.len()];
    let mut queue = BinaryHeap::new();
    // Multiple legal shore entries avoid choosing a disconnected peninsula.
    for id in starts {
        let cost = points[id].distance(plans[0].lake) * 0.1;
        costs[id] = cost;
        parent[id] = id;
        queue.push(Visit {
            cost: cost + heuristic(id),
            id,
        });
    }
    let mut valid = vec![None; points.len()];
    let mut edge_cache = std::collections::BTreeMap::new();
    let mut expanded = 0;
    while let Some(Visit { cost, id }) = queue.pop() {
        budget.checkpoint().await;
        if cost > costs[id] + heuristic(id) + 0.001 {
            continue;
        }
        if goals[id] {
            let mut route = vec![points[id]];
            let mut at = id;
            while parent[at] != at {
                at = parent[at];
                route.push(points[at]);
            }
            route.reverse();
            return route;
        }
        expanded += 1;
        if expanded > 6000 {
            break;
        }
        let mut neighbours = extra.get(&id).cloned().unwrap_or_default();
        if id < RX * RZ {
            let x = (id % RX) as i32;
            let z = (id / RX) as i32;
            for dz in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dz == 0 {
                        continue;
                    }
                    let nx = x + dx;
                    let nz = z + dz;
                    if nx >= 0 && nz >= 0 && nx < RX as i32 && nz < RZ as i32 {
                        neighbours.push(nz as usize * RX + nx as usize);
                    }
                }
            }
        }
        for next in neighbours {
            let a = points[id];
            let b = points[next];
            let distance = a.distance(b);
            let candidate = costs[id] + distance + (ground(b) - ground(a)).abs() * 3.;
            if candidate >= costs[next] {
                continue;
            }
            if !*valid[next].get_or_insert_with(|| walk(b)) {
                continue;
            }
            let edge = (id.min(next), id.max(next));
            let accepted = if let Some(v) = edge_cache.get(&edge) {
                *v
            } else {
                let v = path_walkable(a, b, walk, budget).await;
                edge_cache.insert(edge, v);
                v
            };
            if !accepted {
                continue;
            }
            costs[next] = candidate;
            parent[next] = id;
            queue.push(Visit {
                cost: candidate + heuristic(next),
                id: next,
            });
        }
    }
    Vec::new()
}

async fn path_walkable(
    a: Vec2,
    b: Vec2,
    dry_walk: &impl Fn(Vec2) -> bool,
    budget: &mut WorkBudget,
) -> bool {
    let normal = (b - a).normalize().perp() * 1.25;
    let steps = (a.distance(b) / 0.25).ceil() as usize;
    for i in 0..=steps {
        budget.checkpoint().await;
        let p = a.lerp(b, i as f32 / steps as f32);
        if !dry_walk(p) || !dry_walk(p + normal) || !dry_walk(p - normal) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{terrain::TerrainSettings, watershed::Cache};
    #[test]
    fn paired_lakes_have_a_continuous_downhill_channel_and_repeat_after_eviction() {
        for seed in [42, 20261003, 314159] {
            let g = Generator::new(seed, TerrainSettings::default());
            for owner in [(0, 0), (-4, -2), (4, 2)] {
                g.hydrology.prepare(&g, owner);
                if seed == 314159 && owner == (4, 2) {
                    // This bounded footprint has no descending route with
                    // channel clearance: retain the two original closed basins.
                    assert!(g.hydrology.link(owner).is_none());
                    g.hydrology.retain(&Default::default());
                    g.hydrology.prepare(&g, (owner.0 + 2, owner.1));
                    assert!(g.hydrology.link(owner).is_none());
                    continue;
                }
                let link = g
                    .hydrology
                    .link(owner)
                    .unwrap_or_else(|| panic!("No connection for seed {seed} pair {owner:?}"));
                assert_eq!(link.plan.segments.first().unwrap().a.p, link.upstream);
                assert_eq!(link.plan.segments.last().unwrap().b.p, link.downstream);
                let mut crosses = false;
                let seam = watershed::origin(owner).x + SPAN;
                for s in &link.plan.segments {
                    assert!(s.a.level >= s.b.level);
                    assert!(s.a.flow < s.b.flow);
                    crosses |= (s.a.p.x - seam) * (s.b.p.x - seam) <= 0.;
                    for n in 0..=8 {
                        let p = s.a.p.lerp(s.b.p, n as f32 / 8.);
                        if p.distance(link.upstream) > 60. && p.distance(link.downstream) > 60. {
                            let f = g.water(p);
                            assert!(
                                (f.along - (s.a.flow + (s.b.flow - s.a.flow) * n as f32 / 8.))
                                    .abs()
                                    < 0.25,
                                "Connection intersects another channel at {p:?}"
                            );
                        }
                        assert!(
                            g.water_depth(p) > 0.,
                            "Dry connection {seed} {owner:?} {p:?}"
                        );
                    }
                }
                assert!(crosses);
                let left = g.hydrology.link((owner.0 + 2, owner.1)).unwrap();
                assert!(Arc::ptr_eq(&link, &left));
                let expected = link.plan.segments.clone();
                g.hydrology.retain(&Default::default());
                g.hydrology.prepare(&g, (owner.0 + 2, owner.1));
                assert_eq!(g.hydrology.link(owner).unwrap().plan.segments, expected);
            }
        }
    }
    #[test]
    fn pair_ownership_is_signed_and_not_loading_order_dependent() {
        assert_eq!(watershed::pair((-1, -1)), (-4, -2));
        assert_eq!(watershed::pair((-2, -1)), (-4, -2));
        assert_eq!(watershed::pair((2, 1)), (0, 0));
        let g = Generator::new(42, TerrainSettings::default());
        let cache = Cache::default();
        cache.prepare(&g, (3, 1));
        assert_eq!(
            cache.link((0, 0)).unwrap().plan.segments,
            g.hydrology.link((0, 0)).unwrap().plan.segments
        );
    }
    #[test]
    fn lake_road_connects_both_shores_with_walkable_continuous_edges() {
        let g = Generator::new(42, TerrainSettings::default());
        let road = g.hydrology.lake_road((0, 0));
        assert!(road.len() > 10, "No continuous lake road");
        let lakes = [
            g.hydrology.get((0, 0)).unwrap().lake,
            g.hydrology.get((2, 0)).unwrap().lake,
        ];
        assert!(road[0].distance(lakes[0]) <= 48.);
        assert!(road.last().unwrap().distance(lakes[1]) <= 48.);
        let world = crate::terrain::Meadow::from_stream_generator(g.clone());
        for edge in road.windows(2) {
            assert!(edge[0].distance(edge[1]) > 0.);
            let n = (edge[0].distance(edge[1]) / 0.25).ceil() as usize;
            let normal = (edge[1] - edge[0]).normalize().perp() * 1.25;
            for i in 0..=n {
                let p = edge[0].lerp(edge[1], i as f32 / n as f32);
                assert!(g.hydrology.trail(p) > 0.99);
                for q in [p, p + normal, p - normal] {
                    assert!(world.walkable(q), "Blocked lake road at {q:?}");
                }
            }
        }
        g.hydrology.retain(&std::collections::BTreeSet::new());
        g.hydrology.prepare(&g, (3, 1));
        assert_eq!(road, g.hydrology.lake_road((0, 0)));
        println!(
            "LAKE ROAD: {} edges, {:.0}m",
            road.len() - 1,
            road.windows(2).map(|w| w[0].distance(w[1])).sum::<f32>()
        );
    }
    #[test]
    fn generated_trail_centrelines_are_walkable_on_actual_triangles() {
        for seed in [3, 6, 42] {
            let g = Generator::new(seed, TerrainSettings::default());
            let world = crate::terrain::Meadow::from_stream_generator(g.clone());
            let trails = g.hydrology.trail_segments();
            assert!(!trails.is_empty());
            for (a, b) in trails {
                let steps = (a.distance(b) / 0.25).ceil() as usize;
                for i in 0..=steps {
                    let p = a.lerp(b, i as f32 / steps as f32);
                    let normal = (b - a).normalize().perp() * 1.25;
                    for q in [p, p + normal, p - normal] {
                        assert!(world.walkable(q), "seed {seed}, blocked trail at {q:?}");
                    }
                }
            }
        }
    }
    #[test]
    fn dirt_trails_are_grounded_dry_and_clear_of_vegetation_potential() {
        let g = Generator::new(42, TerrainSettings::default());
        let cache = g.hydrology.trail_segments();
        assert!(!cache.is_empty());
        for (a, b) in cache {
            for n in 0..=8 {
                let p = a.lerp(b, n as f32 / 8.);
                assert!(
                    g.water_depth(p) < -0.015
                        || (g.hydrology.ford_strength(p) > 0.9
                            && g.water_depth(p) <= watershed::MAX_WADING_DEPTH)
                );
                assert!(g.slope(p) < 0.75);
                assert!(g.hydrology.trail(p) > 0.99, "No trail at {p:?}");
                assert!(g.habitat(p).tree_density < 0.001);
                assert!(g.habitat(p).grass_density < 0.001);
            }
        }
    }
}
