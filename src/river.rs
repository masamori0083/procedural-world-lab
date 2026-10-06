//! A constrained Priority-Flood drainage tree with contributing-area tributaries.
//! Reference: Barnes et al., https://arxiv.org/abs/1511.04463.
use crate::terrain::{CELL, EXTENT, SIDE};
use bevy::prelude::*;
use bevy::{asset::RenderAssetUsages, render::render_resource::PrimitiveTopology};
use std::{cmp::Ordering, collections::BinaryHeap};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverPoint {
    pub position: Vec2,
    pub level: f32,
    pub distance: f32,
    pub width: f32,
    /// Shared downstream texture coordinate, continuous across confluences.
    pub flow: f32,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct RiverField {
    pub distance: f32,
    pub level: f32,
    pub width: f32,
    pub along: f32,
    pub lateral: f32,
}
pub struct River {
    /// Main source-to-outlet path, retained for navigation and reports.
    pub points: Vec<RiverPoint>,
    /// Unique reaches, each ending at a confluence or the outlet.
    pub reaches: Vec<Vec<RiverPoint>>,
    pub source_count: usize,
    pub fields: Vec<RiverField>,
}
#[derive(Clone, Copy, Debug)]
pub struct WaterVertex {
    pub position: Vec3,
    pub uv: Vec2,
}
#[derive(Clone, Copy)]
struct FloodCell {
    height: f32,
    index: usize,
}
impl PartialEq for FloodCell {
    fn eq(&self, other: &Self) -> bool {
        self.height == other.height && self.index == other.index
    }
}
impl Eq for FloodCell {}
impl Ord for FloodCell {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .height
            .total_cmp(&self.height)
            .then_with(|| other.index.cmp(&self.index))
    }
}
impl PartialOrd for FloodCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
pub fn position(index: usize) -> Vec2 {
    Vec2::new(
        (index % SIDE) as f32 * CELL - EXTENT,
        (index / SIDE) as f32 * CELL - EXTENT,
    )
}
impl River {
    pub fn generate(heights: &[f32], allowed: &[bool], source: usize) -> Self {
        let mut parents = vec![usize::MAX; heights.len()];
        let mut flooded = vec![f32::INFINITY; heights.len()];
        let mut heap = BinaryHeap::new();
        let mut order = Vec::new();
        for i in 0..heights.len() {
            let x = i % SIDE;
            let z = i / SIDE;
            if allowed[i] && (x == 0 || z == 0 || x == SIDE - 1 || z == SIDE - 1) {
                flooded[i] = heights[i];
                parents[i] = i;
                heap.push(FloodCell {
                    height: heights[i],
                    index: i,
                });
            }
        }
        // Every parent is already visited, so drainage cannot form a cycle.
        while let Some(cell) = heap.pop() {
            order.push(cell.index);
            let x = cell.index % SIDE;
            let z = cell.index / SIDE;
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
                let nx = x as i32 + dx;
                let nz = z as i32 + dz;
                if nx < 0 || nz < 0 || nx >= SIDE as i32 || nz >= SIDE as i32 {
                    continue;
                }
                let next = nz as usize * SIDE + nx as usize;
                if !allowed[next] || parents[next] != usize::MAX {
                    continue;
                }
                let step = position(next).distance(position(cell.index)) / CELL;
                let height = heights[next].max(cell.height + 0.004 * step);
                parents[next] = cell.index;
                flooded[next] = height;
                heap.push(FloodCell {
                    height,
                    index: next,
                });
            }
        }
        assert!(
            allowed[source] && parents[source] != usize::MAX,
            "River source has no outlet"
        );
        // Uniform rainfall proxy: each allowed cell contributes one unit. Parents
        // precede children in the flood order, so reversing it accumulates runoff.
        let mut area = vec![1u32; heights.len()];
        let mut to_outlet = vec![0.0; heights.len()];
        let mut outlets = vec![usize::MAX; heights.len()];
        for &i in &order {
            let parent = parents[i];
            outlets[i] = if parent == i { i } else { outlets[parent] };
            if parent != i {
                to_outlet[i] = to_outlet[parent] + position(i).distance(position(parent));
            }
        }
        for &i in order.iter().rev() {
            if parents[i] != i {
                area[parents[i]] += area[i];
            }
        }
        let outlet = outlets[source];
        let trace = |start: usize, occupied: &[bool]| {
            let mut path = vec![start];
            let mut current = start;
            while parents[current] != current && !occupied[current] {
                current = parents[current];
                path.push(current);
            }
            path
        };
        let mut occupied = vec![false; heights.len()];
        let main = trace(source, &occupied);
        for &i in &main {
            occupied[i] = true;
        }
        let mut sources = vec![source];
        // Prefer sizeable, elevated catchments with long distinct approaches.
        // Four total sources is a readability cap, not four arbitrary lines.
        for _ in 0..3 {
            let network_nodes: Vec<_> = occupied
                .iter()
                .enumerate()
                .filter_map(|(i, &used)| used.then_some(i))
                .collect();
            let mut best: Option<(f32, Vec<usize>)> = None;
            for &i in &order {
                let p = position(i);
                if occupied[i]
                    || outlets[i] != outlet
                    || area[i] < 12
                    || p.x.abs() > EXTENT - 18.0
                    || p.y.abs() > EXTENT - 18.0
                    || sources.iter().any(|&s| position(s).distance(p) < 32.0)
                {
                    continue;
                }
                let path = trace(i, &occupied);
                let join = *path.last().unwrap();
                let length = to_outlet[i] - to_outlet[join];
                if !occupied[join]
                    || sources.contains(&join)
                    || length < 32.0
                    || to_outlet[join] < 20.0
                    || flooded[i] - flooded[join] < 0.12
                    || network_nodes
                        .iter()
                        .any(|&j| position(j).distance_squared(p) < 24.0_f32.powi(2))
                {
                    continue;
                }
                let score = length
                    * (area[i] as f32).sqrt()
                    * (1.0 + (flooded[i] - flooded[join]).min(20.0) * 0.05);
                if best.as_ref().is_none_or(|(value, _)| score > *value) {
                    best = Some((score, path));
                }
            }
            let Some((_, path)) = best else {
                break;
            };
            sources.push(path[0]);
            for i in path {
                occupied[i] = true;
            }
        }
        // Split at junctions BEFORE rounding. Shared endpoints retain identical
        // water levels and widths, and the downstream reach is stored only once.
        let mut incoming = vec![0u8; heights.len()];
        for &i in &order {
            if occupied[i] && parents[i] != i {
                incoming[parents[i]] += 1;
            }
        }
        let mut reaches = Vec::new();
        let mut starts = Vec::new();
        for &start in &order {
            if !occupied[start] || incoming[start] == 1 || parents[start] == start {
                continue;
            }
            let mut ids = vec![start];
            let mut current = start;
            loop {
                current = parents[current];
                ids.push(current);
                if incoming[current] != 1 || parents[current] == current {
                    break;
                }
            }
            let points: Vec<_> = ids
                .iter()
                .map(|&i| RiverPoint {
                    position: position(i),
                    level: flooded[i] - 0.35,
                    distance: 0.0,
                    flow: 0.0,
                    width: 1.6 + 2.0 * (area[i] as f32 / area[outlet] as f32).sqrt(),
                })
                .collect();
            let mut points = round_reach(points);
            // Reaches are visited downstream first by the flood order.
            let end = *points.last().unwrap();
            let end_flow = reaches
                .iter()
                .find_map(|r: &Vec<RiverPoint>| {
                    (r[0].position == end.position).then_some(r[0].flow)
                })
                .unwrap_or(0.0);
            let length = points.last().unwrap().distance;
            for point in &mut points {
                point.flow = end_flow - length + point.distance;
            }
            starts.push(start);
            reaches.push(points);
        }
        let mut points = Vec::new();
        let mut current = source;
        while current != outlet {
            let index = starts.iter().position(|&s| s == current).unwrap();
            let reach = &reaches[index];
            points.extend(reach.iter().take(reach.len() - 1).copied());
            loop {
                current = parents[current];
                if incoming[current] != 1 || current == outlet {
                    break;
                }
            }
            if current == outlet {
                points.push(*reach.last().unwrap());
            }
        }
        set_distances(&mut points);
        let mut river = Self {
            points,
            reaches,
            source_count: sources.len(),
            fields: vec![],
        };
        river.fields = (0..heights.len())
            .map(|i| river.nearest(position(i)))
            .collect();
        river
    }
    fn nearest(&self, p: Vec2) -> RiverField {
        let mut nearest = RiverField {
            distance: f32::INFINITY,
            ..default()
        };
        for pair in self.reaches.iter().flat_map(|r| r.windows(2)) {
            let [a, b] = [pair[0], pair[1]];
            let segment = b.position - a.position;
            let t = ((p - a.position).dot(segment) / segment.length_squared()).clamp(0.0, 1.0);
            let offset = p - a.position.lerp(b.position, t);
            let distance = offset.length();
            if distance < nearest.distance {
                nearest = RiverField {
                    distance,
                    level: a.level + (b.level - a.level) * t,
                    width: a.width + (b.width - a.width) * t,
                    along: a.flow + (b.flow - a.flow) * t,
                    lateral: segment.normalize().perp_dot(offset),
                };
            }
        }
        nearest
    }
    pub fn total_length(&self) -> f32 {
        self.reaches
            .iter()
            .map(|r| r.last().unwrap().distance)
            .sum()
    }
    pub fn at(&self, p: Vec2) -> RiverField {
        let grid =
            ((p + Vec2::splat(EXTENT)) / CELL).clamp(Vec2::ZERO, Vec2::splat((SIDE - 1) as f32));
        let x = (grid.x.floor() as usize).min(SIDE - 2);
        let z = (grid.y.floor() as usize).min(SIDE - 2);
        let u = grid.x - x as f32;
        let v = grid.y - z as f32;
        let (indices, weights) = if u + v <= 1.0 {
            (
                [z * SIDE + x, z * SIDE + x + 1, (z + 1) * SIDE + x],
                [1.0 - u - v, u, v],
            )
        } else {
            (
                [(z + 1) * SIDE + x + 1, (z + 1) * SIDE + x, z * SIDE + x + 1],
                [u + v - 1.0, 1.0 - u, 1.0 - v],
            )
        };
        let mut result = RiverField::default();
        for (index, weight) in indices.into_iter().zip(weights) {
            let f = self.fields[index];
            result.distance += f.distance * weight;
            result.level += f.level * weight;
            result.width += f.width * weight;
            result.along += f.along * weight;
            result.lateral += f.lateral * weight;
        }
        result
    }
    pub fn carve(&self, index: usize, land: f32) -> f32 {
        let f = self.fields[index];
        let smooth = |t: f32| {
            let t = t.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        if f.distance > f.width + 6.0 {
            return land;
        }
        let bowl = f.level - 0.95 + 1.6 * smooth(f.distance / (f.width + 1.0));
        let blend = smooth((f.distance - f.width - 1.0) / 2.0);
        let channel = bowl + (land.max(f.level + 0.6) - bowl) * blend;
        let influence = 1.0 - smooth((f.distance - f.width - 4.0) / 2.0);
        land + (channel - land) * influence
    }
    pub fn length(&self) -> f32 {
        self.points.last().unwrap().distance
    }

    pub fn triangles(&self, heights: &[f32]) -> Vec<[WaterVertex; 3]> {
        let mut triangles = Vec::new();
        for z in 0..SIDE - 1 {
            for x in 0..SIDE - 1 {
                let a = z * SIDE + x;
                let b = a + 1;
                let c = a + SIDE;
                let d = c + 1;
                for indices in [[a, c, b], [b, c, d]] {
                    if !indices
                        .iter()
                        .any(|i| self.fields[*i].distance < self.fields[*i].width + 1.0)
                    {
                        continue;
                    }
                    let vertex = |i: usize| {
                        let p = position(i);
                        let f = self.fields[i];
                        WaterVertex {
                            position: Vec3::new(p.x, f.level, p.y),
                            uv: Vec2::new(0.5 + f.lateral / (2.0 * f.width), f.along / 6.0),
                        }
                    };
                    let mut polygon = Vec::with_capacity(4);
                    for edge in 0..3 {
                        let i = indices[edge];
                        let j = indices[(edge + 1) % 3];
                        let diff_i = heights[i] - self.fields[i].level;
                        let diff_j = heights[j] - self.fields[j].level;
                        let a = vertex(i);
                        let b = vertex(j);
                        if diff_i < 0.0 {
                            polygon.push(a);
                        }
                        if (diff_i < 0.0) != (diff_j < 0.0) {
                            let t = diff_i / (diff_i - diff_j);
                            polygon.push(WaterVertex {
                                position: a.position.lerp(b.position, t),
                                uv: a.uv.lerp(b.uv, t),
                            });
                        }
                    }
                    for i in 1..polygon.len().saturating_sub(1) {
                        let tri = [polygon[0], polygon[i], polygon[i + 1]];
                        if (tri[1].position - tri[0].position)
                            .cross(tri[2].position - tri[0].position)
                            .length_squared()
                            > 1e-10
                        {
                            triangles.push(tri);
                        }
                    }
                }
            }
        }
        triangles
    }
    pub fn mesh(&self, heights: &[f32]) -> Mesh {
        let vertices: Vec<WaterVertex> = self.triangles(heights).into_iter().flatten().collect();
        let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position.to_array()).collect();
        let uvs: Vec<[f32; 2]> = vertices.iter().map(|v| v.uv.to_array()).collect();
        let mut normals = Vec::with_capacity(vertices.len());
        for tri in vertices.chunks_exact(3) {
            let normal = (tri[1].position - tri[0].position)
                .cross(tri[2].position - tri[0].position)
                .normalize()
                .to_array();
            normals.extend([normal; 3]);
        }
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    }
}

fn set_distances(points: &mut [RiverPoint]) {
    points[0].distance = 0.0;
    for i in 1..points.len() {
        points[i].distance =
            points[i - 1].distance + points[i].position.distance(points[i - 1].position);
    }
}
fn round_reach(mut points: Vec<RiverPoint>) -> Vec<RiverPoint> {
    for _ in 0..2 {
        let mut rounded = vec![points[0]];
        for pair in points.windows(2) {
            for t in [0.25, 0.75] {
                rounded.push(RiverPoint {
                    position: pair[0].position.lerp(pair[1].position, t),
                    level: pair[0].level + (pair[1].level - pair[0].level) * t,
                    width: pair[0].width + (pair[1].width - pair[0].width) * t,
                    distance: 0.0,
                    flow: 0.0,
                });
            }
        }
        rounded.push(*points.last().unwrap());
        points = rounded;
    }
    set_distances(&mut points);
    points
}
