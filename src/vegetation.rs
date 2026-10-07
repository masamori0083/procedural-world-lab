//! Three vegetation mesh levels. Placement and collision never depend on the camera.
use crate::{streaming::Batch, trees::TreeShape};
use bevy::prelude::*;
use std::time::Instant;

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
    fn choose(pixels: f32, near: f32, middle: f32, previous: Option<Self>) -> Self {
        // Separate entry/exit thresholds prevent rebuilding at every small camera movement.
        let n = near
            * if previous == Some(Self::Near) {
                0.85
            } else if previous.is_some() {
                1.15
            } else {
                1.0
            };
        let m = middle
            * if matches!(previous, Some(Self::Near | Self::Middle)) {
                0.85
            } else if previous.is_some() {
                1.15
            } else {
                1.0
            };
        if pixels >= n {
            Self::Near
        } else if pixels >= m {
            Self::Middle
        } else {
            Self::Far
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Levels {
    pub trees: Detail,
    pub grass: Detail,
}
impl Levels {
    pub const FULL: Self = Self {
        trees: Detail::Near,
        grass: Detail::Near,
    };
}
#[derive(Resource)]
pub struct LodConfig {
    pub enabled: bool,
}
impl Default for LodConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Clone, Copy)]
pub struct View {
    pub position: Vec3,
    pub rotation: Quat,
    pub fov: f32,
    pub aspect: f32,
    pub height: f32,
}
impl View {
    pub fn from_camera(transform: &Transform, projection: &Projection, camera: &Camera) -> Self {
        let (fov, aspect) = match projection {
            Projection::Perspective(p) => (p.fov, p.aspect_ratio),
            _ => (72_f32.to_radians(), 1.6),
        };
        Self {
            position: transform.translation,
            rotation: transform.rotation,
            fov,
            aspect,
            height: camera.logical_viewport_size().map_or(800., |v| v.y).max(1.),
        }
    }
    pub fn pixels(&self, center: Vec3, diameter: f32) -> f32 {
        let p = self.rotation.inverse() * (center - self.position);
        let depth = -p.z;
        let radius = diameter * 0.5;
        let tan = (self.fov * 0.5).tan();
        // Offscreen regions stay coarse. Keep a sphere-sized margin at the frustum edges.
        if depth + radius <= 0.
            || p.x.abs() > depth.max(0.) * tan * self.aspect + radius
            || p.y.abs() > depth.max(0.) * tan + radius
        {
            return 0.;
        }
        diameter * self.height / (2. * tan * (depth - radius).max(0.08))
    }
}

pub struct PlacedTree {
    pub shape: TreeShape,
    pub offset: Vec3,
    pub center: Vec3,
    pub diameter: f32,
}
impl PlacedTree {
    pub fn new(shape: TreeShape, offset: Vec3) -> Self {
        let (center, diameter) = shape.bounds();
        Self {
            shape,
            offset,
            center: center + offset,
            diameter,
        }
    }
}
#[derive(Clone)]
pub struct Blade {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub color: [f32; 4],
}
pub struct Shrub {
    pub center: Vec3,
    pub radius: f32,
    pub color: [f32; 4],
}
pub struct Blueprint {
    pub trees: Vec<PlacedTree>,
    pub grass: Vec<Blade>,
    pub shrubs: Vec<Shrub>,
    pub origin: Vec3,
}
impl Blueprint {
    pub fn levels(&self, view: Option<View>, previous: Option<Levels>) -> Levels {
        let Some(view) = view else {
            return Levels::FULL;
        };
        let trees = self
            .trees
            .iter()
            .map(|t| view.pixels(t.center + self.origin, t.diameter))
            .fold(0., f32::max);
        let grass = self
            .grass
            .iter()
            .map(|b| view.pixels((b.a + b.b + b.c) / 3. + self.origin, 0.35))
            .chain(
                self.shrubs
                    .iter()
                    .map(|s| view.pixels(s.center + self.origin, s.radius * 2.)),
            )
            .fold(0., f32::max);
        Levels {
            trees: Detail::choose(trees, 100., 28., previous.map(|p| p.trees)),
            grass: Detail::choose(grass, 8., 2., previous.map(|p| p.grass)),
        }
    }
    pub fn full_vertices(&self) -> usize {
        self.trees
            .iter()
            .map(|t| {
                let s = &t.shape;
                let tube = |n: usize| 7 * (6 * (n - 1) + 6);
                tube(s.trunk.len())
                    + s.branches.iter().map(|b| tube(b.len())).sum::<usize>()
                    + s.full_crown_vertices()
            })
            .sum::<usize>()
            + self.grass.len() * 3
            + self.shrubs.len() * 48
    }
    pub fn bytes(&self) -> usize {
        self.trees.capacity() * size_of::<PlacedTree>()
            + self.grass.capacity() * size_of::<Blade>()
            + self.shrubs.capacity() * size_of::<Shrub>()
            + self
                .trees
                .iter()
                .map(|t| {
                    t.shape.trunk.capacity() * size_of::<(Vec3, f32)>()
                        + t.shape.branches.capacity() * size_of::<Vec<(Vec3, f32)>>()
                        + t.shape
                            .branches
                            .iter()
                            .map(|b| b.capacity() * size_of::<(Vec3, f32)>())
                            .sum::<usize>()
                        + t.shape.leaves.capacity() * size_of::<crate::trees::LeafCluster>()
                })
                .sum::<usize>()
    }
    pub fn build(&self, levels: Levels) -> Built {
        let start = Instant::now();
        let template = if levels.trees == Detail::Near {
            Sphere::new(1.).mesh().ico(1).unwrap()
        } else if levels.trees == Detail::Middle {
            Sphere::new(1.).mesh().ico(0).unwrap()
        } else {
            octahedron()
        };
        let mut trees = Batch::default();
        for t in &self.trees {
            trees.append(t.shape.wood_mesh_lod(levels.trees), t.offset);
            trees.append(
                t.shape.crown_mesh(&template),
                t.offset + t.shape.crown_pivot,
            );
        }
        let mut grass = Batch::default();
        if levels.grass != Detail::Far {
            let stride = if levels.grass == Detail::Near { 1 } else { 4 };
            for blade in self.grass.iter().step_by(stride) {
                grass.triangle(blade.a, blade.b, blade.c, blade.color);
            }
        }
        if levels.grass != Detail::Far {
            let sides = if levels.grass == Detail::Near { 8 } else { 4 };
            for shrub in &self.shrubs {
                let base = shrub.center;
                let tip = base + Vec3::Y * shrub.radius * 1.25;
                for i in 0..sides {
                    let point = |i: usize| {
                        let angle = i as f32 * std::f32::consts::TAU / sides as f32;
                        base + Vec3::new(angle.cos(), 0.2, angle.sin()) * shrub.radius
                    };
                    let a = point(i);
                    let b = point((i + 1) % sides);
                    grass.triangle(a, tip, b, shrub.color);
                    grass.triangle(a, b, base, shrub.color);
                }
            }
        }
        let vertices = trees.positions.len() + grass.positions.len();
        Built {
            trees: trees.mesh(),
            grass: grass.mesh(),
            levels,
            vertices,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.,
        }
    }
}
pub struct Built {
    pub trees: Mesh,
    pub grass: Mesh,
    pub levels: Levels,
    pub vertices: usize,
    pub elapsed_ms: f64,
}
fn octahedron() -> Mesh {
    use bevy::{
        asset::RenderAssetUsages, mesh::Indices, render::render_resource::PrimitiveTopology,
    };
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [1., 0., 0.],
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
            [0., 0., -1.],
        ],
    )
    .with_inserted_indices(Indices::U32(vec![
        2, 4, 0, 2, 1, 4, 2, 5, 1, 2, 0, 5, 3, 0, 4, 3, 4, 1, 3, 1, 5, 3, 5, 0,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn screen_size_tracks_camera_zoom_and_landmarks_with_hysteresis() {
        let mut view = View {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            fov: 72_f32.to_radians(),
            aspect: 1.6,
            height: 800.,
        };
        let p = Vec3::new(0., 0., -100.);
        let small = view.pixels(p, 5.);
        let large = view.pixels(p, 25.);
        assert!(large > small * 5.);
        assert_eq!(Detail::choose(small, 100., 28., None), Detail::Middle);
        assert_eq!(Detail::choose(large, 100., 28., None), Detail::Near);
        assert_eq!(view.pixels(-p, 5.), 0.);
        assert_eq!(view.pixels(Vec3::new(1000., 0., -100.), 5.), 0.);
        view.fov *= 0.5;
        assert!(view.pixels(p, 5.) > small * 2.);
        for pixels in [90., 99., 101., 110.] {
            assert_eq!(
                Detail::choose(pixels, 100., 28., Some(Detail::Near)),
                Detail::Near
            );
            assert_eq!(
                Detail::choose(pixels, 100., 28., Some(Detail::Middle)),
                Detail::Middle
            );
        }
        assert_eq!(
            Detail::choose(116., 100., 28., Some(Detail::Far)),
            Detail::Near
        );
        assert_eq!(
            Detail::choose(23., 100., 28., Some(Detail::Middle)),
            Detail::Far
        );
    }
}
