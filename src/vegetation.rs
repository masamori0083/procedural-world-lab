//! Three vegetation mesh levels. Placement and collision never depend on the camera.
use crate::{streaming::Batch, trees::TreeShape};
use bevy::platform::time::Instant;
use bevy::{camera::visibility::VisibilityRange, prelude::*};

// Position, normal, color, UVs and separate GPU wind parameters.
pub const VERTEX_BYTES: usize = 56;

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
#[derive(Clone, Debug, PartialEq)]
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
    pub refined_grass: bool,
}

/// Cached tree levels, independent of camera rotation. Only nearby trees keep
/// detailed meshes; the next level is prepared before its fade band is reached.
#[derive(Clone, Debug, PartialEq)]
pub struct TreePlan {
    pub masks: Vec<u8>,
    pub projection_scale: Option<f32>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct StreamRequest {
    pub levels: Levels,
    pub trees: TreePlan,
    pub grass: Option<GrassPlan>,
}
impl StreamRequest {
    pub fn bytes(&self) -> usize {
        self.trees.masks.capacity()
            + self.grass.as_ref().map_or(0, |p| {
                p.masks.capacity()
                    + p.centers.capacity() * size_of::<Vec3>()
                    + p.occupied.capacity()
            })
    }
}
pub struct TreeLodMesh {
    pub index: usize,
    pub detail: Detail,
    pub center: Vec3,
    pub wood: Mesh,
    pub leaves: Mesh,
}

// Rendering patches are smaller than world chunks. The seeded descriptors and
// collision remain chunk-local; only mesh origins and drawing ranges change.
pub const GRASS_PATCH_SIZE: f32 = 16.;
pub const GRASS_PATCH_COUNT: usize = 36;
const GRASS_NEAR_END: f32 = 30.;
const GRASS_MIDDLE_END: f32 = 72.;
#[derive(Clone, Debug, PartialEq)]
pub struct GrassPlan {
    pub masks: Vec<u8>,
    pub centers: Vec<Vec3>,
    pub occupied: Vec<bool>,
    pub full: bool,
}
pub struct GrassLodMesh {
    pub index: usize,
    pub detail: Detail,
    pub center: Vec3,
    pub visual: Mesh,
    pub shadow: Mesh,
}
fn grass_patch(root: Vec3) -> usize {
    let x = (root.x / GRASS_PATCH_SIZE).floor().clamp(0., 5.) as usize;
    let z = (root.z / GRASS_PATCH_SIZE).floor().clamp(0., 5.) as usize;
    z * 6 + x
}
pub fn grass_range(plan: &GrassPlan, index: usize, detail: Detail) -> Option<VisibilityRange> {
    if plan.full {
        return None;
    }
    Some(VisibilityRange {
        start_margin: if detail == Detail::Middle && plan.masks[index] & 1 != 0 {
            18. ..GRASS_NEAR_END
        } else {
            0. ..0.
        },
        end_margin: if detail == Detail::Near {
            18. ..GRASS_NEAR_END
        } else {
            52. ..GRASS_MIDDLE_END
        },
        use_aabb: false,
    })
}

fn tree_bands(diameter: f32, scale: f32) -> (std::ops::Range<f32>, std::ops::Range<f32>) {
    // Approximately the study's 16–24 / 28–40m transitions for a small tree.
    // Larger landmark trees retain detail farther away at the same screen size.
    let near = diameter * (0.5 + scale / 160.);
    let middle = diameter * (0.5 + scale / 64.);
    (near * 0.85..near * 1.15, middle * 0.85..middle * 1.15)
}
pub fn tree_range(
    diameter: f32,
    plan: &TreePlan,
    index: usize,
    detail: Detail,
) -> Option<VisibilityRange> {
    let scale = plan.projection_scale?;
    let mask = plan.masks[index];
    if mask.count_ones() == 1 {
        return None;
    }
    let (near, middle) = tree_bands(diameter, scale);
    Some(VisibilityRange {
        start_margin: match detail {
            Detail::Middle if mask & 1 != 0 => near.clone(),
            Detail::Far => middle.clone(),
            _ => 0. ..0.,
        },
        end_margin: match detail {
            Detail::Near => near,
            Detail::Middle => middle,
            Detail::Far => f32::MAX..f32::MAX,
        },
        use_aabb: false,
    })
}

impl Blueprint {
    pub fn stream_request(
        &self,
        view: Option<View>,
        previous: Option<&StreamRequest>,
    ) -> StreamRequest {
        let scale = view.map(|v| v.height / (2. * (v.fov * 0.5).tan()));
        let masks: Vec<_> = self
            .trees
            .iter()
            .enumerate()
            .map(|(index, tree)| {
                let Some(v) = view else {
                    return 1;
                };
                let (near, middle) = tree_bands(tree.diameter, scale.unwrap());
                let old = previous.map_or(0, |r| r.trees.masks[index]);
                let distance = (tree.center + self.origin).distance(v.position);
                // 12m prefetch gives workers time to finish. Retain it for another
                // 8m on retreat, avoiding asset churn at the cache boundary.
                let near_limit = near.end + if old & 1 != 0 { 20. } else { 12. };
                let middle_limit = middle.end + if old & 2 != 0 { 20. } else { 12. };
                if distance < near_limit {
                    7
                } else if distance < middle_limit {
                    6
                } else {
                    4
                }
            })
            .collect();
        let mut levels = if self.refined_grass {
            previous.map_or(Levels::FULL, |r| r.levels)
        } else {
            self.levels(view, previous.map(|r| r.levels))
        };
        levels.trees = masks
            .iter()
            .map(|mask| {
                if mask & 1 != 0 {
                    Detail::Near
                } else if mask & 2 != 0 {
                    Detail::Middle
                } else {
                    Detail::Far
                }
            })
            .min_by_key(|d| d.index())
            .unwrap_or(Detail::Far);
        let grass = self.refined_grass.then(|| self.grass_plan(view, previous));
        if grass.is_some() {
            let pixels = view.map_or(f32::MAX, |v| {
                self.shrubs
                    .iter()
                    .map(|s| {
                        s.radius * 2. * scale.unwrap()
                            / (s.center + self.origin).distance(v.position).max(0.08)
                    })
                    .fold(0., f32::max)
            });
            levels.grass = Detail::choose(pixels, 8., 2., previous.map(|r| r.levels.grass));
        }
        StreamRequest {
            levels,
            grass,
            trees: TreePlan {
                masks,
                projection_scale: scale,
            },
        }
    }

    fn grass_plan(&self, view: Option<View>, previous: Option<&StreamRequest>) -> GrassPlan {
        let (centers, occupied) = if let Some(old) = previous.and_then(|p| p.grass.as_ref()) {
            (old.centers.clone(), old.occupied.clone())
        } else {
            let mut centers: Vec<_> = (0..GRASS_PATCH_COUNT)
                .map(|i| {
                    Vec3::new(
                        (i % 6) as f32 * GRASS_PATCH_SIZE + 8.,
                        0.,
                        (i / 6) as f32 * GRASS_PATCH_SIZE + 8.,
                    )
                })
                .collect();
            let mut counts = [0_usize; GRASS_PATCH_COUNT];
            for blade in &self.grass {
                let root = (blade.a + blade.b) * 0.5;
                let i = grass_patch(root);
                centers[i].y += root.y;
                counts[i] += 1;
            }
            for (i, center) in centers.iter_mut().enumerate() {
                center.y /= counts[i].max(1) as f32;
            }
            (centers, counts.into_iter().map(|n| n > 0).collect())
        };
        let masks = centers
            .iter()
            .enumerate()
            .map(|(i, center)| {
                if !occupied[i] {
                    return 0;
                }
                let Some(v) = view else {
                    return 1;
                };
                let old = previous
                    .and_then(|p| p.grass.as_ref())
                    .map_or(0, |p| p.masks[i]);
                let distance = (*center + self.origin).distance(v.position);
                // These use the same patch center as GPU visibility. Prefetch
                // 12m before the fade ends; retain another 8m on retreat.
                if distance < GRASS_NEAR_END + if old & 1 != 0 { 20. } else { 12. } {
                    3
                } else if distance < GRASS_MIDDLE_END + if old & 2 != 0 { 20. } else { 12. } {
                    2
                } else {
                    0
                }
            })
            .collect();
        GrassPlan {
            masks,
            centers,
            occupied,
            full: view.is_none(),
        }
    }

    pub fn build_streamed(
        &self,
        request: StreamRequest,
        previous: Option<&StreamRequest>,
    ) -> Built {
        let start = Instant::now();
        let parts_changed = previous.is_none_or(|p| p.levels.grass != request.levels.grass);
        let mut built = self.build_parts(
            request.levels,
            false,
            parts_changed,
            request.grass.is_none(),
        );
        for (index, tree) in self.trees.iter().enumerate() {
            let old = previous.map_or(0, |p| p.trees.masks[index]);
            let added = request.trees.masks[index] & !old;
            for detail in [Detail::Near, Detail::Middle, Detail::Far] {
                if added & (1 << detail.index()) == 0 {
                    continue;
                }
                let phase = (tree.shape.seed % 10000) as f32 * std::f32::consts::TAU / 10000.;
                let height = tree.shape.trunk.last().unwrap().0.y;
                let mut wood = Batch::default();
                let mut leaves = Batch::default();
                // Both parts/levels use the same origin, so fade distance agrees.
                let offset = tree.offset - tree.center;
                wood.append_wind(
                    tree.shape.wood_mesh_lod(detail),
                    offset,
                    offset.y,
                    height,
                    phase,
                    0.,
                );
                let foliage = if tree.shape.species == crate::trees::Species::Conifer {
                    &mut wood
                } else {
                    &mut leaves
                };
                foliage.append_wind(
                    tree.shape.foliage_mesh(detail),
                    offset + tree.shape.crown_pivot,
                    offset.y,
                    height,
                    phase,
                    tree.shape.foliage_wind_amplitude(detail),
                );
                built.vertices += wood.positions.len() + leaves.positions.len();
                built.tree_additions.push(TreeLodMesh {
                    index,
                    detail,
                    center: tree.center,
                    wood: wood.mesh(),
                    leaves: leaves.mesh(),
                });
            }
        }
        if let Some(plan) = &request.grass {
            let mut batches: Vec<_> = (0..GRASS_PATCH_COUNT)
                .map(|_| [Batch::default(), Batch::default(), Batch::default()])
                .collect();
            for (index, blade) in self.grass.iter().enumerate() {
                let i = grass_patch((blade.a + blade.b) * 0.5);
                let old = previous
                    .and_then(|p| p.grass.as_ref())
                    .map_or(0, |p| p.masks[i]);
                let added = plan.masks[i] & !old;
                if added & 1 != 0 {
                    build_tuft(&mut batches[i][0], blade, index, Detail::Near);
                    build_shadow_tuft(&mut batches[i][2], blade, index);
                }
                if added & 2 != 0 && index % 2 == 0 {
                    // Stable nested roots and directions; a modest wider blade
                    // keeps the distant meadow readable with fewer triangles.
                    build_ribbons_scaled(
                        &mut batches[i][1],
                        blade,
                        index,
                        Detail::Middle,
                        false,
                        1.4,
                    );
                }
            }
            for (i, [near, middle, shadow]) in batches.into_iter().enumerate() {
                for (detail, mut visual, mut proxy) in [
                    (Detail::Near, near, shadow),
                    (Detail::Middle, middle, Batch::default()),
                ] {
                    if visual.positions.is_empty() {
                        continue;
                    }
                    built.vertices += visual.positions.len() + proxy.positions.len();
                    for position in visual
                        .positions
                        .iter_mut()
                        .chain(proxy.positions.iter_mut())
                    {
                        *position = (Vec3::from_array(*position) - plan.centers[i]).to_array();
                    }
                    built.grass_additions.push(GrassLodMesh {
                        index: i,
                        detail,
                        center: plan.centers[i],
                        visual: visual.mesh(),
                        shadow: proxy.mesh(),
                    });
                }
            }
        }
        built.previous_request = previous.cloned();
        built.request = Some(request);
        built.parts_changed = parts_changed;
        built.elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
        built
    }
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
            .map(|b| {
                let root = (b.a + b.b) * 0.5;
                view.pixels((root + b.c) * 0.5 + self.origin, (b.c.y - root.y) * 1.1)
            })
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
                let wood = if s.species == crate::trees::Species::Conifer {
                    tube(3)
                } else {
                    tube(s.trunk.len()) + s.branches.iter().map(|b| tube(b.len())).sum::<usize>()
                };
                wood + s.full_foliage_vertices()
            })
            .sum::<usize>()
            + self.grass.len() * if self.refined_grass { 120 } else { 18 }
            + self.shrubs.len() * 1440
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
        self.build_parts(levels, true, true, true)
    }
    fn build_parts(
        &self,
        levels: Levels,
        include_trees: bool,
        include_parts: bool,
        include_grass: bool,
    ) -> Built {
        let start = Instant::now();
        let mut trees = Batch::default();
        let mut leaves = Batch::default();
        for t in self.trees.iter().filter(|_| include_trees) {
            let phase = (t.shape.seed % 10000) as f32 * std::f32::consts::TAU / 10000.;
            let amplitude = t.shape.foliage_wind_amplitude(levels.trees);
            let height = t.shape.trunk.last().unwrap().0.y;
            trees.append_wind(
                t.shape.wood_mesh_lod(levels.trees),
                t.offset,
                t.offset.y,
                height,
                phase,
                0.,
            );
            let foliage = if t.shape.species == crate::trees::Species::Conifer {
                &mut trees
            } else {
                &mut leaves
            };
            foliage.append_wind(
                t.shape.foliage_mesh(levels.trees),
                t.offset + t.shape.crown_pivot,
                t.offset.y,
                height,
                phase,
                amplitude,
            );
        }
        let mut grass = Batch::default();
        let mut shadow_grass = Batch::default();
        if include_parts && include_grass && levels.grass != Detail::Far {
            let stride = if levels.grass == Detail::Near { 1 } else { 8 };
            for (index, blade) in self.grass.iter().enumerate().step_by(stride) {
                if self.refined_grass {
                    build_tuft(&mut grass, blade, index, levels.grass);
                    if levels.grass == Detail::Near {
                        build_shadow_tuft(&mut shadow_grass, blade, index);
                    }
                } else {
                    build_tuft_legacy(&mut grass, blade, index, levels.grass);
                }
            }
        }
        if include_parts && levels.grass != Detail::Far {
            for shrub in &self.shrubs {
                let mut random = crate::plant_shapes::Random(u64::from(
                    shrub.center.x.to_bits() ^ shrub.center.z.to_bits(),
                ));
                let color = Color::srgb(shrub.color[0], shrub.color[1], shrub.color[2]).to_linear();
                let levels_cards = crate::plant_shapes::leaf_cluster(
                    &mut random,
                    shrub.center + Vec3::Y * shrub.radius * 0.65,
                    Vec3::splat(shrub.radius * 0.65),
                    Quat::IDENTITY,
                    Vec3::new(color.red, color.green, color.blue),
                );
                leaves.append_wind(
                    levels_cards
                        .into_iter()
                        .nth(levels.grass.index())
                        .unwrap()
                        .mesh(),
                    Vec3::ZERO,
                    shrub.center.y,
                    shrub.radius,
                    shrub.center.x * 0.3 + shrub.center.z * 0.2,
                    0.045,
                );
            }
        }
        let vertices = trees.positions.len()
            + leaves.positions.len()
            + grass.positions.len()
            + shadow_grass.positions.len();
        Built {
            trees: trees.mesh(),
            leaves: leaves.mesh(),
            grass: grass.mesh(),
            shadow_grass: shadow_grass.mesh(),
            levels,
            vertices,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.,
            request: None,
            previous_request: None,
            parts_changed: true,
            tree_additions: vec![],
            grass_additions: vec![],
        }
    }
}

// The study's six slender curved ribbons, rooted at the existing seeded tuft.
// Near/Middle retain nested blade directions; Far is represented by ground tint.
pub(crate) fn build_tuft(batch: &mut Batch, blade: &Blade, index: usize, detail: Detail) {
    build_ribbons(batch, blade, index, detail, false);
}
pub(crate) fn build_shadow_tuft(batch: &mut Batch, blade: &Blade, index: usize) {
    build_ribbons(batch, blade, index, Detail::Near, true);
}
fn build_ribbons(batch: &mut Batch, blade: &Blade, index: usize, detail: Detail, shadow: bool) {
    build_ribbons_scaled(batch, blade, index, detail, shadow, 1.);
}
fn build_ribbons_scaled(
    batch: &mut Batch,
    blade: &Blade,
    index: usize,
    detail: Detail,
    shadow: bool,
    width_scale: f32,
) {
    let root = (blade.a + blade.b) * 0.5;
    let height = blade.c.y - root.y;
    let base_side = (blade.b - blade.a).normalize();
    let mut random =
        crate::plant_shapes::Random(u64::from(root.x.to_bits() ^ root.z.to_bits()) ^ index as u64);
    let base = Vec3::from_array([blade.color[0], blade.color[1], blade.color[2]]);
    for i in 0..6 {
        let angle = i as f32 * 2.399 + random.range(-0.25, 0.25);
        let side = Quat::from_rotation_y(angle) * base_side;
        let h = height * random.range(0.70, 1.0);
        let width = random.range(0.025, 0.045) * (height / 0.7).clamp(0.6, 1.4);
        let lean = h * random.range(0.25, 0.65);
        let phase = random.range(0., std::f32::consts::TAU);
        if (detail == Detail::Middle && i % 2 != 0) || (shadow && i != 1 && i != 4) {
            continue;
        }
        let ribbon = crate::plant_shapes::Ribbon {
            root_height: root.extend(h),
            shape: Vec4::new(side.x, side.z, width * width_scale, lean),
            color: base.extend(1.),
        };
        let stations: &[(f32, f32)] = if shadow {
            &[(0., 0.5), (1., 0.04)]
        } else {
            &[(0., 0.5), (0.45, 1.), (0.85, 0.5), (1., 0.04)]
        };
        for pair in stations.windows(2) {
            let [(t0, w0), (t1, w1)] = [pair[0], pair[1]];
            let points = [
                ribbon.point(t0, -w0),
                ribbon.point(t0, w0),
                ribbon.point(t1, w1),
                ribbon.point(t1, -w1),
            ];
            let tint = |t: f32| {
                let c = base * (0.50 + 0.65 * t);
                [c.x, c.y, c.z, 1.]
            };
            let wind = |t: f32| {
                [
                    if height > 0.65 {
                        (h * 0.20).min(0.26) * t * t
                    } else {
                        0.
                    },
                    phase,
                ]
            };
            for [a, b, c] in [[0, 1, 2], [0, 2, 3]] {
                let ts = [t0, t0, t1, t1];
                batch.plant_triangle(
                    [points[a], points[b], points[c]],
                    [tint(ts[a]), tint(ts[b]), tint(ts[c])],
                    [wind(ts[a]), wind(ts[b]), wind(ts[c])],
                );
                batch.plant_normals([
                    ribbon.normal(ts[a]),
                    ribbon.normal(ts[b]),
                    ribbon.normal(ts[c]),
                ]);
                // Root height supports camera-distance fading without CPU rebuilds.
                batch.plant_uvs([[root.y, ts[a]], [root.y, ts[b]], [root.y, ts[c]]]);
            }
        }
    }
}

fn build_tuft_legacy(batch: &mut Batch, blade: &Blade, index: usize, detail: Detail) {
    let root = (blade.a + blade.b) * 0.5;
    let height = blade.c.y - root.y;
    // Tall field grass waves; the short lawn-like ground stays calm.
    let wind = if height > 0.65 {
        (height * 0.10).min(0.14)
    } else {
        0.
    };
    let width = if height > 0.65 { 0.045 } else { 0.022 };
    let side = (blade.b - blade.a).normalize();
    let phase = (index % 29) as f32 * 0.47;
    let tint = |brightness: f32| {
        [
            blade.color[0] * brightness,
            blade.color[1] * brightness,
            blade.color[2] * brightness,
            1.,
        ]
    };
    let count = if detail == Detail::Near { 3 } else { 1 };
    for i in 0..count {
        let rotation = Quat::from_rotation_y(i as f32 * 2.4 + phase * 0.15);
        let sideways = rotation * side;
        let direction = Vec3::new(-sideways.z, 0., sideways.x);
        let h = height * (1. - i as f32 * 0.13);
        let left = root - sideways * width;
        let right = root + sideways * width;
        let lean = direction * h * (0.22 + i as f32 * 0.08);
        let tip = root + Vec3::Y * h + lean;
        if detail == Detail::Near {
            let elbow = root + Vec3::Y * h * 0.57 + lean * 0.26 + sideways * 0.014;
            batch.plant_triangle(
                [left, right, elbow],
                [tint(0.65), tint(0.65), tint(0.95)],
                [[0., phase], [0., phase], [wind * 0.35, phase]],
            );
            batch.plant_triangle(
                [left, elbow, tip],
                [tint(0.65), tint(0.95), tint(1.12)],
                [[0., phase], [wind * 0.35, phase], [wind, phase]],
            );
        } else {
            batch.plant_triangle(
                [left, right, tip],
                [tint(0.65), tint(0.65), tint(1.05)],
                [[0., phase], [0., phase], [wind * 0.5, phase]],
            );
        }
    }
}
pub struct Built {
    pub trees: Mesh,
    pub leaves: Mesh,
    pub shadow_grass: Mesh,
    pub grass: Mesh,
    pub levels: Levels,
    pub vertices: usize,
    pub elapsed_ms: f64,
    pub request: Option<StreamRequest>,
    pub previous_request: Option<StreamRequest>,
    pub parts_changed: bool,
    pub tree_additions: Vec<TreeLodMesh>,
    pub grass_additions: Vec<GrassLodMesh>,
}

impl Built {
    /// A chunk may be reloaded before its old incremental worker finishes.
    pub fn matches_cache(&self, current: &StreamRequest, desired: &StreamRequest) -> bool {
        self.previous_request.as_ref() == Some(current) && self.request.as_ref() == Some(desired)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn grass_scene() -> Blueprint {
        let grass = [
            Vec3::new(8., 3., 8.),
            Vec3::new(9., 3., 8.),
            Vec3::new(88., 3., 88.),
            Vec3::new(89., 3., 88.),
        ]
        .into_iter()
        .map(|root| Blade {
            a: root - Vec3::X * 0.02,
            b: root + Vec3::X * 0.02,
            c: root + Vec3::Y,
            color: [0.2, 0.4, 0.1, 1.],
        })
        .collect();
        Blueprint {
            trees: vec![],
            grass,
            shrubs: vec![],
            origin: Vec3::ZERO,
            refined_grass: true,
        }
    }
    #[test]
    fn near_grass_does_not_promote_distant_patches_and_rotation_keeps_the_cache() {
        let scene = grass_scene();
        let mut view = tree_view(Vec3::new(8., 4., 10.));
        let request = scene.stream_request(Some(view), None);
        let grass = request.grass.as_ref().unwrap();
        assert_eq!(grass.masks[0], 3);
        assert_eq!(grass.masks[35], 0);
        let built = scene.build_streamed(request.clone(), None);
        assert!(built.grass.count_vertices() == 0);
        assert!(built.grass_additions.iter().all(|p| p.index == 0));
        for angle in [0., 1., 3., 5.] {
            view.rotation = Quat::from_rotation_y(angle);
            let rotated = scene.stream_request(Some(view), Some(&request));
            assert_eq!(rotated, request);
            let result = scene.build_streamed(rotated, Some(&request));
            assert!(result.grass_additions.is_empty() && !result.parts_changed);
        }
        let full = scene.stream_request(None, Some(&request));
        assert!(full.grass.as_ref().unwrap().full);
        assert_eq!(full.grass.as_ref().unwrap().masks[35], 1);
    }
    #[test]
    fn grass_cache_prefetch_retention_release_and_loading_fallback_cover_the_bands() {
        let scene = grass_scene();
        let at = |distance: f32, previous: Option<&StreamRequest>| {
            scene.stream_request(Some(tree_view(Vec3::new(8., 3., 8. + distance))), previous)
        };
        let prefetched = at(41., None);
        assert_eq!(prefetched.grass.as_ref().unwrap().masks[0], 3);
        let middle = at(43., None);
        assert_eq!(middle.grass.as_ref().unwrap().masks[0], 2);
        assert_eq!(
            grass_range(middle.grass.as_ref().unwrap(), 0, Detail::Middle)
                .unwrap()
                .start_margin,
            0. ..0.
        );
        let retained = at(49., Some(&prefetched));
        assert_eq!(retained.grass.as_ref().unwrap().masks[0], 3);
        let released = at(51., Some(&retained));
        assert_eq!(released.grass.as_ref().unwrap().masks[0], 2);
        let far = at(93., Some(&released));
        assert_eq!(far.grass.as_ref().unwrap().masks[0], 0);
    }
    #[test]
    fn patch_near_preserves_seeded_geometry_and_complementary_fades() {
        use bevy::mesh::VertexAttributeValues;
        let scene = grass_scene();
        let request = scene.stream_request(Some(tree_view(Vec3::new(8., 4., 10.))), None);
        let plan = request.grass.as_ref().unwrap();
        let near = grass_range(plan, 0, Detail::Near).unwrap();
        let mid = grass_range(plan, 0, Detail::Middle).unwrap();
        assert_eq!(near.end_margin, mid.start_margin);
        assert!(near.end_margin.start < near.end_margin.end);
        let built = scene.build_streamed(request.clone(), None);
        let patch = built
            .grass_additions
            .iter()
            .find(|p| p.detail == Detail::Near)
            .unwrap();
        let mut expected = Batch::default();
        for (i, blade) in scene.grass.iter().take(2).enumerate() {
            build_tuft(&mut expected, blade, i, Detail::Near);
        }
        let expected = expected.mesh();
        for attribute in [
            Mesh::ATTRIBUTE_NORMAL,
            Mesh::ATTRIBUTE_COLOR,
            Mesh::ATTRIBUTE_UV_0,
            Mesh::ATTRIBUTE_UV_1,
        ] {
            assert_eq!(
                patch.visual.attribute(attribute),
                expected.attribute(attribute)
            );
        }
        let Some(VertexAttributeValues::Float32x3(local)) =
            patch.visual.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!()
        };
        let Some(VertexAttributeValues::Float32x3(original)) =
            expected.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!()
        };
        for (a, b) in local.iter().zip(original) {
            assert!((Vec3::from_array(*a) + patch.center).distance(Vec3::from_array(*b)) < 0.00001);
        }
        assert_eq!(patch.visual.count_vertices(), 2 * 36 * 3);
        assert_eq!(patch.shadow.count_vertices(), 2 * 4 * 3);
        let mid = built
            .grass_additions
            .iter()
            .find(|p| p.detail == Detail::Middle)
            .unwrap();
        assert_eq!(mid.visual.count_vertices(), 18 * 3);
        assert_eq!(built.vertices, 2 * (36 + 4) * 3 + 18 * 3);
    }
    fn two_trees() -> Blueprint {
        let world =
            crate::terrain::Meadow::streamed(42, crate::terrain::TerrainSettings::default());
        let trees = [Vec3::ZERO, Vec3::new(88., 0., 88.)]
            .into_iter()
            .enumerate()
            .map(|(i, offset)| {
                let shape = TreeShape::at(
                    &world,
                    Vec2::new(i as f32 * 3., 27.),
                    1.,
                    crate::trees::TreeSettings::default(),
                );
                let mut placed = PlacedTree::new(shape, offset);
                // Controlled common-center diameter isolates policy from seeded shape bounds.
                placed.center = offset;
                placed.diameter = 8.;
                placed
            })
            .collect();
        Blueprint {
            trees,
            grass: vec![],
            shrubs: vec![],
            origin: Vec3::new(960., 0., -480.),
            refined_grass: true,
        }
    }
    fn tree_view(position: Vec3) -> View {
        View {
            position,
            rotation: Quat::IDENTITY,
            fov: 72_f32.to_radians(),
            aspect: 1.6,
            height: 800.,
        }
    }
    #[test]
    fn one_near_tree_does_not_promote_distant_trees_in_the_same_chunk() {
        let data = two_trees();
        let mut view = tree_view(data.origin + Vec3::Z * 8.);
        let plan = data.stream_request(Some(view), None);
        assert_eq!(plan.trees.masks, [7, 4]);
        let initial = data.build_streamed(plan.clone(), None);
        assert_eq!(initial.tree_additions.len(), 4);
        // Orbit in place: all cached handles/geometry remain reusable.
        for angle in [0.4, 1.6, 3.2, 5.8] {
            view.rotation = Quat::from_rotation_y(angle);
            let rotated = data.stream_request(Some(view), Some(&plan));
            assert_eq!(rotated, plan);
            let built = data.build_streamed(rotated, Some(&plan));
            assert!(built.tree_additions.is_empty() && !built.parts_changed);
            assert_eq!(built.vertices, 0);
        }
        let near = data.stream_request(None, Some(&plan));
        assert_eq!(near.trees.masks, [1, 1]);
        assert!(tree_range(8., &near.trees, 0, Detail::Near).is_none());
    }
    #[test]
    fn stale_incremental_result_cannot_install_into_a_reloaded_cache() {
        let data = two_trees();
        let far = data.stream_request(Some(tree_view(data.origin - Vec3::Z * 200.)), None);
        let close = data.stream_request(Some(tree_view(data.origin + Vec3::Z * 8.)), Some(&far));
        let old_worker = data.build_streamed(close.clone(), Some(&far));
        assert!(old_worker.matches_cache(&far, &close));
        // A newly loaded chunk already has these levels. The old Far→Near
        // delta must not duplicate its Near/Middle entities and mesh assets.
        assert!(!old_worker.matches_cache(&close, &close));
    }
    #[test]
    fn tree_cache_prefetch_is_incremental_and_retreat_has_hysteresis() {
        let data = two_trees();
        let scale = 800. / (2. * (72_f32.to_radians() * 0.5).tan());
        let (near, middle) = tree_bands(8., scale);
        let plan_at = |distance, previous| {
            data.stream_request(Some(tree_view(data.origin - Vec3::Z * distance)), previous)
        };
        let far = plan_at(200., None);
        let mid = plan_at(middle.end + 11., Some(&far));
        assert_eq!(mid.trees.masks[0], 6);
        let built = data.build_streamed(mid.clone(), Some(&far));
        assert!(!built.parts_changed);
        assert_eq!(built.tree_additions.len(), 1);
        assert_eq!(built.tree_additions[0].detail, Detail::Middle);
        let close = plan_at(near.end + 11., Some(&mid));
        let built = data.build_streamed(close.clone(), Some(&mid));
        assert_eq!(built.tree_additions.len(), 1);
        assert_eq!(built.tree_additions[0].detail, Detail::Near);
        let retained = plan_at(near.end + 19., Some(&close));
        assert_eq!(retained.trees.masks[0], 7);
        let released = plan_at(near.end + 21., Some(&retained));
        assert_eq!(released.trees.masks[0], 6);
        assert!(
            data.build_streamed(released, Some(&retained))
                .tree_additions
                .is_empty()
        );
        assert_eq!(plan_at(200., Some(&mid)).trees.masks[0], 4);
    }
    #[test]
    fn tree_fades_have_complementary_weights_and_coarse_loading_fallback() {
        let plan = TreePlan {
            masks: vec![7],
            projection_scale: Some(550.),
        };
        let weight = |range: Option<VisibilityRange>, d: f32| {
            range.map_or(1., |r| {
                let fade = |m: std::ops::Range<f32>| {
                    if m.end == m.start {
                        if d >= m.start { 1. } else { 0. }
                    } else {
                        ((d - m.start) / (m.end - m.start)).clamp(0., 1.)
                    }
                };
                fade(r.start_margin) * (1. - fade(r.end_margin))
            })
        };
        for mask in [7, 6, 4] {
            let mut p = plan.clone();
            p.masks[0] = mask;
            for i in 0..2000 {
                let d = i as f32 * 0.1;
                let weights: Vec<_> = [Detail::Near, Detail::Middle, Detail::Far]
                    .into_iter()
                    .filter(|level| mask & (1 << level.index()) != 0)
                    .map(|level| weight(tree_range(8., &p, 0, level), d))
                    .collect();
                assert!((weights.iter().sum::<f32>() - 1.).abs() < 1e-5);
                assert!(weights.iter().filter(|w| **w > 0.).count() <= 2);
            }
        }
    }

    #[test]
    fn only_broadleaf_foliage_sways_and_all_wood_and_conifers_stay_rigid() {
        use crate::trees::{Species, TreeSettings};
        use bevy::mesh::VertexAttributeValues;
        let world =
            crate::terrain::Meadow::streamed(42, crate::terrain::TerrainSettings::default());
        for conifer in [false, true] {
            let shape = (0..2000)
                .map(|i| {
                    TreeShape::at(
                        &world,
                        Vec2::new(i as f32 * 3., 27.),
                        1.,
                        TreeSettings::default(),
                    )
                })
                .find(|shape| (shape.species == Species::Conifer) == conifer)
                .unwrap();
            let blueprint = Blueprint {
                trees: vec![PlacedTree::new(shape, Vec3::new(8., 13., 5.))],
                grass: vec![],
                shrubs: vec![],
                origin: Vec3::ZERO,
                refined_grass: true,
            };
            for detail in [Detail::Near, Detail::Middle, Detail::Far] {
                let built = blueprint.build(Levels {
                    trees: detail,
                    grass: detail,
                });
                // Per-tree meshes use crown-local coordinates but preserve
                // the accepted finite/batched geometry and GPU wind inputs.
                let streamed = blueprint.build_streamed(
                    StreamRequest {
                        levels: built.levels,
                        grass: None,
                        trees: TreePlan {
                            masks: vec![1 << detail.index()],
                            projection_scale: None,
                        },
                    },
                    None,
                );
                let pair = &streamed.tree_additions[0];
                assert_eq!(pair.center, blueprint.trees[0].center);
                for (original, local) in [(&built.trees, &pair.wood), (&built.leaves, &pair.leaves)]
                {
                    assert_eq!(original.count_vertices(), local.count_vertices());
                    if original.count_vertices() == 0 {
                        continue;
                    }
                    for attribute in [
                        Mesh::ATTRIBUTE_NORMAL,
                        Mesh::ATTRIBUTE_UV_0,
                        Mesh::ATTRIBUTE_COLOR,
                    ] {
                        assert_eq!(original.attribute(attribute), local.attribute(attribute));
                    }
                    let Some(VertexAttributeValues::Float32x2(a)) =
                        original.attribute(Mesh::ATTRIBUTE_UV_1)
                    else {
                        panic!()
                    };
                    let Some(VertexAttributeValues::Float32x2(b)) =
                        local.attribute(Mesh::ATTRIBUTE_UV_1)
                    else {
                        panic!()
                    };
                    // Translation changes floating-point height subtraction
                    // slightly; amplitude/phase must stay equivalent.
                    assert!(
                        a.iter()
                            .zip(b)
                            .all(|(a, b)| (a[0] - b[0]).abs() < 1e-6 && a[1] == b[1])
                    );
                    let Some(VertexAttributeValues::Float32x3(a)) =
                        original.attribute(Mesh::ATTRIBUTE_POSITION)
                    else {
                        panic!()
                    };
                    let Some(VertexAttributeValues::Float32x3(b)) =
                        local.attribute(Mesh::ATTRIBUTE_POSITION)
                    else {
                        panic!()
                    };
                    assert!(a.iter().zip(b).all(|(a, b)| {
                        Vec3::from(*a).distance(Vec3::from(*b) + pair.center) < 1e-5
                    }));
                }
                let Some(VertexAttributeValues::Float32x2(wood_wind)) =
                    built.trees.attribute(Mesh::ATTRIBUTE_UV_1)
                else {
                    panic!("Missing wood/conifer wind data")
                };
                assert!(!wood_wind.is_empty());
                assert!(wood_wind.iter().all(|wind| wind[0] == 0.));
                if conifer {
                    assert_eq!(built.leaves.count_vertices(), 0);
                } else {
                    let Some(VertexAttributeValues::Float32x2(leaf_wind)) =
                        built.leaves.attribute(Mesh::ATTRIBUTE_UV_1)
                    else {
                        panic!("Missing broadleaf wind data")
                    };
                    assert_eq!(
                        leaf_wind.iter().any(|wind| wind[0] > 0.),
                        detail != Detail::Far,
                    );
                    assert!(leaf_wind.iter().all(|wind| wind[0] <= 0.045));
                }
            }
        }
    }

    #[test]
    fn rounded_leaves_have_bounded_near_cost_blunt_tips_and_smooth_bends() {
        use bevy::mesh::VertexAttributeValues;
        let blade = Blade {
            a: -Vec3::X * 0.09,
            b: Vec3::X * 0.09,
            c: Vec3::Y * 1.1,
            color: [0.2, 0.35, 0.1, 1.],
        };
        let mut blueprint = Blueprint {
            trees: vec![],
            grass: vec![blade],
            shrubs: vec![],
            origin: Vec3::ZERO,
            refined_grass: true,
        };
        for detail in [Detail::Near, Detail::Middle, Detail::Far] {
            let levels = Levels {
                trees: detail,
                grass: detail,
            };
            blueprint.refined_grass = true;
            let refined = blueprint.build(levels);
            blueprint.refined_grass = false;
            let legacy = blueprint.build(levels);
            assert_eq!(
                refined.grass.count_vertices(),
                legacy.grass.count_vertices() * if detail == Detail::Near { 6 } else { 18 }
            );
            if detail == Detail::Far {
                continue;
            }
            let Some(VertexAttributeValues::Float32x3(normals)) =
                refined.grass.attribute(Mesh::ATTRIBUTE_NORMAL)
            else {
                panic!()
            };
            assert!(
                normals
                    .iter()
                    .all(|n| (Vec3::from(*n).length() - 1.).abs() < 1e-5 && n[1] > 0.)
            );
            if detail == Detail::Near {
                let Some(VertexAttributeValues::Float32x3(p)) =
                    refined.grass.attribute(Mesh::ATTRIBUTE_POSITION)
                else {
                    panic!()
                };
                assert_eq!(p.len(), 108);
                for leaf in p.chunks_exact(18) {
                    let mut widths = Vec::new();
                    let mut centers = Vec::new();
                    for band in leaf.chunks_exact(6) {
                        assert_eq!(band[0], band[3]);
                        assert_eq!(band[2], band[4]);
                        widths.push(Vec3::from(band[1]).distance(Vec3::from(band[0])));
                        centers.push((Vec3::from(band[0]) + Vec3::from(band[1])) * 0.5);
                    }
                    let tip_width = Vec3::from(leaf[14]).distance(Vec3::from(leaf[17]));
                    assert!((0.001..0.006).contains(&tip_width));
                    assert!(widths[0] < widths[1] && widths[2] < widths[1]);
                    let tip = (Vec3::from(leaf[14]) + Vec3::from(leaf[17])) * 0.5;
                    assert!(Vec2::new(tip.x, tip.z).length() > 0.15);
                    centers.push(tip);
                    let slopes: Vec<_> = centers
                        .windows(2)
                        .map(|w| {
                            let delta = w[1] - w[0];
                            Vec2::new(delta.x, delta.z).length() / delta.y
                        })
                        .collect();
                    assert!(slopes.windows(2).all(|w| w[1] > w[0]));
                    assert!(centers.windows(2).all(|w| w[1].y > w[0].y));
                }
            }
        }
    }

    #[test]
    fn tufts_keep_roots_fixed_bend_above_ground_and_reduce_at_distance() {
        use bevy::mesh::VertexAttributeValues;
        let root = Vec3::new(5., 17., -8.);
        let blade = Blade {
            a: root - Vec3::X * 0.09,
            b: root + Vec3::X * 0.09,
            c: root + Vec3::Y * 1.1,
            color: [0.2, 0.35, 0.1, 1.],
        };
        let blueprint = Blueprint {
            trees: vec![],
            grass: vec![blade],
            shrubs: vec![],
            origin: Vec3::ZERO,
            refined_grass: true,
        };
        let near = blueprint.build(Levels::FULL);
        let middle = blueprint.build(Levels {
            trees: Detail::Middle,
            grass: Detail::Middle,
        });
        let far = blueprint.build(Levels {
            trees: Detail::Far,
            grass: Detail::Far,
        });
        assert_eq!(near.vertices, blueprint.full_vertices());
        assert!(middle.vertices < near.vertices && far.vertices == 0);
        for built in [near, middle] {
            let Some(VertexAttributeValues::Float32x3(points)) =
                built.grass.attribute(Mesh::ATTRIBUTE_POSITION)
            else {
                panic!()
            };
            let Some(VertexAttributeValues::Float32x2(wind)) =
                built.grass.attribute(Mesh::ATTRIBUTE_UV_1)
            else {
                panic!()
            };
            let Some(VertexAttributeValues::Float32x4(colors)) =
                built.grass.attribute(Mesh::ATTRIBUTE_COLOR)
            else {
                panic!()
            };
            let bounds = crate::wind::bounds(&built.grass);
            let mut fixed = 0;
            let mut moving = 0;
            for ((p, w), c) in points.iter().zip(wind).zip(colors) {
                assert!(p.iter().chain(w).chain(c).all(|v| v.is_finite()));
                assert!(p[1] >= root.y && w[0] <= 0.26);
                if p[1] == root.y {
                    assert_eq!(w[0], 0.);
                    fixed += 1;
                } else {
                    assert!(w[0] > 0.);
                    moving += 1;
                }
                // Conservative culling includes either extreme of the GPU displacement.
                for sign in [-1., 1.] {
                    let shifted =
                        Vec3A::from(Vec3::from(*p)) + Vec3A::new(sign * 0.20, 0., sign * 0.20);
                    assert!(
                        (shifted - bounds.center)
                            .abs()
                            .cmple(bounds.half_extents + Vec3A::splat(0.0001))
                            .all()
                    );
                }
            }
            assert!(fixed > 0 && moving > 0);
            assert!(
                points
                    .iter()
                    .any(|p| p[1] > root.y
                        && Vec2::new(p[0] - root.x, p[2] - root.z).length() > 0.02)
            );
            let dark = colors.iter().map(|c| c[1]).fold(f32::INFINITY, f32::min);
            let light = colors.iter().map(|c| c[1]).fold(0_f32, f32::max);
            assert!(light > dark * 1.3);
        }
        let mut short = blueprint;
        short.grass[0].c = root + Vec3::Y * 0.45;
        let calm = short.build(Levels::FULL);
        let Some(VertexAttributeValues::Float32x2(wind)) =
            calm.grass.attribute(Mesh::ATTRIBUTE_UV_1)
        else {
            panic!()
        };
        assert!(wind.iter().all(|uv| uv[0] == 0.));
    }
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
