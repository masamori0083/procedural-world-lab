//! Shared Rust-generated leaf masks, seeded leaf cards and curved grass shapes.
use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, PrimitiveTopology, TextureDimension, TextureFormat},
};
pub(crate) const LEAF_PIXELS: u32 = 256;
pub(crate) struct Random(pub(crate) u64);
impl Random {
    pub(crate) fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / 16777216.
    }
    pub(crate) fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.unit()
    }
    pub(crate) fn ball(&mut self) -> Vec3 {
        loop {
            let p = Vec3::new(
                self.range(-1., 1.),
                self.range(-1., 1.),
                self.range(-1., 1.),
            );
            if p.length_squared() < 1. && p.length_squared() > 0.001 {
                return p;
            }
        }
    }
}
#[derive(Default)]
pub(crate) struct Geometry {
    pub(crate) positions: Vec<[f32; 3]>,
    pub(crate) normals: Vec<[f32; 3]>,
    pub(crate) colors: Vec<[f32; 4]>,
    pub(crate) uvs: Vec<[f32; 2]>,
}
impl Geometry {
    pub(crate) fn triangle(
        &mut self,
        points: [Vec3; 3],
        normals: [Vec3; 3],
        colors: [Vec3; 3],
        uvs: [[f32; 2]; 3],
    ) {
        self.positions.extend(points.map(|point| point.to_array()));
        self.normals
            .extend(normals.map(|n| n.normalize().to_array()));
        self.colors.extend(colors.map(|c| [c.x, c.y, c.z, 1.]));
        self.uvs.extend(uvs);
    }
    pub(crate) fn quad(&mut self, p: [Vec3; 4], n: [Vec3; 4], c: [Vec3; 4], uv: [[f32; 2]; 4]) {
        for [a, b, d] in [[0, 1, 2], [0, 2, 3]] {
            self.triangle(
                [p[a], p[b], p[d]],
                [n[a], n[b], n[d]],
                [c[a], c[b], c[d]],
                [uv[a], uv[b], uv[d]],
            );
        }
    }
    pub(crate) fn mesh(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs)
    }
}

pub(crate) fn leaf_texture(seed: u32, tint: Vec3) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: LEAF_PIXELS,
            height: LEAF_PIXELS,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    let bytes = image.data.as_mut().unwrap();
    for y in 0..LEAF_PIXELS {
        for x in 0..LEAF_PIXELS {
            let t = (y as f32 + 0.5) / LEAF_PIXELS as f32;
            let u = (x as f32 + 0.5) / LEAF_PIXELS as f32 * 2. - 1.;
            let center = 0.06 * (t * 4.).sin();
            let width =
                (t * std::f32::consts::PI).sin().powf(0.65) * 0.80 * (1. + 0.018 * (t * 90.).sin());
            let coverage =
                ((width - (u - center).abs()) * LEAF_PIXELS as f32 * 0.5 + 0.5).clamp(0., 1.);
            let noise = ((x * 31 + y * 17).wrapping_add(seed) % 19) as f32 / 19.;
            let vein = (1. - ((u - center).abs() / 0.018)).clamp(0., 1.);
            let shade = 0.80 + 0.15 * t + 0.04 * noise + 0.04 * vein;
            let color = tint * shade;
            let i = ((y * LEAF_PIXELS + x) * 4) as usize;
            bytes[i..i + 4].copy_from_slice(&[
                (color.x * 255.) as u8,
                (color.y * 255.) as u8,
                (color.z * 255.) as u8,
                (coverage * 255.) as u8,
            ]);
        }
    }
    image
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ribbon {
    pub(crate) root_height: Vec4,
    pub(crate) shape: Vec4,
    pub(crate) color: Vec4,
}
impl Ribbon {
    pub(crate) fn point(self, t: f32, side: f32) -> Vec3 {
        let sideways = Vec3::new(self.shape.x, 0., self.shape.y);
        let forward = Vec3::new(-self.shape.y, 0., self.shape.x);
        self.root_height.truncate()
            + Vec3::Y * self.root_height.w * t
            + forward * self.shape.w * t * t
            + sideways * self.shape.z * side
    }
    pub(crate) fn normal(self, t: f32) -> Vec3 {
        (Vec3::new(-self.shape.y, 0., self.shape.x) + Vec3::Y * (0.3 + t)).normalize()
    }
}
impl Geometry {
    pub(crate) fn append(&mut self, other: Geometry) {
        self.positions.extend(other.positions);
        self.normals.extend(other.normals);
        self.colors.extend(other.colors);
        self.uvs.extend(other.uvs);
    }
}
/// Same nested card layout as the study: 240/80/27 cards per seeded envelope.
pub(crate) fn leaf_cluster(
    random: &mut Random,
    center: Vec3,
    radii: Vec3,
    rotation: Quat,
    tint: Vec3,
) -> [Geometry; 3] {
    let mut out = std::array::from_fn(|_| Geometry::default());
    for leaf in 0..240 {
        let offset = random.ball() * radii;
        let position = center + rotation * offset;
        let normal = rotation * (offset.normalize() + Vec3::Y * 0.65).normalize();
        let side = normal.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = normal.cross(side);
        let turn = random.range(0., std::f32::consts::TAU);
        let side = side * turn.cos() + up * turn.sin();
        let up = normal.cross(side);
        let unit = radii.min_element();
        let width = random.range(0.10, 0.16) * unit;
        let height = random.range(0.14, 0.22) * unit;
        let color = tint
            * Vec3::new(
                random.range(0.88, 1.04),
                random.range(0.94, 1.08),
                random.range(0.88, 1.0),
            );
        for (level, stride, scale) in [(0, 1, 1.), (1, 3, 1.45), (2, 9, 2.45)] {
            if leaf % stride != 0 {
                continue;
            }
            let w = width * scale;
            let h = height * scale;
            out[level].quad(
                [
                    position - side * w - up * h,
                    position + side * w - up * h,
                    position + side * w + up * h,
                    position - side * w + up * h,
                ],
                [normal; 4],
                [color; 4],
                [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_leaf_cards_keep_centers_uvs_and_seeded_nested_subsets() {
        let build = || {
            leaf_cluster(
                &mut Random(42),
                Vec3::Y * 4.,
                Vec3::new(1., 1.05, 1.),
                Quat::IDENTITY,
                Vec3::ONE,
            )
        };
        let levels = build();
        let repeated = build();
        assert_eq!(
            levels.iter().map(|g| g.positions.len()).collect::<Vec<_>>(),
            [1440, 480, 162]
        );
        for (level, geometry) in levels.iter().enumerate() {
            assert_eq!(geometry.positions, repeated[level].positions);
            assert_eq!(geometry.uvs, repeated[level].uvs);
            for (index, card) in geometry.positions.chunks_exact(6).enumerate() {
                let center = (Vec3::from(card[0]) + Vec3::from(card[2])) * 0.5;
                let stride = [1, 3, 9][level];
                let original = &levels[0].positions[index * stride * 6..];
                let near_center = (Vec3::from(original[0]) + Vec3::from(original[2])) * 0.5;
                assert!(center.distance(near_center) < 1e-5);
                assert_eq!(&geometry.uvs[index * 6..index * 6 + 6], &levels[0].uvs[..6]);
            }
        }
    }
}
