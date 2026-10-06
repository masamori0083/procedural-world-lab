//! Flow is a moving texture on a fixed downhill surface, not simulated fluid.
use crate::terrain::Meadow;
use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    math::Affine2,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

#[derive(Resource)]
pub struct RiverMaterial(pub Handle<StandardMaterial>);

pub fn setup(
    mut commands: Commands,
    world: Res<Meadow>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut lab: ResMut<crate::LabState>,
) {
    let start = std::time::Instant::now();
    let mut data = Vec::with_capacity(128 * 128 * 4);
    for y in 0..128 {
        for x in 0..128 {
            let u = x as f32 / 128.0;
            let v = y as f32 / 128.0;
            let wave = (std::f32::consts::TAU
                * (v * 3.0 + 0.13 * (u * std::f32::consts::TAU * 2.0).sin()))
            .sin();
            let foam =
                wave.max(0.0).powi(16) * (0.35 + 0.65 * (u * std::f32::consts::PI).sin().powi(2));
            let color =
                Vec3::new(35.0, 114.0, 140.0).lerp(Vec3::new(143.0, 204.0, 207.0), foam * 0.75);
            data.extend([color.x as u8, color.y as u8, color.z as u8, 255]);
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        ..ImageSamplerDescriptor::linear()
    });
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(image)),
        perceptual_roughness: 0.27,
        reflectance: 0.4,
        ..default()
    });
    if !world.is_streaming() {
        commands.spawn((
            Name::new("River network / tributaries to map outlet"),
            Mesh3d(meshes.add(world.river_mesh())),
            MeshMaterial3d(material.clone()),
            Transform::default(),
            bevy::light::NotShadowCaster,
        ));
    }
    commands.insert_resource(RiverMaterial(material));
    let river_ms = start.elapsed().as_secs_f64() * 1000.0;
    lab.scene_setup_ms += river_ms;
    info!(
        "River render setup {:.2} ms; total CPU scene setup {:.2} ms (excludes GPU upload / model loading)",
        river_ms, lab.scene_setup_ms
    );
}

pub fn flow_offset(seconds: f32) -> Vec2 {
    Vec2::new(0.0, -(seconds * 0.8 / 6.0).rem_euclid(1.0))
}

pub fn animate(
    time: Res<Time>,
    river: Option<Res<RiverMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(river) = river else {
        return;
    };
    if let Some(mut material) = materials.get_mut(&river.0) {
        material.uv_transform = Affine2::from_translation(flow_offset(time.elapsed_secs()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_texture_feature_travels_toward_increasing_downstream_coordinate() {
        // StandardMaterial samples uv + offset. Increasing flow is downstream;
        // negative translation therefore moves a fixed feature downstream.
        let upstream = -17.3_f32;
        let downstream = -12.4_f32;
        let time = 0.37;
        let travel_time = (downstream - upstream) / 0.8;
        let a = (upstream / 6. + flow_offset(time).y).rem_euclid(1.);
        let b = (downstream / 6. + flow_offset(time + travel_time).y).rem_euclid(1.);
        assert!((a - b).abs() < 0.00001);
        let wrong = (upstream / 6. + flow_offset(time + travel_time).y).rem_euclid(1.);
        assert!((a - wrong).abs() > 0.1);
    }
}
