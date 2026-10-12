//! Shared GPU wind for chunk-batched vegetation.
use bevy::{
    asset::RenderAssetUsages,
    camera::primitives::{Aabb, MeshAabb},
    image::ImageSampler,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline},
    prelude::*,
    render::{
        mesh::MeshVertexBufferLayoutRef,
        render_resource::{
            AsBindGroup, Extent3d, RenderPipelineDescriptor, SpecializedMeshPipelineError,
            TextureDimension, TextureFormat,
        },
    },
    shader::ShaderDefVal,
    shader::ShaderRef,
};

pub type VegetationMaterial = ExtendedMaterial<StandardMaterial, Wind>;

#[derive(Resource)]
pub struct Config {
    pub enabled: bool,
    pub fast_grass_shadows: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            fast_grass_shadows: true,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
#[bind_group_data(WindShaderKey)]
pub struct Wind {
    #[uniform(100)]
    pub parameters: Vec4,
    fast_shadows: bool,
    pub(crate) full_lighting: bool,
    local_lighting: bool,
}
impl Default for Wind {
    fn default() -> Self {
        Self {
            parameters: Vec4::new(1., 0., 0., 0.),
            fast_shadows: false,
            full_lighting: false,
            local_lighting: false,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindShaderKey {
    fast_shadows: bool,
    grass_lighting: bool,
}
impl From<&Wind> for WindShaderKey {
    fn from(wind: &Wind) -> Self {
        Self {
            fast_shadows: wind.fast_shadows,
            grass_lighting: wind.parameters.y == 2. && !wind.full_lighting && !wind.local_lighting,
        }
    }
}
fn grass_shadow_defs(defs: &mut Vec<ShaderDefVal>) {
    defs.retain(|def| !matches!(def, ShaderDefVal::Bool(name, _) if name.starts_with("SHADOW_FILTER_METHOD_")));
    defs.push("SHADOW_FILTER_METHOD_HARDWARE_2X2".into());
}
#[cfg(any(target_arch = "wasm32", test))]
fn preserve_lod_depth_discard(
    descriptor: &mut RenderPipelineDescriptor,
    mesh_key: bevy::pbr::MeshPipelineKey,
) {
    use bevy::pbr::MeshPipelineKey as Key;
    // Depth-only opaque LOD omits its fragment, and the standard alpha-mask
    // depth fragment omits visibility dither. Both must match forward coverage
    // so discarded pixels cannot occlude the complementary LOD or ground.
    if mesh_key.contains(Key::DEPTH_PREPASS | Key::VISIBILITY_RANGE_DITHER)
        && !mesh_key.intersects(
            Key::NORMAL_PREPASS
                | Key::MOTION_VECTOR_PREPASS
                | Key::DEFERRED_PREPASS
                | Key::UNCLIPPED_DEPTH_ORTHO,
        )
        && descriptor
            .fragment
            .as_ref()
            .is_none_or(|fragment| fragment.targets.is_empty())
    {
        descriptor.fragment = Some(bevy::render::render_resource::FragmentState {
            shader: descriptor.vertex.shader.clone(),
            shader_defs: descriptor.vertex.shader_defs.clone(),
            entry_point: Some("lod_depth_fragment".into()),
            targets: Vec::new(),
        });
    }
}

impl MaterialExtension for Wind {
    fn vertex_shader() -> ShaderRef {
        "shaders/vegetation_wind.wgsl".into()
    }
    fn prepass_vertex_shader() -> ShaderRef {
        Self::vertex_shader()
    }
    fn fragment_shader() -> ShaderRef {
        Self::vertex_shader()
    }
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        #[cfg(target_arch = "wasm32")]
        preserve_lod_depth_discard(descriptor, key.mesh_key);
        // Dense close-up grass pays for shadow filtering at every visible
        // pixel. Keep reception/transmission, but use the study's cheaper
        // filter for grass only. Trees and ground keep the camera's filter.
        if key.bind_group_data.fast_shadows
            && let Some(fragment) = descriptor.fragment.as_mut()
        {
            grass_shadow_defs(&mut fragment.shader_defs);
        }
        if key.bind_group_data.grass_lighting
            && let Some(fragment) = descriptor.fragment.as_mut()
        {
            fragment.shader_defs.push("GRASS_DIFFUSE_LIGHTING".into());
        }
        Ok(())
    }
}

#[derive(Resource)]
pub struct Materials {
    pub wood: Handle<VegetationMaterial>,
    pub leaves: Handle<VegetationMaterial>,
    pub grass: Handle<VegetationMaterial>,
    pub shadow_grass: Handle<VegetationMaterial>,
}
// One tiny generated lookup, shared by drawing and depth/shadow passes.
// X is remaining coverage; Y is the existing seeded blade phase. An odd
// permutation distributes thresholds evenly without adding vertex attributes.
fn grass_fade_mask() -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: 128,
            height: 64,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[255, 255, 255, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::nearest();
    let data = image.data.as_mut().unwrap();
    for row in 0..64usize {
        let threshold = ((row * 37) % 64) * 2 + 1;
        for x in 0..128usize {
            data[(row * 128 + x) * 4 + 3] = if x >= threshold { 255 } else { 0 };
        }
    }
    image
}
pub fn setup(
    mut commands: Commands,
    config: Res<Config>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<VegetationMaterial>>,
) {
    let texture = images.add(crate::plant_shapes::leaf_texture(42, Vec3::ONE));
    let grass_mask = images.add(grass_fade_mask());
    let material = |kind: f32, base: StandardMaterial| VegetationMaterial {
        base,
        extension: Wind {
            parameters: Vec4::new(f32::from(config.enabled), kind, 0., 0.),
            fast_shadows: kind == 2. && config.fast_grass_shadows,
            full_lighting: false,
            local_lighting: false,
        },
    };
    let wood = materials.add(material(
        0.,
        StandardMaterial {
            perceptual_roughness: 1.,
            ..default()
        },
    ));
    let leaves = materials.add(material(
        1.,
        StandardMaterial {
            base_color_texture: Some(texture),
            alpha_mode: AlphaMode::Mask(0.5),
            cull_mode: None,
            double_sided: true,
            perceptual_roughness: 1.,
            diffuse_transmission: 0.3,
            reflectance: 0.15,
            ..default()
        },
    ));
    let grass = materials.add(material(
        2.,
        StandardMaterial {
            base_color_texture: Some(grass_mask.clone()),
            alpha_mode: AlphaMode::Mask(0.5),
            cull_mode: None,
            double_sided: true,
            perceptual_roughness: 1.,
            diffuse_transmission: 0.22,
            thickness: 0.02,
            reflectance: 0.15,
            ..default()
        },
    ));
    let shadow_grass = materials.add(material(
        3.,
        StandardMaterial {
            base_color_texture: Some(grass_mask),
            alpha_mode: AlphaMode::Mask(0.5),
            cull_mode: None,
            double_sided: true,
            perceptual_roughness: 1.,
            ..default()
        },
    ));
    commands.insert_resource(Materials {
        wood,
        leaves,
        grass,
        shadow_grass,
    });
}
// Wind uses the GPU clock. Recompile grass only when the lighting model needs
// to change, not per pixel/frame or when the camera rotates.
#[allow(clippy::type_complexity)]
pub fn animate(
    config: Res<Config>,
    lights: Query<
        Entity,
        Or<(
            With<PointLight>,
            With<SpotLight>,
            With<bevy::light::RectLight>,
            With<bevy::light::LightProbe>,
            With<bevy::light::EnvironmentMapLight>,
            With<bevy::light::GeneratedEnvironmentMapLight>,
            With<bevy::light::IrradianceVolume>,
        )>,
    >,
    mut previous_lighting: Local<Option<bool>>,
    mut materials: ResMut<Assets<VegetationMaterial>>,
) {
    let local_lighting = !lights.is_empty();
    let config_changed = config.is_changed();
    if !config_changed && *previous_lighting == Some(local_lighting) {
        return;
    }
    *previous_lighting = Some(local_lighting);
    let changed: Vec<_> = materials
        .iter()
        .filter(|(_, material)| {
            config_changed
                || (material.extension.parameters.y == 2.
                    && material.extension.local_lighting != local_lighting)
        })
        .map(|(id, _)| id)
        .collect();
    for id in changed {
        let mut material = materials.get_mut(id).unwrap();
        material.extension.parameters.x = f32::from(config.enabled);
        material.extension.fast_shadows =
            material.extension.parameters.y == 2. && config.fast_grass_shadows;
        material.extension.local_lighting = material.extension.parameters.y == 2. && local_lighting;
    }
}

// The GPU can move a vertex by at most 0.26m, including tree crowns. Keep
// render/shadow culling conservative without disabling chunk culling.
pub fn bounds(mesh: &Mesh) -> Aabb {
    let mut bounds = mesh.compute_aabb().unwrap_or_default();
    bounds.half_extents += Vec3A::splat(0.26);
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opaque_lod_depth_prepass_must_discard_the_same_pixels_as_forward() {
        use bevy::pbr::MeshPipelineKey as Key;
        let key = Key::DEPTH_PREPASS | Key::VISIBILITY_RANGE_DITHER;
        let mut descriptor = RenderPipelineDescriptor::default();
        descriptor.vertex.shader_defs =
            vec!["PREPASS_PIPELINE".into(), "VISIBILITY_RANGE_DITHER".into()];
        preserve_lod_depth_discard(&mut descriptor, key);
        let fragment = descriptor
            .fragment
            .as_ref()
            .expect("LOD depth writes must run the dither fragment");
        assert_eq!(fragment.entry_point.as_deref(), Some("lod_depth_fragment"));
        assert!(fragment.targets.is_empty());
        assert_eq!(fragment.shader, descriptor.vertex.shader);
        assert_eq!(fragment.shader_defs, descriptor.vertex.shader_defs);
        for excluded in [
            Key::NONE,
            Key::DEPTH_PREPASS,
            Key::VISIBILITY_RANGE_DITHER,
            key | Key::UNCLIPPED_DEPTH_ORTHO,
        ] {
            let mut other = RenderPipelineDescriptor::default();
            preserve_lod_depth_discard(&mut other, excluded);
            assert!(other.fragment.is_none());
        }
        let original = descriptor.clone();
        preserve_lod_depth_discard(&mut descriptor, key);
        assert_eq!(
            descriptor, original,
            "depth discard specialization must be idempotent"
        );
    }
    #[test]
    fn alpha_mask_lod_depth_prepass_must_keep_alpha_and_dither_discard() {
        use bevy::pbr::MeshPipelineKey as Key;
        let key = Key::DEPTH_PREPASS | Key::VISIBILITY_RANGE_DITHER | Key::MAY_DISCARD;
        let mut descriptor = RenderPipelineDescriptor::default();
        descriptor.vertex.shader_defs = vec![
            "PREPASS_PIPELINE".into(),
            "VISIBILITY_RANGE_DITHER".into(),
            "MAY_DISCARD".into(),
        ];
        descriptor.fragment = Some(bevy::render::render_resource::FragmentState {
            entry_point: Some("fragment".into()),
            targets: Vec::new(),
            ..default()
        });
        preserve_lod_depth_discard(&mut descriptor, key);
        assert_eq!(
            descriptor.fragment.as_ref().unwrap().entry_point.as_deref(),
            Some("lod_depth_fragment")
        );
        assert_eq!(
            descriptor.fragment.as_ref().unwrap().shader_defs,
            descriptor.vertex.shader_defs
        );
        for excluded in [
            key | Key::NORMAL_PREPASS,
            key | Key::MOTION_VECTOR_PREPASS,
            key | Key::DEFERRED_PREPASS,
            key | Key::UNCLIPPED_DEPTH_ORTHO,
        ] {
            let mut other = RenderPipelineDescriptor {
                fragment: Some(bevy::render::render_resource::FragmentState {
                    entry_point: Some("fragment".into()),
                    ..default()
                }),
                ..default()
            };
            let original = other.clone();
            preserve_lod_depth_discard(&mut other, excluded);
            assert_eq!(other, original);
        }
    }
    #[test]
    fn diffuse_lighting_is_grass_only_and_pbr_comparison_remains_available() {
        let mut app = App::new();
        app.init_resource::<Config>()
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<VegetationMaterial>>()
            .add_systems(Startup, setup);
        app.update();
        let shared = app.world().resource::<Materials>();
        let materials = app.world().resource::<Assets<VegetationMaterial>>();
        for (handle, expected) in [
            (&shared.grass, true),
            (&shared.leaves, false),
            (&shared.wood, false),
            (&shared.shadow_grass, false),
        ] {
            assert_eq!(
                WindShaderKey::from(&materials.get(handle).unwrap().extension).grass_lighting,
                expected
            );
        }
        let mut comparison = materials.get(&shared.grass).unwrap().extension.clone();
        comparison.full_lighting = true;
        assert!(!WindShaderKey::from(&comparison).grass_lighting);
        assert!(WindShaderKey::from(&comparison).fast_shadows);
    }
    #[test]
    fn local_lights_restore_pbr_without_rebinding_during_camera_or_light_motion() {
        let mut app = App::new();
        app.init_resource::<Config>()
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<VegetationMaterial>>()
            .add_systems(Startup, setup)
            .add_systems(Update, animate);
        app.update();
        let grass = app.world().resource::<Materials>().grass.clone();
        let key = |app: &App| {
            WindShaderKey::from(
                &app.world()
                    .resource::<Assets<VegetationMaterial>>()
                    .get(&grass)
                    .unwrap()
                    .extension,
            )
        };
        assert!(key(&app).grass_lighting);
        let light = app
            .world_mut()
            .spawn((PointLight::default(), Transform::default()))
            .id();
        app.update();
        assert!(!key(&app).grass_lighting);
        for i in 0..10 {
            app.world_mut().clear_trackers();
            app.world_mut()
                .get_mut::<Transform>(light)
                .unwrap()
                .translation
                .x = i as f32;
            app.update();
            assert!(
                !app.world()
                    .resource_ref::<Assets<VegetationMaterial>>()
                    .is_changed()
            );
        }
        app.world_mut().despawn(light);
        app.update();
        assert!(key(&app).grass_lighting);
    }
    #[test]
    fn visual_and_shadow_grass_share_a_cutout_mask_with_no_residual_at_zero() {
        let mut app = App::new();
        app.init_resource::<Config>()
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<VegetationMaterial>>()
            .add_systems(Startup, setup);
        app.update();
        let shared = app.world().resource::<Materials>();
        let materials = app.world().resource::<Assets<VegetationMaterial>>();
        let visual = materials.get(&shared.grass).unwrap();
        let shadow = materials.get(&shared.shadow_grass).unwrap();
        assert_eq!(visual.base.alpha_mode, AlphaMode::Mask(0.5));
        assert_eq!(shadow.base.alpha_mode, AlphaMode::Mask(0.5));
        let texture = visual.base.base_color_texture.as_ref().unwrap();
        assert_eq!(shadow.base.base_color_texture.as_ref(), Some(texture));
        let image = app
            .world()
            .resource::<Assets<Image>>()
            .get(texture)
            .unwrap();
        let w = image.width() as usize;
        let h = image.height() as usize;
        let bytes = image.data.as_ref().unwrap();
        let visible = |x: usize| (0..h).filter(|y| bytes[(y * w + x) * 4 + 3] >= 128).count();
        assert_eq!(visible(0), 0, "fully faded grass must write no color/depth");
        assert_eq!(visible(w - 1), h, "close grass retains every blade");
        assert!(visible(w / 2).abs_diff(h / 2) <= 1);
        assert!((1..w).all(|x| visible(x - 1) <= visible(x)));
        assert!(bytes.chunks_exact(4).all(|pixel| pixel[..3] == [255; 3]));
        assert!(bytes.len() <= 32 * 1024);
    }
    #[test]
    fn only_visual_grass_uses_the_fast_filter_and_quality_toggle_restores_it() {
        let mut app = App::new();
        app.init_resource::<Config>()
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<VegetationMaterial>>()
            .add_systems(Startup, setup)
            .add_systems(Update, animate);
        app.update();
        let check = |app: &App, fast: bool| {
            let shared = app.world().resource::<Materials>();
            let materials = app.world().resource::<Assets<VegetationMaterial>>();
            for (handle, expected) in [
                (&shared.grass, fast),
                (&shared.wood, false),
                (&shared.leaves, false),
                (&shared.shadow_grass, false),
            ] {
                let m = materials.get(handle).unwrap();
                assert_eq!(WindShaderKey::from(&m.extension).fast_shadows, expected);
            }
            let grass = materials.get(&shared.grass).unwrap();
            assert_eq!(grass.base.diffuse_transmission, 0.22);
            assert_eq!(grass.base.thickness, 0.02);
        };
        check(&app, true);
        app.world_mut().resource_mut::<Config>().fast_grass_shadows = false;
        app.update();
        check(&app, false);
        app.world_mut().resource_mut::<Config>().fast_grass_shadows = true;
        app.update();
        check(&app, true);
        let mut defs = vec![
            ShaderDefVal::Bool("SHADOW_FILTER_METHOD_GAUSSIAN".into(), true),
            ShaderDefVal::Bool("WIND".into(), true),
        ];
        grass_shadow_defs(&mut defs);
        assert_eq!(
            defs,
            vec![
                ShaderDefVal::Bool("WIND".into(), true),
                ShaderDefVal::Bool("SHADOW_FILTER_METHOD_HARDWARE_2X2".into(), true)
            ]
        );
    }
    #[test]
    fn gpu_clock_does_not_modify_shared_materials_and_leaf_texture_is_shared() {
        let mut app = App::new();
        app.init_resource::<Config>()
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<VegetationMaterial>>()
            .add_systems(Startup, setup)
            .add_systems(Update, animate);
        app.update();
        assert_eq!(app.world().resource::<Assets<Image>>().len(), 2);
        assert_eq!(
            app.world().resource::<Assets<VegetationMaterial>>().len(),
            4
        );
        let shared = app.world().resource::<Materials>();
        let leaf = shared.leaves.clone();
        let material = app
            .world()
            .resource::<Assets<VegetationMaterial>>()
            .get(&leaf)
            .unwrap();
        assert_eq!(material.base.alpha_mode, AlphaMode::Mask(0.5));
        assert!(material.base.base_color_texture.is_some());
        app.world_mut().clear_trackers();
        for _ in 0..60 {
            app.update();
            assert!(
                !app.world()
                    .resource_ref::<Assets<VegetationMaterial>>()
                    .is_changed()
            );
        }
        app.world_mut().resource_mut::<Config>().enabled = false;
        app.update();
        assert!(
            app.world()
                .resource::<Assets<VegetationMaterial>>()
                .iter()
                .all(|(_, m)| m.extension.parameters.x == 0.)
        );
    }
}
