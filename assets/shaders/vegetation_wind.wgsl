// UV0 preserves generated leaf texture coordinates. Grass uses the shared
// cutout lookup for distance fading in both color and shadow/depth passes.
// UV1 is bend amplitude and phase. GPU time never rebinds materials.
#import bevy_pbr::{mesh_functions, view_transformations::position_world_to_clip, mesh_view_bindings::view}
#ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::{Vertex, VertexOutput}
#import bevy_render::globals::Globals
@group(0) @binding(1) var<uniform> plant_clock: Globals;
#else
#import bevy_pbr::forward_io::{Vertex, VertexOutput, FragmentOutput}
#import bevy_pbr::mesh_view_bindings::globals as plant_clock
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::{PbrInput, STANDARD_MATERIAL_FLAGS_UNLIT_BIT},
}
#ifdef VISIBILITY_RANGE_DITHER
#import bevy_pbr::pbr_functions::visibility_range_dither
#endif
#ifdef GRASS_DIFFUSE_LIGHTING
#import bevy_pbr::{
    mesh_view_bindings::lights,
    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    shadows::fetch_directional_shadow,
    ambient::ambient_light,
}
#endif
#endif

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> wind: vec4<f32>;

fn displaced(position: vec4<f32>, bend: vec2<f32>, time: f32) -> vec4<f32> {
    var p=position;
    let distance_to_camera=distance(p.xyz, view.lod_view_world_position);
    let attenuation=1.0-smoothstep(16.0,40.0,distance_to_camera);
    if bend.x*wind.x*attenuation > 0.0 {
        let speed=select(1.0,0.65,wind.y == 1.0);
        let phase=dot(p.xz,vec2<f32>(0.38,0.21))+bend.y;
        let wave=sin(time*1.65*speed-phase);
        let gust=0.65+0.20*sin(time*0.43*speed-phase*0.23);
        let ripple=0.10*sin(time*3.1*speed+phase*2.0);
        let offset=vec2<f32>(0.8944272,0.4472136)*(gust*wave+ripple)*bend.x*wind.x*attenuation;
        p.x+=offset.x;
        p.z+=offset.y;
    }
    return p;
}

@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let model = mesh_functions::get_world_from_local(v.instance_index);
    let world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(v.position, 1.0));
    out.world_position = displaced(world_position, v.uv_b, plant_clock.time);
    out.position = position_world_to_clip(out.world_position.xyz);
    out.uv = v.uv;
    if wind.y >= 2.0 {
        let distance_to_camera = distance(world_position.xyz, view.lod_view_world_position);
        let coverage = select(1.0-smoothstep(28.0,60.0,distance_to_camera),1.0-smoothstep(6.0,10.0,distance_to_camera),wind.y == 3.0);
        out.uv = vec2<f32>(coverage, fract(v.uv_b.y / 6.2831853));
    }
#ifdef VERTEX_UVS_B
    out.uv_b = v.uv_b;
#endif
    out.color = v.color;
#ifndef PREPASS_PIPELINE
    out.world_normal = mesh_functions::mesh_normal_local_to_world(v.normal, v.instance_index);
#else
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(v.normal, v.instance_index);
#endif
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif
#ifdef MOTION_VECTOR_PREPASS
    let previous_model = mesh_functions::get_previous_world_from_local(v.instance_index);
    out.previous_world_position = displaced(mesh_functions::mesh_position_local_to_world(previous_model, vec4<f32>(v.position, 1.0)), v.uv_b, plant_clock.time-plant_clock.delta_time);
#endif
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = v.instance_index;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(v.instance_index, model[3]);
#endif
    return out;
}

#ifndef PREPASS_PIPELINE
#ifdef GRASS_DIFFUSE_LIGHTING
// Thin meadow blades use diffuse sunlight and modest backlighting. Their
// 2cm thickness does not need a separate shadow lookup for the reverse lobe.
// Keep ambient occlusion, fog, tone mapping and the existing cutout/LOD mask.
fn grass_lighting(input: PbrInput) -> vec4<f32> {
    let view_z = dot(vec4<f32>(view.view_from_world[0].z,
        view.view_from_world[1].z, view.view_from_world[2].z,
        view.view_from_world[3].z), input.world_position);
    let albedo = input.material.base_color.rgb;
    let transmission = input.material.diffuse_transmission;
    let normal = normalize(input.N);
    var lit = ambient_light(input.world_position, normal, input.V,
        max(dot(normal, input.V), 0.0001), albedo, vec3<f32>(0.0), 1.0,
        input.diffuse_occlusion);
    for (var i = 0u; i < lights.n_directional_lights; i += 1u) {
        let light = lights.directional_lights[i];
        let cosine = dot(normal, light.direction_to_light);
        let diffuse = (1.0-transmission)*max(cosine, 0.0)
            + transmission*max(-cosine, 0.0);
        var visibility = 1.0;
        if (input.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u
            && (light.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            visibility = fetch_directional_shadow(i, input.world_position,
                input.world_normal, view_z, input.frag_coord.xy);
        }
        lit += albedo * light.color.rgb * (diffuse * visibility / 3.14159265);
    }
    let emissive = input.material.emissive.rgb
        * mix(1.0, view.exposure, input.material.emissive.a);
    return vec4<f32>(lit * view.exposure + emissive, input.material.base_color.a);
}
#endif

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
#ifdef VISIBILITY_RANGE_DITHER
    visibility_range_dither(in.position, in.visibility_range_dither);
#endif
    var input = pbr_input_from_standard_material(in, front);
    input.material.base_color = alpha_discard(input.material, input.material.base_color);
    var out: FragmentOutput;
    if (input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) != 0u {
        out.color = input.material.base_color;
    } else {
#ifdef GRASS_DIFFUSE_LIGHTING
        out.color = grass_lighting(input);
#else
        out.color = apply_pbr_lighting(input);
#endif
    }
    out.color = main_pass_post_lighting_processing(input, out.color);
    return out;
}
#endif
