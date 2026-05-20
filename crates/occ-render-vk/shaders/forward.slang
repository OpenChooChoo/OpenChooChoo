// Forward PBR pass. Push constants supply frame-data BDA; everything else is loaded via Ptr<T>.

import common.types;
import common.uniforms;
import common.pbr;

[vk::binding(0, 0)]
Sampler2D g_textures[];

struct VsOut {
    float4 position : SV_Position;
    float3 world_position;
    float3 world_normal;
    float2 uv;
    nointerpolation uint material_index;
};

[shader("vertex")]
VsOut vertexMain(uint vertex_id : SV_VertexID) {
    FrameData frame = g_push.frame_data[0];
    ObjectData object = frame.object_buffer[g_push.object_index];
    Vertex v = frame.vertex_buffer[vertex_id];

    float4 world_pos = mul(object.model, float4(v.position, 1.0));
    float4 clip_pos = mul(object.model_view_proj, float4(v.position, 1.0));

    float3 world_normal = normalize(mul((float3x3)object.normal_matrix, v.normal));

    VsOut output;
    output.position = clip_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.uv = v.uv;
    output.material_index = object.material_index;
    return output;
}

// Reconstruct a tangent-space basis from screen-space derivatives so the
// renderer can sample normal maps even when the source glTF lacks a TANGENT
// attribute. Mikkelsen-style derivation matches the conventional glTF
// normal-map orientation closely enough for static rendering.
float3 sample_normal_map(uint tex_index, float2 uv, float3 world_pos, float3 n, float scale) {
    if (tex_index == 0xFFFFFFFF) {
        return n;
    }
    float3 dpdx = ddx(world_pos);
    float3 dpdy = ddy(world_pos);
    float2 duvdx = ddx(uv);
    float2 duvdy = ddy(uv);

    float3 t = dpdx * duvdy.y - dpdy * duvdx.y;
    // Project T onto the plane orthogonal to N to keep the basis orthonormal.
    t = normalize(t - n * dot(n, t));
    float3 b = normalize(cross(n, t));

    float3 sampled = g_textures[NonUniformResourceIndex(tex_index)].Sample(uv).xyz * 2.0 - 1.0;
    sampled.xy *= scale;
    float3x3 tbn = float3x3(t, b, n);
    return normalize(mul(sampled, tbn));
}

float4 sample_or_default(uint tex_index, float2 uv, float4 default_value) {
    if (tex_index == 0xFFFFFFFF) {
        return default_value;
    }
    return g_textures[NonUniformResourceIndex(tex_index)].Sample(uv);
}

[shader("fragment")]
float4 fragmentMain(VsOut input) : SV_Target {
    FrameData frame = g_push.frame_data[0];
    MaterialData material = frame.material_buffer[input.material_index];

    float4 base_color_sample = sample_or_default(material.base_color_tex, input.uv, float4(1.0, 1.0, 1.0, 1.0));
    float3 base_color = base_color_sample.rgb * material.base_color_factor.rgb;

    float4 mr_sample = sample_or_default(material.metallic_roughness_tex, input.uv, float4(1.0, 1.0, 1.0, 1.0));
    float metallic = saturate(mr_sample.b * material.metallic_factor);
    float roughness = saturate(mr_sample.g * material.roughness_factor);
    float perceptual_roughness = max(roughness, 0.045);
    float a = perceptual_roughness * perceptual_roughness;

    float occlusion_sample = sample_or_default(material.occlusion_tex, input.uv, float4(1.0, 1.0, 1.0, 1.0)).r;
    float occlusion = lerp(1.0, occlusion_sample, material.occlusion_strength);

    float3 emissive = sample_or_default(material.emissive_tex, input.uv, float4(1.0, 1.0, 1.0, 1.0)).rgb * material.emissive_factor;

    float3 n = normalize(input.world_normal);
    float3 normal_world = sample_normal_map(material.normal_tex, input.uv, input.world_position, n, material.normal_scale);

    float3 view_direction = normalize(frame.camera_pos_world.xyz - input.world_position);
    float3 f0 = compute_f0(base_color, metallic, float3(0.04, 0.04, 0.04));

    float3 result = float3(0.0, 0.0, 0.0);
    for (uint i = 0; i < frame.light_count; i++) {
        GpuLight light = frame.light_buffer[i];
        float3 light_direction;
        float3 light_intensity;
        if (light.kind == 0u) {
            light_direction = normalize(-light.position_world.xyz);
            light_intensity = light.color_intensity.rgb * light.color_intensity.a;
        } else {
            float3 to_light = light.position_world.xyz - input.world_position;
            float distance = length(to_light);
            light_direction = to_light / max(distance, 1e-5);
            float radius = max(light.range, 1e-5);
            float s = saturate(distance / radius);
            float s2 = s * s;
            float inv_s2 = 1.0 - s2;
            float attenuation = (inv_s2 * inv_s2) / (1.0 + s2);
            light_intensity = light.color_intensity.rgb * (light.color_intensity.a * attenuation);
        }

        result += shade_surface(
            view_direction,
            base_color * (1.0 - metallic),
            normal_world,
            f0,
            a,
            occlusion,
            light_intensity,
            light_direction
        );
    }

    result += emissive;
    result += base_color * 0.03 * occlusion;

    return float4(result, 1.0);
}
