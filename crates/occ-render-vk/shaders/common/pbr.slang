// Cook-Torrance / GGX helpers, ported from fantasy-rail/triangle.hlsl.

static const float PI = 3.14159265358979323846;

float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

float3 compute_f0(float3 base_color, float metallic, float3 reflectance) {
    return base_color * metallic + (reflectance * (1.0 - metallic));
}

float brdf_d_ggx(float noh, float a) {
    float a2 = a * a;
    float f = (noh * a2 - noh) * noh + 1.0;
    return a2 / (PI * f * f);
}

float3 brdf_f_schlick_vec3(float u, float3 f0, float f90) {
    return f0 + (f90 - f0) * pow5(1.0 - u);
}

float brdf_fd_lambert() {
    return 1.0 / PI;
}

float brdf_v_smith_ggx_correlated(float nov, float nol, float a) {
    float a2 = a * a;
    float ggxl = nov * sqrt((-nol * a2 + nol) * nol + a2);
    float ggxv = nol * sqrt((-nov * a2 + nov) * nov + a2);
    return 0.5 / (ggxl + ggxv);
}

float3 shade_surface(
    float3 view_direction,
    float3 pixel_color,
    float3 pixel_normal,
    float3 pixel_f0,
    float pixel_roughness,
    float occlusion,
    float3 light_intensity,
    float3 light_direction
) {
    float3 n = pixel_normal;
    float3 h = normalize(view_direction + light_direction);

    float nov = max(dot(n, view_direction), 1e-5);
    float nol = saturate(dot(n, light_direction));
    float noh = saturate(dot(n, h));
    float loh = saturate(dot(light_direction, h));

    float f90 = saturate(dot(pixel_f0, float3(50.0 * 0.33, 50.0 * 0.33, 50.0 * 0.33)));

    float d = brdf_d_ggx(noh, pixel_roughness);
    float3 f = brdf_f_schlick_vec3(loh, pixel_f0, f90);
    float v = brdf_v_smith_ggx_correlated(nov, nol, pixel_roughness);

    float3 specular = d * f * v;
    float3 diffuse = pixel_color * brdf_fd_lambert();

    return (diffuse + specular) * light_intensity * (nol * occlusion);
}
