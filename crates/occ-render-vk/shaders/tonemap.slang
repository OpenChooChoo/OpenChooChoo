// Full-screen tonemap. Reads HDR via the first bindless slot (index 0). ACES approximation by Krzysztof Narkowicz.

[vk::binding(0, 0)]
Sampler2D g_textures[];

struct PushConstants {
    uint hdr_texture_index;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[vk::push_constant]
ConstantBuffer<PushConstants> g_push;

struct VsOut {
    float4 position : SV_Position;
    float2 uv;
};

[shader("vertex")]
VsOut vertexMain(uint vertex_id : SV_VertexID) {
    // Standard fullscreen triangle: 3 vertices covering [-1,3] in both axes.
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    VsOut output;
    output.uv = uv;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

float3 aces_narkowicz(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[shader("fragment")]
float4 fragmentMain(VsOut input) : SV_Target {
    float3 hdr = g_textures[g_push.hdr_texture_index].Sample(input.uv).rgb;
    float3 mapped = aces_narkowicz(hdr);
    return float4(mapped, 1.0);
}
