// Mirrors the Rust GPU layout in `occ-render-vk` (std430/scalar buffer device address access).

struct Vertex {
    float3 position;
    float3 normal;
    float2 uv;
};

struct ObjectData {
    float4x4 model;
    float4x4 model_view_proj;
    float4x4 normal_matrix;
    uint material_index;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct MaterialData {
    float4 base_color_factor;
    float3 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float occlusion_strength;
    float normal_scale;
    uint base_color_tex;
    uint normal_tex;
    uint metallic_roughness_tex;
    uint occlusion_tex;
    uint emissive_tex;
    uint _pad0;
};

struct GpuLight {
    float4 position_world;
    float4 color_intensity;
    uint kind;
    float range;
    float _pad0;
    float _pad1;
};

struct FrameData {
    float4 camera_pos_world;
    ObjectData* object_buffer;
    GpuLight* light_buffer;
    MaterialData* material_buffer;
    Vertex* vertex_buffer;
    uint light_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
