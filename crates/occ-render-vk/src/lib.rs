//! Vulkan 1.3 forward renderer for OpenChooChoo.
//!
//! # Environment variables
//!
//! - `OCC_RENDER_VALIDATION` (`0`/`false` force-off, `1`/`true` force-on,
//!   unset/other = default; the default is on when compiled with
//!   `debug_assertions`, off otherwise).

mod device;
mod forward;
mod format;
mod frame;
mod gltf;
mod handle_storage;
mod ktx2_loader;
mod renderer;
mod resource;
mod swapchain;
mod tonemap;

pub use renderer::VulkanRenderer;

use glam::{Mat4, UVec2, Vec3, Vec4};

#[derive(Debug, Clone, Copy)]
pub struct RendererConfig {
    pub initial_size: (u32, u32),
}

#[derive(Debug, Clone)]
pub struct FrameInput<'a> {
    pub view: Mat4,
    pub camera_position_world: Vec3,
    pub aspect_ratio: f32,
    pub projection: Mat4,
    pub objects: &'a [RenderObject],
    pub lights: &'a [RenderLight],
}

#[derive(Debug, Clone, Copy)]
pub struct RenderObject {
    pub transform: Mat4,
    pub mesh: occ_common::MeshHandle,
    pub material: occ_common::MaterialHandle,
}

#[derive(Debug, Clone, Copy)]
pub enum RenderLight {
    Directional {
        direction_world: Vec3,
        color: Vec3,
        illuminance: f32,
    },
    Point {
        position_world: Vec3,
        color: Vec3,
        intensity: f32,
        range: f32,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct LoadedGltfObject {
    pub transform: Mat4,
    pub mesh: occ_common::MeshHandle,
    pub material: occ_common::MaterialHandle,
}

#[derive(Debug, Clone)]
pub struct MeshDescriptor {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    pub size: UVec2,
    pub mipmap_count: u32,
    pub format: TextureFormat,
    pub mip_data: Vec<Vec<u8>>,
    pub debug_label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    BC1Srgb,
    BC3Srgb,
    BC4Unorm,
    BC5Unorm,
    BC7Unorm,
    BC7Srgb,
    Rgba8Unorm,
    Rgba8Srgb,
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialDescriptor {
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec3,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub occlusion_strength: f32,
    pub normal_scale: f32,
    pub base_color: occ_common::TextureHandle,
    pub normal: occ_common::TextureHandle,
    pub metallic_roughness: occ_common::TextureHandle,
    pub occlusion: occ_common::TextureHandle,
    pub emissive: occ_common::TextureHandle,
}
