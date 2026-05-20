use ash::vk;

use crate::TextureFormat;

impl TextureFormat {
    pub(crate) fn to_vk(self) -> vk::Format {
        match self {
            TextureFormat::BC1Srgb => vk::Format::BC1_RGBA_SRGB_BLOCK,
            TextureFormat::BC3Srgb => vk::Format::BC3_SRGB_BLOCK,
            TextureFormat::BC4Unorm => vk::Format::BC4_UNORM_BLOCK,
            TextureFormat::BC5Unorm => vk::Format::BC5_UNORM_BLOCK,
            TextureFormat::BC7Unorm => vk::Format::BC7_UNORM_BLOCK,
            TextureFormat::BC7Srgb => vk::Format::BC7_SRGB_BLOCK,
            TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        }
    }

    pub(crate) fn block_extent(self) -> (u32, u32) {
        match self {
            TextureFormat::BC1Srgb
            | TextureFormat::BC3Srgb
            | TextureFormat::BC4Unorm
            | TextureFormat::BC5Unorm
            | TextureFormat::BC7Unorm
            | TextureFormat::BC7Srgb => (4, 4),
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8Srgb => (1, 1),
        }
    }

    pub(crate) fn block_size_bytes(self) -> u32 {
        match self {
            TextureFormat::BC1Srgb | TextureFormat::BC4Unorm => 8,
            TextureFormat::BC3Srgb
            | TextureFormat::BC5Unorm
            | TextureFormat::BC7Unorm
            | TextureFormat::BC7Srgb => 16,
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8Srgb => 4,
        }
    }
}
