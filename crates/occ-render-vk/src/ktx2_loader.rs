use std::path::Path;

use anyhow::{Context, bail};
use glam::UVec2;
use ktx2::{Format as Ktx2Format, Reader};

use crate::{TextureDescriptor, TextureFormat};

pub(crate) fn load_ktx2(path: &Path, debug_label: String) -> anyhow::Result<TextureDescriptor> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read KTX2 file '{}'", path.display()))?;
    let reader = Reader::new(bytes.as_slice())
        .with_context(|| format!("failed to parse KTX2 '{}'", path.display()))?;
    let header = reader.header();

    if let Some(scheme) = header.supercompression_scheme {
        bail!(
            "KTX2 '{}' uses unsupported supercompression {:?}",
            path.display(),
            scheme
        );
    }

    let format = match header.format {
        Some(f) => map_format(f).with_context(|| {
            format!("KTX2 '{}' uses unsupported VkFormat {:?}", path.display(), f)
        })?,
        None => bail!("KTX2 '{}' has no VkFormat", path.display()),
    };

    let mipmap_count = header.level_count.max(1);
    let mip_data: Vec<Vec<u8>> = reader.levels().map(|level| level.data.to_vec()).collect();

    Ok(TextureDescriptor {
        size: UVec2::new(header.pixel_width, header.pixel_height.max(1)),
        mipmap_count,
        format,
        mip_data,
        debug_label,
    })
}

fn map_format(format: Ktx2Format) -> Option<TextureFormat> {
    Some(match format {
        Ktx2Format::BC1_RGB_SRGB_BLOCK | Ktx2Format::BC1_RGBA_SRGB_BLOCK => TextureFormat::BC1Srgb,
        Ktx2Format::BC3_SRGB_BLOCK => TextureFormat::BC3Srgb,
        Ktx2Format::BC4_UNORM_BLOCK => TextureFormat::BC4Unorm,
        Ktx2Format::BC5_UNORM_BLOCK => TextureFormat::BC5Unorm,
        Ktx2Format::BC7_UNORM_BLOCK => TextureFormat::BC7Unorm,
        Ktx2Format::BC7_SRGB_BLOCK => TextureFormat::BC7Srgb,
        Ktx2Format::R8G8B8A8_UNORM => TextureFormat::Rgba8Unorm,
        Ktx2Format::R8G8B8A8_SRGB => TextureFormat::Rgba8Srgb,
        _ => return None,
    })
}
