use anyhow::{Context, anyhow};
use ash::vk;

use crate::device::Device;

pub(crate) struct Swapchain {
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) images: Vec<vk::Image>,
    pub(crate) views: Vec<vk::ImageView>,
    /// One render-finished binary semaphore per swapchain image. The semaphore
    /// is signaled by the graphics submit that draws into image `i` and waited
    /// on by the matching `vkQueuePresentKHR`. Per-image (rather than
    /// per-frame-in-flight) lifetimes are required because the presentation
    /// engine may hold the semaphore until any future acquire of the same
    /// image, which can be further out than `FRAMES_IN_FLIGHT`.
    pub(crate) render_finished_sems: Vec<vk::Semaphore>,
    pub(crate) format: vk::Format,
    pub(crate) extent: vk::Extent2D,
    pub(crate) present_mode: vk::PresentModeKHR,
}

impl Swapchain {
    pub(crate) fn new(
        device: &Device,
        surface: vk::SurfaceKHR,
        desired_extent: (u32, u32),
    ) -> anyhow::Result<Self> {
        Self::create(device, surface, desired_extent, vk::SwapchainKHR::null())
    }

    pub(crate) fn recreate(
        &mut self,
        device: &Device,
        desired_extent: (u32, u32),
    ) -> anyhow::Result<()> {
        let surface = self.surface;
        let old_swapchain = self.swapchain;
        let new_self = Self::create(device, surface, desired_extent, old_swapchain)?;
        self.destroy_image_views(device);
        self.destroy_render_finished_sems(device);
        unsafe {
            device.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }
        self.swapchain = new_self.swapchain;
        self.images = new_self.images;
        self.views = new_self.views;
        self.render_finished_sems = new_self.render_finished_sems;
        self.format = new_self.format;
        self.extent = new_self.extent;
        self.present_mode = new_self.present_mode;
        Ok(())
    }

    fn create(
        device: &Device,
        surface: vk::SurfaceKHR,
        desired_extent: (u32, u32),
        old_swapchain: vk::SwapchainKHR,
    ) -> anyhow::Result<Self> {
        let capabilities = unsafe {
            device
                .surface_loader
                .get_physical_device_surface_capabilities(device.physical_device, surface)
        }
        .context("failed to query surface capabilities")?;

        let formats = unsafe {
            device
                .surface_loader
                .get_physical_device_surface_formats(device.physical_device, surface)
        }
        .context("failed to query surface formats")?;

        let format = pick_surface_format(&formats)
            .ok_or_else(|| anyhow!("no suitable SRGB surface format available"))?;

        let present_modes = unsafe {
            device
                .surface_loader
                .get_physical_device_surface_present_modes(device.physical_device, surface)
        }
        .context("failed to query present modes")?;
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::FIFO) {
            vk::PresentModeKHR::FIFO
        } else {
            present_modes[0]
        };

        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: desired_extent
                    .0
                    .clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                height: desired_extent.1.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe { device.swapchain_loader.create_swapchain(&create_info, None) }
            .context("failed to create swapchain")?;
        device.set_debug_name(swapchain, "Swapchain");

        let images = unsafe { device.swapchain_loader.get_swapchain_images(swapchain) }?;
        let views = images
            .iter()
            .enumerate()
            .map(|(idx, image)| -> anyhow::Result<vk::ImageView> {
                let info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                let view = unsafe { device.device.create_image_view(&info, None) }?;
                device.set_debug_name(view, &format!("Swapchain Image View {idx}"));
                Ok(view)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let render_finished_sems = (0..images.len())
            .map(|idx| -> anyhow::Result<vk::Semaphore> {
                let info = vk::SemaphoreCreateInfo::default();
                let sem = unsafe { device.device.create_semaphore(&info, None) }
                    .context("create render-finished semaphore")?;
                device.set_debug_name(sem, &format!("Render Finished Sem (image {idx})"));
                Ok(sem)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            surface,
            swapchain,
            images,
            views,
            render_finished_sems,
            format: format.format,
            extent,
            present_mode,
        })
    }

    fn destroy_image_views(&mut self, device: &Device) {
        for view in self.views.drain(..) {
            unsafe { device.device.destroy_image_view(view, None) };
        }
    }

    fn destroy_render_finished_sems(&mut self, device: &Device) {
        for sem in self.render_finished_sems.drain(..) {
            unsafe { device.device.destroy_semaphore(sem, None) };
        }
    }

    pub(crate) fn destroy(&mut self, device: &Device) {
        self.destroy_image_views(device);
        self.destroy_render_finished_sems(device);
        unsafe {
            if self.swapchain != vk::SwapchainKHR::null() {
                device.swapchain_loader.destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
            if self.surface != vk::SurfaceKHR::null() {
                device.surface_loader.destroy_surface(self.surface, None);
                self.surface = vk::SurfaceKHR::null();
            }
        }
    }
}

fn pick_surface_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    let preferred = [vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB];
    for want in preferred {
        if let Some(f) = formats.iter().find(|f| {
            f.format == want && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        }) {
            return Some(*f);
        }
    }
    None
}
