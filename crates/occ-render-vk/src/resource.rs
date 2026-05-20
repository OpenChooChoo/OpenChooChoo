use std::sync::Arc;

use anyhow::Context;
use ash::vk;
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, AllocationCreateInfo, MemoryUsage};

use crate::device::Device;

pub(crate) struct Allocator {
    pub(crate) inner: vk_mem::Allocator,
    pub(crate) device: Arc<Device>,
}

impl Allocator {
    pub(crate) fn new(device: Arc<Device>) -> anyhow::Result<Self> {
        let info = vk_mem::AllocatorCreateInfo::new(
            &device.instance,
            &device.device,
            device.physical_device,
        );
        let mut info = info;
        info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        info.vulkan_api_version = vk::API_VERSION_1_3;
        let inner = unsafe { vk_mem::Allocator::new(info) }
            .context("failed to create vk-mem allocator")?;
        Ok(Self { inner, device })
    }
}

pub(crate) struct Buffer {
    pub(crate) raw: vk::Buffer,
    pub(crate) size: vk::DeviceSize,
    pub(crate) device_address: vk::DeviceAddress,
    pub(crate) mapped: *mut u8,
    allocation: Option<Allocation>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub(crate) fn new_gpu(
        allocator: &Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        debug_name: &str,
    ) -> anyhow::Result<Self> {
        Self::new(allocator, size, usage, MemoryUsage::AutoPreferDevice, AllocationCreateFlags::empty(), debug_name)
    }

    pub(crate) fn new_host_visible(
        allocator: &Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        debug_name: &str,
    ) -> anyhow::Result<Self> {
        Self::new(
            allocator,
            size,
            usage,
            MemoryUsage::AutoPreferHost,
            AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE | AllocationCreateFlags::MAPPED,
            debug_name,
        )
    }

    fn new(
        allocator: &Allocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryUsage,
        flags: AllocationCreateFlags,
        debug_name: &str,
    ) -> anyhow::Result<Self> {
        let needs_address = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let queue_families = [
            allocator.device.graphics_queue_family,
            allocator.device.transfer_queue_family,
        ];
        let mut buffer_info = vk::BufferCreateInfo::default().size(size).usage(usage);
        if allocator.device.graphics_queue_family != allocator.device.transfer_queue_family {
            buffer_info = buffer_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_families);
        } else {
            buffer_info = buffer_info.sharing_mode(vk::SharingMode::EXCLUSIVE);
        }
        let allocation_info = AllocationCreateInfo {
            usage: memory_usage,
            flags,
            ..Default::default()
        };
        let (raw, allocation) =
            unsafe { allocator.inner.create_buffer(&buffer_info, &allocation_info) }
                .with_context(|| format!("failed to create buffer '{debug_name}'"))?;
        allocator.device.set_debug_name(raw, debug_name);

        let device_address = if needs_address {
            unsafe {
                allocator
                    .device
                    .device
                    .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(raw))
            }
        } else {
            0
        };

        let mapped = if flags.contains(AllocationCreateFlags::MAPPED) {
            allocator.inner.get_allocation_info(&allocation).mapped_data as *mut u8
        } else {
            std::ptr::null_mut()
        };

        Ok(Self { raw, size, device_address, mapped, allocation: Some(allocation) })
    }

    pub(crate) fn destroy(&mut self, allocator: &Allocator) {
        if let Some(mut allocation) = self.allocation.take() {
            unsafe { allocator.inner.destroy_buffer(self.raw, &mut allocation) };
            self.raw = vk::Buffer::null();
        }
    }

    pub(crate) fn write_at(&self, offset: usize, bytes: &[u8]) {
        debug_assert!(!self.mapped.is_null(), "buffer is not host-visible");
        debug_assert!(offset + bytes.len() <= self.size as usize, "write out of bounds");
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped.add(offset), bytes.len());
        }
    }
}

pub(crate) struct Image {
    pub(crate) raw: vk::Image,
    pub(crate) view: vk::ImageView,
    allocation: Option<Allocation>,
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    pub(crate) fn new_2d(
        allocator: &Allocator,
        format: vk::Format,
        extent: vk::Extent2D,
        mip_levels: u32,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
        debug_name: &str,
    ) -> anyhow::Result<Self> {
        let queue_families = [
            allocator.device.graphics_queue_family,
            allocator.device.transfer_queue_family,
        ];
        let mut image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        if allocator.device.graphics_queue_family != allocator.device.transfer_queue_family {
            image_info = image_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_families);
        } else {
            image_info = image_info.sharing_mode(vk::SharingMode::EXCLUSIVE);
        }
        let allocation_info = AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (raw, allocation) =
            unsafe { allocator.inner.create_image(&image_info, &allocation_info) }
                .with_context(|| format!("failed to create image '{debug_name}'"))?;
        allocator.device.set_debug_name(raw, debug_name);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(raw)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let view = unsafe { allocator.device.device.create_image_view(&view_info, None) }
            .with_context(|| format!("failed to create view for '{debug_name}'"))?;
        allocator.device.set_debug_name(view, &format!("{debug_name} View"));

        Ok(Self { raw, view, allocation: Some(allocation) })
    }

    pub(crate) fn destroy(&mut self, allocator: &Allocator) {
        unsafe {
            if self.view != vk::ImageView::null() {
                allocator.device.device.destroy_image_view(self.view, None);
                self.view = vk::ImageView::null();
            }
            if let Some(mut allocation) = self.allocation.take() {
                allocator.inner.destroy_image(self.raw, &mut allocation);
                self.raw = vk::Image::null();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn image_memory_barrier(
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    src_stage: vk::PipelineStageFlags2,
    dst_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_access: vk::AccessFlags2,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    mip_levels: u32,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .dst_stage_mask(dst_stage)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(mip_levels)
                .base_array_layer(0)
                .layer_count(1),
        )
}
