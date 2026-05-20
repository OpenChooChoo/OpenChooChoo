use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use ash::vk;
use glam::{Mat4, Vec4};
use occ_common::{MaterialHandle, MeshHandle, TextureHandle};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::device::Device;
use crate::forward::ForwardPipeline;
use crate::frame::{FRAMES_IN_FLIGHT, FrameSet, FrameSubmitter};
use crate::gltf::{DefaultTextures, load_gltf};
use crate::handle_storage::Storage;
use crate::resource::{Allocator, Buffer, Image, image_memory_barrier};
use crate::swapchain::Swapchain;
use crate::tonemap::TonemapPipeline;
use crate::{
    FrameInput, LoadedGltfObject, MaterialDescriptor, MeshDescriptor, RenderLight, RenderObject,
    RendererConfig, TextureDescriptor, TextureFormat,
};

const MESH_BUFFER_SIZE: u64 = 64 * 1024 * 1024;
const INDEX_BUFFER_SIZE: u64 = 64 * 1024 * 1024;
const STAGING_BUFFER_SIZE: u64 = 64 * 1024 * 1024;
const MAX_OBJECTS: u64 = 4096;
const MAX_LIGHTS: u64 = 1024;
const MAX_MATERIALS: u64 = 4096;
const BINDLESS_TEXTURE_COUNT: u32 = 1024;
const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct FrameDataGpu {
    camera_pos_world: Vec4,
    object_buffer_address: u64,
    light_buffer_address: u64,
    material_buffer_address: u64,
    vertex_buffer_address: u64,
    light_count: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct ObjectDataGpu {
    model: Mat4,
    model_view_proj: Mat4,
    normal_matrix: Mat4,
    material_index: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct MaterialDataGpu {
    base_color_factor: [f32; 4],
    emissive_factor: [f32; 3],
    metallic_factor: f32,
    roughness_factor: f32,
    occlusion_strength: f32,
    normal_scale: f32,
    base_color_tex: u32,
    normal_tex: u32,
    metallic_roughness_tex: u32,
    occlusion_tex: u32,
    emissive_tex: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct GpuLight {
    position_world: Vec4,
    color_intensity: Vec4,
    kind: u32,
    range: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct ForwardPushConstants {
    frame_data_address: u64,
    object_index: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct TonemapPushConstants {
    hdr_texture_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct MeshSlot {
    vertex_offset_vertices: u32,
    index_offset_indices: u32,
    index_count: u32,
}

struct MaterialSlot {
    descriptor: MaterialDescriptor,
}

struct PendingMeshUpload {
    mesh_index: u32,
    descriptor: MeshDescriptor,
}

struct PendingTextureUpload {
    texture_index: u32,
    descriptor: TextureDescriptor,
}

struct PendingMaterialUpload {
    material_index: u32,
}

struct TextureSlot {
    image: Image,
    bindless_index: u32,
}

pub struct VulkanRenderer {
    device: Arc<Device>,
    allocator: Allocator,
    swapchain: Swapchain,
    surface_size: (u32, u32),
    pending_resize: Option<(u32, u32)>,
    needs_swapchain_recreate: bool,

    frames: FrameSet,
    submitter: FrameSubmitter,
    frame_counter: u64,

    bindless_pool: vk::DescriptorPool,
    bindless_set_layout: vk::DescriptorSetLayout,
    bindless_set: vk::DescriptorSet,
    bindless_sampler: vk::Sampler,

    vertex_buffer: Buffer,
    index_buffer: Buffer,
    material_buffer: Buffer,
    frame_data_buffers: [Buffer; FRAMES_IN_FLIGHT],
    object_buffers: [Buffer; FRAMES_IN_FLIGHT],
    light_buffers: [Buffer; FRAMES_IN_FLIGHT],
    staging_buffers: [Buffer; FRAMES_IN_FLIGHT],

    forward_pipeline: ForwardPipeline,
    tonemap_pipeline: TonemapPipeline,

    hdr_image: Image,
    depth_image: Image,
    hdr_bindless_index: u32,

    meshes: Storage<MeshSlot>,
    textures: Vec<TextureSlot>,
    materials: Storage<MaterialSlot>,

    pending_meshes: Vec<PendingMeshUpload>,
    pending_textures: Vec<PendingTextureUpload>,
    pending_materials: Vec<PendingMaterialUpload>,
    /// Post-upload image layout transitions queued for the next graphics CB.
    /// Transitions to `SHADER_READ_ONLY_OPTIMAL` cannot happen on the transfer
    /// queue (its queue family doesn't support shader stages as a barrier dst).
    /// Buffers are CONCURRENT-shared and don't need anything analogous — the
    /// transfer-timeline semaphore wait carries their memory dependency over.
    pending_image_acquires: Vec<vk::ImageMemoryBarrier2<'static>>,

    vertex_bump_vertices: u32,
    index_bump_indices: u32,

    pub(crate) defaults: DefaultTextures,
}

impl VulkanRenderer {
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        config: RendererConfig,
    ) -> anyhow::Result<Self> {
        let device = Arc::new(Device::new(display_handle)?);
        let allocator = Allocator::new(device.clone())?;

        let surface = device.create_surface(display_handle, window_handle)?;
        let swapchain = Swapchain::new(&device, surface, config.initial_size)?;
        let surface_size = (swapchain.extent.width, swapchain.extent.height);

        let frames = FrameSet::new(&device)?;
        let submitter = FrameSubmitter::new(&device)?;

        let (bindless_set_layout, bindless_pool, bindless_set, bindless_sampler) =
            create_bindless_resources(&device)?;

        let vertex_buffer = Buffer::new_gpu(
            &allocator,
            MESH_BUFFER_SIZE,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "Vertex Buffer",
        )?;
        let index_buffer = Buffer::new_gpu(
            &allocator,
            INDEX_BUFFER_SIZE,
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "Index Buffer",
        )?;
        let material_buffer = Buffer::new_gpu(
            &allocator,
            std::mem::size_of::<MaterialDataGpu>() as u64 * MAX_MATERIALS,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "Material Buffer",
        )?;

        let make_frame_data = |i: usize| {
            Buffer::new_host_visible(
                &allocator,
                std::mem::size_of::<FrameDataGpu>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                &format!("FrameData {i}"),
            )
        };
        let frame_data_buffers = [make_frame_data(0)?, make_frame_data(1)?];

        let make_object = |i: usize| {
            Buffer::new_host_visible(
                &allocator,
                std::mem::size_of::<ObjectDataGpu>() as u64 * MAX_OBJECTS,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                &format!("Object Buffer {i}"),
            )
        };
        let object_buffers = [make_object(0)?, make_object(1)?];

        let make_light = |i: usize| {
            Buffer::new_host_visible(
                &allocator,
                std::mem::size_of::<GpuLight>() as u64 * MAX_LIGHTS,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                &format!("Light Buffer {i}"),
            )
        };
        let light_buffers = [make_light(0)?, make_light(1)?];

        let make_staging = |i: usize| {
            Buffer::new_host_visible(
                &allocator,
                STAGING_BUFFER_SIZE,
                vk::BufferUsageFlags::TRANSFER_SRC,
                &format!("Staging Buffer {i}"),
            )
        };
        let staging_buffers = [make_staging(0)?, make_staging(1)?];

        let forward_pipeline = ForwardPipeline::new(&device, bindless_set_layout)?;
        let tonemap_pipeline =
            TonemapPipeline::new(&device, bindless_set_layout, swapchain.format)?;

        let (hdr_image, depth_image) = create_screen_targets(&allocator, swapchain.extent)?;
        let hdr_bindless_index = BINDLESS_TEXTURE_COUNT - 1;
        write_bindless_image(
            &device,
            bindless_set,
            bindless_sampler,
            hdr_image.view,
            hdr_bindless_index,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        let placeholder = TextureHandle(0);
        let mut renderer = Self {
            device,
            allocator,
            swapchain,
            surface_size,
            pending_resize: None,
            needs_swapchain_recreate: false,

            frames,
            submitter,
            frame_counter: 0,

            bindless_pool,
            bindless_set_layout,
            bindless_set,
            bindless_sampler,

            vertex_buffer,
            index_buffer,
            material_buffer,
            frame_data_buffers,
            object_buffers,
            light_buffers,
            staging_buffers,

            forward_pipeline,
            tonemap_pipeline,

            hdr_image,
            depth_image,
            hdr_bindless_index,

            meshes: Storage::new(),
            textures: Vec::new(),
            materials: Storage::new(),

            pending_meshes: Vec::new(),
            pending_textures: Vec::new(),
            pending_materials: Vec::new(),
            pending_image_acquires: Vec::new(),

            vertex_bump_vertices: 0,
            index_bump_indices: 0,

            defaults: DefaultTextures {
                white: placeholder,
                flat_normal: placeholder,
            },
        };

        let white = renderer.add_texture(TextureDescriptor {
            size: glam::UVec2::new(1, 1),
            mipmap_count: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_data: vec![vec![255, 255, 255, 255]],
            debug_label: "default-white".to_string(),
        })?;
        let flat_normal = renderer.add_texture(TextureDescriptor {
            size: glam::UVec2::new(1, 1),
            mipmap_count: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_data: vec![vec![128, 128, 255, 255]],
            debug_label: "default-flat-normal".to_string(),
        })?;
        renderer.defaults = DefaultTextures { white, flat_normal };

        Ok(renderer)
    }

    pub fn add_mesh(&mut self, mesh: MeshDescriptor) -> MeshHandle {
        let vertex_count = mesh.vertices.len() as u32;
        let index_count = mesh.indices.len() as u32;
        let vertex_offset_vertices = self.vertex_bump_vertices;
        let index_offset_indices = self.index_bump_indices;
        self.vertex_bump_vertices += vertex_count;
        self.index_bump_indices += index_count;

        let idx = self.meshes.push(MeshSlot {
            vertex_offset_vertices,
            index_offset_indices,
            index_count,
        });
        self.pending_meshes.push(PendingMeshUpload {
            mesh_index: idx,
            descriptor: mesh,
        });
        MeshHandle(idx)
    }

    pub fn add_texture(&mut self, texture: TextureDescriptor) -> anyhow::Result<TextureHandle> {
        let bindless_index = self.textures.len() as u32;
        anyhow::ensure!(
            bindless_index < self.hdr_bindless_index,
            "exceeded bindless texture budget"
        );
        let format = texture.format.to_vk();
        let extent = vk::Extent2D {
            width: texture.size.x,
            height: texture.size.y,
        };
        let mip_levels = texture.mipmap_count.max(1);
        let image = Image::new_2d(
            &self.allocator,
            format,
            extent,
            mip_levels,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
            &texture.debug_label,
        )?;
        self.textures.push(TextureSlot {
            image,
            bindless_index,
        });
        self.pending_textures.push(PendingTextureUpload {
            texture_index: bindless_index,
            descriptor: texture,
        });
        Ok(TextureHandle(bindless_index))
    }

    pub fn add_material(&mut self, material: MaterialDescriptor) -> MaterialHandle {
        let idx = self.materials.push(MaterialSlot {
            descriptor: material,
        });
        self.pending_materials.push(PendingMaterialUpload {
            material_index: idx,
        });
        MaterialHandle(idx)
    }

    pub fn load_gltf(&mut self, path: &Path) -> anyhow::Result<Vec<LoadedGltfObject>> {
        load_gltf(self, path)
    }

    pub fn notify_resize(&mut self, new_size: (u32, u32)) {
        self.pending_resize = Some(new_size);
    }

    pub fn render(&mut self, frame: FrameInput<'_>) -> anyhow::Result<()> {
        if self.pending_resize.is_some() || self.needs_swapchain_recreate {
            unsafe { self.device.device.device_wait_idle() }.ok();
            if let Some(new_size) = self.pending_resize.take() {
                self.surface_size = new_size;
            }
            self.recreate_swapchain()?;
            self.needs_swapchain_recreate = false;
        }

        if frame.objects.len() as u64 > MAX_OBJECTS {
            anyhow::bail!(
                "too many objects: {} > {}",
                frame.objects.len(),
                MAX_OBJECTS
            );
        }
        if frame.lights.len() as u64 > MAX_LIGHTS {
            anyhow::bail!("too many lights: {} > {}", frame.lights.len(), MAX_LIGHTS);
        }
        if self.materials.len() as u64 > MAX_MATERIALS {
            anyhow::bail!(
                "too many materials: {} > {}",
                self.materials.len(),
                MAX_MATERIALS
            );
        }

        let frame_value = self.frame_counter + 1;
        let frame_index = self.frame_counter as usize % FRAMES_IN_FLIGHT;

        unsafe {
            let timeline = self.submitter.graphics_timeline();
            let semaphores = [timeline];
            let values = [self.frames.frames[frame_index].frame_done_value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            self.device.device.wait_semaphores(&wait_info, u64::MAX)?;
        }

        let (image_index, suboptimal) = match unsafe {
            self.device.swapchain_loader.acquire_next_image(
                self.swapchain.swapchain,
                u64::MAX,
                self.frames.frames[frame_index].image_available_sem,
                vk::Fence::null(),
            )
        } {
            Ok(v) => v,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.needs_swapchain_recreate = true;
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };
        if suboptimal {
            self.needs_swapchain_recreate = true;
        }

        let view_proj = frame.projection * frame.view;
        let mut object_data: Vec<ObjectDataGpu> = Vec::with_capacity(frame.objects.len());
        for o in frame.objects {
            let mvp = view_proj * o.transform;
            let normal_matrix = o.transform.inverse().transpose();
            object_data.push(ObjectDataGpu {
                model: o.transform,
                model_view_proj: mvp,
                normal_matrix,
                material_index: o.material.0,
                _pad: [0; 3],
            });
        }

        let light_data: Vec<GpuLight> = frame
            .lights
            .iter()
            .map(|l| match *l {
                RenderLight::Directional {
                    direction_world,
                    color,
                    illuminance,
                } => GpuLight {
                    position_world: direction_world.normalize_or_zero().extend(0.0),
                    color_intensity: color.extend(illuminance),
                    kind: 0,
                    range: 0.0,
                    _pad: [0.0; 2],
                },
                RenderLight::Point {
                    position_world,
                    color,
                    intensity,
                    range,
                } => GpuLight {
                    position_world: position_world.extend(1.0),
                    color_intensity: color.extend(intensity),
                    kind: 1,
                    range,
                    _pad: [0.0; 2],
                },
            })
            .collect();

        let frame_data = FrameDataGpu {
            camera_pos_world: frame.camera_position_world.extend(1.0),
            object_buffer_address: self.object_buffers[frame_index].device_address,
            light_buffer_address: self.light_buffers[frame_index].device_address,
            material_buffer_address: self.material_buffer.device_address,
            vertex_buffer_address: self.vertex_buffer.device_address,
            light_count: light_data.len() as u32,
            _pad: [0; 3],
        };

        self.frame_data_buffers[frame_index].write_at(0, bytemuck::bytes_of(&frame_data));
        if !object_data.is_empty() {
            self.object_buffers[frame_index].write_at(0, bytemuck::cast_slice(&object_data));
        }
        if !light_data.is_empty() {
            self.light_buffers[frame_index].write_at(0, bytemuck::cast_slice(&light_data));
        }

        let graphics_cb = self.frames.frames[frame_index].graphics_cb;
        let transfer_cb = self.frames.frames[frame_index].transfer_cb;

        unsafe {
            self.device.device.reset_command_pool(
                self.frames.frames[frame_index].graphics_cb_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;
            self.device.device.reset_command_pool(
                self.frames.frames[frame_index].transfer_cb_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;
        }

        unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(transfer_cb, &begin)?;
        }

        self.record_transfer_cb(transfer_cb, frame_index)?;

        unsafe {
            self.device.device.end_command_buffer(transfer_cb)?;
        }

        unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .device
                .begin_command_buffer(graphics_cb, &begin)?;
        }

        let object_count = object_data.len();
        let swapchain_image = self.swapchain.images[image_index as usize];
        let swapchain_view = self.swapchain.views[image_index as usize];
        self.record_graphics_cb(
            graphics_cb,
            frame_index,
            object_count as u32,
            frame.objects,
            swapchain_image,
            swapchain_view,
        )?;

        unsafe {
            self.device.device.end_command_buffer(graphics_cb)?;
        }

        let render_finished_sem = self.swapchain.render_finished_sems[image_index as usize];
        self.submitter.submit_frame(
            &self.device,
            frame_value,
            transfer_cb,
            graphics_cb,
            self.frames.frames[frame_index].image_available_sem,
            render_finished_sem,
        )?;

        self.frames.frames[frame_index].frame_done_value = frame_value;

        let swapchains = [self.swapchain.swapchain];
        let image_indices = [image_index];
        let wait_sems = [render_finished_sem];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_sems)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let present_result = unsafe {
            self.device
                .swapchain_loader
                .queue_present(self.submitter.graphics_queue(), &present_info)
        };
        match present_result {
            Ok(suboptimal) => {
                if suboptimal {
                    self.needs_swapchain_recreate = true;
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.needs_swapchain_recreate = true;
            }
            Err(e) => return Err(e.into()),
        }

        self.frame_counter = frame_value;
        Ok(())
    }

    fn record_transfer_cb(
        &mut self,
        transfer_cb: vk::CommandBuffer,
        frame_index: usize,
    ) -> anyhow::Result<()> {
        let staging = &self.staging_buffers[frame_index];
        let mut staging_cursor: u64 = 0;

        let mut buffer_copies: Vec<(vk::Buffer, vk::BufferCopy)> = Vec::new();
        let mut image_copies: Vec<(vk::Image, u32, Vec<vk::BufferImageCopy>)> = Vec::new();

        for upload in self.pending_meshes.drain(..) {
            let slot = self.meshes.get(upload.mesh_index).unwrap();
            let vertex_offset_vertices = slot.vertex_offset_vertices;
            let index_offset_indices = slot.index_offset_indices;
            let vert_bytes = bytemuck::cast_slice::<crate::Vertex, u8>(&upload.descriptor.vertices);
            let idx_bytes = bytemuck::cast_slice::<u32, u8>(&upload.descriptor.indices);
            let vertex_byte_size = std::mem::size_of::<crate::Vertex>() as u64;

            let v_offset = staging_cursor;
            staging.write_at(v_offset as usize, vert_bytes);
            staging_cursor += vert_bytes.len() as u64;

            let i_offset = staging_cursor;
            staging.write_at(i_offset as usize, idx_bytes);
            staging_cursor += idx_bytes.len() as u64;

            buffer_copies.push((
                self.vertex_buffer.raw,
                vk::BufferCopy::default()
                    .src_offset(v_offset)
                    .dst_offset(vertex_offset_vertices as u64 * vertex_byte_size)
                    .size(vert_bytes.len() as u64),
            ));
            buffer_copies.push((
                self.index_buffer.raw,
                vk::BufferCopy::default()
                    .src_offset(i_offset)
                    .dst_offset(index_offset_indices as u64 * 4)
                    .size(idx_bytes.len() as u64),
            ));
        }

        let texture_uploads = std::mem::take(&mut self.pending_textures);
        for upload in texture_uploads {
            let slot = &self.textures[upload.texture_index as usize];
            let image_raw = slot.image.raw;
            let (block_w, block_h) = upload.descriptor.format.block_extent();
            let block_size = upload.descriptor.format.block_size_bytes() as u64;
            let mip_count = upload.descriptor.mipmap_count.max(1);

            let mut regions = Vec::with_capacity(upload.descriptor.mip_data.len());
            for (mip_index, mip_bytes) in upload.descriptor.mip_data.iter().enumerate() {
                let mip_u32 = mip_index as u32;
                let mip_w = (upload.descriptor.size.x >> mip_u32).max(1);
                let mip_h = (upload.descriptor.size.y >> mip_u32).max(1);
                let blocks_w = mip_w.div_ceil(block_w);
                let blocks_h = mip_h.div_ceil(block_h);
                let expected_size = blocks_w as u64 * blocks_h as u64 * block_size;
                anyhow::ensure!(
                    mip_bytes.len() as u64 >= expected_size,
                    "mip {mip_index} too small: {} < {} for {}",
                    mip_bytes.len(),
                    expected_size,
                    upload.descriptor.debug_label
                );

                // Per the Vulkan spec, `bufferOffset` for a compressed (block)
                // format must be a multiple of the format's block byte size
                // (16 for BC2/3/5/6/7, 8 for BC1/4). Round the staging cursor
                // up to that alignment before each mip.
                let mip_offset = staging_cursor.next_multiple_of(block_size);
                staging.write_at(mip_offset as usize, mip_bytes);
                staging_cursor = mip_offset + mip_bytes.len() as u64;

                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(mip_offset)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(mip_u32)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width: mip_w,
                            height: mip_h,
                            depth: 1,
                        }),
                );
            }
            image_copies.push((image_raw, mip_count, regions));

            write_bindless_image(
                &self.device,
                self.bindless_set,
                self.bindless_sampler,
                slot.image.view,
                slot.bindless_index,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        }

        let material_uploads = std::mem::take(&mut self.pending_materials);
        for upload in material_uploads {
            let slot = self.materials.get(upload.material_index).unwrap();
            let descriptor = &slot.descriptor;
            let gpu = MaterialDataGpu {
                base_color_factor: descriptor.base_color_factor.to_array(),
                emissive_factor: descriptor.emissive_factor.to_array(),
                metallic_factor: descriptor.metallic_factor,
                roughness_factor: descriptor.roughness_factor,
                occlusion_strength: descriptor.occlusion_strength,
                normal_scale: descriptor.normal_scale,
                base_color_tex: descriptor.base_color.0,
                normal_tex: descriptor.normal.0,
                metallic_roughness_tex: descriptor.metallic_roughness.0,
                occlusion_tex: descriptor.occlusion.0,
                emissive_tex: descriptor.emissive.0,
                _pad: 0,
            };
            let bytes = bytemuck::bytes_of(&gpu);
            let staging_offset = staging_cursor;
            staging.write_at(staging_offset as usize, bytes);
            staging_cursor += bytes.len() as u64;

            buffer_copies.push((
                self.material_buffer.raw,
                vk::BufferCopy::default()
                    .src_offset(staging_offset)
                    .dst_offset(
                        upload.material_index as u64
                            * std::mem::size_of::<MaterialDataGpu>() as u64,
                    )
                    .size(bytes.len() as u64),
            ));
        }

        if staging_cursor > STAGING_BUFFER_SIZE {
            anyhow::bail!(
                "staging buffer overflow: needed {}, have {}",
                staging_cursor,
                STAGING_BUFFER_SIZE
            );
        }

        if image_copies.is_empty() && buffer_copies.is_empty() {
            return Ok(());
        }

        // Single batched UNDEFINED -> TRANSFER_DST_OPTIMAL barrier for all
        // texture uploads this frame.
        let pre_image_barriers: Vec<vk::ImageMemoryBarrier2> = image_copies
            .iter()
            .map(|(image, mip_count, _)| {
                image_memory_barrier(
                    *image,
                    vk::ImageAspectFlags::COLOR,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::empty(),
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    *mip_count,
                )
            })
            .collect();
        if !pre_image_barriers.is_empty() {
            unsafe {
                let dep =
                    vk::DependencyInfo::default().image_memory_barriers(&pre_image_barriers);
                self.device.device.cmd_pipeline_barrier2(transfer_cb, &dep);
            }
        }

        for (dst, region) in &buffer_copies {
            unsafe {
                self.device
                    .device
                    .cmd_copy_buffer(transfer_cb, staging.raw, *dst, std::slice::from_ref(region));
            }
        }
        for (image, _, regions) in &image_copies {
            unsafe {
                self.device.device.cmd_copy_buffer_to_image(
                    transfer_cb,
                    staging.raw,
                    *image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions,
                );
            }
        }

        // Queue the TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL transition
        // for the graphics CB: the transfer queue's queue family doesn't
        // support shader pipeline stages as a barrier dst. The cross-submit
        // transfer-timeline semaphore wait carries the memory dependency
        // through to the graphics submit; the graphics CB then materializes
        // the layout transition in one batched barrier.
        self.pending_image_acquires
            .extend(image_copies.iter().map(|(image, mip_count, _)| {
                image_memory_barrier(
                    *image,
                    vk::ImageAspectFlags::COLOR,
                    vk::PipelineStageFlags2::COPY,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::AccessFlags2::SHADER_READ,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    *mip_count,
                )
            }));

        Ok(())
    }

    fn record_graphics_cb(
        &mut self,
        cb: vk::CommandBuffer,
        frame_index: usize,
        object_count: u32,
        objects: &[RenderObject],
        swapchain_image: vk::Image,
        swapchain_view: vk::ImageView,
    ) -> anyhow::Result<()> {
        let extent = self.swapchain.extent;
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        // Drain post-upload image layout transitions queued by the transfer
        // CB. Batched into a single pipeline_barrier2 call.
        let image_acquires = std::mem::take(&mut self.pending_image_acquires);
        if !image_acquires.is_empty() {
            unsafe {
                let dep = vk::DependencyInfo::default().image_memory_barriers(&image_acquires);
                self.device.device.cmd_pipeline_barrier2(cb, &dep);
            }
        }

        unsafe {
            let hdr_to_color = image_memory_barrier(
                self.hdr_image.raw,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                1,
            );
            let depth_to_attachment = image_memory_barrier(
                self.depth_image.raw,
                vk::ImageAspectFlags::DEPTH,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                1,
            );
            let barriers = [hdr_to_color, depth_to_attachment];
            let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            self.device.device.cmd_pipeline_barrier2(cb, &dep);
        }

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.hdr_image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_image.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });
        let color_attachments = [color_attachment];
        let rendering = vk::RenderingInfo::default()
            .render_area(scissor)
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);

        unsafe {
            self.device.device.cmd_begin_rendering(cb, &rendering);
            self.device.device.cmd_set_viewport(cb, 0, &[viewport]);
            self.device.device.cmd_set_scissor(cb, 0, &[scissor]);
            self.device.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.forward_pipeline.pipeline,
            );
            self.device.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.forward_pipeline.layout,
                0,
                &[self.bindless_set],
                &[],
            );
            self.device.device.cmd_bind_index_buffer(
                cb,
                self.index_buffer.raw,
                0,
                vk::IndexType::UINT32,
            );

            let frame_address = self.frame_data_buffers[frame_index].device_address;
            for (idx, object) in objects.iter().enumerate().take(object_count as usize) {
                let mesh = self
                    .meshes
                    .get(object.mesh.0)
                    .with_context(|| format!("invalid mesh handle {:?}", object.mesh))?;
                let push = ForwardPushConstants {
                    frame_data_address: frame_address,
                    object_index: idx as u32,
                    _pad: 0,
                };
                self.device.device.cmd_push_constants(
                    cb,
                    self.forward_pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push),
                );
                self.device.device.cmd_draw_indexed(
                    cb,
                    mesh.index_count,
                    1,
                    mesh.index_offset_indices,
                    mesh.vertex_offset_vertices as i32,
                    0,
                );
            }

            self.device.device.cmd_end_rendering(cb);

            let hdr_to_sample = image_memory_barrier(
                self.hdr_image.raw,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                1,
            );
            let swap_to_color = image_memory_barrier(
                swapchain_image,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                1,
            );
            let barriers = [hdr_to_sample, swap_to_color];
            let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            self.device.device.cmd_pipeline_barrier2(cb, &dep);
        }

        let swap_color = vk::RenderingAttachmentInfo::default()
            .image_view(swapchain_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });
        let color_attachments = [swap_color];
        let rendering_swap = vk::RenderingInfo::default()
            .render_area(scissor)
            .layer_count(1)
            .color_attachments(&color_attachments);

        unsafe {
            self.device.device.cmd_begin_rendering(cb, &rendering_swap);
            self.device.device.cmd_set_viewport(cb, 0, &[viewport]);
            self.device.device.cmd_set_scissor(cb, 0, &[scissor]);
            self.device.device.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.tonemap_pipeline.pipeline,
            );
            self.device.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.tonemap_pipeline.layout,
                0,
                &[self.bindless_set],
                &[],
            );
            let push = TonemapPushConstants {
                hdr_texture_index: self.hdr_bindless_index,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            self.device.device.cmd_push_constants(
                cb,
                self.tonemap_pipeline.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            self.device.device.cmd_draw(cb, 3, 1, 0, 0);
            self.device.device.cmd_end_rendering(cb);

            let swap_to_present = image_memory_barrier(
                swapchain_image,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::empty(),
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                1,
            );
            let barriers = [swap_to_present];
            let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            self.device.device.cmd_pipeline_barrier2(cb, &dep);
        }

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> anyhow::Result<()> {
        self.swapchain.recreate(&self.device, self.surface_size)?;
        self.hdr_image.destroy(&self.allocator);
        self.depth_image.destroy(&self.allocator);
        let (hdr, depth) = create_screen_targets(&self.allocator, self.swapchain.extent)?;
        self.hdr_image = hdr;
        self.depth_image = depth;
        write_bindless_image(
            &self.device,
            self.bindless_set,
            self.bindless_sampler,
            self.hdr_image.view,
            self.hdr_bindless_index,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        Ok(())
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.device_wait_idle().ok();

            self.forward_pipeline.destroy(&self.device);
            self.tonemap_pipeline.destroy(&self.device);

            self.hdr_image.destroy(&self.allocator);
            self.depth_image.destroy(&self.allocator);

            for buf in self.frame_data_buffers.iter_mut() {
                buf.destroy(&self.allocator);
            }
            for buf in self.object_buffers.iter_mut() {
                buf.destroy(&self.allocator);
            }
            for buf in self.light_buffers.iter_mut() {
                buf.destroy(&self.allocator);
            }
            for buf in self.staging_buffers.iter_mut() {
                buf.destroy(&self.allocator);
            }
            self.vertex_buffer.destroy(&self.allocator);
            self.index_buffer.destroy(&self.allocator);
            self.material_buffer.destroy(&self.allocator);

            for slot in self.textures.iter_mut() {
                slot.image.destroy(&self.allocator);
            }

            self.device
                .device
                .destroy_descriptor_pool(self.bindless_pool, None);
            self.device
                .device
                .destroy_descriptor_set_layout(self.bindless_set_layout, None);
            self.device
                .device
                .destroy_sampler(self.bindless_sampler, None);

            self.submitter.destroy(&self.device);
            self.frames.destroy(&self.device);

            self.swapchain.destroy(&self.device);
        }
    }
}

fn create_bindless_resources(
    device: &Device,
) -> anyhow::Result<(
    vk::DescriptorSetLayout,
    vk::DescriptorPool,
    vk::DescriptorSet,
    vk::Sampler,
)> {
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(8.0)
        .min_lod(0.0)
        .max_lod(vk::LOD_CLAMP_NONE);
    let sampler = unsafe { device.device.create_sampler(&sampler_info, None) }
        .context("create bindless sampler")?;
    device.set_debug_name(sampler, "Bindless Sampler");

    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(BINDLESS_TEXTURE_COUNT)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX)];
    let flags =
        [vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];
    let mut flag_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
        .bindings(&bindings)
        .push_next(&mut flag_info);
    let layout = unsafe {
        device
            .device
            .create_descriptor_set_layout(&layout_info, None)
    }
    .context("create bindless set layout")?;
    device.set_debug_name(layout, "Bindless Set Layout");

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(BINDLESS_TEXTURE_COUNT)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe { device.device.create_descriptor_pool(&pool_info, None) }
        .context("create bindless pool")?;
    device.set_debug_name(pool, "Bindless Pool");

    let set_layouts = [layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&set_layouts);
    let sets = unsafe { device.device.allocate_descriptor_sets(&alloc_info) }
        .context("allocate bindless set")?;
    let set = sets[0];
    device.set_debug_name(set, "Bindless Set");

    Ok((layout, pool, set, sampler))
}

fn write_bindless_image(
    device: &Device,
    set: vk::DescriptorSet,
    sampler: vk::Sampler,
    view: vk::ImageView,
    array_index: u32,
    layout: vk::ImageLayout,
) {
    let image_info = [vk::DescriptorImageInfo::default()
        .sampler(sampler)
        .image_view(view)
        .image_layout(layout)];
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .dst_array_element(array_index)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_info);
    unsafe {
        device.device.update_descriptor_sets(&[write], &[]);
    }
}

fn create_screen_targets(
    allocator: &Allocator,
    extent: vk::Extent2D,
) -> anyhow::Result<(Image, Image)> {
    let hdr = Image::new_2d(
        allocator,
        HDR_FORMAT,
        extent,
        1,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        "HDR Color",
    )?;
    let depth = Image::new_2d(
        allocator,
        DEPTH_FORMAT,
        extent,
        1,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
        "Depth",
    )?;
    Ok((hdr, depth))
}
