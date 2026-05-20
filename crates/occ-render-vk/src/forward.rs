use anyhow::Context;
use ash::vk;

use crate::device::Device;

const FORWARD_VERT_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/forward.vert.spv"));
const FORWARD_FRAG_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/forward.frag.spv"));

const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

pub(crate) struct ForwardPipeline {
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
}

impl ForwardPipeline {
    pub(crate) fn new(
        device: &Device,
        bindless_layout: vk::DescriptorSetLayout,
    ) -> anyhow::Result<Self> {
        let vert = create_module(device, FORWARD_VERT_SPV)?;
        device.set_debug_name(vert, "Forward Vertex SPV");
        let frag = create_module(device, FORWARD_FRAG_SPV)?;
        device.set_debug_name(frag, "Forward Fragment SPV");

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(16);

        let set_layouts = [bindless_layout];
        let pc_ranges = [push_range];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&pc_ranges);
        let layout =
            unsafe { device.device.create_pipeline_layout(&layout_info, None) }
                .context("forward pipeline layout")?;
        device.set_debug_name(layout, "Forward Pipeline Layout");

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(c"vertexMain"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(c"fragmentMain"),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        // glTF triangles are CCW. The model→world rotation and view matrix are
        // det=+1, and the clip-space Y-flip plus Vulkan's Y-down framebuffer
        // convention cancel out in the rasterizer's signed-area test, so front
        // faces remain CCW in Vulkan's framebuffer coordinates.
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);
        let attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend_state =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats = [HDR_FORMAT];
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(DEPTH_FORMAT);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&blend_state)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);

        let pipelines = unsafe {
            device.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, err)| anyhow::anyhow!("forward pipeline creation failed: {err:?}"))?;
        let pipeline = pipelines[0];
        device.set_debug_name(pipeline, "Forward Pipeline");

        Ok(Self { layout, pipeline, vert, frag })
    }

    pub(crate) fn destroy(&mut self, device: &Device) {
        unsafe {
            device.device.destroy_pipeline(self.pipeline, None);
            device.device.destroy_pipeline_layout(self.layout, None);
            device.device.destroy_shader_module(self.vert, None);
            device.device.destroy_shader_module(self.frag, None);
        }
    }
}

fn create_module(device: &Device, bytes: &[u8]) -> anyhow::Result<vk::ShaderModule> {
    // SPIR-V requires 4-byte alignment for the `code` slice. `include_bytes!`
    // returns a `&[u8]` with no guaranteed alignment, so copy into a `Vec<u32>`
    // if the source isn't already 4-byte aligned.
    anyhow::ensure!(
        bytes.len().is_multiple_of(4),
        "shader SPIR-V byte length {} is not a multiple of 4",
        bytes.len()
    );
    let owned_words: Vec<u32>;
    let words: &[u32] = if (bytes.as_ptr() as usize).is_multiple_of(4) {
        bytemuck::cast_slice::<u8, u32>(bytes)
    } else {
        owned_words = bytes
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        &owned_words
    };
    let info = vk::ShaderModuleCreateInfo::default().code(words);
    let module = unsafe { device.device.create_shader_module(&info, None) }
        .context("failed to create shader module")?;
    Ok(module)
}
