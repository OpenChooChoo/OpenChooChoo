use anyhow::Context;
use ash::vk;

use crate::device::Device;

pub(crate) const FRAMES_IN_FLIGHT: usize = 2;

pub(crate) struct Frame {
    pub(crate) graphics_cb_pool: vk::CommandPool,
    pub(crate) graphics_cb: vk::CommandBuffer,
    pub(crate) transfer_cb_pool: vk::CommandPool,
    pub(crate) transfer_cb: vk::CommandBuffer,
    pub(crate) image_available_sem: vk::Semaphore,
    pub(crate) frame_done_value: u64,
}

impl Frame {
    fn new(device: &Device, i: usize) -> anyhow::Result<Self> {
        let graphics_cb_pool = create_command_pool(
            device,
            device.graphics_queue_family,
            &format!("Graphics CB Pool {i}"),
        )?;
        let transfer_cb_pool = create_command_pool(
            device,
            device.transfer_queue_family,
            &format!("Transfer CB Pool {i}"),
        )?;
        let graphics_cb =
            allocate_primary(device, graphics_cb_pool, &format!("Graphics CB {i}"))?;
        let transfer_cb =
            allocate_primary(device, transfer_cb_pool, &format!("Transfer CB {i}"))?;
        let image_available_sem =
            create_binary_semaphore(device, &format!("Image Available Sem {i}"))?;
        Ok(Self {
            graphics_cb_pool,
            graphics_cb,
            transfer_cb_pool,
            transfer_cb,
            image_available_sem,
            frame_done_value: 0,
        })
    }
}

pub(crate) struct FrameSet {
    pub(crate) frames: [Frame; FRAMES_IN_FLIGHT],
}

impl FrameSet {
    pub(crate) fn new(device: &Device) -> anyhow::Result<Self> {
        let frames = [Frame::new(device, 0)?, Frame::new(device, 1)?];
        Ok(Self { frames })
    }

    pub(crate) fn destroy(&mut self, device: &Device) {
        for frame in self.frames.iter_mut() {
            unsafe {
                device.device.destroy_semaphore(frame.image_available_sem, None);
                device.device.destroy_command_pool(frame.graphics_cb_pool, None);
                if frame.transfer_cb_pool != frame.graphics_cb_pool {
                    device.device.destroy_command_pool(frame.transfer_cb_pool, None);
                }
            }
        }
    }
}

fn create_command_pool(
    device: &Device,
    family: u32,
    label: &str,
) -> anyhow::Result<vk::CommandPool> {
    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(family);
    let pool = unsafe { device.device.create_command_pool(&info, None) }
        .with_context(|| format!("failed to create command pool '{label}'"))?;
    device.set_debug_name(pool, label);
    Ok(pool)
}

fn allocate_primary(
    device: &Device,
    pool: vk::CommandPool,
    label: &str,
) -> anyhow::Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb =
        unsafe { device.device.allocate_command_buffers(&info) }?.first().copied().unwrap();
    device.set_debug_name(cb, label);
    Ok(cb)
}

fn create_binary_semaphore(device: &Device, label: &str) -> anyhow::Result<vk::Semaphore> {
    let info = vk::SemaphoreCreateInfo::default();
    let sem = unsafe { device.device.create_semaphore(&info, None) }?;
    device.set_debug_name(sem, label);
    Ok(sem)
}

pub(crate) fn create_timeline_semaphore(
    device: &Device,
    label: &str,
) -> anyhow::Result<vk::Semaphore> {
    let mut tinfo = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut tinfo);
    let sem = unsafe { device.device.create_semaphore(&info, None) }?;
    device.set_debug_name(sem, label);
    Ok(sem)
}

/// Per-frame submitter that schedules a transfer batch followed by a graphics
/// batch using timeline semaphores. In single-queue mode `transfer_queue`
/// mirrors `graphics_queue`; both submits go to the graphics queue and the
/// same wait/signal pattern still applies.
pub(crate) struct FrameSubmitter {
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) transfer_queue: vk::Queue,
    pub(crate) graphics_timeline: vk::Semaphore,
    pub(crate) transfer_timeline: vk::Semaphore,
}

impl FrameSubmitter {
    pub(crate) fn new(device: &Device) -> anyhow::Result<Self> {
        let graphics_timeline = create_timeline_semaphore(device, "Graphics Timeline")?;
        let transfer_timeline = create_timeline_semaphore(device, "Transfer Timeline")?;
        Ok(Self {
            graphics_queue: device.graphics_queue,
            transfer_queue: device.transfer_queue,
            graphics_timeline,
            transfer_timeline,
        })
    }

    pub(crate) fn graphics_timeline(&self) -> vk::Semaphore {
        self.graphics_timeline
    }

    pub(crate) fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    pub(crate) fn destroy(&mut self, device: &Device) {
        unsafe {
            device.device.destroy_semaphore(self.graphics_timeline, None);
            device.device.destroy_semaphore(self.transfer_timeline, None);
        }
    }

    pub(crate) fn submit_frame(
        &mut self,
        device: &Device,
        frame_value: u64,
        transfer_cb: vk::CommandBuffer,
        graphics_cb: vk::CommandBuffer,
        image_available_sem: vk::Semaphore,
        render_finished_sem: vk::Semaphore,
    ) -> anyhow::Result<()> {
        let prev_graphics_wait = frame_value.saturating_sub(FRAMES_IN_FLIGHT as u64);

        let transfer_wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.graphics_timeline)
            .value(prev_graphics_wait)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let transfer_signal = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.transfer_timeline)
            .value(frame_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let transfer_cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(transfer_cb);

        let transfer_waits = [transfer_wait];
        let transfer_signals = [transfer_signal];
        let transfer_cbs = [transfer_cb_info];
        let transfer_submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&transfer_waits)
            .command_buffer_infos(&transfer_cbs)
            .signal_semaphore_infos(&transfer_signals);

        let graphics_wait_transfer = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.transfer_timeline)
            .value(frame_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let graphics_wait_image = vk::SemaphoreSubmitInfo::default()
            .semaphore(image_available_sem)
            .value(0)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        let graphics_signal_render = vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished_sem)
            .value(0)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let graphics_signal_timeline = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.graphics_timeline)
            .value(frame_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        let graphics_cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(graphics_cb);

        let graphics_waits = [graphics_wait_transfer, graphics_wait_image];
        let graphics_signals = [graphics_signal_render, graphics_signal_timeline];
        let graphics_cbs = [graphics_cb_info];
        let graphics_submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&graphics_waits)
            .command_buffer_infos(&graphics_cbs)
            .signal_semaphore_infos(&graphics_signals);

        unsafe {
            device
                .device
                .queue_submit2(self.transfer_queue, &[transfer_submit], vk::Fence::null())
                .context("transfer queue_submit2")?;
            device
                .device
                .queue_submit2(self.graphics_queue, &[graphics_submit], vk::Fence::null())
                .context("graphics queue_submit2")?;
        }
        Ok(())
    }
}
