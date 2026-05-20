use std::ffi::{CStr, CString, c_void};

use anyhow::{Context, anyhow};
use ash::{Entry, Instance, ext, khr, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub(crate) struct Device {
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) graphics_queue_family: u32,
    pub(crate) transfer_queue: vk::Queue,
    pub(crate) transfer_queue_family: u32,
    pub(crate) surface_loader: khr::surface::Instance,
    pub(crate) swapchain_loader: khr::swapchain::Device,
    debug_utils_instance: Option<ext::debug_utils::Instance>,
    debug_utils_device: Option<ext::debug_utils::Device>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl Device {
    pub(crate) fn new(display_handle: RawDisplayHandle) -> anyhow::Result<Self> {
        let validation_enabled = should_enable_validation();

        let entry = unsafe { Entry::load() }.context("failed to load Vulkan loader")?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"OpenChooChoo")
            .application_version(0)
            .engine_name(c"OpenChooChoo")
            .engine_version(0)
            .api_version(vk::API_VERSION_1_3);

        let mut instance_extensions: Vec<*const i8> =
            ash_window::enumerate_required_extensions(display_handle)
                .context("failed to enumerate required window extensions")?
                .to_vec();
        if validation_enabled {
            instance_extensions.push(ext::debug_utils::NAME.as_ptr());
        }

        let layer_ptrs: Vec<*const i8> = if validation_enabled {
            vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
        } else {
            Vec::new()
        };

        let mut messenger_info = build_debug_messenger_create_info();

        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&layer_ptrs);
        if validation_enabled {
            instance_create_info = instance_create_info.push_next(&mut messenger_info);
        }

        let instance = unsafe { entry.create_instance(&instance_create_info, None) }
            .context("failed to create Vulkan instance")?;

        let (debug_utils_instance, debug_messenger) = if validation_enabled {
            let dbg = ext::debug_utils::Instance::new(&entry, &instance);
            let msg = unsafe { dbg.create_debug_utils_messenger(&messenger_info, None) }
                .context("failed to create debug messenger")?;
            (Some(dbg), Some(msg))
        } else {
            (None, None)
        };

        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let (physical_device, graphics_queue_family, transfer_queue_family, has_dedicated_transfer) =
            pick_physical_device(&instance)?;

        let mut queue_infos = Vec::with_capacity(2);
        let priorities = [1.0_f32];
        queue_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_queue_family)
                .queue_priorities(&priorities),
        );
        if has_dedicated_transfer {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(transfer_queue_family)
                    .queue_priorities(&priorities),
            );
        }

        let device_extensions = [khr::swapchain::NAME.as_ptr()];

        let mut vulkan_11_features = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(true);
        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .buffer_device_address(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_indexing(true)
            .scalar_block_layout(true);
        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let core_features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);
        let mut core_features2 = vk::PhysicalDeviceFeatures2::default()
            .features(core_features)
            .push_next(&mut vulkan_11_features)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut core_features2);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .context("failed to create logical device")?;

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };
        let transfer_queue = if has_dedicated_transfer {
            unsafe { device.get_device_queue(transfer_queue_family, 0) }
        } else {
            graphics_queue
        };

        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

        let debug_utils_device = if validation_enabled {
            Some(ext::debug_utils::Device::new(&instance, &device))
        } else {
            None
        };

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            graphics_queue,
            graphics_queue_family,
            transfer_queue,
            transfer_queue_family,
            surface_loader,
            swapchain_loader,
            debug_utils_instance,
            debug_utils_device,
            debug_messenger,
        })
    }

    pub(crate) fn create_surface(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> anyhow::Result<vk::SurfaceKHR> {
        unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.instance,
                display_handle,
                window_handle,
                None,
            )
        }
        .context("failed to create window surface")
    }

    pub(crate) fn set_debug_name<H: vk::Handle>(&self, handle: H, name: &str) {
        let Some(dbg) = self.debug_utils_device.as_ref() else {
            return;
        };
        let name_cstr = CString::new(name).unwrap();
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(handle)
            .object_name(&name_cstr);
        unsafe { dbg.set_debug_utils_object_name(&info) }.ok();
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            if let (Some(dbg), Some(msg)) =
                (self.debug_utils_instance.as_ref(), self.debug_messenger)
            {
                dbg.destroy_debug_utils_messenger(msg, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn should_enable_validation() -> bool {
    match std::env::var("OCC_RENDER_VALIDATION") {
        Ok(v) if v.eq_ignore_ascii_case("false") || v == "0" => false,
        Ok(v) if v.eq_ignore_ascii_case("true") || v == "1" => true,
        _ => cfg!(debug_assertions),
    }
}

fn build_debug_messenger_create_info<'a>() -> vk::DebugUtilsMessengerCreateInfoEXT<'a> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe {
        if callback_data.is_null() || (*callback_data).p_message.is_null() {
            return vk::FALSE;
        }
        CStr::from_ptr((*callback_data).p_message).to_string_lossy().into_owned()
    };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("vk: {message}");
    } else {
        log::warn!("vk: {message}");
    }
    vk::FALSE
}

fn pick_physical_device(
    instance: &Instance,
) -> anyhow::Result<(vk::PhysicalDevice, u32, u32, bool)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .context("failed to enumerate physical devices")?;

    let mut chosen: Option<(vk::PhysicalDevice, u32, u32, bool, i32)> = None;
    let mut rejected_for_features: Vec<String> = Vec::new();
    for pd in physical_devices {
        let properties = unsafe { instance.get_physical_device_properties(pd) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        if properties.api_version < vk::API_VERSION_1_3 {
            log::debug!("skipping {name}: api < 1.3");
            continue;
        }

        let extensions = unsafe { instance.enumerate_device_extension_properties(pd) }?;
        let has_swapchain = extensions.iter().any(|e| {
            let s = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            s == khr::swapchain::NAME
        });
        if !has_swapchain {
            continue;
        }

        let has_draw_params = {
            let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut v11);
            unsafe { instance.get_physical_device_features2(pd, &mut features2) };
            v11.shader_draw_parameters == vk::TRUE
        };
        if !has_draw_params {
            rejected_for_features.push(format!("{name}: missing shaderDrawParameters"));
            continue;
        }

        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let mut graphics_family: Option<u32> = None;
        let mut dedicated_transfer: Option<u32> = None;
        for (idx, family) in families.iter().enumerate() {
            let idx = idx as u32;
            let graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let transfer = family.queue_flags.contains(vk::QueueFlags::TRANSFER);
            if graphics && graphics_family.is_none() {
                graphics_family = Some(idx);
            }
            if transfer && !graphics && dedicated_transfer.is_none() {
                dedicated_transfer = Some(idx);
            }
        }

        let Some(graphics_family) = graphics_family else {
            continue;
        };
        let (transfer_family, has_dedicated_transfer) = match dedicated_transfer {
            Some(f) => (f, true),
            None => (graphics_family, false),
        };

        let score = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 100,
            _ => 1,
        };

        if chosen.as_ref().is_none_or(|c| c.4 < score) {
            chosen = Some((pd, graphics_family, transfer_family, has_dedicated_transfer, score));
        }
    }

    chosen.map(|c| (c.0, c.1, c.2, c.3)).ok_or_else(|| {
        if rejected_for_features.is_empty() {
            anyhow!("no suitable Vulkan 1.3 physical device found")
        } else {
            anyhow!(
                "no suitable Vulkan 1.3 physical device found; candidates rejected for missing features: [{}]",
                rejected_for_features.join("; ")
            )
        }
    })
}

