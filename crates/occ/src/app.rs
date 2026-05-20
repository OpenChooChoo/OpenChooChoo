use std::collections::HashSet;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Instant;

use anyhow::Context;
use glam::{EulerRot, Quat, Vec3};
use occ_components::{Camera, Light, MeshRenderer, Transform};
use occ_render_vk::{RendererConfig, VulkanRenderer};
use occ_systems::{CameraOrbit, run_camera_orbit, run_render_frame};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use sdl3::keyboard::Keycode;

use crate::input::InputEvent;

pub enum AppControl {
    Shutdown,
}

/// Raw window/display handles passed across the thread boundary.
///
/// SDL3 keeps the window alive on the main thread until shutdown; the raw
/// handle is just a HWND/Xlib id and is safe to use from another thread for
/// the lifetime of the window.
pub struct WindowHandles {
    pub window: RawWindowHandle,
    pub display: RawDisplayHandle,
}

// SAFETY: see doc comment on `WindowHandles`. The handles are valid for as
// long as the window lives on the main thread, which outlives the game thread.
unsafe impl Send for WindowHandles {}

pub struct App {
    events: Receiver<InputEvent>,
    control: Sender<AppControl>,
    game: occ_common::Game,
    renderer: VulkanRenderer,
    running: bool,
    window_size: (u32, u32),
    pixel_size: (u32, u32),
    display_scale: f32,
    held_keys: HashSet<Keycode>,
    mouse_pos: (f32, f32),
    last_tick: Instant,
}

impl App {
    pub fn new(
        events: Receiver<InputEvent>,
        control: Sender<AppControl>,
        handles: WindowHandles,
        initial_logical: (u32, u32),
        initial_pixel: (u32, u32),
        initial_scale: f32,
    ) -> anyhow::Result<Self> {
        let mut renderer = VulkanRenderer::new(
            handles.display,
            handles.window,
            RendererConfig {
                initial_size: initial_pixel,
            },
        )
        .context("failed to create Vulkan renderer")?;

        let helmet_objects = renderer
            .load_gltf(Path::new("assets/DamagedHelmet/DamagedHelmet.gltf"))
            .context("failed to load DamagedHelmet.gltf")?;

        let mut game = occ_common::Game::new();

        for obj in helmet_objects {
            game.world.spawn((
                Transform::from_matrix(obj.transform),
                MeshRenderer {
                    mesh: obj.mesh,
                    material: obj.material,
                },
            ));
        }

        // Camera
        let camera = Camera::default();
        game.world.spawn((
            Transform::default(),
            camera,
            CameraOrbit {
                target: Vec3::ZERO,
                radius: 3.0,
                height: 1.0,
                angular_speed: 0.5,
                angle: 0.0,
            },
        ));

        // Lights
        // Key light: bright directional from above-front-right
        game.world.spawn((
            Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.3, 0.0)),
            Light::Directional {
                color: Vec3::new(1.0, 0.95, 0.9),
                illuminance: 8.0,
            },
        ));
        // Fill point light: warm, low
        game.world.spawn((
            Transform::from_translation(Vec3::new(-2.0, -1.5, 1.5)),
            Light::Point {
                color: Vec3::new(1.0, 0.6, 0.3),
                intensity: 4.0,
                range: 8.0,
            },
        ));

        Ok(Self {
            events,
            control,
            game,
            renderer,
            running: true,
            window_size: initial_logical,
            pixel_size: initial_pixel,
            display_scale: initial_scale,
            held_keys: HashSet::new(),
            mouse_pos: (0.0, 0.0),
            last_tick: Instant::now(),
        })
    }

    pub fn run(&mut self) {
        while self.running {
            loop {
                match self.events.try_recv() {
                    Ok(event) => self.handle_event(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.running = false;
                        break;
                    }
                }
            }
            self.tick();
        }
        let _ = self.control.send(AppControl::Shutdown);
    }

    fn handle_event(&mut self, event: InputEvent) {
        match event {
            InputEvent::Quit | InputEvent::WindowCloseRequested => {
                self.running = false;
            }
            InputEvent::KeyDown {
                keycode: Some(kc), ..
            } => {
                self.held_keys.insert(kc);
            }
            InputEvent::KeyUp {
                keycode: Some(kc), ..
            } => {
                self.held_keys.remove(&kc);
            }
            InputEvent::MouseMotion { x, y, .. } => {
                self.mouse_pos = (x, y);
            }
            InputEvent::WindowSizeChanged {
                logical,
                pixel,
                scale,
            } => {
                self.window_size = logical;
                self.pixel_size = pixel;
                self.display_scale = scale;
                self.renderer.notify_resize(pixel);
            }
            _ => {}
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32();
        self.last_tick = now;
        self.game.tick(dt);

        // Systems
        run_camera_orbit(&mut self.game.world, dt);
        if let Err(e) = run_render_frame(&mut self.game.world, &mut self.renderer, self.pixel_size)
        {
            log::error!("render failed: {e}");
            self.running = false;
        }
    }
}
