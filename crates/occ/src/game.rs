use std::collections::HashSet;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};

use sdl3::keyboard::Keycode;

use crate::input::InputEvent;

pub enum GameControl {
    Shutdown,
}

pub struct Game {
    events: Receiver<InputEvent>,
    control: Sender<GameControl>,
    running: bool,
    frame: u64,
    window_logical: (u32, u32),
    window_pixel: (u32, u32),
    display_scale: f32,
    held_keys: HashSet<Keycode>,
    mouse_pos: (f32, f32),
}

impl Game {
    pub fn new(
        events: Receiver<InputEvent>,
        control: Sender<GameControl>,
        initial_logical: (u32, u32),
        initial_pixel: (u32, u32),
        initial_scale: f32,
    ) -> Self {
        Self {
            events,
            control,
            running: true,
            frame: 0,
            window_logical: initial_logical,
            window_pixel: initial_pixel,
            display_scale: initial_scale,
            held_keys: HashSet::new(),
            mouse_pos: (0.0, 0.0),
        }
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
            self.frame = self.frame.wrapping_add(1);
        }
        let _ = self.control.send(GameControl::Shutdown);
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
                self.window_logical = logical;
                self.window_pixel = pixel;
                self.display_scale = scale;
            }
            _ => {}
        }
    }

    fn tick(&mut self) {}
}
