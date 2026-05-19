use sdl3::event::{Event, WindowEvent};
use sdl3::keyboard::{Keycode, Mod, Scancode};
use sdl3::mouse::{MouseButton, MouseWheelDirection};

#[derive(Debug, Clone)]
pub enum InputEvent {
    Quit,
    #[expect(dead_code)]
    KeyDown {
        keycode: Option<Keycode>,
        scancode: Option<Scancode>,
        keymod: Mod,
        repeat: bool,
    },
    #[expect(dead_code)]
    KeyUp {
        keycode: Option<Keycode>,
        scancode: Option<Scancode>,
        keymod: Mod,
    },
    #[expect(dead_code)]
    MouseMotion {
        x: f32,
        y: f32,
        xrel: f32,
        yrel: f32,
    },
    #[expect(dead_code)]
    MouseButtonDown {
        button: MouseButton,
        x: f32,
        y: f32,
        clicks: u8,
    },
    #[expect(dead_code)]
    MouseButtonUp {
        button: MouseButton,
        x: f32,
        y: f32,
    },
    #[expect(dead_code)]
    MouseWheel {
        x: f32,
        y: f32,
        direction: MouseWheelDirection,
    },
    WindowSizeChanged {
        logical: (u32, u32),
        pixel: (u32, u32),
        scale: f32,
    },
    WindowCloseRequested,
}

pub fn translate_event(event: Event) -> Option<InputEvent> {
    Some(match event {
        Event::Quit { .. } => InputEvent::Quit,
        Event::KeyDown {
            keycode,
            scancode,
            keymod,
            repeat,
            ..
        } => InputEvent::KeyDown {
            keycode,
            scancode,
            keymod,
            repeat,
        },
        Event::KeyUp {
            keycode,
            scancode,
            keymod,
            ..
        } => InputEvent::KeyUp {
            keycode,
            scancode,
            keymod,
        },
        Event::MouseMotion {
            x, y, xrel, yrel, ..
        } => InputEvent::MouseMotion { x, y, xrel, yrel },
        Event::MouseButtonDown {
            mouse_btn,
            x,
            y,
            clicks,
            ..
        } => InputEvent::MouseButtonDown {
            button: mouse_btn,
            x,
            y,
            clicks,
        },
        Event::MouseButtonUp {
            mouse_btn, x, y, ..
        } => InputEvent::MouseButtonUp {
            button: mouse_btn,
            x,
            y,
        },
        Event::MouseWheel {
            x, y, direction, ..
        } => InputEvent::MouseWheel { x, y, direction },
        Event::Window {
            win_event: WindowEvent::CloseRequested,
            ..
        } => InputEvent::WindowCloseRequested,
        _ => return None,
    })
}
