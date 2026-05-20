mod app;
mod input;

use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::Duration;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use sdl3::event::{Event, WindowEvent};
use sdl3::video::Window;

use crate::app::{App, AppControl, WindowHandles};
use crate::input::{InputEvent, translate_event};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let sdl_context = sdl3::init()?;
    let video = sdl_context.video()?;

    let window = video
        .window("OpenChooChoo", 1280, 720)
        .position_centered()
        .resizable()
        .high_pixel_density()
        .vulkan()
        .build()?;

    let (mut last_logical, mut last_pixel, mut last_scale) = window_size_snapshot(&window);

    let window_handle = window.window_handle()?.as_raw();
    let display_handle = window.display_handle()?.as_raw();
    let handles = WindowHandles {
        window: window_handle,
        display: display_handle,
    };

    let (event_tx, event_rx) = mpsc::channel();
    let (control_tx, control_rx) = mpsc::channel();

    let game_thread = thread::Builder::new().name("game".into()).spawn(move || {
        let mut app = match App::new(
            event_rx,
            control_tx,
            handles,
            last_logical,
            last_pixel,
            last_scale,
        ) {
            Ok(app) => app,
            Err(e) => {
                log::error!("failed to initialize app: {e:?}");
                return;
            }
        };
        app.run();
    })?;

    let mut event_pump = sdl_context.event_pump()?;

    'main: loop {
        match control_rx.try_recv() {
            Ok(AppControl::Shutdown) | Err(TryRecvError::Disconnected) => break 'main,
            Err(TryRecvError::Empty) => {}
        }
        for sdl_event in event_pump.poll_iter() {
            let size_may_have_changed = matches!(
                &sdl_event,
                Event::Window {
                    win_event: WindowEvent::Resized(_, _)
                        | WindowEvent::PixelSizeChanged(_, _)
                        | WindowEvent::DisplayChanged(_),
                    ..
                }
            );

            if let Some(input_event) = translate_event(sdl_event)
                && event_tx.send(input_event).is_err()
            {
                break 'main;
            }

            if size_may_have_changed {
                let (logical, pixel, scale) = window_size_snapshot(&window);
                if logical != last_logical || pixel != last_pixel || scale != last_scale {
                    last_logical = logical;
                    last_pixel = pixel;
                    last_scale = scale;
                    if event_tx
                        .send(InputEvent::WindowSizeChanged {
                            logical,
                            pixel,
                            scale,
                        })
                        .is_err()
                    {
                        break 'main;
                    }
                }
            }
        }
        thread::sleep(Duration::from_millis(1));
    }

    drop(event_tx);
    let _ = game_thread.join();

    Ok(())
}

fn window_size_snapshot(window: &Window) -> ((u32, u32), (u32, u32), f32) {
    (
        window.size(),
        window.size_in_pixels(),
        window.display_scale(),
    )
}
