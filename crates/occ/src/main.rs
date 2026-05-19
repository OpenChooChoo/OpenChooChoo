mod game;
mod input;

use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::Duration;

use sdl3::event::{Event, WindowEvent};
use sdl3::video::Window;

use crate::game::{Game, GameControl};
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
        .build()?;

    let (mut last_logical, mut last_pixel, mut last_scale) = window_size_snapshot(&window);

    let (event_tx, event_rx) = mpsc::channel();
    let (control_tx, control_rx) = mpsc::channel();

    let game_thread = thread::Builder::new().name("game".into()).spawn(move || {
        let mut game = Game::new(event_rx, control_tx, last_logical, last_pixel, last_scale);
        game.run();
    })?;

    let mut event_pump = sdl_context.event_pump()?;

    'main: loop {
        match control_rx.try_recv() {
            Ok(GameControl::Shutdown) | Err(TryRecvError::Disconnected) => break 'main,
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
