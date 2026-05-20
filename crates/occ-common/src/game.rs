#[derive(Debug, Default, Clone, Copy)]
pub struct Time {
    pub frame: u64,
    pub elapsed_secs: f32,
    pub delta_secs: f32,
}

pub struct Game {
    pub world: hecs::World,
    pub time: Time,
}

impl Game {
    pub fn new() -> Self {
        Self {
            world: hecs::World::new(),
            time: Time::default(),
        }
    }

    pub fn tick(&mut self, dt: f32) {
        self.time.frame = self.time.frame.wrapping_add(1);
        self.time.delta_secs = dt;
        self.time.elapsed_secs += dt;
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}
