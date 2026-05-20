use glam::{Mat4, Vec4};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub fov_y_radians: f32,
    pub z_near: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov_y_radians: 60.0_f32.to_radians(),
            z_near: 0.05,
        }
    }
}

// The world->view basis swap is performed by the view matrix itself (e.g.
// `Mat4::look_at_rh` returns an OpenGL-convention view: Y up, looking down -Z),
// so projection only needs to flip Y for Vulkan's clip space.
const VULKAN_Y_FLIP: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, -1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 1.0, 0.0),
    Vec4::new(0.0, 0.0, 0.0, 1.0),
);

impl Camera {
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        let perspective =
            Mat4::perspective_infinite_reverse_rh(self.fov_y_radians, aspect_ratio, self.z_near);
        VULKAN_Y_FLIP * perspective
    }
}
