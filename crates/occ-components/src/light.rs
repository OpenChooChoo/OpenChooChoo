use glam::Vec3;

// Directional lights use `transform.rotation * Vec3::Y` (world +Y forward) as the
// outgoing light direction; position is taken from `transform.translation` for point lights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Light {
    Directional { color: Vec3, illuminance: f32 },
    Point { color: Vec3, intensity: f32, range: f32 },
}
