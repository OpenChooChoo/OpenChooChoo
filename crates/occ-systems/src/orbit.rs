use glam::{Mat4, Vec3};
use occ_components::{Camera, Transform};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraOrbit {
    pub target: Vec3,
    pub radius: f32,
    pub height: f32,
    pub angular_speed: f32,
    pub angle: f32,
}

pub fn run_camera_orbit(world: &mut hecs::World, dt: f32) {
    for (transform, orbit, _camera) in
        world.query_mut::<(&mut Transform, &mut CameraOrbit, &Camera)>()
    {
        orbit.angle += orbit.angular_speed * dt;

        let (sin, cos) = orbit.angle.sin_cos();
        let position =
            orbit.target + Vec3::new(cos * orbit.radius, sin * orbit.radius, orbit.height);

        let view = Mat4::look_at_rh(position, orbit.target, Vec3::Z);
        let world_from_view = view.inverse();
        let (_scale, rotation, translation) = world_from_view.to_scale_rotation_translation();

        transform.translation = translation;
        transform.rotation = rotation;
    }
}
