use glam::Vec3;
use occ_components::{Camera, Light, MeshRenderer, Transform};
use occ_render_vk::{FrameInput, RenderLight, RenderObject, VulkanRenderer};

pub fn run_render_frame(
    world: &mut hecs::World,
    renderer: &mut VulkanRenderer,
    pixel_size: (u32, u32),
) -> anyhow::Result<()> {
    let objects: Vec<RenderObject> = world
        .query_mut::<(&Transform, &MeshRenderer)>()
        .into_iter()
        .map(|(transform, mr)| RenderObject {
            transform: transform.matrix(),
            mesh: mr.mesh,
            material: mr.material,
        })
        .collect();

    let lights: Vec<RenderLight> = world
        .query_mut::<(&Transform, &Light)>()
        .into_iter()
        .map(|(transform, light)| match *light {
            Light::Directional { color, illuminance } => RenderLight::Directional {
                direction_world: transform.rotation * Vec3::Y,
                color,
                illuminance,
            },
            Light::Point {
                color,
                intensity,
                range,
            } => RenderLight::Point {
                position_world: transform.translation,
                color,
                intensity,
                range,
            },
        })
        .collect();

    let (camera_transform, camera) = world
        .query_mut::<(&Transform, &Camera)>()
        .into_iter()
        .next()
        .map(|(t, c)| (*t, *c))
        .expect("no camera in world");

    let aspect_ratio = pixel_size.0 as f32 / pixel_size.1.max(1) as f32;
    let view = camera_transform.matrix().inverse();
    let projection = camera.projection_matrix(aspect_ratio);

    renderer.render(FrameInput {
        view,
        camera_position_world: camera_transform.translation,
        aspect_ratio,
        projection,
        objects: &objects,
        lights: &lights,
    })
}
