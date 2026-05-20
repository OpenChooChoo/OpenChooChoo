use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, bail};
use glam::{Mat4, Vec4};
use occ_common::{MaterialHandle, MeshHandle, TextureHandle};

use crate::ktx2_loader::load_ktx2;
use crate::{LoadedGltfObject, MaterialDescriptor, MeshDescriptor, Vertex, VulkanRenderer};

#[derive(Clone, Copy)]
pub(crate) struct DefaultTextures {
    pub(crate) white: TextureHandle,
    pub(crate) flat_normal: TextureHandle,
}

pub(crate) fn load_gltf(
    renderer: &mut VulkanRenderer,
    path: &Path,
) -> anyhow::Result<Vec<LoadedGltfObject>> {
    let defaults = renderer.defaults;

    let document = gltf::Gltf::open(path)
        .with_context(|| format!("failed to open glTF '{}'", path.display()))?;
    let directory = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("glTF path has no parent: {}", path.display()))?
        .to_path_buf();

    let mut buffer_blobs: Vec<Vec<u8>> = Vec::with_capacity(document.buffers().count());
    for buffer in document.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Uri(uri) => {
                let buf_path = directory.join(uri);
                let data = std::fs::read(&buf_path).with_context(|| {
                    format!("failed to read glTF buffer '{}'", buf_path.display())
                })?;
                buffer_blobs.push(data);
            }
            gltf::buffer::Source::Bin => {
                bail!("embedded BIN buffers not supported (path: {})", path.display());
            }
        }
    }

    let mut texture_cache: HashMap<usize, TextureHandle> = HashMap::new();
    let mut load_texture =
        |renderer: &mut VulkanRenderer,
         tex_index: usize,
         label: &str|
         -> anyhow::Result<TextureHandle> {
            if let Some(handle) = texture_cache.get(&tex_index) {
                return Ok(*handle);
            }
            let texture = document
                .textures()
                .nth(tex_index)
                .ok_or_else(|| anyhow::anyhow!("texture index {tex_index} out of bounds"))?;
            let image_index = texture.source().index();
            let image = document
                .images()
                .nth(image_index)
                .ok_or_else(|| anyhow::anyhow!("image index {image_index} out of bounds"))?;
            let uri = match image.source() {
                gltf::image::Source::Uri { uri, .. } => uri.to_string(),
                gltf::image::Source::View { .. } => {
                    bail!("inline glTF image views are not supported (texture '{label}')");
                }
            };
            let texture_path = directory.join(&uri);
            let descriptor = load_ktx2(&texture_path, format!("{label} ({uri})"))?;
            let handle = renderer.add_texture(descriptor)?;
            texture_cache.insert(tex_index, handle);
            Ok(handle)
        };

    let mut mesh_primitive_handles: Vec<Vec<MeshHandle>> = Vec::new();
    let mut mesh_material_indices: Vec<Vec<Option<usize>>> = Vec::new();
    for mesh in document.meshes() {
        let mut primitives_handles = Vec::new();
        let mut primitive_material_indices = Vec::new();
        for primitive in mesh.primitives() {
            let reader =
                primitive.reader(|buffer| buffer_blobs.get(buffer.index()).map(|v| v.as_slice()));
            let positions = reader
                .read_positions()
                .ok_or_else(|| anyhow::anyhow!("primitive missing POSITION"))?
                .collect::<Vec<_>>();
            let normals = reader
                .read_normals()
                .ok_or_else(|| anyhow::anyhow!("primitive missing NORMAL"))?
                .collect::<Vec<_>>();
            // Tangents are not required: the forward fragment shader reconstructs
            // the TBN from screen-space derivatives, so any TANGENT attribute in
            // the source asset is intentionally ignored.
            let uvs = reader
                .read_tex_coords(0)
                .ok_or_else(|| anyhow::anyhow!("primitive missing TEXCOORD_0"))?
                .into_f32()
                .collect::<Vec<_>>();
            let indices: Vec<u32> = reader
                .read_indices()
                .ok_or_else(|| anyhow::anyhow!("primitive missing indices"))?
                .into_u32()
                .collect();

            anyhow::ensure!(
                positions.len() == normals.len() && positions.len() == uvs.len(),
                "primitive attribute counts mismatch"
            );

            let vertices = positions
                .iter()
                .zip(normals.iter())
                .zip(uvs.iter())
                .map(|((p, n), uv)| Vertex { position: *p, normal: *n, uv: *uv })
                .collect::<Vec<_>>();

            let mesh_handle = renderer.add_mesh(MeshDescriptor { vertices, indices });
            primitives_handles.push(mesh_handle);
            primitive_material_indices.push(primitive.material().index());
        }
        mesh_primitive_handles.push(primitives_handles);
        mesh_material_indices.push(primitive_material_indices);
    }

    let mut material_cache: HashMap<Option<usize>, MaterialHandle> = HashMap::new();
    let mut load_material =
        |renderer: &mut VulkanRenderer,
         material_index: Option<usize>|
         -> anyhow::Result<MaterialHandle> {
            if let Some(handle) = material_cache.get(&material_index) {
                return Ok(*handle);
            }
            let descriptor = match material_index {
                Some(idx) => {
                    let m = document
                        .materials()
                        .nth(idx)
                        .ok_or_else(|| anyhow::anyhow!("material index {idx} out of bounds"))?;
                    let pbr = m.pbr_metallic_roughness();
                    let base_color_factor = Vec4::from_array(pbr.base_color_factor());
                    let emissive_factor = glam::Vec3::from_array(m.emissive_factor());
                    let metallic_factor = pbr.metallic_factor();
                    let roughness_factor = pbr.roughness_factor();
                    let occlusion_strength =
                        m.occlusion_texture().map(|t| t.strength()).unwrap_or(1.0);
                    let normal_scale = m.normal_texture().map(|t| t.scale()).unwrap_or(1.0);
                    let base_color = match pbr.base_color_texture() {
                        Some(info) => load_texture(renderer, info.texture().index(), "base_color")?,
                        None => defaults.white,
                    };
                    let metallic_roughness = match pbr.metallic_roughness_texture() {
                        Some(info) => load_texture(renderer, info.texture().index(), "mr")?,
                        None => defaults.white,
                    };
                    let normal = match m.normal_texture() {
                        Some(info) => load_texture(renderer, info.texture().index(), "normal")?,
                        None => defaults.flat_normal,
                    };
                    let occlusion = match m.occlusion_texture() {
                        Some(info) => load_texture(renderer, info.texture().index(), "occlusion")?,
                        None => defaults.white,
                    };
                    let emissive = match m.emissive_texture() {
                        Some(info) => load_texture(renderer, info.texture().index(), "emissive")?,
                        None => defaults.white,
                    };
                    MaterialDescriptor {
                        base_color_factor,
                        emissive_factor,
                        metallic_factor,
                        roughness_factor,
                        occlusion_strength,
                        normal_scale,
                        base_color,
                        normal,
                        metallic_roughness,
                        occlusion,
                        emissive,
                    }
                }
                None => MaterialDescriptor {
                    base_color_factor: Vec4::ONE,
                    emissive_factor: glam::Vec3::ZERO,
                    metallic_factor: 0.0,
                    roughness_factor: 1.0,
                    occlusion_strength: 1.0,
                    normal_scale: 1.0,
                    base_color: defaults.white,
                    normal: defaults.flat_normal,
                    metallic_roughness: defaults.white,
                    occlusion: defaults.white,
                    emissive: defaults.white,
                },
            };
            let handle = renderer.add_material(descriptor);
            material_cache.insert(material_index, handle);
            Ok(handle)
        };

    let mut output = Vec::new();
    let scene = document.default_scene().or_else(|| document.scenes().next());
    let nodes: Vec<_> = match scene {
        Some(scene) => scene.nodes().collect(),
        None => document.nodes().collect(),
    };
    for node in nodes {
        visit_node(
            renderer,
            &node,
            occ_common::coords::GLTF_TO_WORLD_MATRIX,
            &mesh_primitive_handles,
            &mesh_material_indices,
            &mut load_material,
            &mut output,
        )?;
    }

    Ok(output)
}

fn visit_node<'a, F>(
    renderer: &mut VulkanRenderer,
    node: &gltf::Node<'a>,
    parent_transform: Mat4,
    mesh_primitive_handles: &[Vec<MeshHandle>],
    mesh_material_indices: &[Vec<Option<usize>>],
    load_material: &mut F,
    output: &mut Vec<LoadedGltfObject>,
) -> anyhow::Result<()>
where
    F: FnMut(&mut VulkanRenderer, Option<usize>) -> anyhow::Result<MaterialHandle>,
{
    let local: Mat4 = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world_transform = parent_transform * local;

    if let Some(mesh) = node.mesh() {
        let mesh_idx = mesh.index();
        let primitive_handles = &mesh_primitive_handles[mesh_idx];
        let mat_indices = &mesh_material_indices[mesh_idx];
        for (prim_idx, mesh_handle) in primitive_handles.iter().enumerate() {
            let material_handle = load_material(renderer, mat_indices[prim_idx])?;
            output.push(LoadedGltfObject {
                transform: world_transform,
                mesh: *mesh_handle,
                material: material_handle,
            });
        }
    }

    for child in node.children() {
        visit_node(
            renderer,
            &child,
            world_transform,
            mesh_primitive_handles,
            mesh_material_indices,
            load_material,
            output,
        )?;
    }
    Ok(())
}
