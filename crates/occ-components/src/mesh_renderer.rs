use occ_common::{MaterialHandle, MeshHandle};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshRenderer {
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}
