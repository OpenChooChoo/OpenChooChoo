//! Coordinate spaces in the game:
//!
//! World:
//! +X right
//! +Y forward
//! +Z up
//!
//! glTF:
//! +X right
//! +Y up
//! -Z forward
//!
//! Vulkan clip:
//! +X right
//! +Y down
//! +Z forward in [0, 1]

use glam::{Mat4, Vec4};

/// Matrix to convert from glTF space to world space.
///
/// X -> X
/// Y -> Z
/// Z -> -Y
/// W -> W
pub const GLTF_TO_WORLD_MATRIX: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 1.0, 0.0),
    Vec4::new(0.0, -1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 0.0, 1.0),
);

/// Matrix to convert from world space to glTF space (inverse of `GLTF_TO_WORLD_MATRIX`).
///
/// X -> X
/// Y -> -Z
/// Z -> Y
/// W -> W
pub const WORLD_TO_GLTF: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, -1.0, 0.0),
    Vec4::new(0.0, 1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 0.0, 1.0),
);

// Informational only: documents the world (Y forward, Z up) -> OpenGL-style
// view (Y up, -Z forward) basis swap. In practice this swap is produced by
// `Mat4::look_at_rh` directly, so the projection chain does not multiply by
// this matrix. Kept here as a reference for any future view-matrix derivations.
//
// X -> X
// Y -> -Z
// Z -> Y
// W -> W
pub const WORLD_TO_VULKAN_CLIP_MATRIX: Mat4 = Mat4::from_cols(
    Vec4::new(1.0, 0.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, -1.0, 0.0),
    Vec4::new(0.0, 1.0, 0.0, 0.0),
    Vec4::new(0.0, 0.0, 0.0, 1.0),
);

#[test]
fn gltf_world_round_trip() {
    assert_eq!(GLTF_TO_WORLD_MATRIX.inverse(), WORLD_TO_GLTF);
}
