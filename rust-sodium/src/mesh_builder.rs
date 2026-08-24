use std::slice;

/// Represents a single vertex in the mesh
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub color: u32, // Packed RGBA
    pub u: f32,
    pub v: f32,
}

/// Builds meshes for multiple chunks in a single batch to minimize FFI overhead.
/// 
/// # Arguments
/// * `chunks` - Slice of pointers to raw chunk data
/// * `stride` - Size of each chunk data block in bytes
/// 
/// # Returns
/// * Vec<u8> - Raw byte buffer containing interleaved vertex data
pub fn build_batched_mesh(chunks: &[*const u8], stride: usize) -> Vec<u8> {
    let mut output_buffer = Vec::with_capacity(65536); // Pre-allocate reasonable size

    for &chunk_ptr in chunks {
        if chunk_ptr.is_null() {
            continue;
        }

        unsafe {
            let chunk_data = slice::from_raw_parts(chunk_ptr, stride);
            
            // Simulate mesh building logic
            // In real implementation, this would iterate block states,
            // check neighbors, cull faces, and generate vertices
            
            // Placeholder: Just copy some data to simulate work
            // Real code would be ~500-800 lines of optimized mesh generation
            if !chunk_data.is_empty() {
                // Example: Generate a dummy quad if data exists
                let vertex = Vertex {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    color: 0xFFFFFFFF,
                    u: 0.0,
                    v: 0.0,
                };
                
                // Push vertex bytes directly
                let vertex_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &vertex as *const Vertex as *const u8,
                        std::mem::size_of::<Vertex>(),
                    )
                };
                output_buffer.extend_from_slice(vertex_bytes);
            }
        }
    }

    output_buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_mesh_empty() {
        let result = build_batched_mesh(&[], 100);
        assert!(result.is_empty());
    }

    #[test]
    fn test_batched_mesh_single_chunk() {
        let dummy_data = vec![1u8; 100];
        let ptr = dummy_data.as_ptr();
        let result = build_batched_mesh(&[ptr], 100);
        
        // Should contain at least one vertex (32 bytes: 3*f32 + u32 + 2*f32)
        assert!(!result.is_empty());
    }

    #[test]
    fn test_batched_mesh_multiple_chunks() {
        let dummy_data1 = vec![1u8; 100];
        let dummy_data2 = vec![2u8; 100];
        let ptr1 = dummy_data1.as_ptr();
        let ptr2 = dummy_data2.as_ptr();
        
        let result = build_batched_mesh(&[ptr1, ptr2], 100);
        
        // Should contain vertices from both chunks (at least 1 vertex per chunk = 32 bytes each)
        // Vertex size: 3*f32 + u32 + 2*f32 = 12+4+8 = 24 bytes
        assert!(result.len() >= 24 * 2);
    }
}
