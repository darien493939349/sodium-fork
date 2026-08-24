//! Sodium Rust - High-performance math and memory utilities ported from Sodium
//! 
//! This module provides JNI bindings for performance-critical code that can be
//! executed from Java Minecraft through Project Panama FFI.
//!
//! ## Modules
//! 
//! - `math_util`: Fast mathematical operations (power-of-two checks, alignment, etc.)
//! - `bitwise_math`: Branchless bitwise comparison operations
//! - `native_buffer`: Safe native memory allocation with leak detection
//! - `frustum_culling`: High-performance frustum culling with SIMD support

pub mod math_util;
pub mod bitwise_math;
pub mod native_buffer;
pub mod frustum_culling;

use jni::JNIEnv;
use jni::objects::{JClass, JByteBuffer};
use jni::sys::{jint, jlong, jobject, JNI_TRUE, JNI_FALSE, jboolean};

/// Initialize the native library - called once when the library loads
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_NativeLib_init(
    _env: JNIEnv,
    _class: JClass,
) -> jboolean {
    // Initialize any global state if needed
    JNI_TRUE
}

// ============================================================================
// MathUtil JNI Bindings
// ============================================================================

/// Check if a number is a power of two
/// Java signature: public static native boolean isPowerOfTwo(int n);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_MathUtil_isPowerOfTwo(
    _env: JNIEnv,
    _class: JClass,
    n: jint,
) -> jboolean {
    if math_util::is_power_of_two(n as i32) {
        JNI_TRUE
    } else {
        JNI_FALSE
    }
}

/// Align a number to the next multiple of alignment (must be power-of-two)
/// Java signature: public static native int align(int num, int alignment);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_MathUtil_align(
    _env: JNIEnv,
    _class: JClass,
    num: jint,
    alignment: jint,
) -> jint {
    math_util::align(num as i32, alignment as i32) as jint
}

/// Convert float to comparable integer for sorting/comparison
/// Java signature: public static native int floatToComparableInt(float f);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_MathUtil_floatToComparableInt(
    _env: JNIEnv,
    _class: JClass,
    f: f32,
) -> jint {
    math_util::float_to_comparable_int(f) as jint
}

/// Convert comparable integer back to float
/// Java signature: public static native float comparableIntToFloat(int i);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_MathUtil_comparableIntToFloat(
    _env: JNIEnv,
    _class: JClass,
    i: jint,
) -> f32 {
    math_util::comparable_int_to_float(i as i32)
}

/// Calculate exponential moving average for doubles
/// Java signature: public static native double exponentialMovingAverageDouble(double old, double new, double contribution);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_MathUtil_exponentialMovingAverageDouble(
    _env: JNIEnv,
    _class: JClass,
    old_value: f64,
    new_value: f64,
    contribution: f64,
) -> f64 {
    math_util::exponential_moving_average_f64(old_value, new_value, contribution)
}

/// Calculate exponential moving average for longs
/// Java signature: public static native long exponentialMovingAverageLong(long old, long new, float contribution);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_MathUtil_exponentialMovingAverageLong(
    _env: JNIEnv,
    _class: JClass,
    old_value: jlong,
    new_value: jlong,
    contribution: f32,
) -> jlong {
    math_util::exponential_moving_average_i64(old_value, new_value, contribution)
}

// ============================================================================
// BitwiseMath JNI Bindings
// ============================================================================

/// Branchless less-than comparison: returns 1 if a < b, otherwise 0
/// Java signature: public static native int lessThan(int a, int b);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_BitwiseMath_lessThan(
    _env: JNIEnv,
    _class: JClass,
    a: jint,
    b: jint,
) -> jint {
    bitwise_math::less_than(a as i32, b as i32) as jint
}

/// Branchless greater-than comparison: returns 1 if a > b, otherwise 0
/// Java signature: public static native int greaterThan(int a, int b);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_BitwiseMath_greaterThan(
    _env: JNIEnv,
    _class: JClass,
    a: jint,
    b: jint,
) -> jint {
    bitwise_math::greater_than(a as i32, b as i32) as jint
}

// ============================================================================
// NativeBuffer JNI Bindings
// ============================================================================

/// Allocate a native buffer with the specified capacity
/// Java signature: public static native long allocateNativeBuffer(int capacity);
/// Returns: Direct address to the allocated memory
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_NativeBuffer_allocateNativeBuffer(
    _env: JNIEnv,
    _class: JClass,
    capacity: jint,
) -> jlong {
    match native_buffer::allocate(capacity as usize) {
        Ok(ptr) => ptr as jlong,
        Err(_) => 0, // Return NULL on failure
    }
}

/// Free a previously allocated native buffer
/// Java signature: public static native void freeNativeBuffer(long address);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_NativeBuffer_freeNativeBuffer(
    _env: JNIEnv,
    _class: JClass,
    address: jlong,
) {
    if address != 0 {
        unsafe {
            native_buffer::deallocate(address as *mut u8);
        }
    }
}

/// Copy data from a Java ByteBuffer to native memory
/// Java signature: public static native long copyFromByteBuffer(java.nio.ByteBuffer src);
/// Returns: Address of the newly allocated native buffer
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_NativeBuffer_copyFromByteBuffer(
    env: JNIEnv,
    _class: JClass,
    src: JByteBuffer,
) -> jlong {
    // Get the direct buffer address and capacity from Java
    if let Ok((addr, capacity)) = env.get_direct_buffer_address(&src).and_then(|ptr| {
        env.get_direct_buffer_capacity(&src).map(|cap| (ptr, cap))
    }) {
        if !addr.is_null() && capacity > 0 {
            match native_buffer::allocate_and_copy(addr as *const u8, capacity) {
                Ok(ptr) => return ptr as jlong,
                Err(_) => return 0,
            }
        }
    }
    0
}

/// Get total allocated native memory in bytes
/// Java signature: public static native long getTotalAllocated();
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_NativeBuffer_getTotalAllocated(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    native_buffer::get_total_allocated() as jlong
}

/// Reclaim leaked buffers (force GC if requested)
/// Java signature: public static native void reclaim(boolean forceGc);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_util_NativeBuffer_reclaim(
    _env: JNIEnv,
    _class: JClass,
    force_gc: jboolean,
) {
    if force_gc == JNI_TRUE {
        // Note: We can't directly trigger JVM GC from native code
        // This would need to be handled by the Java side
        log::warn!("Force GC requested but must be handled by JVM");
    }
    native_buffer::reclaim_leaked_buffers();
}

// ============================================================================
// FrustumCulling JNI Bindings
// ============================================================================

use frustum_culling::{Frustum, Plane, Aabb, build_frustum_planes, normalize_frustum};

/// Create a new frustum from 6 planes
/// Java signature: public static native long createFrustum(float[] planes);
/// planes array format: [left_x, left_y, left_z, left_w, right_x, ...] (24 floats total)
/// Returns: Pointer to Frustum struct
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_render_FrustumCulling_createFrustum(
    env: JNIEnv,
    _class: JClass,
    planes_array: jni::objects::JFloatArray,
) -> jlong {
    let mut planes = [0.0f32; 24];
    if env.get_float_array_region(&planes_array, 0, &mut planes).is_err() {
        return 0;
    }

    let frustum_planes = [
        Plane::new(planes[0], planes[1], planes[2], planes[3]),
        Plane::new(planes[4], planes[5], planes[6], planes[7]),
        Plane::new(planes[8], planes[9], planes[10], planes[11]),
        Plane::new(planes[12], planes[13], planes[14], planes[15]),
        Plane::new(planes[16], planes[17], planes[18], planes[19]),
        Plane::new(planes[20], planes[21], planes[22], planes[23]),
    ];

    let frustum = Box::new(Frustum::new(frustum_planes));
    Box::into_raw(frustum) as jlong
}

/// Free a previously created frustum
/// Java signature: public static native void freeFrustum(long frustumPtr);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_render_FrustumCulling_freeFrustum(
    _env: JNIEnv,
    _class: JClass,
    frustum_ptr: jlong,
) {
    if frustum_ptr != 0 {
        unsafe {
            let _ = Box::from_raw(frustum_ptr as *mut Frustum);
        }
    }
}

/// Test if an AABB is visible to the frustum
/// Java signature: public static native int testAabb(long frustumPtr, float minX, float minY, float minZ, float maxX, float maxY, float maxZ);
/// Returns: 0 = outside, 1 = inside, 2 = intersecting
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_render_FrustumCulling_testAabb(
    _env: JNIEnv,
    _class: JClass,
    frustum_ptr: jlong,
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
) -> jint {
    if frustum_ptr == 0 {
        return 0;
    }

    unsafe {
        let frustum = &*(frustum_ptr as *const Frustum);
        let aabb = Aabb::new(min_x, min_y, min_z, max_x, max_y, max_z);
        frustum.test_aabb(&aabb) as jint
    }
}

/// Batch test multiple AABBs and return visibility bitmask
/// Java signature: public static native long testAabbBatch(long frustumPtr, float[] minCoords, float[] maxCoords, int count);
/// minCoords/maxCoords format: [x0, y0, z0, x1, y1, z1, ...] (count * 3 floats each)
/// Returns: Bitmask where bit i is set if chunk i is visible
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_render_FrustumCulling_testAabbBatch(
    env: JNIEnv,
    _class: JClass,
    frustum_ptr: jlong,
    min_coords: jni::objects::JFloatArray,
    max_coords: jni::objects::JFloatArray,
    count: jint,
) -> jlong {
    if frustum_ptr == 0 || count <= 0 || count > 64 {
        return 0;
    }

    let count = count as usize;
    let mut mins = vec![0.0f32; count * 3];
    let mut maxs = vec![0.0f32; count * 3];

    if env.get_float_array_region(&min_coords, 0, &mut mins).is_err()
        || env.get_float_array_region(&max_coords, 0, &mut maxs).is_err()
    {
        return 0;
    }

    unsafe {
        let frustum = &*(frustum_ptr as *const Frustum);
        
        let mut aabbs = Vec::with_capacity(count);
        for i in 0..count {
            let idx = i * 3;
            aabbs.push(Aabb::new(
                mins[idx], mins[idx + 1], mins[idx + 2],
                maxs[idx], maxs[idx + 1], maxs[idx + 2],
            ));
        }

        frustum.test_aabb_batch(&aabbs) as jlong
    }
}

/// Build frustum from view-projection matrix
/// Java signature: public static native long buildFrustumFromMatrix(float[] viewProj);
/// viewProj format: 4x4 matrix in row-major order (16 floats)
/// Returns: Pointer to Frustum struct
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_mods_sodium_client_render_FrustumCulling_buildFrustumFromMatrix(
    env: JNIEnv,
    _class: JClass,
    view_proj: jni::objects::JFloatArray,
) -> jlong {
    let mut matrix = [0.0f32; 16];
    if env.get_float_array_region(&view_proj, 0, &mut matrix).is_err() {
        return 0;
    }

    // Convert to 4x4 matrix format
    let mut vp = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            vp[i][j] = matrix[i * 4 + j];
        }
    }

    let mut frustum = build_frustum_planes(vp);
    normalize_frustum(&mut frustum);

    let frustum_box = Box::new(frustum);
    Box::into_raw(frustum_box) as jlong
}

// ============================================================================
// Rust Integration / Heartbeat / Batched Mesh Building
// ============================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::slice;

mod mesh_builder;

// Global Heartbeat Counter to verify Rust is actually running
// Check this value in Java to ensure integration works
static RUST_EXECUTION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Get the current execution count (heartbeat)
/// Java signature: public static native long getExecutionCount();
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_sodium_render_native_RustIntegration_getExecutionCount() -> u64 {
    RUST_EXECUTION_COUNTER.load(Ordering::Relaxed)
}

/// Reset the execution count
/// Java signature: public static native void resetExecutionCount();
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_sodium_render_native_RustIntegration_resetExecutionCount() {
    RUST_EXECUTION_COUNTER.store(0, Ordering::Relaxed);
}

/// Batched Mesh Building Entry Point
/// Receives pointers to multiple chunk data arrays, builds them all in one go,
/// and returns a pointer to the unified vertex buffer.
/// 
/// Java signature: public static native long buildChunkMeshesBatched(long[] chunkDataPtrs, int chunkCount, int stride, int[] outBufferSize);
/// 
/// Args:
///   chunk_data_ptrs: Pointer to an array of pointers (each points to a chunk's block data)
///   chunk_count: Number of chunks to process in this batch
///   stride: Size of each chunk data block
///   out_buffer_size: Output parameter for the resulting buffer size
/// Returns:
///   Pointer to the raw vertex buffer (Java must manage this memory or copy it)
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_sodium_render_native_RustIntegration_buildChunkMeshesBatched(
    chunk_data_ptrs: *const *const u8,
    chunk_count: i32,
    stride: i32,
    out_buffer_size: *mut i32,
) -> *mut std::ffi::c_void {
    // Increment heartbeat - THIS PROVES RUST IS RUNNING
    RUST_EXECUTION_COUNTER.fetch_add(1, Ordering::Relaxed);

    if chunk_data_ptrs.is_null() || chunk_count <= 0 {
        return std::ptr::null_mut();
    }

    unsafe {
        // Create a slice of the input pointers
        let chunks = slice::from_raw_parts(chunk_data_ptrs, chunk_count as usize);
        
        // Call the optimized mesh builder
        let mesh_data = mesh_builder::build_batched_mesh(chunks, stride as usize);

        if mesh_data.is_empty() {
            *out_buffer_size = 0;
            return std::ptr::null_mut();
        }

        // Set output size
        *out_buffer_size = mesh_data.len() as i32;

        // Leak the vector intentionally to transfer ownership to Java
        // Java must call 'freeBuffer' later to avoid leaks
        let ptr = mesh_data.as_ptr() as *mut std::ffi::c_void;
        std::mem::forget(mesh_data); 
        ptr
    }
}

/// Frees a buffer allocated by Rust
/// Java signature: public static native void freeBuffer(long ptr, int size);
#[no_mangle]
pub extern "system" fn Java_net_caffeinemc_sodium_render_native_RustIntegration_freeBuffer(
    ptr: *mut std::ffi::c_void,
    size: i32,
) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        // Reconstruct the vector to drop it and free memory
        let _vec = Vec::from_raw_parts(ptr as *mut u8, size as usize, size as usize);
        // Vector drops here automatically
    }
}
