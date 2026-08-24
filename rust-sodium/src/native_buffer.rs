//! Safe native memory management ported from Sodium's NativeBuffer.java
//!
//! This module provides safe allocation and deallocation of native memory with
//! leak detection capabilities similar to the original Java implementation.

use std::alloc::{self, Layout};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Global counter for total allocated bytes
static TOTAL_ALLOCATED: AtomicU64 = AtomicU64::new(0);

/// Track active allocations for leak detection
lazy_static::lazy_static! {
    static ref ACTIVE_ALLOCATIONS: Mutex<HashMap<u64, AllocationInfo>> = Mutex::new(HashMap::new());
}

/// Information about an allocation for debugging and leak detection
#[derive(Debug)]
struct AllocationInfo {
    size: usize,
    timestamp: u64,
    backtrace: Option<String>,
}

/// Error type for allocation failures
#[derive(Debug)]
pub enum AllocationError {
    OutOfMemory(usize),
    InvalidSize,
}

impl std::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocationError::OutOfMemory(size) => {
                write!(f, "Failed to allocate {} bytes: out of memory", size)
            }
            AllocationError::InvalidSize => {
                write!(f, "Invalid allocation size")
            }
        }
    }
}

impl std::error::Error for AllocationError {}

/// Allocate a block of native memory with the specified capacity
/// 
/// Returns the pointer address on success, or an error if allocation fails.
pub fn allocate(capacity: usize) -> Result<*mut u8, AllocationError> {
    if capacity == 0 {
        return Err(AllocationError::InvalidSize);
    }

    const MAX_ATTEMPTS: usize = 3;
    let mut attempts = 0;
    let mut ptr: *mut u8 = ptr::null_mut();

    while attempts < MAX_ATTEMPTS {
        unsafe {
            let layout = Layout::from_size_align_unchecked(capacity, 16);
            ptr = alloc::alloc(layout);
            
            if !ptr.is_null() {
                break;
            }
        }

        eprintln!(
            "EMERGENCY: Tried to allocate {} bytes but the allocator reports failure",
            capacity
        );
        eprintln!(
            "EMERGENCY: ... Attempting to reclaim leaked buffers (attempt {}/{})",
            attempts + 1,
            MAX_ATTEMPTS
        );

        // Try to reclaim any leaked buffers
        reclaim_leaked_buffers();
        attempts += 1;
    }

    if ptr.is_null() {
        return Err(AllocationError::OutOfMemory(capacity));
    }

    // Track the allocation
    let ptr_addr = ptr as u64;
    let info = AllocationInfo {
        size: capacity,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        backtrace: None, // Could add backtrace capture here if needed
    };

    if let Ok(mut allocations) = ACTIVE_ALLOCATIONS.lock() {
        allocations.insert(ptr_addr, info);
    }

    TOTAL_ALLOCATED.fetch_add(capacity as u64, Ordering::SeqCst);

    Ok(ptr)
}

/// Allocate memory and copy data from a source buffer
pub fn allocate_and_copy(src: *const u8, capacity: usize) -> Result<*mut u8, AllocationError> {
    let dst = allocate(capacity)?;
    
    unsafe {
        ptr::copy_nonoverlapping(src, dst, capacity);
    }
    
    Ok(dst)
}

/// Deallocate a previously allocated block of memory
/// 
/// # Safety
/// The pointer must have been returned by `allocate` and not already freed.
pub unsafe fn deallocate(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }

    let ptr_addr = ptr as u64;
    
    // Remove from tracking and get the size
    let size = if let Ok(mut allocations) = ACTIVE_ALLOCATIONS.lock() {
        allocations.remove(&ptr_addr).map(|info| info.size)
    } else {
        None
    };

    if let Some(size) = size {
        TOTAL_ALLOCATED.fetch_sub(size as u64, Ordering::SeqCst);
        
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, 16);
            alloc::dealloc(ptr, layout);
        }
    } else {
        eprintln!("Warning: Attempted to free untracked pointer {:p}", ptr);
    }
}

/// Get the total amount of currently allocated memory in bytes
pub fn get_total_allocated() -> u64 {
    TOTAL_ALLOCATED.load(Ordering::SeqCst)
}

/// Get the number of active allocations (for debugging)
pub fn get_active_allocation_count() -> usize {
    ACTIVE_ALLOCATIONS
        .lock()
        .map(|allocs| allocs.len())
        .unwrap_or(0)
}

/// Reclaim leaked buffers by checking for allocations that are too old
/// 
/// In a real implementation, this would use weak references or a GC integration.
/// For now, it just reports statistics about potentially leaked memory.
pub fn reclaim_leaked_buffers() {
    if let Ok(allocations) = ACTIVE_ALLOCATIONS.lock() {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let old_allocations: Vec<_> = allocations
            .iter()
            .filter(|(_, info)| current_time - info.timestamp > 300) // 5 minutes
            .collect();

        if !old_allocations.is_empty() {
            eprintln!(
                "Warning: Found {} potentially leaked allocations totaling {} bytes",
                old_allocations.len(),
                old_allocations.iter().map(|(_, info)| info.size).sum::<usize>()
            );
        }
    }
}

/// Copy data between native buffers
/// 
/// # Safety
/// Both pointers must be valid and the destination must have sufficient capacity.
pub unsafe fn copy_memory(src: *const u8, dst: *mut u8, len: usize) {
    ptr::copy_nonoverlapping(src, dst, len);
}

/// Set a block of memory to a specific value
/// 
/// # Safety
/// The pointer must be valid and have at least `len` bytes of capacity.
pub unsafe fn set_memory(ptr: *mut u8, val: u8, len: usize) {
    ptr::write_bytes(ptr, val, len);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let ptr = allocate(1024).expect("Failed to allocate");
        assert!(!ptr.is_null());
        
        let total = get_total_allocated();
        assert!(total >= 1024);
        
        unsafe {
            deallocate(ptr);
        }
        
        assert_eq!(get_total_allocated(), 0);
    }

    #[test]
    fn test_allocate_and_copy() {
        let data = [1u8, 2, 3, 4, 5];
        let ptr = allocate_and_copy(data.as_ptr(), data.len()).expect("Failed to allocate");
        assert!(!ptr.is_null());
        
        unsafe {
            for i in 0..data.len() {
                assert_eq!(*ptr.add(i), data[i]);
            }
        }
        
        unsafe {
            deallocate(ptr);
        }
    }

    #[test]
    fn test_multiple_allocations() {
        let ptr1 = allocate(100).expect("Failed to allocate");
        let ptr2 = allocate(200).expect("Failed to allocate");
        let ptr3 = allocate(300).expect("Failed to allocate");
        
        assert_eq!(get_total_allocated(), 600);
        
        unsafe {
            deallocate(ptr1);
            deallocate(ptr2);
            deallocate(ptr3);
        }
        
        assert_eq!(get_total_allocated(), 0);
    }

    #[test]
    fn test_zero_allocation_fails() {
        let result = allocate(0);
        assert!(matches!(result, Err(AllocationError::InvalidSize)));
    }

    #[test]
    fn test_active_allocation_count() {
        assert_eq!(get_active_allocation_count(), 0);
        
        let ptr1 = allocate(100).expect("Failed to allocate");
        assert_eq!(get_active_allocation_count(), 1);
        
        let ptr2 = allocate(200).expect("Failed to allocate");
        assert_eq!(get_active_allocation_count(), 2);
        
        unsafe {
            deallocate(ptr1);
            deallocate(ptr2);
        }
        
        assert_eq!(get_active_allocation_count(), 0);
    }
}
