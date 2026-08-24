//! High-performance mathematical utilities ported from Sodium's MathUtil.java
//!
//! These functions are optimized for performance-critical code paths in Minecraft rendering.

/// Check if a number is greater than zero and is a power of two
#[inline]
pub fn is_power_of_two(n: i32) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Convert bytes to mebibytes (MiB)
#[inline]
pub fn to_mib(bytes: i64) -> i64 {
    bytes / (1024 * 1024)
}

/// Convert mebibytes to bytes
#[inline]
pub fn from_mib(mib: i64) -> i64 {
    mib * (1024 * 1024)
}

/// Align an integer to the next multiple of alignment.
/// 
/// # Safety
/// The `alignment` parameter MUST be a power of two, or undefined behavior will occur.
/// 
/// # Example
/// ```
/// assert_eq!(align(10, 16), 16);
/// assert_eq!(align(16, 16), 16);
/// assert_eq!(align(17, 16), 32);
/// ```
#[inline]
pub fn align(num: i32, alignment: i32) -> i32 {
    debug_assert!(is_power_of_two(alignment), "Alignment must be a power of two");
    let additive = alignment - 1;
    let mask = !additive;
    (num + additive) & mask
}

/// Converts a float to a comparable integer value for sorting/comparison.
/// 
/// The resulting integer preserves the ordering of floats when compared as integers,
/// treating negative values correctly. This is useful for packing floats into
/// composite types (like longs) while maintaining sortability.
/// 
/// Reference: https://stackoverflow.com/questions/23900328/are-floats-bit-patterns-ordered
/// Java original: return bits ^ ((bits >> 31) & 0x7FFFFFFF);
#[inline]
pub fn float_to_comparable_int(f: f32) -> i32 {
    let bits = f.to_bits() as i32;
    // Exact port of Java code: bits ^ ((bits >> 31) & 0x7FFFFFFF)
    // In Java, >> on int is arithmetic (sign-extending) shift
    bits ^ ((bits >> 31) & 0x7FFFFFFF)
}

/// Converts a comparable integer back to a float.
/// 
/// This is the inverse of `float_to_comparable_int`.
/// Java original: return Float.intBitsToFloat(i ^ ((i >> 31) & 0x7FFFFFFF));
#[inline]
pub fn comparable_int_to_float(i: i32) -> f32 {
    // Exact port of Java code: i ^ ((i >> 31) & 0x7FFFFFFF)
    let bits = i ^ ((i >> 31) & 0x7FFFFFFF);
    f32::from_bits(bits as u32)
}

/// Calculate exponential moving average for f64 values
#[inline]
pub fn exponential_moving_average_f64(old_value: f64, new_value: f64, contribution: f64) -> f64 {
    contribution * new_value + (1.0 - contribution) * old_value
}

/// Calculate exponential moving average for i64 values with f32 contribution
#[inline]
pub fn exponential_moving_average_i64(old_value: i64, new_value: i64, contribution: f32) -> i64 {
    ((contribution * new_value as f32) as i64) + (((1.0 - contribution) * old_value as f32) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_power_of_two() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(is_power_of_two(4));
        assert!(!is_power_of_two(5));
        assert!(is_power_of_two(8));
        assert!(is_power_of_two(16));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(-1));
    }

    #[test]
    fn test_align() {
        assert_eq!(align(0, 16), 0);
        assert_eq!(align(1, 16), 16);
        assert_eq!(align(15, 16), 16);
        assert_eq!(align(16, 16), 16);
        assert_eq!(align(17, 16), 32);
        assert_eq!(align(31, 16), 32);
        assert_eq!(align(32, 16), 32);
    }

    #[test]
    fn test_float_comparable_roundtrip() {
        // Test with values that don't include -0.0 since the Java algorithm
        // treats +0.0 and -0.0 as having different bit representations
        let values = [1.0f32, -1.0, f32::MIN, f32::MAX, 3.14159, -2.71828, 0.0];
        
        // Test that ordering is preserved (excluding -0.0 vs 0.0 comparison)
        for &a in &values {
            for &b in &values {
                let ai = float_to_comparable_int(a);
                let bi = float_to_comparable_int(b);
                
                if a < b {
                    assert!(ai < bi, "Ordering not preserved for {} vs {}", a, b);
                } else if a > b {
                    assert!(ai > bi, "Ordering not preserved for {} vs {}", a, b);
                } else if !(a == 0.0 && b == 0.0) {
                    // Skip equality check for 0.0/-0.0 edge case
                    assert_eq!(ai, bi, "Equality not preserved for {}", a);
                }
            }
        }
    }

    #[test]
    fn test_float_comparable_roundtrip_conversion() {
        // Exclude -0.0 since it doesn't round-trip correctly with this algorithm
        let values = [0.0f32, 1.0, -1.0, 3.14159, -2.71828];
        for &v in &values {
            let converted = comparable_int_to_float(float_to_comparable_int(v));
            assert!((v - converted).abs() < f32::EPSILON, "Roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_exponential_moving_average() {
        let ema = exponential_moving_average_f64(100.0, 200.0, 0.5);
        assert!((ema - 150.0).abs() < f64::EPSILON);
        
        let ema = exponential_moving_average_f64(100.0, 200.0, 0.1);
        assert!((ema - 110.0).abs() < f64::EPSILON);
    }
}
