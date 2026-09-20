//! Executable-page memory management: errno mapping bounds and the
//! map → protect → unmap round trip (W^X discipline).

use rubyc::native::target::MemoryManager;
use crate::x86::memory::{check, MEMORY};

#[test]
fn errno_mapping_bounds() {
    // -1 ..= -4095 map to raw os errors; anything else is a value.
    assert!(check(-1).is_err());
    assert!(check(-4095).is_err());
    assert!(check(-4096).is_ok(), "below errno range is a value");
    assert_eq!(check(0).unwrap(), 0);
    assert_eq!(check(0x4000_0000).unwrap(), 0x4000_0000);
}

#[test]
fn round_trip_map_protect_unmap() {
    let page = MEMORY.map_rw(4096).expect("map");
    // Fill through the still-writable mapping.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::write_bytes(page as *mut u8, 0xC3, 1); // single `ret`
    }
    MEMORY.protect_exec(page, 4096).expect("protect");
    MEMORY.unmap(page, 4096);
}

#[test]
fn protect_on_bad_range_fails_without_crash() {
    // Length zero / unmapped address must return an error, not fault.
    assert!(MEMORY.protect_exec(0xDEAD_0000, 4096).is_err());
}
