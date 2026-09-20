//! Executes the hand-written `scan_int()` snippet with real stdin plumbing
//! (pipe + dup2, raw syscalls only — the crate stays libc-free).

use crate::x86::memory::{mprotect_rx, syscall3, syscall6_mmap};
use crate::x86::runtime::SCAN_INT;

/// Execute SCAN_INT with stdin redirected from a pipe containing `input`.
#[allow(unsafe_code)]
fn exec_scan(input: &[u8]) -> i64 {
    let page = syscall6_mmap(SCAN_INT.len() + 4096);
    assert!(page > 0, "mmap failed");
    let page = page as *mut u8;
    unsafe {
        // SAFETY: page is a fresh mapping of exactly this size.
        std::ptr::copy_nonoverlapping(SCAN_INT.as_ptr(), page, SCAN_INT.len());
    }
    mprotect_rx(page, SCAN_INT.len());

    // pipe → write input → close write end → dup2 onto fd 0
    let mut fds = [0i32; 2];
    assert_eq!(syscall3(22, fds.as_mut_ptr() as isize, 0, 0), 0);
    for b in input {
        assert_eq!(syscall3(1, fds[1] as isize, (b as *const u8) as isize, 1), 1);
    }
    assert_eq!(syscall3(3, fds[1] as isize, 0, 0), 0);
    assert_eq!(syscall3(33, fds[0] as isize, 0, 0), 0);

    let f: extern "C" fn() -> i64 = unsafe {
        // SAFETY: bytes above are our own assembled snippet, W^X'd.
        std::mem::transmute(page as usize)
    };
    f()
}

#[test]
fn parses_positive() {
    assert_eq!(exec_scan(b"42\n"), 42);
}

#[test]
fn parses_negative() {
    assert_eq!(exec_scan(b"-25\n"), -25);
}

#[test]
fn parses_multi_digit_and_zero() {
    assert_eq!(exec_scan(b"007\n"), 7);
    assert_eq!(exec_scan(b"0\n"), 0);
}

#[test]
fn eof_returns_zero() {
    assert_eq!(exec_scan(b""), 0);
}
