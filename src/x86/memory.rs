//! Executable-page memory management for linux_x86_64.
//!
//! Direct syscalls — no libc anywhere in this project. The JIT maps pages
//! writable, fills them, then flips them to read+execute; W and X never
//! coexist (W^X).

use rubyc::native::target::MemoryManager;
use std::io;

/// Syscall numbers (x86-64 Linux).
const SYS_MMAP: isize = 9;
const SYS_MPROTECT: isize = 10;
const SYS_MUNMAP: isize = 11;
const PROT_READ: isize = 1;
const PROT_WRITE: isize = 2;
const PROT_EXEC: isize = 4;
const MAP_PRIVATE: isize = 2;
const MAP_ANONYMOUS: isize = 0x20;
/// Kernel returns -errno in [-4095, -1].
const ERR_MAX: isize = -4095;

pub struct Memory;

impl MemoryManager for Memory {
    fn map_rw(&self, len: usize) -> Result<u64, io::Error> {
        let ret = syscall6_mmap(len);
        check(ret).map(|p| p as u64)
    }

    fn protect_exec(&self, addr: u64, len: usize) -> Result<(), io::Error> {
        let ret = syscall3(
            SYS_MPROTECT,
            addr as isize,
            len as isize,
            PROT_READ | PROT_EXEC,
        );
        check(ret).map(|_| ())
    }

    fn unmap(&self, addr: u64, len: usize) {
        #[allow(unsafe_code)]
        let _ = syscall3(SYS_MUNMAP, addr as isize, len as isize, 0);
    }
}

pub const MEMORY: Memory = Memory;

/// Map a raw kernel return value to `Result`: values in `-4095..=-1` are
/// negative errno, everything else is a success value.
pub(crate) fn check(ret: isize) -> Result<usize, io::Error> {
    if (ERR_MAX..0).contains(&ret) {
        Err(io::Error::from_raw_os_error((-ret) as i32))
    } else {
        Ok(ret as usize)
    }
}

/// Issue a 3-argument syscall. Returns the raw kernel result.
#[allow(unsafe_code)]
pub(crate) fn syscall3(nr: isize, a: isize, b: isize, c: isize) -> isize {
    // SAFETY: register-level syscall per the x86-64 ABI; clobbers rcx/r11.
    unsafe {
        let ret: isize;
        std::arch::asm!(
            "syscall",
            inlateout("rax") nr => ret,
            in("rdi") a,
            in("rsi") b,
            in("rdx") c,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack)
        );
        ret
    }
}

/// mmap takes six arguments; issue it directly. Anonymous private mapping,
/// null hint, no file descriptor — the OS picks the address (JIT regions
/// land above 4 GB, which our codegen already accounts for via movabs).
#[allow(unsafe_code)]
pub(crate) fn syscall6_mmap(len: usize) -> isize {
    // SAFETY: anonymous private RW mapping of `len` bytes at any address.
    unsafe {
        let ret: isize;
        std::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MMAP => ret,
            in("rdi") 0usize,
            in("rsi") len,
            in("rdx") PROT_WRITE,
            in("r10") MAP_PRIVATE | MAP_ANONYMOUS,
            in("r8") (-1isize),
            in("r9") 0isize,
            lateout("rcx") _,
            lateout("r11") _,
        );
        ret
    }
}

/// Flip a scratch region to READ|EXEC after writing machine code into it.
/// Test-only: production code goes through [`Memory::protect_exec`].
#[allow(unsafe_code)]
#[cfg(test)]
pub(crate) fn mprotect_rx(addr: *mut u8, len: usize) {
    // SAFETY: register-level syscall per the x86-64 ABI; clobbers rcx/r11.
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") 10isize => _,
            in("rdi") addr as usize,
            in("rsi") len,
            in("rdx") 5, // PROT_READ | PROT_EXEC
            lateout("rcx") _,
            lateout("r11") _,
        );
    }
}
