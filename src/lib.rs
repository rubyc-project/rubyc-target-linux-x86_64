//! linux_x86_64 target backend — owns the x86-64 machine-code emission path.
//!
//! The complete x86-64 emit pipeline (instruction encoder, register allocator,
//! ELF image writer, shared-library writer, bytecode loader, and executable
//! memory manager) lives in this crate under [`x86`]. It implements the four
//! port traits defined by `rubyc::native::target` so the core can drive it
//! through a `NativePipeline` without depending on this crate.

pub mod mc_serialize;
pub mod x86;

/// The target this backend emits for.
pub use x86::TARGET;
/// Machine-code backend (port: `CodeGenBackend`).
pub use x86::BACKEND;
/// Executable-image writer (port: `ImageWriter`).
pub use x86::IMAGE_WRITER;
/// Input classifier (port: `BytecodeLoader`).
pub use x86::LOADER;
/// Executable-page memory manager (port: `MemoryManager`).
pub use x86::MEMORY;

pub struct LinuxX86_64Backend;

impl rubyc::extensions::TargetExtension for LinuxX86_64Backend {
    fn metadata(&self) -> &rubyc::extensions::ExtensionMetadata {
        static META: std::sync::LazyLock<rubyc::extensions::ExtensionMetadata> =
            std::sync::LazyLock::new(|| rubyc::extensions::ExtensionMetadata {
                name: "linux_x86_64".into(), display_name: "Linux x86-64".into(),
                author: "RubyC contributors".into(), version: env!("CARGO_PKG_VERSION").into(),
                description: "Static ELF and JIT target adapter".into(), license: "MIT".into(),
                homepage: String::new(), repository: String::new(),
            });
        &META
    }

    fn triple(&self) -> &str { "x86_64-unknown-linux-gnu" }
}

impl rubyc_target::TargetBackend for LinuxX86_64Backend {
    fn name(&self) -> &'static str {
        "linux_x86_64"
    }

    fn triple(&self) -> &'static str {
        "x86_64-unknown-linux-gnu"
    }

    fn lower(&self, _program: &[u8]) -> Result<Vec<u8>, String> {
        // The core pipeline handles this; this stub exists so the
        // discovery registry can enumerate the target.
        Err("delegated to core pipeline".into())
    }

    fn write_image(&self, _mc: &[u8], _entry_preamble: bool) -> Result<Vec<u8>, String> {
        Err("delegated to core pipeline".into())
    }
}

// ── Plugin ABI entry points ────────────────────────────────────────────────
//
// These are exported as C symbols so the host binary can load this crate as a
// `.so` plugin and call them through the function pointers in
// `rubyc::plugin::TargetPlugin`.

use std::os::raw::c_char;
use rubyc::native::target::{CodeGenBackend, ImageWriter, MemoryManager};

#[unsafe(no_mangle)]
pub extern "C" fn rbxt_name() -> *const c_char {
    static NAME: &[u8] = b"linux_x86_64\0";
    NAME.as_ptr() as *const c_char
}

#[unsafe(no_mangle)]
pub extern "C" fn rbxt_triple() -> *const c_char {
    static TRIPLE: &[u8] = b"x86_64-unknown-linux-gnu\0";
    TRIPLE.as_ptr() as *const c_char
}

#[unsafe(no_mangle)]
pub extern "C" fn rbxt_base_addr() -> u64 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn rbxt_lower(
    ir: *const c_char,
    len: u64,
) -> *mut rubyc::plugin::PluginBytes {
    let ir = unsafe { std::slice::from_raw_parts(ir as *const u8, len as usize) };
    match lower_target(ir) {
        Ok(bytes) => Box::into_raw(Box::new(rubyc::plugin::PluginBytes::success(bytes))),
        Err(msg) => Box::into_raw(Box::new(rubyc::plugin::PluginBytes::error(msg))),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rbxt_write_image(
    mc: *const c_char,
    len: u64,
    entry_preamble: u8,
) -> *mut rubyc::plugin::PluginBytes {
    let mc = unsafe { std::slice::from_raw_parts(mc as *const u8, len as usize) };
    match write_image_target(mc, entry_preamble != 0) {
        Ok(bytes) => Box::into_raw(Box::new(rubyc::plugin::PluginBytes::success(bytes))),
        Err(msg) => Box::into_raw(Box::new(rubyc::plugin::PluginBytes::error(msg))),
    }
}

/// Lower IR bytes to machine-code bytes.
pub fn lower_target(ir: &[u8]) -> Result<Vec<u8>, String> {
    let program = rubyc::native::ir::decode_program(ir)
        .map_err(|e| format!("IR decode: {e}"))?;
    let mc = x86::BACKEND.lower(&program).map_err(|e| e.to_string())?;
    Ok(mc_serialize::encode(&mc))
}

/// Write machine-code bytes to a shared object (ET_DYN).
#[unsafe(no_mangle)]
pub extern "C" fn rbxt_write_shared(mc: *const c_char, len: u64) -> *mut rubyc::plugin::PluginBytes {
    let mc = unsafe { std::slice::from_raw_parts(mc as *const u8, len as usize) };
    match write_shared_target(mc) {
        Ok(bytes) => Box::into_raw(Box::new(rubyc::plugin::PluginBytes::success(bytes))),
        Err(msg) => Box::into_raw(Box::new(rubyc::plugin::PluginBytes::error(msg))),
    }
}

/// Write machine-code bytes to an executable image.
pub fn write_image_target(mc: &[u8], entry_preamble: bool) -> Result<Vec<u8>, String> {
    let code = mc_serialize::decode(mc)?;
    x86::IMAGE_WRITER
        .write_image(&code, entry_preamble)
        .map_err(|e| e.to_string())
}

/// Write machine-code bytes to a shared object.
pub fn write_shared_target(mc: &[u8]) -> Result<Vec<u8>, String> {
    let code = mc_serialize::decode(mc)?;
    x86::IMAGE_WRITER
        .write_shared_library(&code)
        .map_err(|e| e.to_string())
}

/// Execute machine-code bytes in-memory (JIT). Returns the exit status.
#[unsafe(no_mangle)]
pub extern "C" fn rbxt_run(mc: *const c_char, len: u64) -> i32 {
    let mc = unsafe { std::slice::from_raw_parts(mc as *const u8, len as usize) };
    match run_machine_code(mc) {
        Ok(status) => status,
        Err(_) => 1,
    }
}

pub fn run_machine_code(bytes: &[u8]) -> Result<i32, String> {
    let mut code = mc_serialize::decode(bytes)?;


    // Map pages, relocate, copy, protect, invoke.
    const PAGE: usize = 4096;
    let code_r = code.code.len().div_ceil(PAGE) * PAGE;
    let data_r = code.data.len().div_ceil(PAGE) * PAGE;
    let heap_r = (code.heap_size as usize).div_ceil(PAGE) * PAGE;
    let total = code_r + data_r + heap_r;

    let page = x86::MEMORY.map_rw(total).map_err(|e| format!("mmap: {e}"))?;
    let code_base = page;
    let data_base = page + code_r as u64;
    let heap_base = page + (code_r + data_r) as u64;

    code.relocate(code_base, data_base, heap_base);

    #[allow(unsafe_code)]
    unsafe {
        std::ptr::copy_nonoverlapping(code.code.as_ptr(), page as *mut u8, code.code.len());
        std::ptr::copy_nonoverlapping(code.data.as_ptr(), data_base as *mut u8, code.data.len());
        if heap_r > 0 {
            (heap_base as *mut u64).write_unaligned(0);
            ((heap_base + 8) as *mut u64).write_unaligned(heap_base + 4096);
        }
    }
    x86::MEMORY
        .protect_exec(page, code_r)
        .map_err(|e| format!("mprotect: {e}"))?;

    let status = invoke(page);
    x86::MEMORY.unmap(page, total);
    Ok(status)
}

/// Call the JIT entry point: a zero-argument `() -> i32` function.
fn invoke(addr: u64) -> i32 {

    // SAFETY: addr is RX memory holding backend-emitted code for this ABI.
    #[allow(unsafe_code)]
    unsafe {
        let f: extern "C" fn() -> i32 = std::mem::transmute(addr);
        let result = f();

        result
    }
}

#[cfg(test)]
#[path = "../tests/mod.rs"]
mod tests;
