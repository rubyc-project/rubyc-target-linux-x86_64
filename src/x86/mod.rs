//! Linux x86-64 platform support.
//!
//! Owns everything platform-specific for this target: the hand-rolled
//! x86-64 instruction encoder, the ELF64 executable image writer, and the
//! input classifier (shebang + magic sniffing).

pub mod elf;
pub mod encode_x64;
pub mod loader;
pub mod memory;
pub mod regalloc;
pub mod runtime;
pub mod shlib;

use rubyc::native::target::Target;

/// Compile-time witness that keeps the implementation wired to the registry.
pub const TARGET: Target = Target::LinuxX86_64;

/// Machine-code backend for this target.
pub const BACKEND: encode_x64::Backend = encode_x64::Backend;

/// Executable-image writer for this target.
pub const IMAGE_WRITER: elf::Writer = elf::Writer;

/// Input classifier for this target.
pub const LOADER: loader::Loader = loader::Loader;

/// Executable-page memory manager for this target.
pub const MEMORY: memory::Memory = memory::Memory;
