//! Linux input classification: shebang handling and magic detection.
//!
//! Unix kernels execute `#!` interpreter lines, so a "saved bytecode"
//! script may carry one; the loader must skip it before checking the RubyC
//! preamble. Other platforms implement their own policies here.

use rubyc::native::container;
use rubyc::native::target::{BytecodeLoader, InputKind};

pub struct Loader;

impl BytecodeLoader for Loader {
    fn sniff(&self, bytes: &[u8]) -> InputKind {
        let mut rest = bytes;

        // Optional `#!interpreter\n` line — the kernel contract on Unix.
        if rest.starts_with(b"#!") {
            match rest.iter().position(|&b| b == b'\n') {
                Some(nl) => rest = &rest[nl + 1..],
                None => return InputKind::Source,
            }
        }

        // Optional jump-over prefix (present in files we emit).
        if rest.starts_with(&container::JUMP_OVER_MAGIC) {
            rest = &rest[container::JUMP_OVER_MAGIC.len()..];
        }

        // Magic is what makes it bytecode; anything else stays source.
        if rest.starts_with(&container::MAGIC) {
            InputKind::Bytecode {
                payload_start: bytes.len() - rest.len() + container::MAGIC.len(),
            }
        } else {
            InputKind::Source
        }
    }
}

pub const LOADER: Loader = Loader;
