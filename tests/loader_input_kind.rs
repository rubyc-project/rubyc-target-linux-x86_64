//! Input classification: shebang skipping and 🐕🇨 magic detection decide
//! whether stdin bytes are source or bytecode.

use rubyc::native::container;
use rubyc::native::target::BytecodeLoader;
use crate::x86::loader::{Loader, LOADER};

fn wrapped(shebang: bool, jump: bool) -> Vec<u8> {
    let mut v = Vec::new();
    if shebang {
        v.extend_from_slice(b"#!/usr/bin/env rubyc\n");
    }
    if jump {
        v.extend_from_slice(&container::JUMP_OVER_MAGIC);
    }
    v.extend_from_slice(&container::MAGIC);
    v.extend_from_slice(&[0x01, 0x00, 0x00]); // kind + version stub
    v
}

#[test]
fn classifies_all_wrappings() {
    assert_eq!(
        LOADER.sniff(&wrapped(true, true)),
        rubyc::native::target::InputKind::Bytecode { payload_start: 31 }
    );
    assert_eq!(
        LOADER.sniff(&wrapped(false, true)),
        rubyc::native::target::InputKind::Bytecode { payload_start: 10 }
    );
    // Without the jump prefix the magic sits right after the shebang.
    assert_eq!(
        LOADER.sniff(&wrapped(true, false)),
        rubyc::native::target::InputKind::Bytecode { payload_start: 29 }
    );
    assert_eq!(
        LOADER.sniff(&wrapped(false, false)),
        rubyc::native::target::InputKind::Bytecode { payload_start: 8 }
    );
}

#[test]
fn plain_bytes_are_source() {
    assert_eq!(
        LOADER.sniff(b"namespace t;"),
        rubyc::native::target::InputKind::Source
    );
    assert_eq!(
        LOADER.sniff(b"#!\nno newline magic"),
        rubyc::native::target::InputKind::Source
    );
    assert_eq!(LOADER.sniff(b""), rubyc::native::target::InputKind::Source);
}

#[test]
fn loader_is_a_zero_sized_singleton() {
    // Both spellings are the same unit struct; the const exists for
    // embedding call sites.
    let _ = Loader;
    let _: Loader = LOADER;
}
