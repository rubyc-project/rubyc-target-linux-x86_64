//! JIT size-cap guard: an oversized program is rejected before executing.
//!
//! Runs through this target's backend + memory manager, so it lives in the
//! target crate's test tree rather than rubyc-core (which stays
//! target-agnostic).
//!
//! Note: slow (~100s) — it parses/lowers/codegens ~8k functions to cross the
//! 1 MiB cap. Run it on its own if you only need a quick signal.

use crate::x86::{BACKEND, MEMORY};
use rubyc::native::jit::jit_run;
use rubyc::parser;

#[test]
fn jit_size_cap_is_enforced() {
    // Many small helper functions, each called from `main`, emit real
    // machine code (call sites are not DCE'd). The parser stays happy
    // because no single block is huge, while the total code size exceeds
    // the 1 MiB JIT cap without allocating huge strings in diagnostics.
    let fns: usize = 8_000;
    let helpers: String = (0..fns)
        .map(|i| format!(
            "int f{i}(int v) {{ int a = v + {i}; int b = a * 2; int c = b + a; int d = c - {i}; return d; }}"
        ))
        .collect::<Vec<_>>()
        .join(" ");
    let calls: String = (0..fns)
        .map(|i| format!("f{i}(1);"))
        .collect::<Vec<_>>()
        .join(" ");
    let src = format!(
        "namespace t; class c {{ {helpers} int main() {{ {calls} return 0; }} }}"
    );
    let unit = parser::parse(&src).expect("parse");
    match jit_run(&unit, &BACKEND, &MEMORY) {
        Err(e) => assert!(e.to_string().contains("cap"), "{e}"),
        Ok(_) => panic!("expected size-cap rejection"),
    }
}
