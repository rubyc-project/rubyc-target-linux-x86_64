//! Hand-written runtime snippets: byte-stable sizes, placeholder layout,
//! and branch targets inside ALLOC_REVISED / RELEASE.

use crate::x86::runtime::{
    alloc_snippet, ALLOC_HEAP_BASE_OFFSET, ALLOC_REVISED, ITOA, RELEASE, RETAIN,
};

#[test]
fn snippet_sizes_are_stable() {
    // The encoder's helper-offset bookkeeping relies on these being
    // fixed; a change here must be deliberate.
    assert_eq!(ITOA.len(), 88);
    assert_eq!(alloc_snippet().len(), 42);
    assert_eq!(ALLOC_HEAP_BASE_OFFSET, 2);
}

#[test]
fn alloc_snippet_has_placeholder_for_heap_base() {
    let snip = alloc_snippet();
    assert_eq!(&snip[0..2], &[0x48, 0xB9]); // movabs rcx
    assert_eq!(&snip[2..10], &[0u8; 8]); // zeroed, relocated later
}

#[test]
fn retain_release_are_recognizable_prologues() {
    // __rubyc_retain: test rax,rax then jz over the lock increment.
    assert_eq!(&RETAIN[..5], &[0x48, 0x85, 0xC0, 0x74, 0x05]);
    // __rubyc_release opens with the same null check.
    assert_eq!(&RELEASE[..5], &[0x48, 0x85, 0xC0, 0x74, 0x1E]);
}

#[test]
fn alloc_revised_jumps() {
    // je .bump at offset 0x14+? — locate `74` after test rcx,rcx
    let i = ALLOC_REVISED
        .windows(3)
        .position(|w| w == [0x48, 0x85, 0xC9])
        .unwrap();
    assert_eq!(ALLOC_REVISED[i + 3], 0x74);
    let rel = ALLOC_REVISED[i + 4] as i8 as i64;
    // next instr after jle is i+5; .bump at 0x32 → rel must be 0x32-(i+5)
    assert_eq!((i + 5) as i64 + rel, 0x44); // .bump: mov rax,[r10+8]
}

#[test]
fn release_jnz_skips_freelist_push() {
    // lock dec … then jnz .skip with rel8 0x1e landing past the push seq
    let i = RELEASE
        .windows(5)
        .position(|w| w == [0xF0, 0x48, 0xFF, 0x48, 0x08])
        .unwrap();
    assert_eq!(RELEASE[i + 5], 0x75); // jnz
    let rel = RELEASE[i + 6] as i8 as i64;
    // Both skip paths must land on a `ret`, not past the snippet.
    let target = ((i + 7) as i64 + rel) as usize;
    assert_eq!(RELEASE[target], 0xC3);
    assert!(target < RELEASE.len());
}
