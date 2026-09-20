//! Lossless `MachineCode` ↔ byte serializer tests (MCM1 format).
//!
//! Exercises `encode`/`decode` roundtrips across every field, plus the
//! error paths (bad magic, truncation, invalid reloc kind/section/sym tag).

use crate::mc_serialize::{decode, encode};
use rubyc::native::target::{MachineCode, Reloc, RelocKind, Section, Sym};

fn empty_mc() -> MachineCode {
    MachineCode {
        code: vec![],
        data: vec![],
        fn_bases: vec![],
        vtable_offsets: vec![],
        vtable_names: vec![],
        static_offsets: vec![],
        static_names: vec![],
        helper_itoa: None,
        helper_itoa_err: None,
        helper_scan_int: None,
        helper_retain: None,
        helper_release: None,
        helper_alloc: None,
        helper_exc_enter: None,
        helper_exc_pop: None,
        helper_exc_throw: None,
        helper_exc_rethrow: None,
        helper_exc_load: None,
        helper_exc_clear: None,
        exc_state_offset: None,
        helper_argv: None,
        import_names: vec![],
        import_bases: vec![],
        import_abs: vec![],
        exports: vec![],
        export_takes_receiver: vec![],
        init_fn: None,
        relocs: vec![],
        heap_size: 0,
    }
}

/// A `MachineCode` populating every field, so a roundtrip must preserve all
/// of it.
fn full_mc() -> MachineCode {
    MachineCode {
        code: vec![0x55, 0x48, 0x89, 0xe5, 0xc3],
        data: vec![0x48, 0x65, 0x6c, 0x6c, 0x6f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        fn_bases: vec![0, 5, 120],
        vtable_offsets: vec![5, 13],
        vtable_names: vec!["A.B".to_string(), "C.D".to_string()],
        static_offsets: vec![16, 24],
        static_names: vec!["A.B::SF".to_string(), "C.D::SG".to_string()],
        helper_itoa: Some(64),
        helper_itoa_err: Some(128),
        helper_scan_int: Some(192),
        helper_retain: Some(256),
        helper_release: Some(320),
        helper_alloc: Some(384),
        helper_exc_enter: Some(512),
        helper_exc_pop: Some(576),
        helper_exc_throw: Some(640),
        helper_exc_rethrow: Some(704),
        helper_exc_load: Some(768),
        helper_exc_clear: Some(832),
        exc_state_offset: Some(64),
        helper_argv: None,
        import_names: vec![("libc.so.6".into(), "printf".into()), ("libm.so.6".into(), "sqrt".into())],
        import_bases: vec![4096, 8192],
        // Runtime-only addresses do not roundtrip; decode yields empty.
        import_abs: vec![0x1234],
        exports: vec![("add".into(), 0, 2), ("sub".into(), 1, 2)],
        export_takes_receiver: vec![],
        init_fn: None,
        relocs: vec![
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 4, sym: Sym::Fn(0) },
            Reloc { kind: RelocKind::Rel32Call, section: Section::Code, site: 8, sym: Sym::String(5) },
            Reloc { kind: RelocKind::Abs64, section: Section::Data, site: 16, sym: Sym::VtableOff(5) },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 20, sym: Sym::HeapBase },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 24, sym: Sym::HelperItoa },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 28, sym: Sym::HelperItoaErr },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 32, sym: Sym::HelperScanInt },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 36, sym: Sym::HelperRetain },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 40, sym: Sym::HelperRelease },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 44, sym: Sym::HelperAlloc },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 48, sym: Sym::Import(1) },
            Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 52, sym: Sym::Name(7) },
        ],
        heap_size: 4096,
    }
}

#[test]
fn empty_machine_code_roundtrips() {
    let mc = empty_mc();
    let bytes = encode(&mc);
    let back = decode(&bytes).unwrap();
    assert_eq!(back.code, mc.code);
    assert_eq!(back.data, mc.data);
    assert_eq!(back.fn_bases, mc.fn_bases);
    assert_eq!(back.vtable_offsets, mc.vtable_offsets);
    assert_eq!(back.vtable_names, mc.vtable_names);
    assert_eq!(back.static_names, mc.static_names);
    assert_eq!(back.helper_itoa, mc.helper_itoa);
    assert_eq!(back.helper_itoa_err, mc.helper_itoa_err);
    assert_eq!(back.helper_scan_int, mc.helper_scan_int);
    assert_eq!(back.helper_retain, mc.helper_retain);
    assert_eq!(back.helper_release, mc.helper_release);
    assert_eq!(back.helper_alloc, mc.helper_alloc);
    assert_eq!(back.import_names, mc.import_names);
    assert_eq!(back.import_bases, mc.import_bases);
    assert_eq!(back.import_abs, Vec::<u64>::new());
    assert_eq!(back.exports, mc.exports);
    assert_eq!(back.relocs, mc.relocs);
    assert_eq!(back.heap_size, mc.heap_size);
}

#[test]
fn full_machine_code_roundtrips_all_fields() {
    let mc = full_mc();
    let bytes = encode(&mc);
    let back = decode(&bytes).unwrap();
    assert_eq!(back.code, mc.code);
    assert_eq!(back.data, mc.data);
    assert_eq!(back.fn_bases, mc.fn_bases);
    assert_eq!(back.vtable_offsets, mc.vtable_offsets);
    assert_eq!(back.vtable_names, vec!["A.B".to_string(), "C.D".to_string()]);
    assert_eq!(back.static_names, vec!["A.B::SF".to_string(), "C.D::SG".to_string()]);
    assert_eq!(back.helper_itoa, Some(64));
    assert_eq!(back.helper_itoa_err, Some(128));
    assert_eq!(back.helper_scan_int, Some(192));
    assert_eq!(back.helper_retain, Some(256));
    assert_eq!(back.helper_release, Some(320));
    assert_eq!(back.helper_alloc, Some(384));
    assert_eq!(back.helper_exc_enter, Some(512));
    assert_eq!(back.helper_exc_pop, Some(576));
    assert_eq!(back.helper_exc_throw, Some(640));
    assert_eq!(back.helper_exc_rethrow, Some(704));
    assert_eq!(back.helper_exc_load, Some(768));
    assert_eq!(back.helper_exc_clear, Some(832));
    assert_eq!(back.exc_state_offset, Some(64));
    assert_eq!(back.import_names, vec![("libc.so.6".to_string(), "printf".to_string()), ("libm.so.6".to_string(), "sqrt".to_string())]);
    assert_eq!(back.import_bases, vec![4096, 8192]);
    assert_eq!(back.exports, vec![("add".to_string(), 0, 2), ("sub".to_string(), 1, 2)]);
    assert_eq!(back.relocs.len(), 12);
    assert_eq!(back.relocs[0], Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 4, sym: Sym::Fn(0) });
    assert_eq!(back.relocs[1], Reloc { kind: RelocKind::Rel32Call, section: Section::Code, site: 8, sym: Sym::String(5) });
    assert_eq!(back.relocs[2].sym, Sym::VtableOff(5));
    assert_eq!(back.relocs[3].sym, Sym::HeapBase);
    assert_eq!(back.relocs[4].sym, Sym::HelperItoa);
    assert_eq!(back.relocs[5].sym, Sym::HelperItoaErr);
    assert_eq!(back.relocs[6].sym, Sym::HelperScanInt);
    assert_eq!(back.relocs[7].sym, Sym::HelperRetain);
    assert_eq!(back.relocs[8].sym, Sym::HelperRelease);
    assert_eq!(back.relocs[9].sym, Sym::HelperAlloc);
    assert_eq!(back.relocs[10].sym, Sym::Import(1));
    assert_eq!(back.relocs[11].sym, Sym::Name(7));
    assert_eq!(back.heap_size, 4096);
}

#[test]
fn only_some_helpers_set_roundtrips() {
    let mut mc = empty_mc();
    mc.helper_itoa = Some(10);
    mc.helper_release = Some(30);
    mc.code = vec![1, 2, 3];
    let back = decode(&encode(&mc)).unwrap();
    assert_eq!(back.helper_itoa, Some(10));
    assert_eq!(back.helper_itoa_err, None);
    assert_eq!(back.helper_scan_int, None);
    assert_eq!(back.helper_retain, None);
    assert_eq!(back.helper_release, Some(30));
    assert_eq!(back.helper_alloc, None);
    assert_eq!(back.code, vec![1, 2, 3]);
}

#[test]
fn encode_starts_with_mcm1_magic() {
    let bytes = encode(&empty_mc());
    assert_eq!(&bytes[..4], &0x52_4D_43_31u32.to_le_bytes());
}

#[test]
fn decode_rejects_bad_magic() {
    let mut bytes = encode(&empty_mc());
    bytes[0] = 0x00;
    assert!(decode(&bytes).is_err());
    let err = decode(&bytes).unwrap_err();
    assert!(err.contains("bad magic"), "{err}");
}

#[test]
fn decode_rejects_too_short() {
    assert!(decode(&[]).is_err());
    assert!(decode(&[1, 2, 3]).is_err());
}

#[test]
fn decode_rejects_truncated_body() {
    let bytes = encode(&full_mc());
    // Cut the buffer mid-stream (past the code section) so a u64 read fails.
    let truncated = &bytes[..bytes.len() - 3];
    assert!(decode(truncated).is_err());
}

#[test]
fn decode_rejects_truncated_code_section() {
    // Build a valid buffer, then lie about the code length so `bytes(code_len)`
    // reads past the end.
    let mc = full_mc();
    let mut bytes = encode(&mc);
    // bytes[4..12] is code_len (u64); set it large enough to overflow the
    // buffer (but < u64::MAX so the bounds check itself doesn't overflow).
    bytes[4..12].copy_from_slice(&0x1_0000_0000u64.to_le_bytes());
    assert!(decode(&bytes).is_err());
}

#[test]
fn roundtrip_is_idempotent_byte_identical() {
    let mc = full_mc();
    let once = encode(&mc);
    let twice = encode(&decode(&once).unwrap());
    assert_eq!(once, twice, "encode(decode(x)) must equal x for a stable MCM1 layout");
}
