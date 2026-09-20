//! `ET_DYN` shared-object writer tests.
//!
//! `write_shared_library` builds a dlopen-able image from a `MachineCode`.
//! These tests feed synthetic `MachineCode`s (no native exec needed) and
//! assert the structural facts dlopen depends on: the ELF/class/machine
//! header, the 🐕🇨 container preamble at offset 64, the five program
//! headers, the code/trampoline blob, and the dynamic/relocation tables.

use crate::x86::shlib::write_shared_library;
use rubyc::native::target::{MachineCode, Reloc, RelocKind, Section, Sym};

fn read_u16(img: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([img[at], img[at + 1]])
}

fn read_u32(img: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([img[at], img[at + 1], img[at + 2], img[at + 3]])
}

fn read_u64(img: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(img[at..at + 8].try_into().unwrap())
}

/// Minimal `MachineCode`: a single 5-byte `ret`-style body, no exports, no
/// relocations. Enough to walk the full layout computation.
fn bare_mc() -> MachineCode {
    MachineCode {
        code: vec![0xc3],
        data: vec![],
        fn_bases: vec![0],
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

/// `MachineCode` with one export (arity 2), so a trampoline is emitted and
/// `.dynsym`/`.dynstr`/`.hash` get a real symbol.
fn one_export_mc() -> MachineCode {
    let mut mc = bare_mc();
    // Body is at fn_bases[0] = 0; give it a couple of bytes.
    mc.code = vec![0xc3, 0x90];
    mc.fn_bases = vec![0];
    mc.exports = vec![("my_add".to_string(), 0, 2)];
    mc
}

/// `MachineCode` exercising the GOT/RELA path: a code-absolute relocation to
/// the data section (so `count_code_abs64` > 0 and a GOT slot is written),
/// plus a data-absolute relocation.
fn with_relocs_mc() -> MachineCode {
    let mut mc = bare_mc();
    // Code blob: a 10-byte movabs window at site 0 (immediate at 2..10),
    // padded so `encode_got_load`'s reads (at-2 .. at+5) stay in bounds.
    mc.code = vec![0x48, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0, 0xc3];
    mc.data = vec![0, 0, 0, 0, 0, 0, 0, 0];
    mc.fn_bases = vec![0];
    mc.relocs = vec![
        // Abs64 in Code → gets a GOT slot + rewritten mov.
        Reloc { kind: RelocKind::Abs64, section: Section::Code, site: 2, sym: Sym::String(0) },
        // Abs64 in Data → emits a RELATIVE relocation record (no GOT).
        Reloc { kind: RelocKind::Abs64, section: Section::Data, site: 0, sym: Sym::Fn(0) },
    ];
    mc
}

#[test]
fn elf_header_is_x86_64_et_dyn() {
    let img = write_shared_library(&bare_mc()).unwrap();
    assert_eq!(&img[0..4], b"\x7fELF");
    assert_eq!(img[4], 2, "ELFCLASS64");
    assert_eq!(img[5], 1, "little-endian");
    assert_eq!(read_u16(&img, 16), 3, "ET_DYN");
    assert_eq!(read_u16(&img, 18), 62, "EM_X86_64");
    assert_eq!(read_u32(&img, 20), 1, "e_version");
    assert_eq!(read_u64(&img, 24), 0, "e_entry = none");
    assert_eq!(read_u64(&img, 32), 80, "e_phoff = 80 (after 13-byte preamble)");
    assert_eq!(read_u16(&img, 54), 56, "e_phentsize");
    assert_eq!(read_u16(&img, 56), 5, "e_phnum");
    assert_eq!(read_u16(&img, 58), 64, "e_shentsize");
    assert_eq!(read_u16(&img, 60), 9, "e_shnum");
}

#[test]
fn container_preamble_at_offset_64_is_shared_library() {
    let img = write_shared_library(&bare_mc()).unwrap();
    // 2-byte jump, 8-byte magic, kind byte, 2-byte version.
    assert_eq!(&img[64..66], &[0xEB, 0x08], "JUMP_OVER_MAGIC");
    assert_eq!(&img[66..74], &[0xF0, 0x9F, 0x90, 0x95, 0xF0, 0x9F, 0x87, 0xA8], "MAGIC");
    assert_eq!(img[74], 0x03, "Kind::SharedLibrary");
    assert_eq!(
        read_u16(&img, 75),
        rubyc::bytecode::format::FORMAT_VERSION,
        "FORMAT_VERSION"
    );
}

#[test]
fn program_headers_cover_rx_rw_dynamic_and_gnu_stack() {
    let img = write_shared_library(&one_export_mc()).unwrap();
    let at = |i: usize| 80 + i * 56;
    assert_eq!(read_u32(&img, at(0)), 1, "PT_LOAD");
    assert_eq!(read_u32(&img, at(0) + 4), 5, "RX: PF_R|PF_X");
    assert_eq!(read_u64(&img, at(0) + 16), 0, "RX vaddr 0");
    assert_eq!(read_u32(&img, at(1)), 1, "PT_LOAD");
    assert_eq!(read_u32(&img, at(1) + 4), 6, "RW: PF_R|PF_W");
    assert_eq!(read_u32(&img, at(2)), 1, "PT_LOAD (heap, memsz only)");
    assert_eq!(read_u64(&img, at(2) + 32), 0, "heap filesz = 0");
    assert_eq!(read_u32(&img, at(3)), 2, "PT_DYNAMIC");
    assert_eq!(read_u32(&img, at(4)), 0x6474_e551, "PT_GNU_STACK");
    assert_eq!(read_u32(&img, at(4) + 4), 6, "GNU_STACK RW (no X)");
}

#[test]
fn code_blob_is_embedded_at_360() {
    let code: Vec<u8> = vec![0x55, 0x48, 0x89, 0xe5, 0xc3];
    let mut mc = bare_mc();
    mc.code = code.clone();
    let img = write_shared_library(&mc).unwrap();
    let CODE_BASE: usize = 80 + 5 * 56;
    assert_eq!(&img[CODE_BASE..CODE_BASE + code.len()], &code[..]);
}

#[test]
fn single_export_emits_trampoline_and_dynsym_entry() {
    let img = write_shared_library(&one_export_mc()).unwrap();
    let CODE_BASE: usize = 80 + 5 * 56;
    // Trampoline sits immediately after the 2-byte code blob. For arity 2
    // (n=2) the emitted trampoline is:
    //   2× `mov r64,r64` (3B each) + `xor edi,edi` (2B) + `call rel32` (5B)
    //   + `ret` (1B) = 14 bytes, ending in 0xC3.
    let tramp_at = CODE_BASE + 2;
    let tail = &img[tramp_at..tramp_at + 14];
    assert!(tail.iter().any(|&b| b == 0xE8), "trampoline has a rel32 call");
    assert_eq!(*tail.last().unwrap(), 0xC3, "trampoline ends with ret");

    // The `xor edi,edi` NULL-receiver prefix.
    assert_eq!(&tail[6..8], &[0x31, 0xFF], "trampoline nulls the receiver");

    // .dynsym must reference the export name in .dynstr. Find "my_add\0".
    let needle: Vec<u8> = b"my_add\0".to_vec();
    let off = img.windows(needle.len()).position(|w| w == &needle).unwrap();
    assert!(off > 0, "export name lives in the string table");
}

#[test]
fn dynamic_table_has_rela_and_symtab_pointers() {
    let img = write_shared_library(&one_export_mc()).unwrap();
    // Walk the .dynamic section (10 entries of 16 bytes, terminated by DT_NULL)
    // and confirm the required tags are present with nonzero values.
    let shnum = read_u16(&img, 60) as usize;
    let shoff = read_u64(&img, 40) as usize;
    let dynamic_idx = (0..shnum)
        .find(|&i| read_u32(&img, shoff + i * 64 + 4) == 6) // SHT_DYNAMIC
        .unwrap();
    let dyn_at = read_u64(&img, shoff + dynamic_idx * 64 + 24) as usize;
    let dyn_size = read_u64(&img, shoff + dynamic_idx * 64 + 32) as usize;

    let mut seen = std::collections::HashMap::new();
    for e in 0..(dyn_size / 16) {
        let tag = read_u64(&img, dyn_at + e * 16);
        let val = read_u64(&img, dyn_at + e * 16 + 8);
        if tag == 0 {
            break;
        }
        seen.insert(tag, val);
    }
    assert!(seen.get(&6).copied().unwrap_or(0) != 0, "DT_SYMTAB set");
    assert!(seen.get(&5).copied().unwrap_or(0) != 0, "DT_STRTAB set");
    assert!(seen.get(&7).copied().unwrap_or(0) != 0, "DT_RELA set");
    assert!(seen.get(&4).copied().unwrap_or(0) != 0, "DT_HASH set");
    assert_eq!(seen.get(&11).copied().unwrap_or(0), 24, "DT_SYMENT = 24");
    assert_eq!(seen.get(&9).copied().unwrap_or(0), 24, "DT_RELAENT = 24");
}

#[test]
fn code_reloc_produces_got_slot_and_rela_record() {
    let img = write_shared_library(&with_relocs_mc()).unwrap();
    // The code movabs window (10 bytes) at CODE_BASE must be rewritten into
    // a `mov rX,[rip+d32]` form: REX.W (0x48) then 0x8B at the opcode slot.
    let CODE_BASE: usize = 80 + 5 * 56;
    // site = 2 → rewritten bytes at site-2=0 (REX), site-1=1 (0x8B), site=2 (modrm).
    assert_eq!(img[CODE_BASE], 0x48, "rewritten REX.W");
    assert_eq!(img[CODE_BASE + 1], 0x8B, "rewritten mov opcode");

    // The 3 bytes trailing the 7-byte `mov [rip+d32]` (the high bytes of the
    // original 8-byte imm64) must be NOPs, not stale zeros — otherwise the CPU
    // executes `00 00 00` (add %al,(%rax)) and segfaults on the first call.
    assert_eq!(img[CODE_BASE + 7], 0x90, "trailing imm64 byte 0 → nop");
    assert_eq!(img[CODE_BASE + 8], 0x90, "trailing imm64 byte 1 → nop");
    assert_eq!(img[CODE_BASE + 9], 0x90, "trailing imm64 byte 2 → nop");

    // A RELATIVE relocation (tag 8) must exist for the code GOT slot plus one
    // for the data-absolute reference → 2 records total.
    let shnum = read_u16(&img, 60) as usize;
    let shoff = read_u64(&img, 40) as usize;
    let rela_idx = (0..shnum)
        .find(|&i| read_u32(&img, shoff + i * 64 + 4) == 4) // SHT_RELA
        .unwrap();
    let rela_at = read_u64(&img, shoff + rela_idx * 64 + 24) as usize;
    let rela_size = read_u64(&img, shoff + rela_idx * 64 + 32) as usize;
    let n_records = rela_size / 24;
    assert_eq!(n_records, 2, "one RELATIVE for the code GOT + one for the data ref");

    // The records must actually be written into the image (a buffer that is
    // built but never copied leaves .rela.dyn all zeros and the GOT never gets
    // fixed up by the loader). Each: r_info = R_X86_64_RELATIVE (8), and
    // r_offset/r_addend are nonzero.
    for i in 0..n_records {
        let r_offset = read_u64(&img, rela_at + i * 24);
        let r_info = read_u64(&img, rela_at + i * 24 + 8);
        let r_addend = read_u64(&img, rela_at + i * 24 + 16);
        assert_eq!(r_info, 8, "record {i} is R_X86_64_RELATIVE");
        assert_ne!(r_offset, 0, "record {i} r_offset nonzero");
        assert_ne!(r_addend, 0, "record {i} r_addend nonzero");
    }
}

/// `MachineCode` with one vtable row, so a `<path>::__vtable` OBJECT
/// symbol is emitted for downstream images to dlsym.
fn one_vtable_mc() -> MachineCode {
    let mut mc = bare_mc();
    mc.data = vec![0u8; 16];
    mc.vtable_offsets = vec![0];
    mc.vtable_names = vec!["du.W".to_string()];
    mc
}

#[test]
fn vtable_row_emits_object_dynsym_entry() {
    let img = write_shared_library(&one_vtable_mc()).unwrap();
    // The `__vtable` name lives in .dynstr.
    let needle: Vec<u8> = b"du.W::__vtable\0".to_vec();
    let name_off = img.windows(needle.len()).position(|w| w == &needle).unwrap();

    // Section headers: locate .dynstr (the STRTAB containing the name)
    // and .dynsym (SHT_DYNSYM = 11).
    let shnum = read_u16(&img, 60) as usize;
    let shoff = read_u64(&img, 40) as usize;
    let sh = |i: usize| shoff + i * 64;
    let dynstr_off = (0..shnum)
        .map(|i| (read_u64(&img, sh(i) + 24) as usize, read_u64(&img, sh(i) + 32) as usize))
        .find(|(off, size)| {
            *off <= name_off && name_off + needle.len() <= *off + *size
        })
        .map(|(off, _)| off)
        .unwrap();
    let dynsym_idx = (0..shnum)
        .find(|&i| read_u32(&img, sh(i) + 4) == 11)
        .unwrap();
    let dynsym_off = read_u64(&img, sh(dynsym_idx) + 24) as usize;
    let dynsym_size = read_u64(&img, sh(dynsym_idx) + 32) as usize;
    let strx = (name_off - dynstr_off) as u32;

    // Walk the 24-byte entries for the OBJECT symbol referencing the name.
    let mut found = false;
    for e in 0..(dynsym_size / 24) {
        let at = dynsym_off + e * 24;
        let st_name = read_u32(&img, at);
        let st_info = img[at + 4];
        let st_shndx = read_u16(&img, at + 6);
        let st_value = read_u64(&img, at + 8);
        if st_name == strx {
            assert_eq!(st_info, 0x11, "STB_GLOBAL|STT_OBJECT");
            assert_eq!(st_shndx, 2, "row lives in .data");
            assert_ne!(st_value, 0, "row vaddr is set");
            found = true;
        }
    }
    assert!(found, "du.W::__vtable has a dynsym entry");
}

/// `MachineCode` with one static slot, so a `<path>::<field>` OBJECT
/// symbol is emitted for downstream images to dlsym.
fn one_static_mc() -> MachineCode {
    let mut mc = bare_mc();
    mc.data = vec![0u8; 16];
    mc.static_offsets = vec![8];
    mc.static_names = vec!["du.W::SF".to_string()];
    mc
}

#[test]
fn static_slot_emits_object_dynsym_entry() {
    let img = write_shared_library(&one_static_mc()).unwrap();
    let needle: Vec<u8> = b"du.W::SF\0".to_vec();
    let name_off = img.windows(needle.len()).position(|w| w == &needle).unwrap();

    let shnum = read_u16(&img, 60) as usize;
    let shoff = read_u64(&img, 40) as usize;
    let sh = |i: usize| shoff + i * 64;
    let dynstr_off = (0..shnum)
        .map(|i| (read_u64(&img, sh(i) + 24) as usize, read_u64(&img, sh(i) + 32) as usize))
        .find(|(off, size)| {
            *off <= name_off && name_off + needle.len() <= *off + *size
        })
        .map(|(off, _)| off)
        .unwrap();
    let dynsym_idx = (0..shnum)
        .find(|&i| read_u32(&img, sh(i) + 4) == 11)
        .unwrap();
    let dynsym_off = read_u64(&img, sh(dynsym_idx) + 24) as usize;
    let dynsym_size = read_u64(&img, sh(dynsym_idx) + 32) as usize;
    let strx = (name_off - dynstr_off) as u32;

    let mut found = false;
    for e in 0..(dynsym_size / 24) {
        let at = dynsym_off + e * 24;
        if read_u32(&img, at) == strx {
            assert_eq!(img[at + 4], 0x11, "STB_GLOBAL|STT_OBJECT");
            assert_eq!(read_u16(&img, at + 6), 2, "slot lives in .data");
            assert_ne!(read_u64(&img, at + 8), 0, "slot vaddr is set");
            found = true;
        }
    }
    assert!(found, "du.W::SF has a dynsym entry");
}

/// `MachineCode` with an `@init` function: DT_INIT must carry its
/// address so the loader runs static initializers at dlopen.
fn init_mc() -> MachineCode {
    let mut mc = bare_mc();
    mc.init_fn = Some(0);
    mc
}

#[test]
fn dt_init_points_at_init_fn() {
    let img = write_shared_library(&init_mc()).unwrap();
    let shnum = read_u16(&img, 60) as usize;
    let shoff = read_u64(&img, 40) as usize;
    let sh = |i: usize| shoff + i * 64;
    // Dynamic table carries DT_INIT (12) with a nonzero address.
    let dyn_idx = (0..shnum)
        .find(|&i| read_u32(&img, sh(i) + 4) == 6) // SHT_DYNAMIC
        .unwrap();
    let dyn_at = read_u64(&img, sh(dyn_idx) + 24) as usize;
    let dyn_size = read_u64(&img, sh(dyn_idx) + 32) as usize;
    let mut saw_init = false;
    for e in 0..(dyn_size / 16) {
        if read_u64(&img, dyn_at + e * 16) == 12 {
            // DT_INIT: address of the @init body (function 0 starts
            // the code blob).
            assert_ne!(read_u64(&img, dyn_at + e * 16 + 8), 0);
            saw_init = true;
        }
    }
    assert!(saw_init, "DT_INIT present");

    // Without @init the entry is zero and the loader skips it.
    let img = write_shared_library(&bare_mc()).unwrap();
    let shnum = read_u16(&img, 60) as usize;
    let shoff = read_u64(&img, 40) as usize;
    let sh = |i: usize| shoff + i * 64;
    let dyn_idx = (0..shnum)
        .find(|&i| read_u32(&img, sh(i) + 4) == 6)
        .unwrap();
    let dyn_at = read_u64(&img, sh(dyn_idx) + 24) as usize;
    let dyn_size = read_u64(&img, sh(dyn_idx) + 32) as usize;
    for e in 0..(dyn_size / 16) {
        if read_u64(&img, dyn_at + e * 16) == 12 {
            assert_eq!(read_u64(&img, dyn_at + e * 16 + 8), 0);
        }
    }
}
