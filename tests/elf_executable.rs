//! Static `ET_EXEC` ELF writer tests (`Writer::write_image`).
//!
//! Covers both `entry_preamble` modes, the three-segment layout, and the
//! relocation-patching paths (Abs64 into code/data, Rel32Call). All inputs
//! are synthetic `MachineCode`s — no native exec required.

use crate::x86::elf::Writer;
use rubyc::native::container::{JUMP_OVER_MAGIC, MAGIC, PREAMBLE_LEN};
use rubyc::native::target::{ImageWriter, MachineCode, Reloc, RelocKind, Section, Sym};

const BASE: u64 = 0x0040_0000;
const CODE_OFF: usize = 64 + 3 * 56; // 232
const PAGE: u64 = 0x1000;

fn read_u16(img: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([img[at], img[at + 1]])
}

fn read_u32(img: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([img[at], img[at + 1], img[at + 2], img[at + 3]])
}

fn read_u64(img: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(img[at..at + 8].try_into().unwrap())
}

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

fn mc_with_data(code: Vec<u8>, data: Vec<u8>) -> MachineCode {
    let mut mc = bare_mc();
    mc.code = code;
    mc.data = data;
    mc
}

#[test]
fn elf_header_is_x86_64_et_exec() {
    let img = Writer.write_image(&bare_mc(), false).unwrap();
    assert_eq!(&img[0..4], b"\x7fELF");
    assert_eq!(img[4], 2, "ELFCLASS64");
    assert_eq!(img[5], 1, "little-endian");
    assert_eq!(read_u16(&img, 16), 2, "ET_EXEC");
    assert_eq!(read_u16(&img, 18), 62, "EM_X86_64");
    assert_eq!(read_u32(&img, 20), 1, "e_version");
    assert_eq!(read_u64(&img, 32), 64, "e_phoff");
    assert_eq!(read_u16(&img, 54), 56, "e_phentsize");
    assert_eq!(read_u16(&img, 56), 3, "e_phnum");
    assert_eq!(read_u16(&img, 58), 6, "e_shnum");
}

#[test]
fn entry_point_points_at_code_start() {
    let img = Writer.write_image(&bare_mc(), false).unwrap();
    let entry = read_u64(&img, 24);
    assert_eq!(entry, BASE + CODE_OFF as u64, "e_entry = BASE + code_off (no preamble)");
}

#[test]
fn entry_preamble_shifts_code_and_entry() {
    let img = Writer.write_image(&bare_mc(), true).unwrap();
    // Preamble sits at code_off; the 2-byte jump + 8-byte magic are written.
    assert_eq!(&img[CODE_OFF..CODE_OFF + 2], &JUMP_OVER_MAGIC);
    assert_eq!(&img[CODE_OFF + 2..CODE_OFF + 10], &MAGIC);
    // e_entry is the fixed segment start (BASE + code_off) in both modes; the
    // jump-over-magic preamble at that address redirects into the real code.
    let entry = read_u64(&img, 24);
    assert_eq!(entry, BASE + CODE_OFF as u64);
}

#[test]
fn code_is_placed_after_optional_preamble() {
    let code: Vec<u8> = vec![0x55, 0x48, 0x89, 0xe5, 0xc3];

    // No preamble: code at CODE_OFF.
    let img = Writer.write_image(&mc_with_data(code.clone(), vec![]), false).unwrap();
    assert_eq!(&img[CODE_OFF..CODE_OFF + code.len()], &code[..]);

    // With preamble: code pushed PREAMBLE_LEN bytes later.
    let img = Writer.write_image(&mc_with_data(code.clone(), vec![]), true).unwrap();
    let at = CODE_OFF + PREAMBLE_LEN;
    assert_eq!(&img[at..at + code.len()], &code[..]);
}

#[test]
fn data_segment_is_page_aligned_and_embedded() {
    // 5-byte code, 16-byte data. data_vaddr = align(0x4000e8+5, 0x1000) = 0x401000.
    let code = vec![0x90u8; 5];
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let img = Writer.write_image(&mc_with_data(code, data.clone()), false).unwrap();

    let code_vaddr = BASE + CODE_OFF as u64;
    let data_vaddr = (code_vaddr + 5).div_ceil(PAGE) * PAGE;
    assert_eq!(data_vaddr, BASE + PAGE, "data vaddr is the next page");

    // RW segment (phdr index 1) carries the data vaddr + size.
    let ph_at = 64 + 1 * 56;
    assert_eq!(read_u32(&img, ph_at), 1, "PT_LOAD");
    assert_eq!(read_u32(&img, ph_at + 4), 6, "R|W");
    assert_eq!(read_u64(&img, ph_at + 16), data_vaddr, "data vaddr");
    assert_eq!(read_u64(&img, ph_at + 32), 16, "data filesz");

    // Data blob is physically at the page-aligned file offset.
    let data_file_off = ((CODE_OFF as u64 + 5).div_ceil(PAGE)) * PAGE;
    assert_eq!(data_file_off, PAGE);
    assert_eq!(&img[PAGE as usize..PAGE as usize + 16], &data[..]);
}

#[test]
fn heap_bss_segment_has_zero_filesz() {
    let mut mc = bare_mc();
    mc.heap_size = 0x2000; // 2 pages
    let img = Writer.write_image(&mc, false).unwrap();

    let code_vaddr = BASE + CODE_OFF as u64;
    let data_vaddr = (code_vaddr + 1).div_ceil(PAGE) * PAGE; // data empty
    let heap_vaddr = (data_vaddr + 0).div_ceil(PAGE) * PAGE; // == data_vaddr

    let ph_at = 64 + 2 * 56;
    assert_eq!(read_u32(&img, ph_at), 1, "PT_LOAD");
    assert_eq!(read_u64(&img, ph_at + 16), heap_vaddr, "heap vaddr");
    assert_eq!(read_u64(&img, ph_at + 32), 0, "BSS filesz = 0");
    assert_eq!(read_u64(&img, ph_at + 40), 0x2000, "heap memsz = 2 pages");
}

#[test]
fn abs64_reloc_patches_code_with_symbol_address() {
    // Code blob: a 10-byte window with the 8-byte immediate at site 2.
    let mut mc = mc_with_data(vec![0x48, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0, 0xc3], vec![0u8; 8]);
    // String(0) lives at data_vaddr + 0.
    mc.relocs = vec![Reloc {
        kind: RelocKind::Abs64,
        section: Section::Code,
        site: 2,
        sym: Sym::String(0),
    }];
    let img = Writer.write_image(&mc, false).unwrap();

    let code_vaddr = BASE + CODE_OFF as u64;
    let data_vaddr = (code_vaddr + 11).div_ceil(PAGE) * PAGE;
    // Immediate at file offset CODE_OFF + 2.
    let patched = read_u64(&img, CODE_OFF + 2);
    assert_eq!(patched, data_vaddr, "Abs64 in code resolved to the data symbol");
}

#[test]
fn abs64_reloc_patches_data_blob() {
    let mut mc = mc_with_data(vec![0x90u8; 4], vec![0u8; 8]);
    // Fn(0) lives at code_blob_vaddr + fn_bases[0]=0 → code_vaddr.
    mc.relocs = vec![Reloc {
        kind: RelocKind::Abs64,
        section: Section::Data,
        site: 0,
        sym: Sym::Fn(0),
    }];
    let img = Writer.write_image(&mc, false).unwrap();

    let code_vaddr = BASE + CODE_OFF as u64;
    let data_vaddr = (code_vaddr + 4).div_ceil(PAGE) * PAGE;
    let data_file_off = ((CODE_OFF as u64 + 4).div_ceil(PAGE)) * PAGE;
    let patched = read_u64(&img, data_file_off as usize);
    assert_eq!(patched, code_vaddr, "Abs64 in data resolved to the function");
    let _ = data_vaddr;
}

#[test]
fn rel32call_reloc_patches_relative_offset() {
    // A 5-byte `call rel32` at code offset 0 (opcode 0xE8, imm at 1..5).
    // The reloc `site` is the offset of the 4-byte immediate = 1.
    let mut mc = mc_with_data(vec![0xe8, 0, 0, 0, 0], vec![]);
    mc.fn_bases = vec![10]; // target body at code_blob + 10
    mc.relocs = vec![Reloc {
        kind: RelocKind::Rel32Call,
        section: Section::Code,
        site: 1,
        sym: Sym::Fn(0),
    }];
    let img = Writer.write_image(&mc, false).unwrap();

    let code_vaddr = BASE + CODE_OFF as u64;
    // The reloc engine computes instr_vaddr = code_blob_vaddr + site = code_vaddr + 1.
    // rel = target - (instr_vaddr + 4) = (code_vaddr+10) - (code_vaddr+1+4) = 5.
    let expected_rel = 5u32;
    let patched = read_u32(&img, CODE_OFF + 1);
    assert_eq!(patched, expected_rel, "Rel32Call resolved to a relative displacement");
}

#[test]
fn heap_base_symbol_resolves_to_heap_vaddr() {
    let mut mc = mc_with_data(vec![0x90u8; 4], vec![0u8; 8]);
    mc.heap_size = 0x1000;
    mc.relocs = vec![Reloc {
        kind: RelocKind::Abs64,
        section: Section::Data,
        site: 0,
        sym: Sym::HeapBase,
    }];
    let img = Writer.write_image(&mc, false).unwrap();

    let code_vaddr = BASE + CODE_OFF as u64;
    let data_vaddr = (code_vaddr + 4).div_ceil(PAGE) * PAGE;
    let heap_vaddr = (data_vaddr + 8).div_ceil(PAGE) * PAGE;
    let data_file_off = ((CODE_OFF as u64 + 4).div_ceil(PAGE)) * PAGE;
    let patched = read_u64(&img, data_file_off as usize);
    assert_eq!(patched, heap_vaddr, "HeapBase resolved to the heap segment vaddr");
}

#[test]
fn import_reloc_panics_when_unlinked() {
    // Sym::Import requires a populated import_bases; an empty one must panic
    // with the documented message.
    let mut mc = mc_with_data(vec![0x90u8; 4], vec![]);
    mc.relocs = vec![Reloc {
        kind: RelocKind::Abs64,
        section: Section::Code,
        site: 0,
        sym: Sym::Import(0),
    }];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Writer.write_image(&mc, false)
    }));
    assert!(result.is_err(), "unlinked import must panic");
}
