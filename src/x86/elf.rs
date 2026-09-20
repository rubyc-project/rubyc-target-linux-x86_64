//! ELF executable image writer for linux_x86_64.
//!
//! Three contiguous segments laid out after the ELF headers:
//!
//! ```text
//! [ELF hdr][3x phdr][pad][jump+🐕🇨 preamble][code+helpers]  R+X
//! [strings + vtable words]                                   RW
//! [object heap (BSS-style, no file bytes)]                   RW
//! ```
//!
//! Relocations from the `MachineCode` are resolved while writing: every
//! symbol is placed at a known vaddr in this layout, so the image bytes are
//! final when written — no dynamic linker involved.

use rubyc::native::target::{ImageWriter, MachineCode, TargetError};

const BASE_VADDR: u64 = 0x0040_0000;
const PAGE: u64 = 0x1000;

pub struct Writer;

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

impl ImageWriter for Writer {
    fn write_image(&self, mc: &MachineCode, entry_preamble: bool) -> Result<Vec<u8>, TargetError> {
        const EH: usize = 64;
        const PH: usize = 56;
        const PHNUM: usize = 3;
        const SH: usize = 64; // section header size

        let preamble_len = if entry_preamble {
            rubyc::native::container::PREAMBLE_LEN
        } else {
            0
        };
        let code_off = EH + PHNUM * PH; // file offset of the RX segment content
        let rx_len = preamble_len + mc.code.len();

        // Vaddrs: RX segment starts at BASE; data and heap follow page-aligned.
        let code_vaddr = BASE_VADDR + code_off as u64;
        let data_vaddr = align_up(code_vaddr + rx_len as u64, PAGE);
        let heap_vaddr = align_up(data_vaddr + mc.data.len() as u64, PAGE);

        let data_file_off = align_up((code_off + rx_len) as u64, PAGE);

        // Build symbol table (always emitted; gives `nm`/`readelf` visibility).
        let mut strtab: Vec<u8> = vec![0u8]; // index 0 = NUL
        let mut symbols: Vec<[u64; 4]> = Vec::new(); // [name_off, info, other, shndx]
        // Index 0: null symbol
        symbols.push([0, 0, 0, 0]);
        // Index 1: _start at code_vaddr (text section = 1)
        let _start_off = add_string(&mut strtab, "_start");
        symbols.push([_start_off, sym_info(2, 0), 0, 1]); // STT_FUNC, STB_LOCAL, shndx=1
        // Function symbols
        for (i, _base) in mc.fn_bases.iter().enumerate() {
            let name = if i == 0 { "main".to_string() } else { format!("fn_{i}") };
            let off = add_string(&mut strtab, &name);
            symbols.push([off, sym_info(2, 0), 0, 1]); // STT_FUNC, shndx=1 (text)
        }
        // Heap base symbol
        let heap_off = add_string(&mut strtab, "__heap_base");
        symbols.push([heap_off, sym_info(1, 3), 0, 3]); // STT_OBJECT, STB_COMMON, shndx=3 (BSS)

        let symtab_off = align_up(data_file_off + mc.data.len() as u64, 8);
        let symtab_len = (symbols.len() * 24) as u64;
        let strtab_off = align_up(symtab_off + symtab_len, 8);
        let strtab_len = strtab.len() as u64;
        // shstrtab: section name strings (inlined, NUL-terminated)
        let shstrtab: &[u8] = b"\0.text\0.data\0.bss\0.symtab\0.strtab";
        let shstrtab_off = align_up(strtab_off + strtab_len, 8);
        let shstrtab_len = shstrtab.len() as u64;
        let shnum = 6u16; // 0=null, 1=text, 2=data, 3=bss, 4=symtab, 5=strtab
        let shoff = align_up(shstrtab_off + shstrtab_len, 8);
        let total_len = (shoff + (shnum as u64) * (SH as u64)) as usize;

        // Apply relocations against final addresses, patching copies of the
        // code and data blobs before assembling. Each relocation targets
        // exactly one blob, keyed by its section.
        let mut code_bytes = mc.code.clone();
        let mut data_bytes = mc.data.clone();
        // Encoder offsets are relative to its own code blob which sits
        // after the preamble in the RX segment.
        let code_blob_vaddr = code_vaddr
            + if entry_preamble {
                rubyc::native::container::PREAMBLE_LEN as u64
            } else {
                0
            };
        for r in &mc.relocs {
            let target = match &r.sym {
                rubyc::native::target::Sym::Fn(i) => code_blob_vaddr + mc.fn_bases[*i] as u64,
                rubyc::native::target::Sym::String(off) => data_vaddr + *off as u64,
                rubyc::native::target::Sym::VtableOff(off) => data_vaddr + *off as u64,
                rubyc::native::target::Sym::Static(idx) => {
                    data_vaddr + mc.static_offsets.get(*idx).copied().unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HeapBase => heap_vaddr,
                rubyc::native::target::Sym::HelperItoa => {
                    code_blob_vaddr + mc.helper_itoa.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperItoaErr => {
                    code_blob_vaddr + mc.helper_itoa_err.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperScanInt => {
                    code_blob_vaddr + mc.helper_scan_int.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperRetain => {
                    code_blob_vaddr + mc.helper_retain.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperRelease => {
                    code_blob_vaddr + mc.helper_release.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperAlloc => {
                    code_blob_vaddr + mc.helper_alloc.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperExcEnter => {
                    code_blob_vaddr + mc.helper_exc_enter.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperExcPop => {
                    code_blob_vaddr + mc.helper_exc_pop.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperExcThrow => {
                    code_blob_vaddr + mc.helper_exc_throw.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperExcRethrow => {
                    code_blob_vaddr + mc.helper_exc_rethrow.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperExcLoad => {
                    code_blob_vaddr + mc.helper_exc_load.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::HelperExcClear => {
                    code_blob_vaddr + mc.helper_exc_clear.unwrap_or(0) as u64
                }
                rubyc::native::target::Sym::ExcState(off) => data_vaddr + *off as u64,
                // Startup-saved rsp slot: last 8 data bytes whenever the
                // argv helper exists (backend appends slot with helper).
                rubyc::native::target::Sym::ArgvRsp => {
                    if mc.helper_argv.is_some() && !mc.data.is_empty() {
                        data_vaddr + mc.data.len() as u64 - 8
                    } else {
                        data_vaddr
                    }
                }
                rubyc::native::target::Sym::Import(i) => {
                    code_blob_vaddr
                        + mc.import_bases
                            .get(*i)
                            .copied()
                            .expect("imports must be linked before image writing")
                            as u64
                }
                // Name symbols are not yet resolved by the x86_64 backend.
                rubyc::native::target::Sym::Name(_) => 0,
            };
            let (code_site, data_site) = match r.section {
                rubyc::native::target::Section::Code => (Some(r.site), None),
                rubyc::native::target::Section::Data => (None, Some(r.site)),
            };
            if let Some(site) = code_site {
                match r.kind {
                    rubyc::native::target::RelocKind::Abs64 => {
                        code_bytes[site..site + 8].copy_from_slice(&target.to_le_bytes());
                    }
                    rubyc::native::target::RelocKind::Rel32Call => {
                        let instr_vaddr = code_blob_vaddr + site as u64;
                        let rel = target as i64 - (instr_vaddr as i64 + 4);
                        code_bytes[site..site + 4].copy_from_slice(&(rel as i32).to_le_bytes());
                    }
                }
            }
            if let Some(site) = data_site {
                match r.kind {
                    rubyc::native::target::RelocKind::Abs64 => {
                        data_bytes[site..site + 8].copy_from_slice(&target.to_le_bytes());
                    }
                    rubyc::native::target::RelocKind::Rel32Call => {}
                }
            }
        }

        // NOTE: relocations are applied exactly once, against the code-blob
        // vaddrs above. A second pass over the same list with segment-level
        // vaddrs used to overwrite them off by the preamble length — calls
        // into function starts only survived by accidentally executing the
        // jump-over-magic preamble, while helper calls crashed.

        let mut img = vec![0u8; total_len];

        // --- ELF header ---
        img[0..4].copy_from_slice(b"\x7fELF");
        img[4] = 2; // 64-bit
        img[5] = 1; // little-endian
        img[6] = 1; // version
        img[7] = 0; // SYSV
        img[16..18].copy_from_slice(&2u16.to_le_bytes()); // EXEC
        img[18..20].copy_from_slice(&62u16.to_le_bytes()); // x86-64
        img[20..24].copy_from_slice(&1u32.to_le_bytes());
        img[24..32].copy_from_slice(&(BASE_VADDR + entry_offset() as u64).to_le_bytes());
        img[32..40].copy_from_slice(&(EH as u64).to_le_bytes()); // phoff
        img[40..48].copy_from_slice(&(shoff as u64).to_le_bytes()); // shoff
        img[48..52].copy_from_slice(&0u32.to_le_bytes()); // flags
        img[52..54].copy_from_slice(&(EH as u16).to_le_bytes());
        img[54..56].copy_from_slice(&(PH as u16).to_le_bytes());
        img[56..58].copy_from_slice(&(PHNUM as u16).to_le_bytes());

        // --- Program headers ---
        let mut ph = EH;
        // 1) RX segment: headers + preamble + code
        write_ph(
            &mut img,
            ph,
            1,
            5,
            0,
            BASE_VADDR,
            (code_off + rx_len) as u64,
            (code_off + rx_len) as u64,
        );
        ph += PH;
        // 2) RW segment: data
        write_ph(
            &mut img,
            ph,
            1,
            6, /*R|W*/
            data_file_off,
            data_vaddr,
            mc.data.len() as u64,
            mc.data.len() as u64,
        );
        ph += PH;
        // 3) BSS-style heap: no file bytes
        let heap_pages = heap_size_pages(mc.heap_size);
        write_ph(
            &mut img,
            ph,
            1,
            6,
            data_file_off, // offset irrelevant with filesz=0
            heap_vaddr,
            0,
            heap_pages,
        );

        // --- Content ---
        let preamble_at = code_off;
        if entry_preamble {
            img[preamble_at..preamble_at + 2]
                .copy_from_slice(&rubyc::native::container::JUMP_OVER_MAGIC);
            img[preamble_at + 2..preamble_at + 10]
                .copy_from_slice(&rubyc::native::container::MAGIC);
        }
        let cstart = preamble_at + preamble_len_for(entry_preamble);
        img[cstart..cstart + code_bytes.len()].copy_from_slice(&code_bytes);
        let dstart = data_file_off as usize;
        img[dstart..dstart + data_bytes.len()].copy_from_slice(&data_bytes);

        img[58..60].copy_from_slice(&(shnum as u16).to_le_bytes()); // shnum

        // Write symtab entries (Elf64_Sym: 24 bytes each)
        let symtab_start = symtab_off as usize;
        for (i, [name_off, info, other, shndx]) in symbols.iter().enumerate() {
            let at = symtab_start + i * 24;
            let mut entry = [0u8; 24];
            entry[0..8].copy_from_slice(&name_off.to_le_bytes());
            entry[8] = *info as u8;
            entry[9] = *other as u8;
            entry[10..12].copy_from_slice(&(*shndx as u16).to_le_bytes());
            img[at..at + 24].copy_from_slice(&entry);
        }

        // Write strtab
        img[strtab_off as usize..(strtab_off + strtab_len) as usize].copy_from_slice(&strtab);

        // Write shstrtab
        img[shstrtab_off as usize..(shstrtab_off + shstrtab_len) as usize].copy_from_slice(shstrtab);

        // --- Section headers (at shoff) ---
        // sh_name offsets into shstrtab: .text=1, .data=6, .bss=11, .symtab=15, .strtab=22
        let sh_base = shoff as usize;
        // 0: null (already zero)
        // 1: .text — type=1(PROG), flags=6(RX), link=0, info=0
        write_sh(&mut img, sh_base + SH, 1, 1, code_vaddr, code_off as u64, rx_len as u64, 6, 0, 0, 0, 8, 0);
        // 2: .data — type=3(NOBITS→DATA), flags=6(RW), link=0, info=0
        write_sh(&mut img, sh_base + 2 * SH, 6, 1, data_vaddr, data_file_off, mc.data.len() as u64, 6, 0, 0, 0, 8, 0);
        // 3: .bss — type=8(NOBITS), flags=6(RW), link=0, info=0
        write_sh(&mut img, sh_base + 3 * SH, 11, 8, heap_vaddr, 0, heap_size_pages(mc.heap_size), 6, 0, 0, 0, 8, 0);
        // 4: .symtab — type=2(SYMTAB), link=5(strtab), info=1 (first non-local)
        write_sh(&mut img, sh_base + 4 * SH, 15, 2, 0, symtab_off, symtab_len, 5, 1, 0, 24, 0, 0);
        // 5: .strtab — type=3(STRTAB)
        write_sh(&mut img, sh_base + 5 * SH, 22, 3, 0, strtab_off, strtab_len, 0, 0, 0, 0, 0, 0);

        Ok(img)
    }

    fn write_shared_library(&self, mc: &MachineCode) -> Result<Vec<u8>, TargetError> {
        super::shlib::write_shared_library(mc)
    }
}

fn entry_offset() -> usize {
    64 + 3 * 56
}

/// STT_FUNC=2, STB_LOCAL=0 → info byte 0x02; STT_OBJECT=1, STB_COMMON=3 → 0x31.
const fn sym_info(typ: u8, bind: u8) -> u64 {
    ((bind as u64) << 4) | typ as u64
}

/// Append a NUL-terminated string to `strtab`; returns its offset.
fn add_string(strtab: &mut Vec<u8>, s: &str) -> u64 {
    let off = strtab.len() as u64;
    strtab.extend_from_slice(s.as_bytes());
    strtab.push(0);
    off
}

/// Write an Elf64_Shdr at `at` in `img`.
#[allow(clippy::too_many_arguments)]
fn write_sh(
    img: &mut [u8],
    at: usize,
    name: u32,
    type_: u32,
    flags: u64,
    addr: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addr_align: u64,
    ent_size: u64,
    _unused: u64,
) {
    img[at..at + 4].copy_from_slice(&name.to_le_bytes());
    img[at + 4..at + 8].copy_from_slice(&type_.to_le_bytes());
    img[at + 8..at + 16].copy_from_slice(&flags.to_le_bytes());
    img[at + 16..at + 24].copy_from_slice(&addr.to_le_bytes());
    img[at + 24..at + 32].copy_from_slice(&offset.to_le_bytes());
    img[at + 32..at + 40].copy_from_slice(&size.to_le_bytes());
    img[at + 40..at + 44].copy_from_slice(&link.to_le_bytes());
    img[at + 44..at + 48].copy_from_slice(&info.to_le_bytes());
    img[at + 48..at + 56].copy_from_slice(&addr_align.to_le_bytes());
    img[at + 56..at + 64].copy_from_slice(&ent_size.to_le_bytes());
}

fn preamble_len_for(entry_preamble: bool) -> usize {
    if entry_preamble {
        rubyc::native::container::PREAMBLE_LEN
    } else {
        0
    }
}

fn heap_size_pages(heap_size: u64) -> u64 {
    heap_size.div_ceil(PAGE) * PAGE
}

#[allow(clippy::too_many_arguments)]
fn write_ph(
    img: &mut [u8],
    at: usize,
    p_type: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
) {
    img[at..at + 4].copy_from_slice(&p_type.to_le_bytes());
    img[at + 4..at + 8].copy_from_slice(&flags.to_le_bytes());
    img[at + 8..at + 16].copy_from_slice(&offset.to_le_bytes());
    img[at + 16..at + 24].copy_from_slice(&vaddr.to_le_bytes());
    img[at + 24..at + 32].copy_from_slice(&vaddr.to_le_bytes()); // paddr
    img[at + 32..at + 40].copy_from_slice(&filesz.to_le_bytes());
    img[at + 40..at + 48].copy_from_slice(&memsz.to_le_bytes());
    img[at + 48..at + 56].copy_from_slice(&PAGE.to_le_bytes());
}
