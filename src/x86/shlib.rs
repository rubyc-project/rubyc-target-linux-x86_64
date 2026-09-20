//! Shared-object emission (`-O shared`): a dlopen-able `ET_DYN` image that
//! exports public RubyC methods as plain symbols so C hosts can consume
//! them with `dlsym`.
//!
//! Layout is identity-mapped (vaddr == file offset), and the 🐕🇨 container
//! header sits at offset 64, immediately after the ELF header — the same
//! 13-byte preamble every other artifact carries:
//!
//! ```text
//! [ELF hdr][🐕🇨 preamble][4x phdr][pad][code+helpers]             R+X
//! [strings+vtables][.rela.dyn][.dynsym][.dynstr][.hash][.dynamic] RW
//! [object heap]                                                   RW
//! [section headers]
//! ```
//!
//! Absolute addresses are resolved by the dynamic loader through
//! `R_X86_64_RELATIVE` records generated from `MachineCode::relocs`; rel32
//! call sites are position-independent already.

use rubyc::native::container::{self, Kind};
use rubyc::native::target::{MachineCode, RelocKind, Section, Sym, TargetError};

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
/// Marks the stack non-executable; without it glibc refuses to dlopen.
const PT_GNU_STACK: u32 = 0x6474_e551;
const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;
/// `R_X86_64_RELATIVE`.
const R_RELATIVE: u64 = 8;

const DT_NULL: u64 = 0;
const DT_HASH: u64 = 4;
const DT_STRTAB: u64 = 5;
const DT_SYMTAB: u64 = 6;
const DT_RELA: u64 = 7;
const DT_RELASZ: u64 = 8;
const DT_RELAENT: u64 = 9;
const DT_STRSZ: u64 = 10;
const DT_SYMENT: u64 = 11;
const DT_RELACOUNT: u64 = 19;
const DT_INIT: u64 = 12;

const SHT_NULL: u32 = 0;
const SHT_PROGBITS: u32 = 1;
const SHT_STRTAB: u32 = 3;
const SHT_RELA: u32 = 4;
const SHT_HASH: u32 = 5;
const SHT_DYNAMIC: u32 = 6;
const SHT_DYNSYM: u32 = 11;

const PAGE: usize = 0x1000;

#[inline]
fn rex(w: bool, r: bool, b: bool) -> u8 {
    let mut byte = 0x40;
    if w {
        byte |= 0x08;
    }
    if r {
        byte |= 0x04;
    }
    if b {
        byte |= 0x01;
    }
    byte
}

#[inline]
fn modrm(mod_: u8, reg: u8, rm: u8) -> u8 {
    (mod_ << 6) | ((reg & 7) << 3) | (rm & 7)
}

fn align(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

#[allow(dead_code)]
fn push_u16(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn push_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn push_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn put_u16(img: &mut [u8], at: usize, x: u16) {
    img[at..at + 2].copy_from_slice(&x.to_le_bytes());
}

fn put_u32(img: &mut [u8], at: usize, x: u32) {
    img[at..at + 4].copy_from_slice(&x.to_le_bytes());
}

fn put_u64(img: &mut [u8], at: usize, x: u64) {
    img[at..at + 8].copy_from_slice(&x.to_le_bytes());
}

/// SysV hash of a symbol name for the `.hash` table dlopen requires.
fn elf_hash(name: &str) -> u32 {
    let mut h: u32 = 0;
    for b in name.bytes() {
        h = (h << 4).wrapping_add(b as u32);
        let g = h & 0xF000_0000;
        if g != 0 {
            h ^= g >> 24;
        }
        h &= !g;
    }
    h
}

/// Number of absolute-address references living in the code blob. Each one
/// gets a GOT slot in shared libraries (text is never relocated directly).
fn count_code_abs64(mc: &MachineCode) -> usize {
    mc.relocs
        .iter()
        .filter(|r| r.kind == RelocKind::Abs64 && r.section == Section::Code)
        .count()
}

/// Build a SysV→internal-convention trampoline placed at vaddr `at`.
///
/// C calls it with parameters in registers (rdi, rsi, …). Internal bodies
/// expect the receiver in rdi (NULL here) and parameter i in
/// rsi/rdx/rcx/r8/r9 — so the trampoline shifts the register window right
/// by one and jumps into a `call body; ret` stub.
///
/// Returns the encoded bytes.
fn build_trampoline(nargs: usize, at: u64, body_vaddr: u64, takes_receiver: bool) -> Vec<u8> {
    // Receiverless (static) bodies take SysV args directly: no
    // shifting, just a jump stub (call + ret so returns work).
    if !takes_receiver {
        let mut out: Vec<u8> = Vec::new();
        out.push(0xE8);
        let next = at + 5;
        push_u32(&mut out, (body_vaddr as i64 - next as i64) as i32 as u32);
        out.push(0xC3); // ret
        return out;
    }
    let mut out: Vec<u8> = Vec::new();
    const SYSV_REGS: [u8; 6] = [7, 6, 2, 1, 8, 9]; // rdi rsi rdx rcx r8 r9
    let n = nargs.min(5); // receiver consumes rdi; params fill rsi..r9

    // Shift params right: for i = n-1..=0: ARG[i+1] = ARG[i]
    //   mov rXX_{i+1}, rXX_i  (reg-to-reg, full-width)
    for i in (0..n).rev() {
        let dst = SYSV_REGS[i + 1];
        let src = SYSV_REGS[i];
        out.push(rex_byte(true, src, dst));
        out.push(0x89);
        out.push(modrm(0b11, src, dst));
    }
    // xor edi, edi — NULL receiver.
    out.extend_from_slice(&[0x31, 0xFF]);

    // call body (rel32)
    out.push(0xE8);
    let next = at + (out.len() + 4) as u64;
    push_u32(&mut out, (body_vaddr as i64 - next as i64) as i32 as u32);

    out.push(0xC3); // ret
    out
}

fn rex_byte(w: bool, src: u8, dst: u8) -> u8 {
    rex(w, src >= 8, dst >= 8)
}

/// Rewrite one 10-byte `movabs rX, imm64` window (immediate at `site`)
/// into `mov rX, [rip+d32]`, preserving exact length.
///
/// `d32` addresses the GOT slot holding the final value, relative to RIP
/// just after the 7-byte load instruction.
fn encode_got_load(img: &mut [u8], code_base: usize, site: usize, reg_dst: usize, got_vaddr: u64) {
    // mov rX, [rip+d32]: REX.W | 8B | modrm(reg=dst, rm=101) | disp32
    let rex_w: u8 = if reg_dst >= 8 { 0x4C } else { 0x48 };
    let modrm: u8 = 0x05 | (((reg_dst & 7) as u8) << 3);
    let rip_after = (code_base + site + 5) as u64;
    let d32 = got_vaddr as i64 - rip_after as i64;
    let at = code_base + site;
    img[at - 2..at - 1].copy_from_slice(&[rex_w]);
    img[at - 1..at].copy_from_slice(&[0x8B]);
    img[at..at + 1].copy_from_slice(&[modrm]);
    img[at + 1..at + 5].copy_from_slice(&(d32 as i32).to_le_bytes());
    // The original `movabs` window is 10 bytes; `mov [rip+d32]` is 7. Pad the
    // remaining 3 bytes with NOPs so execution never falls into stale imm64.
    img[at + 5..at + 8].copy_from_slice(&[0x90, 0x90, 0x90]);
}

/// Trampoline size for `n` arguments (fixed-width encoding).
/// Receiverless entries emit a bare jump stub instead.
fn trampoline_size(nargs: usize, takes_receiver: bool) -> usize {
    if !takes_receiver {
        return 5 + 1;
    }
    let n = nargs.min(6);
    10 + 8 + n * 6 + 5 + 10 + 1
}

struct Phdr {
    p_type: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
}

struct Shdr {
    sh_type: u32,
    flags: u64,
    addr: u64,
    offset: u64,
    size: u64,
    link: u32,
    entsize: u64,
}

/// Emit the shared object image.
pub fn write_shared_library(mc: &MachineCode) -> Result<Vec<u8>, TargetError> {
    const EH: usize = 64;
    const PREAMBLE_AT: usize = 64;
    const PHDR_AT: usize = 80; // after the 13-byte preamble, 16-aligned
    const PHNUM: usize = 5;
    const CODE_BASE: usize = PHDR_AT + PHNUM * 56;

    // ---------- RX segment ----------
    // Trampolines (SysV entry stubs, one per export) follow the code blob.
    let trampoline_sizes: Vec<usize> = mc
        .exports
        .iter()
        .enumerate()
        .map(|(i, (_, _, arity))| {
            trampoline_size(
                *arity,
                mc.export_takes_receiver.get(i).copied().unwrap_or(true),
            )
        })
        .collect();
    let tramps_total: usize = trampoline_sizes.iter().sum();
    let tramp_base = CODE_BASE + mc.code.len();
    let rx_end = tramp_base + tramps_total;

    // ---------- RW segment layout (sizes first so heap placement works) --
    // Segment boundaries must be page-aligned for dlopen.
    let data_at = align(rx_end, PAGE);
    let n_got = count_code_abs64(mc); // one slot per code-absolute reference
    let got_start = align(data_at + mc.data.len(), 8);
    let rela_start = align(got_start + n_got * 8, 8);
    let nrela = n_got
        + mc.relocs
            .iter()
            .filter(|r| r.kind == RelocKind::Abs64 && r.section == Section::Data)
            .count();
    // OBJECT symbols (data section): vtable rows (`<path>::__vtable`)
    // then static slots (full `<path>::<field>` symbols) — one entry per
    // row/slot so downstream images can dlsym them via `LoadImport`.
    let mut object_syms: Vec<(String, usize)> = Vec::new();
    for (i, class_path) in mc.vtable_names.iter().enumerate() {
        let off = mc.vtable_offsets.get(i).copied().unwrap_or(0);
        object_syms.push((rubyc::native::target::vtable_symbol(class_path), off));
    }
    for (i, sym) in mc.static_names.iter().enumerate() {
        let off = mc.static_offsets.get(i).copied().unwrap_or(0);
        object_syms.push((sym.clone(), off));
    }
    let nsyms = mc.exports.len() + object_syms.len() + 1; // leading null symbol
    let dynsym_size = nsyms * 24;
    let dynstr_size = 1 + mc
        .exports
        .iter()
        .map(|(n, _, _)| n.len() + 1)
        .sum::<usize>()
        + object_syms
            .iter()
            .map(|(n, _)| n.len() + 1)
            .sum::<usize>();
    let nbucket = nsyms.max(1);
    let hash_size = (nbucket + nsyms + 2) * 4;
    let dynamic_size = 11 * 16; // ten entries plus DT_NULL

    let dynsym_start = align(rela_start + nrela * 24, 8);
    let dynstr_start = align(dynsym_start + dynsym_size, 8);
    let hash_start = align(dynstr_start + dynstr_size, 8);
    let dynamic_start = align(hash_start + hash_size, 16);
    let rw_end = dynamic_start + dynamic_size;
    let heap_vaddr = align(rw_end, PAGE) as u64;
    let heap_memsz = align(mc.heap_size as usize, PAGE);

    // Section header table after the RW segment.
    const SHDR_COUNT: usize = 9; // NULL, text, data, rela, dynsym, dynstr, hash, dynamic, shstrtab
    let names: [&[u8]; 7] = [
        b".text\0",
        b".data\0",
        b".rela.dyn\0",
        b".dynsym\0",
        b".dynstr\0",
        b".hash\0",
        b".dynamic\0",
    ];
    let mut shstrtab: Vec<u8> = vec![0];
    let mut name_offsets: Vec<usize> = Vec::new();
    for n in names {
        name_offsets.push(shstrtab.len());
        shstrtab.extend_from_slice(n);
    }
    let shstrtab_name_off = shstrtab.len();
    shstrtab.extend_from_slice(b".shstrtab\0");

    let mut export_values: Vec<u64> = Vec::with_capacity(mc.exports.len());
    let shoff = align(rw_end, 8);
    let shstrtab_at = shoff + SHDR_COUNT * 64;
    // The file must physically contain both the RW segment and the tables.
    let image_len = shstrtab_at + shstrtab.len().max(PAGE - (rw_end % PAGE));

    let mut img = vec![0u8; image_len];

    // ---------- ELF header ----------
    img[0..4].copy_from_slice(b"\x7fELF");
    img[4] = 2; // ELFCLASS64
    img[5] = 1; // little-endian
    img[6] = 1; // EV_CURRENT
    img[7] = 0; // SYSV
    put_u16(&mut img, 16, 3); // ET_DYN
    put_u16(&mut img, 18, 62); // EM_X86_64
    put_u32(&mut img, 20, 1); // e_version
    put_u64(&mut img, 24, 0); // e_entry — none
    put_u64(&mut img, 32, PHDR_AT as u64); // e_phoff
    put_u16(&mut img, 52, EH as u16); // e_ehsize
    put_u16(&mut img, 54, 56); // e_phentsize
    put_u16(&mut img, 56, PHNUM as u16);
    put_u16(&mut img, 58, 64); // e_shentsize
    put_u16(&mut img, 60, SHDR_COUNT as u16);

    // ---------- 🐕🇨 container preamble ----------
    img[PREAMBLE_AT..PREAMBLE_AT + 2].copy_from_slice(&container::JUMP_OVER_MAGIC);
    img[PREAMBLE_AT + 2..PREAMBLE_AT + 10].copy_from_slice(&container::MAGIC);
    img[PREAMBLE_AT + 10] = Kind::SharedLibrary.to_byte();
    let version_bytes = rubyc::bytecode::format::FORMAT_VERSION.to_le_bytes();
    img[PREAMBLE_AT + 11..PREAMBLE_AT + 13].copy_from_slice(&version_bytes);

    // ---------- program headers ----------
    let phdrs = [
        Phdr {
            p_type: PT_LOAD,
            flags: PF_R | PF_X,
            offset: 0,
            vaddr: 0,
            filesz: rx_end as u64,
            memsz: rx_end as u64,
        },
        Phdr {
            p_type: PT_LOAD,
            flags: PF_R | PF_W,
            offset: data_at as u64,
            vaddr: data_at as u64,
            filesz: (rw_end - data_at) as u64,
            memsz: (rw_end - data_at) as u64,
        },
        Phdr {
            p_type: PT_LOAD,
            flags: PF_R | PF_W,
            // Zero-file-size anonymous-style segment: offset must stay
            // page-congruent but points at nothing.
            offset: 0,
            vaddr: heap_vaddr,
            filesz: 0,
            memsz: heap_memsz as u64,
        },
        Phdr {
            p_type: PT_DYNAMIC,
            flags: PF_R | PF_W,
            offset: dynamic_start as u64,
            vaddr: dynamic_start as u64,
            filesz: dynamic_size as u64,
            memsz: dynamic_size as u64,
        },
        Phdr {
            p_type: PT_GNU_STACK,
            flags: PF_R | PF_W, // no PF_X: modern stacks are W^X
            offset: 0,
            vaddr: 0,
            filesz: 0,
            memsz: 0,
        },
    ];
    for (i, p) in phdrs.iter().enumerate() {
        let at = PHDR_AT + i * 56;
        put_u32(&mut img, at, p.p_type);
        put_u32(&mut img, at + 4, p.flags);
        put_u64(&mut img, at + 8, p.offset);
        put_u64(&mut img, at + 16, p.vaddr);
        put_u64(&mut img, at + 24, p.vaddr); // paddr
        put_u64(&mut img, at + 32, p.filesz);
        put_u64(&mut img, at + 40, p.memsz);
        put_u64(&mut img, at + 48, PAGE as u64);
    }

    // ---------- code ----------
    img[CODE_BASE..CODE_BASE + mc.code.len()].copy_from_slice(&mc.code);

    // ---------- export entry trampolines ----------
    // Receiver-taking entries get a register-shifting trampoline;
    // receiverless (static) entries export the body address directly
    // (SysV caller and body conventions already agree: params from
    // rdi on). trampoline_size mirrors the branch below.
    let takes_recv = |i: usize| mc.export_takes_receiver.get(i).copied().unwrap_or(true);
    let mut tramp_at = tramp_base;
    for (i, (name, fn_index, arity)) in mc.exports.iter().enumerate() {
        let size = trampoline_size(*arity, takes_recv(i));
        let body_vaddr = (CODE_BASE + mc.fn_bases[*fn_index]) as u64;
        let code = build_trampoline(*arity, tramp_at as u64, body_vaddr, takes_recv(i));
        img[tramp_at..tramp_at + code.len()].copy_from_slice(&code);
        export_values.push(if takes_recv(i) { tramp_at as u64 } else { body_vaddr });
        let _ = name;
        tramp_at += size;
    }
    debug_assert_eq!(export_values.len(), mc.exports.len());

    // ---------- data blob ----------
    img[data_at..data_at + mc.data.len()].copy_from_slice(&mc.data);

    // ---------- GOT + RELA records ----------
    //
    // Text is mapped R+X, so the dynamic loader must never write there.
    // Every absolute reference in code is redirected through a GOT slot in
    // the writable segment; only GOT entries and vtable words receive
    // R_X86_64_RELATIVE records.
    let final_value = |r: &rubyc::native::target::Reloc| -> Option<u64> {
        match (&r.sym, r.section) {
            (Sym::String(off), Section::Code) => Some((data_at + off) as u64),
            (Sym::VtableOff(off), Section::Code) => Some((data_at + off) as u64),
            (Sym::Static(idx), Section::Code) => mc
                .static_offsets
                .get(*idx)
                .map(|o| (data_at + o) as u64),
            (Sym::HeapBase, _) => Some(heap_vaddr),
            (Sym::ExcState(off), Section::Code) => Some((data_at + off) as u64),
            (Sym::Import(i), Section::Code) => mc
                .import_bases
                .get(*i)
                .filter(|b| **b != usize::MAX)
                .map(|b| (CODE_BASE + b) as u64),
            (Sym::Fn(i), Section::Data) => Some((CODE_BASE + mc.fn_bases[*i]) as u64),
            _ => None,
        }
    };

    let mut rela: Vec<u8> = Vec::with_capacity(nrela * 24);
    let mut got_slot = got_start;
    for r in &mc.relocs {
        if !(r.kind == RelocKind::Abs64 && r.section == Section::Code) {
            continue;
        }
        let value = final_value(r).unwrap_or(0);
        // GOT slot: initialized with the object-relative address and fixed
        // up by the loader with load-base-relative RELATIVE relocation.
        img[got_slot..got_slot + 8].copy_from_slice(&value.to_le_bytes());
        push_u64(&mut rela, got_slot as u64);
        push_u64(&mut rela, R_RELATIVE);
        push_u64(&mut rela, value);
        // Destination register is implied by which instruction produced
        // the reference (WriteStr→rsi, VTableStore/alloc→rcx, call→rax).
        let reg_dst: usize = match &r.sym {
            Sym::String(_) => 6,                                  // rsi
            Sym::VtableOff(_) | Sym::Static(_) | Sym::HeapBase => 1, // rcx
            Sym::ExcState(_) => 8,                                // r8
            _ => 0,                                               // rax (imports)
        };
        encode_got_load(&mut img, CODE_BASE, r.site, reg_dst, got_slot as u64);
        got_slot += 8;
    }
    for r in &mc.relocs {
        if !(r.kind == RelocKind::Abs64 && r.section == Section::Data) {
            continue;
        }
        if let Some(value) = final_value(r) {
            let site_vaddr = (data_at + r.site) as u64;
            push_u64(&mut rela, site_vaddr);
            push_u64(&mut rela, R_RELATIVE);
            push_u64(&mut rela, value);
        }
    }
    debug_assert_eq!(rela.len(), nrela * 24);
    img[rela_start..rela_start + nrela * 24].copy_from_slice(&rela);

    // ---------- dynsym ----------
    let mut symtab: Vec<u8> = vec![0u8; dynsym_size];
    let mut strtab: Vec<u8> = vec![0]; // NUL at offset 0
    for (i, (name, fn_index, _arity)) in mc.exports.iter().enumerate() {
        let name_off = strtab.len();
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
        let at = (i + 1) * 24;
        symtab[at..at + 4].copy_from_slice(&(name_off as u32).to_le_bytes()); // st_name
        symtab[at + 4] = 0x12; // STB_GLOBAL << 4 | STT_FUNC
        symtab[at + 5] = 0; // STV_DEFAULT — required for dlsym visibility
        symtab[at + 6..at + 8].copy_from_slice(&1u16.to_le_bytes()); // st_shndx = .text
        let entry_vaddr = export_values
            .get(i)
            .copied()
            .unwrap_or_else(|| (CODE_BASE + mc.fn_bases[*fn_index]) as u64);
        symtab[at + 8..at + 16].copy_from_slice(&entry_vaddr.to_le_bytes());
        // st_size stays 0: dlsym does not need it.
    }
    // Data-section OBJECT symbols follow the function exports: vtable
    // rows and static slots alike (see `object_syms`). Downstream images
    // dlsym these addresses and point constructed objects' slot 0
    // (`LoadImport` of `<path>::__vtable`) or static accesses
    // (`LoadImport` of `<path>::<field>`) at them, so virtual dispatch
    // and statics ride the owner's storage. st_shndx = .data (section 2).
    for (i, (sym_name, off)) in object_syms.iter().enumerate() {
        let name_off = strtab.len();
        strtab.extend_from_slice(sym_name.as_bytes());
        strtab.push(0);
        let at = (mc.exports.len() + 1 + i) * 24;
        symtab[at..at + 4].copy_from_slice(&(name_off as u32).to_le_bytes()); // st_name
        symtab[at + 4] = 0x11; // STB_GLOBAL << 4 | STT_OBJECT
        symtab[at + 5] = 0; // STV_DEFAULT — required for dlsym visibility
        symtab[at + 6..at + 8].copy_from_slice(&2u16.to_le_bytes()); // st_shndx = .data
        symtab[at + 8..at + 16].copy_from_slice(&((data_at + off) as u64).to_le_bytes());
        // st_size stays 0: dlsym does not need it.
    }
    img[dynsym_start..dynsym_start + dynsym_size].copy_from_slice(&symtab);

    // ---------- dynstr ----------
    img[dynstr_start..dynstr_start + strtab.len()].copy_from_slice(&strtab);

    // ---------- hash ----------
    let mut hash: Vec<u8> = Vec::with_capacity(hash_size);
    push_u32(&mut hash, nbucket as u32);
    push_u32(&mut hash, nsyms as u32);
    let buckets = vec![0u32; nbucket];
    let mut chains = vec![0u32; nsyms];
    let mut bucket_vec = buckets.clone();
    for (i, (name, _, _)) in mc.exports.iter().enumerate() {
        let sym_index = (i + 1) as u32;
        let b = (elf_hash(name) % nbucket as u32) as usize;
        chains[sym_index as usize] = bucket_vec[b];
        bucket_vec[b] = sym_index;
    }
    // Data-section OBJECT symbols hash under their full names.
    for (i, (sym_name, _)) in object_syms.iter().enumerate() {
        let sym_index = (mc.exports.len() + 1 + i) as u32;
        let b = (elf_hash(sym_name) % nbucket as u32) as usize;
        chains[sym_index as usize] = bucket_vec[b];
        bucket_vec[b] = sym_index;
    }
    for b in &bucket_vec {
        push_u32(&mut hash, *b);
    }
    for c in &chains {
        push_u32(&mut hash, *c);
    }
    debug_assert_eq!(hash.len(), hash_size);
    img[hash_start..hash_start + hash.len()].copy_from_slice(&hash);

    // ---------- dynamic ----------
    // DT_INIT runs `@init` at dlopen (static initializers) when the
    // program has one; otherwise the entry is zero and the loader
    // skips it. Single pointer, no array parsing involved.
    let init_vaddr = mc
        .init_fn
        .and_then(|i| mc.fn_bases.get(i))
        .map(|b| (CODE_BASE + b) as u64)
        .unwrap_or(0);
    let entries: [(u64, u64); 11] = [
        (DT_HASH, hash_start as u64),
        (DT_STRTAB, dynstr_start as u64),
        (DT_SYMTAB, dynsym_start as u64),
        (DT_STRSZ, dynstr_size as u64),
        (DT_SYMENT, 24),
        (DT_RELA, rela_start as u64),
        (DT_RELASZ, (nrela * 24) as u64),
        (DT_RELAENT, 24),
        (DT_RELACOUNT, nrela as u64),
        (DT_INIT, init_vaddr),
        (DT_NULL, 0),
    ];
    let mut dynamic: Vec<u8> = Vec::with_capacity(dynamic_size);
    for (tag, val) in entries {
        push_u64(&mut dynamic, tag);
        push_u64(&mut dynamic, val);
    }
    img[dynamic_start..dynamic_start + dynamic.len()].copy_from_slice(&dynamic);

    // ---------- section headers ----------
    let shstrtab_at = shoff + SHDR_COUNT * 64;
    let sections = [
        Shdr {
            sh_type: SHT_NULL,
            flags: 0,
            addr: 0,
            offset: 0,
            size: 0,
            link: 0,
            entsize: 0,
        },
        Shdr {
            sh_type: SHT_PROGBITS,
            flags: 6, // SHF_ALLOC | SHF_EXECINSTR
            addr: CODE_BASE as u64,
            offset: CODE_BASE as u64,
            size: mc.code.len() as u64,
            link: 0,
            entsize: 0,
        },
        Shdr {
            sh_type: SHT_PROGBITS,
            flags: 3, // SHF_ALLOC | SHF_WRITE
            addr: data_at as u64,
            offset: data_at as u64,
            size: (rela_start - data_at) as u64,
            link: 0,
            entsize: 0,
        },
        Shdr {
            sh_type: SHT_RELA,
            flags: 0x40, // SHF_ALLOC (INFO_LINK not needed for RELATIVE)
            addr: rela_start as u64,
            offset: rela_start as u64,
            size: (nrela * 24) as u64,
            link: 4, // .dynsym
            entsize: 24,
        },
        Shdr {
            sh_type: SHT_DYNSYM,
            flags: 2, // SHF_ALLOC
            addr: dynsym_start as u64,
            offset: dynsym_start as u64,
            size: dynsym_size as u64,
            link: 5, // .dynstr
            entsize: 24,
        },
        Shdr {
            sh_type: SHT_STRTAB,
            flags: 2,
            addr: dynstr_start as u64,
            offset: dynstr_start as u64,
            size: dynstr_size as u64,
            link: 0,
            entsize: 0,
        },
        Shdr {
            sh_type: SHT_HASH,
            flags: 2,
            addr: hash_start as u64,
            offset: hash_start as u64,
            size: hash_size as u64,
            link: 4, // .dynsym
            entsize: 4,
        },
        Shdr {
            sh_type: SHT_DYNAMIC,
            flags: 3,
            addr: dynamic_start as u64,
            offset: dynamic_start as u64,
            size: dynamic_size as u64,
            link: 5, // .dynstr
            entsize: 16,
        },
        Shdr {
            sh_type: SHT_STRTAB,
            flags: 0,
            addr: 0,
            offset: shstrtab_at as u64,
            size: shstrtab.len() as u64,
            link: 0,
            entsize: 0,
        },
    ];
    for (i, s) in sections.iter().enumerate() {
        let at = shoff + i * 64;
        let name_off = match i {
            0 => 0,
            8 => shstrtab_name_off, // .shstrtab names itself
            _ => name_offsets[i - 1],
        };
        put_u32(&mut img, at, name_off as u32);
        put_u32(&mut img, at + 4, s.sh_type);
        put_u64(&mut img, at + 8, s.flags);
        put_u64(&mut img, at + 16, s.addr);
        put_u64(&mut img, at + 24, s.offset);
        put_u64(&mut img, at + 32, s.size);
        put_u32(&mut img, at + 40, s.link);
        put_u32(&mut img, at + 44, 0); // info
        put_u64(&mut img, at + 48, 8); // addralign
        put_u64(&mut img, at + 56, s.entsize);
    }
    img[shstrtab_at..shstrtab_at + shstrtab.len()].copy_from_slice(&shstrtab);

    // Patch remaining ELF header fields now that offsets are final.
    put_u64(&mut img, 40, shoff as u64); // e_shoff
    put_u32(&mut img, 48, 0); // e_flags
    put_u16(&mut img, 62, 8); // e_shstrndx -> .shstrtab

    Ok(img)
}
