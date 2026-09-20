//! x86-64 machine-code emitter for Linux (raw syscall ABI).
//!
//! Consumes allocated virtual-register IR: the register allocator has
//! placed every value in a callee-saved register (`rbx`, `r12`–`r15`) or a
//! frame slot, so emission is a straight walk with no liveness reasoning.
//! Caller-saved registers serve only argument passing, syscall ABI,
//! division operands, and emission scratch (`rax` value path, `r10`/`r11`
//! address arithmetic) — which is why a call can never clobber a live
//! allocated value.
//!
//! Absolute-address windows keep exact shapes the shared-library GOT pass
//! depends on: `movabs rsi/rcx/rax, imm64` for string pointers, vtable
//! rows, heap base, and import targets respectively, with the relocation
//! site pointing at the immediate.

use crate::x86::regalloc::{self, Loc, Phys};
use rubyc::native::ir::{
    BinOp, BlockId, CmpOp, FBinOp, FCmpOp, Instr, IrFunction, Program, Terminator, VReg,
};
use rubyc::native::target::{
    CodeGenBackend, MachineCode, Reloc, RelocKind, Section, Sym, TargetError,
};
use crate::x86::runtime;

pub struct Backend;

/// Build-time trace switch, read exactly once.
fn trace_enabled() -> bool {
    use std::sync::OnceLock;
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("RUBYC_TRACE").is_some())
}

/// True when any function builds the argv array, so the startup-saved
/// rsp slot and the rsp-save stub must be emitted.
fn program_uses_argv(program: &Program) -> bool {
    for f in &program.functions {
        for b in &f.blocks {
            if b.instrs.iter().any(|i| matches!(i, Instr::BuildArgv { .. })) {
                return true;
            }
        }
    }
    false
}

/// True when any function references an exception instruction or terminator,
/// so the handler state block and the six exc helpers must be emitted.
fn program_uses_exc(program: &Program) -> bool {
    for f in &program.functions {
        for b in &f.blocks {
            if b
                .instrs
                .iter()
                .any(|i| {
                    matches!(
                        i,
                        Instr::HandlerEnter { .. }
                            | Instr::HandlerPop
                            | Instr::LoadExc { .. }
                            | Instr::ClearInflight
                    )
                })
                || matches!(&b.term, Terminator::Unwind { .. } | Terminator::Rethrow)
            {
                return true;
            }
        }
    }
    false
}

impl CodeGenBackend for Backend {
    fn lower(&self, program: &Program) -> Result<MachineCode, TargetError> {
        let mut e = Emitter::new(program);
        e.uses_exc = program_uses_exc(program);
        e.uses_argv = program_uses_argv(program);

        // Data layout is fixed before emission: literal strings first, then
        // one 8-byte word per vtable slot. VTable byte offsets are therefore
        // known up front and each `VTableAddr` carries its final relocation.
        let mut data = program.strings.clone();
        let vtable_offsets: Vec<usize> = program
            .vtables
            .iter()
            .scan(data.len(), |cursor, (_, row)| {
                let off = *cursor;
                *cursor += row.len() * 8;
                Some(off)
            })
            .collect();
        for row in program.vtables.iter().map(|(_, r)| r) {
            data.extend(std::iter::repeat_n(0u8, row.len() * 8));
        }
        e.vtable_offsets = vtable_offsets.clone();

        // Static field storage: one zero-initialized 8-byte slot per
        // `Program::statics`, appended after the vtable words. Zero is the
        // default value for i64/bool and a null pointer for refs.
        let static_offsets: Vec<usize> = program
            .statics
            .iter()
            .scan(data.len(), |cursor, _| {
                let off = *cursor;
                *cursor += 8;
                Some(off)
            })
            .collect();
        for _ in &program.statics {
            data.extend(std::iter::repeat_n(0u8, 8));
        }
        e.static_offsets = static_offsets.clone();
        // Static slot export symbols (`<path>::<field>`, parallel to
        // static_offsets): `Program::statics[].class` is the qualified
        // class path. Format shared with lowering (`target::static_symbol`).
        let static_names: Vec<String> = program
            .statics
            .iter()
            .map(|s| rubyc::native::target::static_symbol(&s.class, &s.name))
            .collect();

        for f in &program.functions {
            e.emit_function(f);
        }

        // Runtime helpers: appended only when referenced (code size).
        let mut helper_itoa = None;
        let mut helper_alloc = None;
        let mut helper_retain: Option<usize> = None;
        let mut helper_release: Option<usize> = None;
        if e.uses_int {
            helper_itoa = Some(e.code.len());
            e.bytes(runtime::ITOA);
        }
        let mut helper_itoa_err = None;
        if e.uses_int_err {
            helper_itoa_err = Some(e.code.len());
            e.bytes(&runtime::itoa_to_fd(2));
        }
        let mut helper_scan_int = None;
        if e.uses_scan {
            helper_scan_int = Some(e.code.len());
            e.bytes(runtime::SCAN_INT);
        }
        let mut heap_reloc_site = None;
        let mut release_heap_reloc_site = None;
        if e.uses_retain {
            helper_retain = Some(e.code.len());
            e.bytes(runtime::RETAIN);
        }
        if e.uses_release {
            helper_release = Some(e.code.len());
            release_heap_reloc_site =
                Some(helper_release.unwrap() + runtime::RELEASE_HEAPBASE_OFFSET);
            e.bytes(runtime::RELEASE);
        }

        if e.uses_alloc {
            heap_reloc_site = Some(e.code.len() + runtime::ALLOC_REVISED_HEAPBASE_OFFSET);
            helper_alloc = Some(e.code.len());
            e.bytes(runtime::ALLOC_REVISED);

        }

        // Exception runtime: a zero-initialized state block at the end of the
        // data section plus the six handlers, emitted only when a program uses
        // try/catch. Each helper's `movabs r8, <base>` is relocated to the
        // state block (an `Abs64` reloc at [`runtime::EXC_BASE_OFFSET`]).
        let mut exc_state_offset: Option<usize> = None;
        // Startup-saved rsp slot (8 zero bytes): the `@start` stub writes
        // the kernel rsp here before the prologue; `BuildArgv` reads argv
        // through it. Last data word whenever argv is used.
        let mut argv_rsp_off: Option<usize> = None;
        if e.uses_argv {
            argv_rsp_off = Some(data.len());
            data.extend(std::iter::repeat_n(0u8, 8));
        }
        let mut helper_exc_enter = None;
        let mut helper_exc_pop = None;
        let mut helper_exc_throw = None;
        let mut helper_exc_rethrow = None;
        let mut helper_exc_load = None;
        let mut helper_exc_clear = None;
        if e.uses_exc {
            exc_state_offset = Some(data.len());
            data.extend(std::iter::repeat_n(0u8, runtime::EXC_STATE_SIZE));

            helper_exc_enter = Some(e.code.len());
            e.bytes(runtime::EXC_ENTER);
            helper_exc_pop = Some(e.code.len());
            e.bytes(runtime::EXC_POP);
            helper_exc_load = Some(e.code.len());
            e.bytes(runtime::EXC_LOAD);
            helper_exc_clear = Some(e.code.len());
            e.bytes(runtime::EXC_CLEAR);
            helper_exc_throw = Some(e.code.len());
            e.bytes(runtime::EXC_THROW);
            helper_exc_rethrow = Some(e.code.len());
            e.bytes(runtime::EXC_RETHROW);
        }

        let mut all_relocs = std::mem::take(&mut e.relocs);

        // Call sites become rel32 relocations resolved by the writer/JIT.
        for (site, target) in e.call_sites {
            let sym = match target {
                CallTarget::Fn(i) => Sym::Fn(i as usize),
                CallTarget::HelperItoa => Sym::HelperItoa,
                CallTarget::HelperItoaErr => Sym::HelperItoaErr,
                CallTarget::HelperScanInt => Sym::HelperScanInt,
                CallTarget::HelperRetain => Sym::HelperRetain,
                CallTarget::HelperRelease => Sym::HelperRelease,
                CallTarget::HelperAlloc => Sym::HelperAlloc,
                CallTarget::HelperExcEnter => Sym::HelperExcEnter,
                CallTarget::HelperExcPop => Sym::HelperExcPop,
                CallTarget::HelperExcThrow => Sym::HelperExcThrow,
                CallTarget::HelperExcRethrow => Sym::HelperExcRethrow,
                CallTarget::HelperExcLoad => Sym::HelperExcLoad,
                CallTarget::HelperExcClear => Sym::HelperExcClear,
            };
            all_relocs.push(Reloc {
                kind: RelocKind::Rel32Call,
                section: Section::Code,
                site,
                sym,
            });
        }

        // Vtable words relocate to their implementing function; abstract
        // holes (`usize::MAX`) stay zero and are never dispatched through.
        for (row_i, (_, row)) in program.vtables.iter().enumerate() {
            for (slot, &fn_idx) in row.iter().enumerate() {
                if fn_idx == usize::MAX {
                    continue;
                }
                all_relocs.push(Reloc {
                    kind: RelocKind::Abs64,
                    section: Section::Data,
                    site: vtable_offsets[row_i] + slot * 8,
                    sym: Sym::Fn(fn_idx),
                });
            }
        }
        if let Some(site) = heap_reloc_site {
            all_relocs.push(Reloc {
                kind: RelocKind::Abs64,
                section: Section::Code,
                site,
                sym: Sym::HeapBase,
            });
        }
        if let Some(site) = release_heap_reloc_site {
            all_relocs.push(Reloc {
                kind: RelocKind::Abs64,
                section: Section::Code,
                site,
                sym: Sym::HeapBase,
            });
        }

        // Each exc helper's `movabs r8, <base>` points at the state block.
        if let Some(off) = exc_state_offset {
            for &h in [
                helper_exc_enter,
                helper_exc_pop,
                helper_exc_load,
                helper_exc_clear,
                helper_exc_throw,
                helper_exc_rethrow,
            ]
            .iter()
            .flatten()
            {
                all_relocs.push(Reloc {
                    kind: RelocKind::Abs64,
                    section: Section::Code,
                    site: h + runtime::EXC_BASE_OFFSET,
                    sym: Sym::ExcState(off),
                });
            }
        }

        Ok(MachineCode {
            code: e.code,
            data,
            fn_bases: e.fn_starts,
            vtable_offsets,
            vtable_names: program.vtables.iter().map(|(n, _)| n.clone()).collect(),
            static_offsets,
            static_names,
            helper_itoa,
            helper_itoa_err,
            helper_scan_int,
            helper_retain,
            helper_release,
            helper_alloc,
            helper_exc_enter,
            helper_exc_pop,
            helper_exc_throw,
            helper_exc_rethrow,
            helper_exc_load,
            helper_exc_clear,
            exc_state_offset,
            helper_argv: argv_rsp_off,
            import_names: program.imports.clone(),
            import_bases: Vec::new(),
            // Freshly lowered code has no load-time addresses yet.
            import_abs: Vec::new(),
            exports: program.exports.clone(),
            export_takes_receiver: program
                .exports
                .iter()
                .map(|(_, fi, _)| {
                    program
                        .functions
                        .get(*fi)
                        .is_some_and(|f| f.receiver != rubyc::native::ir::ReceiverKind::None)
                })
                .collect(),
            // @init (shared libraries): fn index in the emitted order,
            // matching fn_bases positions below.
            init_fn: program.init,
            relocs: all_relocs,
            heap_size: program.heap_size,
        })
    }

    fn encode_object(&self, mc: &MachineCode) -> Result<Vec<u8>, TargetError> {
        Ok(crate::mc_serialize::encode(mc))
    }

    fn decode_object(&self, bytes: &[u8]) -> Result<MachineCode, TargetError> {
        crate::mc_serialize::decode(bytes).map_err(TargetError::BadObject)
    }
}

/// Internal calling convention: receiver in `rdi`, parameters in
/// `rsi, rdx, rcx, r8, r9`, remainder pushed right-to-left, return in
/// `rax`. SysV-shaped so external imports share the path.
const ARG: [u8; 6] = [7, 6, 2, 1, 8, 9];

const RAX: u8 = 0;
const RCX: u8 = 1;
/// ARG[0]: first integer argument / return-value carrier for exit.
const RDI: u8 = 7;
const RSI: u8 = 6;
const R10: u8 = 10;
const R11: u8 = 11;

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

enum Disp {
    D8(i8),
    D32(i32),
}

/// `[rbp - 8*(slot+1)]` addressing with disp8/disp32 selection.
fn slot_addr(slot: u32, reg_field: u8) -> (u8, Disp) {
    let d = -8i32 * (slot as i32 + 1);
    if (-120..=-1).contains(&d) {
        (modrm(0b01, reg_field, 0b101), Disp::D8(d as i8))
    } else {
        (modrm(0b10, reg_field, 0b101), Disp::D32(d))
    }
}

struct Emitter<'p> {
    code: Vec<u8>,
    #[allow(dead_code)]
    strings: &'p [u8],
    vtable_offsets: Vec<usize>,
    static_offsets: Vec<usize>,
    fn_starts: Vec<usize>,
    /// Whether each function takes a receiver (position matches
    /// `program.functions`): static call sites must match the
    /// callee's prologue contract (args from rdi when there is no
    /// receiver, from rsi after a zeroed rdi otherwise).
    fn_takes_receiver: Vec<bool>,
    relocs: Vec<Reloc>,
    call_sites: Vec<(usize, CallTarget)>,
    uses_int: bool,
    uses_int_err: bool,
    uses_scan: bool,
    uses_retain: bool,
    uses_release: bool,
    uses_alloc: bool,
    uses_exc: bool,
    uses_argv: bool,
}

#[derive(Clone, Copy)]
enum CallTarget {
    Fn(u32),
    HelperItoa,
    HelperItoaErr,
    HelperScanInt,
    HelperRetain,
    HelperRelease,
    HelperAlloc,
    HelperExcEnter,
    HelperExcPop,
    HelperExcThrow,
    HelperExcRethrow,
    HelperExcLoad,
    HelperExcClear,
}

impl<'p> Emitter<'p> {
    fn new(program: &'p Program) -> Self {
        Emitter {
            code: Vec::new(),
            strings: &program.strings,
            vtable_offsets: Vec::new(),
            static_offsets: Vec::new(),
            fn_starts: Vec::new(),
            fn_takes_receiver: program
                .functions
                .iter()
                .map(|f| {
                    f.receiver != rubyc::native::ir::ReceiverKind::None
                })
                .collect(),
            relocs: Vec::new(),
            call_sites: Vec::new(),
            uses_int: false,
            uses_int_err: false,
            uses_scan: false,
            uses_retain: false,
            uses_release: false,
            uses_alloc: false,
            uses_exc: false,
            uses_argv: false,
        }
    }

    fn bytes(&mut self, b: &[u8]) {
        self.code.extend_from_slice(b);
    }
    fn imm32(&mut self, v: i32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }
    fn imm64(&mut self, v: u64) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }
    fn disp(&mut self, d: Disp) {
        match d {
            Disp::D8(v) => self.code.push(v as u8),
            Disp::D32(v) => self.imm32(v),
        }
    }

    /// `mov dst, src` between full-width registers. Opcode 89 /r takes the
    /// SOURCE in the reg field and the DESTINATION in r/m.
    fn mov_rr(&mut self, dst: u8, src: u8) {
        if dst == src {
            return;
        }
        self.bytes(&[rex(true, src >= 8, dst >= 8), 0x89]);
        self.code.push(modrm(0b11, src, dst));
    }

    /// `mov dst, [rbp - slot_disp]`
    fn load_slot(&mut self, dst: u8, slot: u32) {
        let (m, d) = slot_addr(slot, dst & 7);
        self.bytes(&[rex(true, dst >= 8, false), 0x8B]);
        self.code.push(m);
        self.disp(d);
    }

    /// `mov [rbp - slot_disp], src`
    fn store_slot(&mut self, slot: u32, src: u8) {
        let (m, d) = slot_addr(slot, src & 7);
        self.bytes(&[rex(true, src >= 8, false), 0x89]);
        self.code.push(m);
        self.disp(d);
    }

    /// Load a location's value into a physical register.
    fn get(&mut self, into: u8, l: Loc) {
        match l {
            Loc::Reg(r) => self.mov_rr(into, r.id()),
            Loc::Slot(k) => self.load_slot(into, k),
        }
    }

    /// Store `from` into a destination location.
    fn put(&mut self, from: u8, l: Loc) {
        match l {
            Loc::Reg(r) => self.mov_rr(r.id(), from),
            Loc::Slot(k) => self.store_slot(k, from),
        }
    }

    /// Emit a `movabs rX, imm64` placeholder carrying an Abs64 relocation.
    /// Returns the immediate's offset within `code`.
    fn abs64_window(&mut self, op_low3: u8, sym: Sym) -> usize {
        self.code.push(rex(true, false, false));
        self.code.push(0xB8 | (op_low3 & 7));
        let site = self.code.len();
        self.imm64(0);
        self.relocs.push(Reloc {
            kind: RelocKind::Abs64,
            section: Section::Code,
            site,
            sym,
        });
        site
    }

    /// `write(fd, str_ptr, len)` inline syscall for a data-section string.
    fn write_str_syscall(&mut self, off: u32, len: u32, fd: u32) {
        self.bytes(&[0xBF]); // mov edi, fd
        self.imm32(fd as i32);
        self.abs64_window(0xBE, Sym::String(off as usize)); // movabs rsi
        self.bytes(&[0xBA]); // mov edx, len
        self.imm32(len as i32);
        self.bytes(&[0xB8]); // mov eax, 1 (SYS_write)
        self.imm32(1);
        self.bytes(&[0x0F, 0x05]);
    }

    fn call_rel32(&mut self, target: CallTarget) {
        self.bytes(&[0xE8]);
        let site = self.code.len();
        self.imm32(0);
        self.call_sites.push((site, target));
    }

    /// Place call arguments: stack tail first (reverse), then the six
    /// register arguments.
    fn place_args(&mut self, alloc: &regalloc::Allocation, args: &[VReg]) {
        if args.len() > 6 {
            for a in args[6..].iter().rev() {
                self.get(RAX, alloc.loc[a.0 as usize]);
                self.bytes(&[0x50]); // push rax
            }
        }
        for (i, a) in args.iter().take(6).enumerate() {
            self.get(ARG[i], alloc.loc[a.0 as usize]);
        }
    }

    fn cleanup_tail(&mut self, n: usize) {
        if n > 6 {
            let bytes = (n - 6) * 8;
            // Wide calls (64-element vector factories) exceed imm8:
            // use the imm32 form, mirroring frame reservation below.
            if bytes <= 120 {
                self.bytes(&[0x48, 0x83, 0xC4, bytes as u8]); // add rsp, imm8
            } else {
                self.bytes(&[0x48, 0x81, 0xC4]); // add rsp, imm32
                self.imm32(bytes as i32);
            }
        }
    }

    fn emit_function(&mut self, f: &IrFunction) {
        let trace = trace_enabled();
        if trace {
            eprintln!(
                "EMIT {} vregs={} blocks={}",
                f.name,
                f.vregs,
                f.blocks.len()
            );
            for (bi, blk) in f.blocks.iter().enumerate() {
                for i in &blk.instrs {
                    eprintln!("  b{bi} {:?}", i);
                }
                eprintln!("  b{bi} term {:?}", blk.term);
            }
        }
        let alloc = regalloc::allocate(f);

        if trace {
            eprintln!("FN {} @{}", f.name, self.code.len());
            for (v, l) in alloc.loc.iter().enumerate() {
                eprintln!("  v{:<3} = {:?}", v, l);
            }
        }
        self.fn_starts.push(self.code.len());
        // `@start` with argv: stash the kernel rsp (argc/`argv`/`envp`
        // live above it) into the startup slot BEFORE the prologue
        // pushes anything. Only `rax` is touched (dead at entry), so no
        // register state is disturbed. Gated on argv use: paramless
        // programs keep byte-identical entry code.
        if f.name == "@start" && self.uses_argv {
            self.abs64_window(0xB8, Sym::ArgvRsp); // movabs rax, slot
            self.bytes(&[0x48, 0x89, 0x20]); // mov [rax], rsp
        }
        self.bytes(&[0x55]); // push rbp
        self.bytes(&[0x48, 0x89, 0xE5]); // mov rbp, rsp

        // Callee-saved registers we touch must be preserved for the caller
        // (SysV). Saved after the frame reservation so slots stay put.
        let mut savers: Vec<Phys> = alloc
            .loc
            .iter()
            .filter_map(|l| match l {
                Loc::Reg(r) => Some(*r),
                _ => None,
            })
            .collect();
        savers.sort_unstable();
        savers.dedup();

        // SysV ABI requires 16-byte stack alignment at call sites.
        // After `push rbp` (8) + frame (16n), rsp ≡ 8 (mod 16).
        // Each callee-saved push adds 8 bytes. Need odd total pushes for alignment.
        // If savers.len() is even, add 8 bytes to frame to compensate.
        let align_pad = if savers.len().is_multiple_of(2) { 8 } else { 0 };
        let frame_bytes = (alloc.spills as u64 * 8).div_ceil(16) * 16 + align_pad;
        if frame_bytes > 0 && frame_bytes <= 120 {
            self.bytes(&[0x48, 0x83, 0xEC, frame_bytes as u8]);
        } else if frame_bytes > 0 {
            self.bytes(&[0x48, 0x81, 0xEC]);
            self.imm32(frame_bytes as i32);
        }
        for r in &savers {
            self.code.push(rex(true, false, r.needs_rex()));
            self.bytes(&[0x50 | (r.id() & 7)]); // push r
        }

        // Incoming ABI values materialize into their homes: receiver from
        // ARG[0], parameter i from ARG[1+i].
        let mut arg_i = 0usize;
        if f.receiver != rubyc::native::ir::ReceiverKind::None {
            self.put(ARG[arg_i], alloc.loc[0]);
            arg_i += 1;
        }
        for p in 0..f.arity {
            // Params live at vregs 1..=arity (vreg 0 is the receiver).
            // Clamp to available locations to avoid out-of-bounds on
            // functions with more params than the allocator tracked.
            let idx = (p + 1).min(alloc.loc.len().saturating_sub(1));
            let arg_idx = (arg_i + p).min(ARG.len().saturating_sub(1));
            self.put(ARG[arg_idx], alloc.loc[idx]);
        }

        // Blocks emit in layout order; branch targets patch after the
        // whole function is laid out.
        let mut block_starts: Vec<usize> = Vec::with_capacity(f.blocks.len());
        let mut jumps: Vec<(usize, BlockId)> = Vec::new();
        for block in &f.blocks {
            block_starts.push(self.code.len());
            let pos_base = block_starts.len();
            for (pi, instr) in block.instrs.iter().enumerate() {
                if trace {
                    eprintln!("  @{:#x} pos{:02} {:?}", self.code.len(), pi, instr);
                }
                let _ = pos_base;
                self.instr(&alloc, instr);
            }
            match &block.term {
                Terminator::Ret(Some(v)) => {
                    self.get(RAX, alloc.loc[v.0 as usize]);
                    for r in savers.iter().rev() {
                        self.code.push(rex(true, false, r.needs_rex()));
                        let pop = 0x58 | (r.id() & 7); // pop r
                        self.code.push(pop);
                    }
                    self.bytes(&[0xC9, 0xC3]); // leave; ret
                }
                Terminator::Ret(None) => {
                    self.code.push(0xB8);
                    self.imm32(0);
                    for r in savers.iter().rev() {
                        self.code.push(rex(true, false, r.needs_rex()));
                        let pop = 0x58 | (r.id() & 7);
                        self.code.push(pop);
                    }
                    self.bytes(&[0xC9, 0xC3]);
                }
                Terminator::Exit => {
                    self.code.push(0xB8);
                    self.imm32(231); // exit_group
                    self.bytes(&[0x31, 0xFF]); // xor edi, edi
                    self.bytes(&[0x0F, 0x05]);
                }
                Terminator::ExitWith(v) => {
                    // mov edi, <main's return>; exit_group
                    self.get(RDI, alloc.loc[v.0 as usize]);
                    self.code.push(0xB8);
                    self.imm32(231);
                    self.bytes(&[0x0F, 0x05]);
                }
                Terminator::Branch {
                    cond,
                    if_true,
                    if_false,
                } => match cond {
                    None => {
                        self.code.push(0xE9); // jmp rel32
                        let site = self.code.len();
                        self.imm32(0);
                        jumps.push((site, *if_true));
                    }
                    Some(c) => {
                        // test c, c; jnz true; jmp false
                        self.get(RAX, alloc.loc[c.0 as usize]);
                        self.bytes(&[rex(true, false, false), 0x85]);
                        self.code.push(modrm(0b11, RAX, RAX));
                        self.code.extend_from_slice(&[0x0F, 0x85]); // jnz rel32
                        let site_t = self.code.len();
                        self.imm32(0);
                        jumps.push((site_t, *if_true));
                        self.code.push(0xE9); // jmp rel32
                        let site_f = self.code.len();
                        self.imm32(0);
                        jumps.push((site_f, *if_false));
                    }
                },
                Terminator::Unwind { src } => {
                    // Exception object in rdi; the runtime longjmps to the
                    // nearest active handler (or exits if none). Never returns.
                    self.get(RDI, alloc.loc[src.0 as usize]);
                    self.call_rel32(CallTarget::HelperExcThrow);
                }
                Terminator::Rethrow => {
                    self.call_rel32(CallTarget::HelperExcRethrow);
                }
            }
        }

        // Patch every recorded rel32 against final block offsets.
        for (site, target) in jumps {
            let rel = (block_starts[target as usize] as isize - (site + 4) as isize) as i32;
            self.code[site..site + 4].copy_from_slice(&rel.to_le_bytes());
        }
    }

    /// `mov dst, [r10 + d]`
    fn field_load(&mut self, dst: u8, d: i32) {
        let small = (-120..=120).contains(&d);
        self.code.push(rex(true, dst >= 8, true)); // r10 base needs B
        self.code.push(0x8B);
        if small {
            self.code.push(modrm(0b01, dst, 0b010));
            self.code.push(d as i8 as u8);
        } else {
            self.code.push(modrm(0b10, dst, 0b010));
            self.imm32(d);
        }
    }

    /// `movzx dst, byte [r10 + d]` (zero-extended byte load).
    fn field_load_byte(&mut self, dst: u8, d: i32) {
        let small = (-120..=120).contains(&d);
        self.code.push(rex(true, dst >= 8, true)); // W=1, R=(dst>=8), B=1 (r10 base)
        self.code.push(0x0F);
        self.code.push(0xB6);
        if small {
            self.code.push(modrm(0b01, dst, 0b010));
            self.code.push(d as i8 as u8);
        } else {
            self.code.push(modrm(0b10, dst, 0b010));
            self.imm32(d);
        }
    }

    /// `mov [r10 + d], r11`
    fn field_store(&mut self, d: i32) {
        let small = (-120..=120).contains(&d);
        self.code.push(rex(true, true, true)); // r11 reg, r10 base
        self.code.push(0x89);
        if small {
            self.code.push(modrm(0b01, R11, 0b010));
            self.code.push(d as i8 as u8);
        } else {
            self.code.push(modrm(0b10, R11, 0b010));
            self.imm32(d);
        }
    }

    /// Byte-wise `memcpy([r10], [r11], r8)`. Clobbers rax/r8/r10/r11 (all
    /// outside the regalloc pool: rbx, r12..r15).
    fn memcpy_loop(&mut self) {
        self.bytes(&[0x4D, 0x85, 0xC0]); // test r8, r8
        self.bytes(&[0x74, 0x12]); // jz +18 (skip: len 0)
        self.bytes(&[0x49, 0x0F, 0xB6, 0x03]); // movzx rax, byte [r11]
        self.bytes(&[0x49, 0x88, 0x02]); // mov [r10], al
        self.bytes(&[0x49, 0xFF, 0xC2]); // inc r10
        self.bytes(&[0x49, 0xFF, 0xC3]); // inc r11
        self.bytes(&[0x49, 0xFF, 0xC8]); // dec r8
        self.bytes(&[0x75, 0xE9]); // jnz -23 (loop)
    }

    /// Byte-wise `memcmp([r10], [r11], r8)`. Returns 1 (equal) or 0 (differ)
    /// in rax. Clobbers rax/r8/r10/r11.
    fn memcmp_loop(&mut self) {
        self.bytes(&[0x4D, 0x85, 0xC0]); // test r8, r8
        self.bytes(&[0x74, 0x14]); // jz +20 (equal: len 0)
        self.bytes(&[0x49, 0x0F, 0xB6, 0x03]); // loop: movzx rax, byte [r11]
        self.bytes(&[0x49, 0x38, 0x02]); // cmp al, [r10]
        self.bytes(&[0x75, 0x12]); // jne +18 (differ)
        self.bytes(&[0x49, 0xFF, 0xC2]); // inc r10
        self.bytes(&[0x49, 0xFF, 0xC3]); // inc r11
        self.bytes(&[0x49, 0xFF, 0xC8]); // dec r8
        self.bytes(&[0x75, 0xE7]); // jnz -25 (loop)
        self.bytes(&[0xB8, 0x01, 0x00, 0x00, 0x00]); // mov eax, 1
        self.bytes(&[0xEB, 0x05]); // jmp +5 (end)
        self.bytes(&[0xB8, 0x00, 0x00, 0x00, 0x00]); // mov eax, 0
    }

    fn instr(&mut self, alloc: &regalloc::Allocation, i: &Instr) {
        let _trace = trace_enabled();
        match i {
            Instr::Const { dst, imm } => {
                let fits32 = (-2147483648i64..=2147483647i64).contains(imm);
                match alloc.loc[dst.0 as usize] {
                    Loc::Reg(r) => {
                        if fits32 {
                            self.code.push(rex(true, false, r.needs_rex()));
                            self.code.push(0xC7);
                            self.code.push(modrm(0b11, 0, r.id()));
                            self.imm32(*imm as i32);
                        } else {
                            self.code.push(rex(true, false, r.needs_rex()));
                            self.code.push(0xB8 | (r.id() & 7));
                            self.imm64(*imm as u64);
                        }
                    }
                    Loc::Slot(k) => {
                        if fits32 {
                            let (m, d) = slot_addr(k, 0);
                            self.code.push(rex(true, false, false));
                            self.code.push(0xC7);
                            self.code.push(m);
                            self.disp(d);
                            self.imm32(*imm as i32);
                        } else {
                            self.code.push(rex(true, false, false));
                            self.code.push(0xB8);
                            self.imm64(*imm as u64);
                            self.store_slot(k, RAX);
                        }
                    }
                }
            }
            Instr::Mov { dst, a } => {
                let dl = alloc.loc[dst.0 as usize];
                let al = alloc.loc[a.0 as usize];
                match (dl, al) {
                    (Loc::Reg(dr), Loc::Reg(sr)) => {
                        if !matches!(dl, Loc::Reg(d2) if d2 == sr) {
                            self.mov_rr(dr.id(), sr.id());
                        }
                    }
                    (Loc::Reg(dr), Loc::Slot(ks)) => self.load_slot(dr.id(), ks),
                    (Loc::Slot(kd), Loc::Reg(sr)) => self.store_slot(kd, sr.id()),
                    (Loc::Slot(kd), Loc::Slot(ks)) => {
                        self.load_slot(RAX, ks);
                        self.store_slot(kd, RAX);
                    }
                }
            }
            Instr::Raw(bytes) => self.code.extend_from_slice(bytes),
            Instr::Bin { dst, op, a, b: bv } => self.bin(alloc, *dst, *op, *a, *bv),
            Instr::Cmp { dst, op, a, b } => self.cmp(alloc, *dst, *op, *a, *b),
            Instr::Neg { dst, a } => {
                let dl = alloc.loc[dst.0 as usize];
                let al = alloc.loc[a.0 as usize];
                if let (Loc::Reg(dr), Loc::Reg(sr)) = (dl, al)
                    && dr == sr
                {
                    self.bytes(&[rex(true, false, dr.needs_rex()), 0xF7]);
                    self.code.push(modrm(0b11, 3, dr.id())); // neg r
                    return;
                }
                self.get(RAX, al);
                self.bytes(&[0x48, 0xF7, 0xD8]); // neg rax
                self.put(RAX, dl);
            }
            Instr::CallFn { dst, fidx, args } => {
                self.place_args(alloc, args);
                self.call_rel32(CallTarget::Fn(*fidx));
                self.cleanup_tail(args.len());
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            // Static call: match the callee's prologue contract. A
            // receiver-taking callee gets rdi = 0 with params from rsi
            // on (the CallVirt call-site contract); a receiverless
            // (static) callee takes params straight from rdi on, like
            // an ordinary call.
            Instr::CallStatic { dst, fidx, args } => {
                if self.fn_takes_receiver.get(*fidx as usize).copied().unwrap_or(true) {
                    self.bytes(&[0x31, 0xff]); // xor edi, edi — rdi = 0 (receiver)
                    for (i, a) in args.iter().take(5).enumerate() {
                        self.get(ARG[i + 1], alloc.loc[a.0 as usize]);
                    }
                    if args.len() > 5 {
                        for a in args[5..].iter().rev() {
                            self.get(RAX, alloc.loc[a.0 as usize]);
                            self.bytes(&[0x50]); // push rax
                        }
                    }
                } else {
                    self.place_args(alloc, args);
                }
                self.call_rel32(CallTarget::Fn(*fidx));
                self.cleanup_tail(args.len());
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::CallVirt {
                dst,
                slot,
                recv,
                args,
            } => {
                // Receiver goes to rdi first; declared parameters then
                // fill rsi onward (stack tail beyond five). Dispatch:
                //   mov r10, [rdi]; call qword [r10 + slot*8]
                self.get(ARG[0], alloc.loc[recv.0 as usize]);
                for (i, a) in args.iter().take(5).enumerate() {
                    self.get(ARG[i + 1], alloc.loc[a.0 as usize]);
                }
                if args.len() > 5 {
                    for a in args[5..].iter().rev() {
                        self.get(RAX, alloc.loc[a.0 as usize]);
                        self.bytes(&[0x50]); // push rax
                    }
                }
                self.bytes(&[rex(true, true, false), 0x8B, modrm(0b00, 2, 0b111)]); // mov r10,[rdi]
                self.bytes(&[rex(true, false, true), 0xFF]);
                self.code.push(modrm(0b10, 0b010, 0b100)); // /2, SIB follows
                self.code.push(modrm(0b00, 0b100, 0b010)); // sib: none, base r10
                self.imm32((*slot * 8) as i32);
                self.cleanup_tail(args.len().max(1));
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::CallImport { dst, import, args } => {
                self.place_args(alloc, args);
                self.abs64_window(0xB8, Sym::Import(*import as usize)); // movabs rax
                self.bytes(&[0xFF, 0xD0]); // call rax
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            // Address import: dst = absolute address of the imported
            // entity (vtable row / static slot), resolved by the linker
            // into the import slot — same slot mechanism as CallImport,
            // loaded instead of called.
            Instr::LoadImport { dst, import } => {
                self.abs64_window(0xB8, Sym::Import(*import as usize)); // movabs rax
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::Alloc { dst, bytes } => {
                self.uses_alloc = true;
                self.bytes(&[0xBE]); // mov esi, imm32
                self.imm32(*bytes as i32);
                self.call_rel32(CallTarget::HelperAlloc);
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::VTableAddr { dst, row } => {
                let r = (*row as usize).min(self.vtable_offsets.len().saturating_sub(1));
                let off = self.vtable_offsets[r];
                self.abs64_window(0xB9, Sym::VtableOff(off)); // movabs rcx
                self.put(RCX, alloc.loc[dst.0 as usize]);
            }
            Instr::LoadStaticAddr { dst, idx } => {
                let r = (*idx as usize).min(self.static_offsets.len().saturating_sub(1));
                self.abs64_window(0xB9, Sym::Static(r)); // movabs rcx
                self.put(RCX, alloc.loc[dst.0 as usize]);
            }
            Instr::LoadField { dst, base, off } => {
                let d = *off as i32;
                self.get(R10, alloc.loc[base.0 as usize]);
                match alloc.loc[dst.0 as usize] {
                    Loc::Reg(r) => self.field_load(r.id(), d),
                    Loc::Slot(k) => {
                        self.field_load(RAX, d);
                        self.store_slot(k, RAX);
                    }
                }
            }
            Instr::LoadByte { dst, base, off } => {
                let d = *off as i32;
                self.get(R10, alloc.loc[base.0 as usize]);
                match alloc.loc[dst.0 as usize] {
                    Loc::Reg(r) => self.field_load_byte(r.id(), d),
                    Loc::Slot(k) => {
                        self.field_load_byte(RAX, d);
                        self.store_slot(k, RAX);
                    }
                }
            }
            Instr::StoreField { base, off, src } => {
                let d = *off as i32;
                self.get(R10, alloc.loc[base.0 as usize]);
                self.get(R11, alloc.loc[src.0 as usize]);
                self.field_store(d);
            }
            Instr::StoreField { base, off, src } => {
                let d = *off as i32;
                self.get(R10, alloc.loc[base.0 as usize]);
                self.get(R11, alloc.loc[src.0 as usize]);
                self.field_store(d);
            }
            Instr::BuildArgv { dst } => {
                // Process argv → `string[]` (array layout mirrors
                // `NewArray`: length at 0, elements from 8; strings in
                // literal layout). Reads the kernel stack through the
                // startup-saved slot; a zero slot (JIT) or wild argc
                // yields an empty/truncated array, never a fault.
                // Scratch: rax, rcx, rdx, rsi, rdi, r8, r9, r11.
                // rbx, rbp, r12-r15 are never touched. Stack is
                // balanced (save + restore around the sequence); alloc
                // calls are individually realigned (see below).
                self.uses_alloc = true;
                self.bytes(&[0x48, 0x89, 0xE0]); // mov rax, rsp
                self.bytes(&[0x48, 0x83, 0xE4, 0xF0]); // and rsp, -16
                self.bytes(&[0x50]); // push rax (orig rsp); rsp%16==8 past here
                self.abs64_window(0xB8, Sym::ArgvRsp); // movabs rax, slot
                // REX by helper, not hand bytes: r9 needs REX.R while
                // the [rax] base needs REX.B *clear* (0x4D here would
                // silently redirect the read to [r8] — caught live).
                self.bytes(&[rex(true, true, false), 0x8B, modrm(0, 1, 0)]); // mov r9, [rax]
                // Same trap: r9 as a memory base needs REX.B.
                self.bytes(&[rex(true, false, true), 0x8B, modrm(0, 1, 1)]); // mov rcx, [r9] (argc)
                self.bytes(&[0x48, 0xC7, 0xC0, 0x00, 0x04, 0x00, 0x00]); // mov rax, 1024
                self.bytes(&[0x48, 0x39, 0xC1]); // cmp rcx, rax
                let jbe_argc = self.code.len();
                self.bytes(&[0x76, 0x00]); // jbe argc_ok
                self.bytes(&[0x48, 0x89, 0xC1]); // mov rcx, rax (truncate)
                let argc_ok = self.code.len();
                self.code[jbe_argc + 1] = (argc_ok - (jbe_argc + 2)) as i8 as u8;
                self.bytes(&[rex(true, false, false), 0x8D, 0x34, 0xCD, 0x08, 0x00, 0x00, 0x00]); // lea rsi, [rcx*8+8] (rsi is reg 6: REX.R clear)
                self.bytes(&[0x48, 0x83, 0xEC, 0x08]); // sub rsp, 8 (align call)
                self.call_rel32(CallTarget::HelperAlloc);
                self.bytes(&[0x48, 0x83, 0xC4, 0x08]); // add rsp, 8
                self.bytes(&[0x48, 0x89, 0xC7]); // mov rdi, rax (array base)
                // Reload argc: rcx died in the alloc call (SysV
                // caller-saved). r9 (saved rsp) survives it, so reread
                // from the slot instead of trusting the stale register
                // (shipped the stale version once: Length read 0).
                self.bytes(&[rex(true, false, true), 0x8B, modrm(0, 1, 1)]); // mov rcx, [r9] (argc, reloaded)
                // ModRM via helper: mod=00 (memory [rdi]), NOT mod=11
                // (which would move rcx into rdi and lose the array —
                // shipped that way once, faulted on the Length read).
                self.bytes(&[rex(true, false, false), 0x89, modrm(0, 1, 7)]); // mov [rdi], rcx (length; rdi is reg 7: REX.B clear)
                self.bytes(&[0x4D, 0x31, 0xDB]); // xor r11d, r11d (i = 0)
                self.bytes(&[rex(true, true, false), 0x3B, modrm(0, 3, 7)]); // cmp r11, [rdi]: opcode 0x3B (r64, r/m64) so jge skips when i >= length; 0x39 would compare backwards and skip whenever length > 0
                let jge_done = self.code.len();
                self.bytes(&[0x0F, 0x8D, 0x00, 0x00, 0x00, 0x00]); // jge done
                let per_arg = self.code.len();
                self.bytes(&[0x4B, 0x8B, 0x74, 0xD9, 0x08]); // mov rsi, [r9+r11*8+8] (rsi is reg 6: REX.R clear; X/B set)
                self.bytes(&[0x49, 0x89, 0xF0]); // mov r8, rsi (rsi source: REX.R clear)
                self.bytes(&[0x31, 0xD2]); // xor edx, edx (len = 0)
                let strlen1 = self.code.len();
                self.bytes(&[0x41, 0x8A, 0x04, 0x10]); // mov al, [r8+rdx]
                self.bytes(&[0x84, 0xC0]); // test al, al
                let jz_s1 = self.code.len();
                self.bytes(&[0x74, 0x00]); // jz strlen1_done
                self.bytes(&[0x48, 0xFF, 0xC2]); // inc rdx
                let jmp_s1 = self.code.len();
                self.bytes(&[0xEB, 0x00]); // jmp strlen1
                let strlen1_done = self.code.len();
                self.code[jz_s1 + 1] = (strlen1_done - (jz_s1 + 2)) as i8 as u8;
                self.code[jmp_s1 + 1] = (strlen1 as i32 - (jmp_s1 + 2) as i32) as i8 as u8;
                self.bytes(&[rex(true, false, false), 0x8D, 0x72, 0x10]); // lea rsi, [rdx+16] (rsi is reg 6: REX.R clear)
                self.bytes(&[0x48, 0x83, 0xEC, 0x08]); // sub rsp, 8
                self.call_rel32(CallTarget::HelperAlloc);
                self.bytes(&[0x48, 0x83, 0xC4, 0x08]); // add rsp, 8
                // strlen2 and the copy loop below both use `al` as
                // scratch, which would clobber strobj's low byte in rax
                // (shipped that way once: every element pointed 0-255
                // bytes low). Save/restore around each clobber region;
                // balanced pushes, no calls between, rsp parity intact.
                self.bytes(&[0x50]); // push rax (save strobj)
                // strlen again: rdx died in alloc, r8 src survived.
                self.bytes(&[0x31, 0xD2]); // xor edx, edx
                let strlen2 = self.code.len();
                self.bytes(&[0x41, 0x8A, 0x04, 0x10]); // mov al, [r8+rdx]
                self.bytes(&[0x84, 0xC0]); // test al, al
                let jz_s2 = self.code.len();
                self.bytes(&[0x74, 0x00]); // jz strlen2_done
                self.bytes(&[0x48, 0xFF, 0xC2]); // inc rdx
                let jmp_s2 = self.code.len();
                self.bytes(&[0xEB, 0x00]); // jmp strlen2
                let strlen2_done = self.code.len();
                self.code[jz_s2 + 1] = (strlen2_done - (jz_s2 + 2)) as i8 as u8;
                self.code[jmp_s2 + 1] = (strlen2 as i32 - (jmp_s2 + 2) as i32) as i8 as u8;
                self.bytes(&[0x58]); // pop rax (restore strobj)
                self.bytes(&[0x48, 0x89, 0x10]); // mov [rax], rdx (length)
                self.bytes(&[0xC7, 0x40, 0x08, 0x01, 0x00, 0x00, 0x00]); // mov dword [rax+8], 1
                self.bytes(&[0x50]); // push rax (save strobj; copy uses al; before the empty-string jz so push/pop always pair)
                self.bytes(&[0x48, 0x85, 0xD2]); // test rdx, rdx
                let jz_copy = self.code.len();
                self.bytes(&[0x74, 0x00]); // jz copy_done (empty string)
                self.bytes(&[0x31, 0xC9]); // xor ecx, ecx
                let copy_head = self.code.len();
                self.bytes(&[0x41, 0x8A, 0x34, 0x08]); // mov sil, [r8+rcx] (sil, not al: al is rax's low byte and rax holds strobj live through the copy; loading a char into al rewrites the pointer and every later store lands wild)
                self.bytes(&[0x48, 0x88, 0xB4, 0x08, 0x10, 0x00, 0x00, 0x00]); // mov [rax+rcx+16], sil (rcx index: REX.X clear)
                self.bytes(&[0x48, 0xFF, 0xC1]); // inc rcx
                self.bytes(&[0x48, 0x39, 0xD1]); // cmp rcx, rdx
                let jl_copy = self.code.len();
                self.bytes(&[0x7C, 0x00]); // jl copy_head
                let copy_done = self.code.len();
                self.code[jz_copy + 1] = (copy_done - (jz_copy + 2)) as i8 as u8;
                self.code[jl_copy + 1] = (copy_head as i32 - (jl_copy + 2) as i32) as i8 as u8;
                self.bytes(&[0x58]); // pop rax (restore strobj)
                self.bytes(&[0x4A, 0x89, 0x44, 0xDF, 0x08]); // mov [rdi+r11*8+8], rax (rdi base is reg 7: REX.B clear; X set)
                self.bytes(&[0x49, 0xFF, 0xC3]); // inc r11
                self.bytes(&[rex(true, true, false), 0x3B, modrm(0, 3, 7)]); // cmp r11, [rdi]: opcode 0x3B (r64, r/m64) so jge skips when i >= length; 0x39 would compare backwards and skip whenever length > 0
                let jl_per = self.code.len();
                self.bytes(&[0x0F, 0x8C, 0x00, 0x00, 0x00, 0x00]); // jl per_arg
                let done = self.code.len();
                for (pos, target) in [(jge_done, done), (jl_per, per_arg)] {
                    let rel = target as i32 - (pos + 6) as i32;
                    self.code[pos + 2..pos + 6].copy_from_slice(&rel.to_le_bytes());
                }
                self.bytes(&[0x48, 0x89, 0xF8]); // mov rax, rdi (result)
                self.bytes(&[0x5C]); // pop rsp (restore original)
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::WriteStr { off, len } => {
                self.write_str_syscall(*off, *len, 1);
            }
            Instr::EWriteStr { off, len } => {
                self.write_str_syscall(*off, *len, 2);
            }
            Instr::LoadStrAddr { dst, off } => {
                self.abs64_window(0xB9, Sym::String(*off as usize)); // movabs rcx, str_addr
                self.put(RCX, alloc.loc[dst.0 as usize]);
            }
            Instr::MemCpy { dst, src, len } => {
                self.get(R10, alloc.loc[dst.0 as usize]); // r10 = dst
                self.get(R11, alloc.loc[src.0 as usize]); // r11 = src
                self.get(8, alloc.loc[len.0 as usize]); // r8 = len
                self.memcpy_loop();
            }
            Instr::MemCmp { dst, a, b, len } => {
                self.get(R10, alloc.loc[a.0 as usize]); // r10 = a
                self.get(R11, alloc.loc[b.0 as usize]); // r11 = b
                self.get(8, alloc.loc[len.0 as usize]); // r8 = len
                self.memcmp_loop(); // result (0/1) in rax
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::AllocDyn { dst, bytes } => {
                self.uses_alloc = true;
                // rsi = size
                self.get(RSI, alloc.loc[bytes.0 as usize]);
                self.call_rel32(CallTarget::HelperAlloc);
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::WriteString { src } => {
                // write(1, obj+16, len) where len = *(obj+0).
                // Syscall convention: rdi=fd, rsi=buf, rdx=count, rax=nr.
                self.get(RSI, alloc.loc[src.0 as usize]); // rsi = obj
                self.bytes(&[0x48, 0x8B, 0x16]); // mov rdx, [rsi]    (rdx = len)
                self.bytes(&[0x48, 0x83, 0xC6, 0x10]); // add rsi, 0x10 (rsi = obj+16 = buf)
                self.bytes(&[0xBF, 0x01, 0x00, 0x00, 0x00]); // mov edi, 1   (fd)
                self.bytes(&[0xB8, 0x01, 0x00, 0x00, 0x00]); // mov eax, 1 (syscall nr)
                self.bytes(&[0x0F, 0x05]); // syscall
            }
            Instr::WriteInt { src } => {
                self.uses_int = true;
                self.get(RAX, alloc.loc[src.0 as usize]);
                self.call_rel32(CallTarget::HelperItoa);
            }
            Instr::EWriteInt { src } => {
                self.uses_int_err = true;
                self.get(RAX, alloc.loc[src.0 as usize]);
                self.call_rel32(CallTarget::HelperItoaErr);
            }
            Instr::ScanInt { dst } => {
                self.uses_scan = true;
                self.call_rel32(CallTarget::HelperScanInt);
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::Retain { src } => {
                self.uses_retain = true;
                self.get(RAX, alloc.loc[src.0 as usize]);
                self.call_rel32(CallTarget::HelperRetain);
            }
            Instr::Release { src, bytes } => {
                self.uses_release = true;
                self.get(RAX, alloc.loc[src.0 as usize]);
                self.bytes(&[0xBE]); // mov esi, imm32 (capacity)
                self.imm32(*bytes as i32);
                self.call_rel32(CallTarget::HelperRelease);
            }
            Instr::Exit => unreachable!("Exit is a terminator"),
            // v2 instructions not yet supported by the x86_64 backend.
            Instr::CallByName { .. } => unreachable!("CallByName not yet supported by x86_64"),
            Instr::FConst { dst, bits } => {
                // Emit the f64 bit pattern as an i64 constant (the vreg holds
                // the double's bit pattern; FBin/FCmp move it to XMM on use).
                let imm = *bits as i64;
                let fits32 = (-2147483648i64..=2147483647i64).contains(&imm);
                match alloc.loc[dst.0 as usize] {
                    Loc::Reg(r) => {
                        if fits32 {
                            self.code.push(rex(true, false, r.needs_rex()));
                            self.code.push(0xC7);
                            self.code.push(modrm(0b11, 0, r.id()));
                            self.imm32(imm as i32);
                        } else {
                            self.code.push(rex(true, false, r.needs_rex()));
                            self.code.push(0xB8 | (r.id() & 7));
                            self.imm64(imm as u64);
                        }
                    }
                    Loc::Slot(k) => {
                        if fits32 {
                            let (m, d) = slot_addr(k, 0);
                            self.code.push(rex(true, false, false));
                            self.code.push(0xC7);
                            self.code.push(m);
                            self.disp(d);
                            self.imm32(imm as i32);
                        } else {
                            self.code.push(rex(true, false, false));
                            self.code.push(0xB8);
                            self.imm64(imm as u64);
                            self.store_slot(k, RAX);
                        }
                    }
                }
            }
            Instr::FBin { dst, op, a, b } => {
                self.fbin(alloc, *dst, *op, *a, *b)
            }
            Instr::FCmp { dst, op, a, b } => {
                self.fcmp(alloc, *dst, *op, *a, *b)
            }
            Instr::FNeg { dst, a } => {
                self.get(RAX, alloc.loc[a.0 as usize]);
                // Negate the IEEE 754 double by flipping the sign bit (bit 63).
                // A `xor rax, imm8/imm32` cannot flip only bit 63 (the immediate
                // sign-extends), so load the constant into a scratch register
                // (rcx is not a vreg) and XOR.
                self.bytes(&[0x48, 0xB9]); // movabs rcx, 0x8000000000000000
                self.bytes(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80]);
                self.bytes(&[0x48, 0x31, 0xC8]); // xor rax, rcx
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::StackAlloc { .. } => unreachable!("StackAlloc not yet supported by x86_64"),
            Instr::Drop { .. } => unreachable!("Drop not yet supported by x86_64"),
            // ---- exception handling (setjmp/longjmp handler frames) ----
            Instr::HandlerEnter { dst } => {
                // Push the handler frame. rax = 0 on first entry, 1 when an
                // exception longjmps back here (registers restored by runtime).
                self.call_rel32(CallTarget::HelperExcEnter);
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::HandlerPop => {
                self.call_rel32(CallTarget::HelperExcPop);
            }
            Instr::LoadExc { dst } => {
                self.call_rel32(CallTarget::HelperExcLoad);
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::ClearInflight => {
                self.call_rel32(CallTarget::HelperExcClear);
            }
            Instr::TaskAlloc { dst, bytes } => {
                // Reuse the bump allocator: allocate a task frame from the
                // program heap. The frame layout is:
                //   +0  state (0 = Ready, 1 = Running, 2 = Suspended, 3 = Done)
                //   +8  saved return address
                //   +16 current state-machine state
                //   +24 locals (vregs)
                self.uses_alloc = true;
                self.bytes(&[0xBE]); // mov esi, imm32
                self.imm32(*bytes as i32);
                self.call_rel32(CallTarget::HelperAlloc);
                // Initialize state to 1 (Running) at offset 0.
                self.bytes(&[0xC7, 0x00, 0x01, 0x00, 0x00, 0x00]); // mov [rax], 1
                // Store the task pointer in the destination.
                self.put(RAX, alloc.loc[dst.0 as usize]);
            }
            Instr::Suspend { state } => {
                // Stub: save the state value to the task frame (offset 16).
                // The backend will intercept this to actually suspend the
                // coroutine and return to the scheduler. For now, we just
                // record the state; the function continues (the scheduler
                // integration is a later phase).
                let _ = state;
                self.bytes(&[0x90]); // nop
            }
            Instr::CoroutineResume { task, dst } => {
                // Stub: load the task pointer into dst. The backend will
                // read the saved state from the task frame and jump to the
                // appropriate state-machine block. For now, just pass the
                // task pointer through.
                let tl = alloc.loc[task.0 as usize];
                match alloc.loc[dst.0 as usize] {
                    Loc::Reg(r) => {
                        self.get(r.id(), tl);
                    }
                    Loc::Slot(k) => {
                        self.get(RAX, tl);
                        self.store_slot(k, RAX);
                    }
                }
            }
        }
    }

    fn bin(&mut self, alloc: &regalloc::Allocation, dst: VReg, op: BinOp, a: VReg, b: VReg) {
        let dl = alloc.loc[dst.0 as usize];

        if matches!(op, BinOp::Div | BinOp::Rem) {
            // Hardware idiv semantics; zero divisor traps. Div takes the
            // quotient (rax), Rem the remainder (rdx).
            self.get(RAX, alloc.loc[a.0 as usize]);
            self.get(R11, alloc.loc[b.0 as usize]);
            self.bytes(&[0x48, 0x99]); // cqo
            self.bytes(&[rex(true, false, true), 0xF7, modrm(0b11, 7, R11)]);
            if op == BinOp::Rem {
                // Remainder rides in rdx; route it into the destination.
                match dl {
                    Loc::Reg(dr) => self.mov_rr(dr.id(), 2),
                    Loc::Slot(k) => self.store_slot(k, 2),
                }
            } else {
                self.put(RAX, dl);
            }
            return;
        }

        if matches!(op, BinOp::Shl | BinOp::Shr | BinOp::ShrShr) {
            // Variable shifts need the count in cl. Allocated values never
            // live in rcx, so the scratch use is safe between calls.
            self.get(RAX, alloc.loc[a.0 as usize]);
            self.get(RCX, alloc.loc[b.0 as usize]);
            let op_ext = match op {
                BinOp::Shl => 4u8,     // SHL group /4
                BinOp::Shr => 7u8,     // SAR group /7 — arithmetic, signed ints
                BinOp::ShrShr => 5u8,  // SHR group /5 — logical, unsigned
                _ => unreachable!("non-shift op in shift path"),
            };
            self.bytes(&[0x48, 0xD3]);
            self.code.push(modrm(0b11, op_ext, RAX));
            self.put(RAX, dl);
            return;
        }

        // One opcode covers both register and memory source operands:
        // add/sub/and/or/xor read r/m (03/2B/23/0B/33), imul reads r/m (0F AF).
        let opbytes: &[u8] = match op {
            BinOp::Add => &[0x03],
            BinOp::Sub => &[0x2B],
            BinOp::Mul => &[0x0F, 0xAF],
            BinOp::And => &[0x23],
            BinOp::Or => &[0x0B],
            BinOp::BitAnd => &[0x23],
            BinOp::BitOr => &[0x0B],
            BinOp::BitXor => &[0x33],
            BinOp::Div | BinOp::Rem => unreachable!("handled above"),
            BinOp::Shl | BinOp::Shr | BinOp::ShrShr => unreachable!("handled above"),
            BinOp::Is => unreachable!("lowered to Const in IR"),
        };

        let bl = alloc.loc[b.0 as usize];
        let emit_operand = |e: &mut Self, rd: u8| match bl {
            Loc::Reg(br) => {
                e.code.push(rex(true, rd >= 8, br.needs_rex()));
                e.bytes(opbytes);
                e.code.push(modrm(0b11, rd, br.id()));
            }
            Loc::Slot(ks) => {
                let (m, d) = slot_addr(ks, rd & 7);
                e.code.push(rex(true, rd >= 8, false));
                e.bytes(opbytes);
                e.code.push(m);
                e.disp(d);
            }
        };

        match dl {
            Loc::Reg(dr) => {
                // Hazard: if b lives in the destination register itself,
                // loading `a` into it would destroy b. Stash b in r11
                // first, then compute against the stash.
                let b_in_dst = matches!(bl, Loc::Reg(br) if br.id() == dr.id());
                if b_in_dst {
                    self.mov_rr(R11, dr.id());
                    self.get(dr.id(), alloc.loc[a.0 as usize]);
                    let rex_b = rex(true, dr.id() >= 8, true);
                    self.code.push(rex_b);
                    self.bytes(match op {
                        BinOp::Add => &[0x03],
                        BinOp::Sub => &[0x2B],
                        BinOp::Mul => &[0x0F, 0xAF],
                        BinOp::And | BinOp::BitAnd => &[0x23],
                        BinOp::Or | BinOp::BitOr => &[0x0B],
                        BinOp::BitXor => &[0x33],
                        BinOp::Div
                        | BinOp::Rem
                        | BinOp::Shl
                        | BinOp::Shr
                        | BinOp::ShrShr
                        | BinOp::Is => {
                            unreachable!("handled above")
                        }
                    });
                    self.code.push(modrm(0b11, dr.id(), R11));
                } else {
                    self.get(dr.id(), alloc.loc[a.0 as usize]);
                    emit_operand(self, dr.id());
                }
            }
            Loc::Slot(_) => {
                self.get(RAX, alloc.loc[a.0 as usize]);
                emit_operand(self, RAX);
                self.put(RAX, dl);
            }
        }
    }

    /// `movq xmm0, rax` / `movq xmm1, r11` — load the two f64 bit patterns
    /// (held in integer registers) into XMM registers for a scalar-double
    /// operation.
    fn load_fops(&mut self) {
        self.bytes(&[0x66, 0x48, 0x0F, 0x6E]);
        self.code.push(modrm(0b11, 0, RAX)); // reg=xmm0, rm=rax
        // r11 extends the r/m field (REX.B), not the reg field (REX.R).
        self.bytes(&[0x66, 0x49, 0x0F, 0x6E]);
        self.code.push(modrm(0b11, 1, R11)); // reg=xmm1, rm=r11
    }

    /// `movq rax, xmm0` — move the scalar-double result back to the integer
    /// register that holds the vreg's bit pattern.
    fn store_fop(&mut self) {
        // `66 0F 7E /r` is movq r64, xmm (xmm -> int). Without the 66 prefix it
        // is movq xmm, r/m (int -> xmm), i.e. the wrong direction.
        self.bytes(&[0x66, 0x48, 0x0F, 0x7E]);
        self.code.push(modrm(0b11, 0, RAX)); // reg=xmm0(src), rm=rax(dst)
    }

    /// dst = a op b (IEEE 754 double). Hybrid: the vregs hold the double's
    /// bit pattern in 64-bit integer registers; we move them to XMM0/XMM1,
    /// perform the scalar-double op, and move the result back.
    fn fbin(&mut self, alloc: &regalloc::Allocation, dst: VReg, op: FBinOp, a: VReg, b: VReg) {
        self.get(RAX, alloc.loc[a.0 as usize]);
        self.get(R11, alloc.loc[b.0 as usize]);
        self.load_fops();
        let opcode = match op {
            FBinOp::Add => 0x58, // addsd
            FBinOp::Sub => 0x5C, // subsd
            FBinOp::Mul => 0x59, // mulsd
            FBinOp::Div => 0x5E, // divsd
        };
        // 66 prefix makes these scalar-DOUBLE ops (ADDSD/SUBSD/MULSD/DIVSD);
        // without it they are the scalar-single (...SS) variants.
        self.bytes(&[0x66, 0x48, 0x0F, opcode]);
        self.code.push(modrm(0b11, 0, 1)); // reg=xmm0, rm=xmm1
        self.store_fop();
        self.put(RAX, alloc.loc[dst.0 as usize]);
    }

    /// dst = (a `op` b) as 0/1 (IEEE 754 double). `comisd xmm0, xmm1` sets
    /// CF/ZF/PF (xmm0 = a, xmm1 = b):
    ///   a>b   → CF=0, ZF=0
    ///   a==b  → CF=0, ZF=1
    ///   a<b   → CF=1, ZF=1
    ///   NaN   → CF=1, ZF=1, PF=1
    /// Each comparison maps to a `setcc` into `al`; IEEE-754 then requires
    /// that every *ordered* comparison is false for NaN while `!=` is true:
    ///   Ge = SETAE (CF==0)               [already 0 for NaN]
    ///   Gt = SETA  (CF==0 & ZF==0)       [already 0 for NaN]
    ///   Lt = SETB  (CF==1)  & SETNP      [SETB is 1 for NaN → mask w/ setnp]
    ///   Le = SETBE (CF==1 | ZF==1) & SETNP
    ///   Eq = SETE  (ZF==1)   & SETNP
    ///   Ne = SETNE (ZF==0)   | SETP      [NaN must be true]
    /// `cl` (low byte of rcx) is free scratch: rcx is not a vreg register.
    fn fcmp(&mut self, alloc: &regalloc::Allocation, dst: VReg, op: FCmpOp, a: VReg, b: VReg) {
        self.get(RAX, alloc.loc[a.0 as usize]);
        self.get(R11, alloc.loc[b.0 as usize]);
        self.load_fops();
        // comisd xmm0, xmm1 (scalar DOUBLE). The 66 prefix is mandatory:
        // `0F 2F` alone is comiss (single) and would only compare the low 32
        // bits of each double.
        self.bytes(&[0x66, 0x48, 0x0F, 0x2F]);
        self.code.push(modrm(0b11, 0, 1)); // reg=xmm0, rm=xmm1
        let setcc: u8 = match op {
            FCmpOp::Ge => 0x93, // setae (CF==0)
            FCmpOp::Gt => 0x97, // seta  (CF==0 & ZF==0)
            FCmpOp::Lt => 0x92, // setb  (CF==1)
            FCmpOp::Le => 0x96, // setbe (CF==1 | ZF==1)
            FCmpOp::Eq => 0x94, // sete  (ZF==1)
            FCmpOp::Ne => 0x95, // setne (ZF==0)
        };
        self.bytes(&[0x0F, setcc]);
        self.code.push(modrm(0b11, 0, 0)); // dest = al
        match op {
            // Ge / Gt already yield 0 for NaN.
            FCmpOp::Ge | FCmpOp::Gt => {}
            // Lt / Le / Eq must be false for NaN: al &= setnp (cl).
            FCmpOp::Lt | FCmpOp::Le | FCmpOp::Eq => {
                self.bytes(&[0x0F, 0x9B]); self.code.push(modrm(0b11, 0, 2)); // setnp cl
                self.bytes(&[0x20]); self.code.push(modrm(0b11, 2, 0));       // and al, cl
            }
            // Ne must be TRUE for NaN: al |= setp (cl).
            FCmpOp::Ne => {
                self.bytes(&[0x0F, 0x9A]); self.code.push(modrm(0b11, 0, 2)); // setp cl
                self.bytes(&[0x08]); self.code.push(modrm(0b11, 2, 0));       // or al, cl
            }
        }
        self.bytes(&[0x48, 0x0F, 0xB6, 0xC0]); // movzx rax, al
        self.put(RAX, alloc.loc[dst.0 as usize]);
    }

    /// dst = (a `op` b) as 0/1. Sequence: load `a` into rax, compare
    /// against `b` (register or slot), materialize the flag with setcc on
    /// al, then zero-extend into the destination.
    fn cmp(&mut self, alloc: &regalloc::Allocation, dst: VReg, op: CmpOp, a: VReg, b: VReg) {
        let setcc: u8 = match op {
            CmpOp::Eq => 0x94, // sete
            CmpOp::Ne => 0x95, // setne
            CmpOp::Lt => 0x9C, // setl
            CmpOp::Gt => 0x9F, // setg
            CmpOp::Le => 0x9E, // setle
            CmpOp::Ge => 0x9D, // setge
        };

        self.get(RAX, alloc.loc[a.0 as usize]);
        match alloc.loc[b.0 as usize] {
            Loc::Reg(br) => {
                // cmp rax, r/m — 0x3B keeps the loaded `a` as the LEFT
                // operand so the condition codes match `a op b`.
                // (0x39 would compute b−a and invert every ordering.)
                self.bytes(&[rex(true, false, br.needs_rex()), 0x3B]);
                self.code.push(modrm(0b11, RAX, br.id()));
            }
            Loc::Slot(ks) => {
                let (m, d) = slot_addr(ks, RAX);
                self.bytes(&[rex(true, false, false), 0x3B]); // cmp rax, [mem]
                self.code.push(m);
                self.disp(d);
            }
        }
        self.bytes(&[0x0F, setcc, 0xC0]); // setcc al
        self.bytes(&[rex(true, false, false), 0x0F, 0xB6, modrm(0b11, RAX, RAX)]); // movzx rax, al
        self.put(RAX, alloc.loc[dst.0 as usize]);
    }
}
