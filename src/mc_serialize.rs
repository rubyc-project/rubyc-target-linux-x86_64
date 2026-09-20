//! Lossless `MachineCode` ↔ byte serializer for the target plugin ABI.
//!
//! The target plugin must pass a full `MachineCode` (code, data, relocations,
//! function bases, vtable offsets, helper offsets, import tables, exports,
//! heap size) across the FFI boundary so the image writer can apply
//! relocations against final addresses. This module encodes that structure as
//! a flat little-endian byte buffer with a simple section layout:
//!
//! ```text
//!   [u32 magic 0x524D4331]  "MCM1"
//!   [u64 code_len][code]
//!   [u64 data_len][data]
//!   [u32 fn_bases_count][u64 x N]
//!   [u32 vtable_offsets_count][u64 x N]
//!   [u8 helper_flags: 5 bits][u64 x set helpers]
//!   [u32 imports_count][ (u32 lib_name_len, lib_name, u32 entry_name_len, entry_name) x N ]
//!   [u32 import_bases_count][u64 x N]
//!   [u32 exports_count][ (u32 name_len, name, u32 fn_idx, u32 arity) x N ]
//!   [u32 relocs_count][ (u8 kind, u8 section, u64 site, u8 sym_tag, u64 sym_arg) x N ]
//!   [u64 heap_size]
//!   [u32 vtable_names_count][ (u32 len, name) x N ]  (trailing; tolerant read)
//!   [u32 static_names_count][ (u32 len, symbol) x N ]  (trailing; tolerant read)
//!   [u32 export_recv_count][ u8 x N ]  (trailing; tolerant read)
//!   [u32 init_fn or 0xFFFFFFFF]  (trailing; tolerant read)
//! ```

use rubyc::native::target::{MachineCode, Reloc, RelocKind, Section, Sym};

const MAGIC: u32 = 0x52_4D_43_31; // "MCM1"

/// Sym tags for the relocation table.
const SYM_FN: u8 = 0;
const SYM_STRING: u8 = 1;
const SYM_VTABLE: u8 = 2;
const SYM_HEAP: u8 = 3;
const SYM_ITOA: u8 = 4;
const SYM_ITOA_ERR: u8 = 5;
const SYM_SCAN_INT: u8 = 6;
const SYM_RETAIN: u8 = 7;
const SYM_RELEASE: u8 = 8;
const SYM_ALLOC: u8 = 9;
const SYM_IMPORT: u8 = 10;
const SYM_NAME: u8 = 11;
const SYM_STATIC: u8 = 12;
const SYM_EXC_ENTER: u8 = 13;
const SYM_EXC_POP: u8 = 14;
const SYM_EXC_THROW: u8 = 15;
const SYM_EXC_RETHROW: u8 = 16;
const SYM_EXC_LOAD: u8 = 17;
const SYM_EXC_CLEAR: u8 = 18;
const SYM_EXC_STATE: u8 = 19;
const SYM_ARGV_RSP: u8 = 20;

fn sym_tag(sym: &Sym) -> (u8, u64) {
    match sym {
        Sym::Fn(i) => (SYM_FN, *i as u64),
        Sym::String(o) => (SYM_STRING, *o as u64),
        Sym::VtableOff(o) => (SYM_VTABLE, *o as u64),
        Sym::Static(i) => (SYM_STATIC, *i as u64),
        Sym::HeapBase => (SYM_HEAP, 0),
        Sym::HelperItoa => (SYM_ITOA, 0),
        Sym::HelperItoaErr => (SYM_ITOA_ERR, 0),
        Sym::HelperScanInt => (SYM_SCAN_INT, 0),
        Sym::HelperRetain => (SYM_RETAIN, 0),
        Sym::HelperRelease => (SYM_RELEASE, 0),
        Sym::HelperAlloc => (SYM_ALLOC, 0),
        Sym::Import(i) => (SYM_IMPORT, *i as u64),
        Sym::Name(i) => (SYM_NAME, *i as u64),
        Sym::HelperExcEnter => (SYM_EXC_ENTER, 0),
        Sym::HelperExcPop => (SYM_EXC_POP, 0),
        Sym::HelperExcThrow => (SYM_EXC_THROW, 0),
        Sym::HelperExcRethrow => (SYM_EXC_RETHROW, 0),
        Sym::HelperExcLoad => (SYM_EXC_LOAD, 0),
        Sym::HelperExcClear => (SYM_EXC_CLEAR, 0),
        Sym::ExcState(o) => (SYM_EXC_STATE, *o as u64),
        Sym::ArgvRsp => (SYM_ARGV_RSP, 0),
    }
}

fn sym_from(tag: u8, arg: u64) -> Option<Sym> {
    Some(match tag {
        SYM_FN => Sym::Fn(arg as usize),
        SYM_STRING => Sym::String(arg as usize),
        SYM_VTABLE => Sym::VtableOff(arg as usize),
        SYM_STATIC => Sym::Static(arg as usize),
        SYM_HEAP => Sym::HeapBase,
        SYM_ITOA => Sym::HelperItoa,
        SYM_ITOA_ERR => Sym::HelperItoaErr,
        SYM_SCAN_INT => Sym::HelperScanInt,
        SYM_RETAIN => Sym::HelperRetain,
        SYM_RELEASE => Sym::HelperRelease,
        SYM_ALLOC => Sym::HelperAlloc,
        SYM_IMPORT => Sym::Import(arg as usize),
        SYM_NAME => Sym::Name(arg as usize),
        SYM_EXC_ENTER => Sym::HelperExcEnter,
        SYM_EXC_POP => Sym::HelperExcPop,
        SYM_EXC_THROW => Sym::HelperExcThrow,
        SYM_EXC_RETHROW => Sym::HelperExcRethrow,
        SYM_EXC_LOAD => Sym::HelperExcLoad,
        SYM_EXC_CLEAR => Sym::HelperExcClear,
        SYM_EXC_STATE => Sym::ExcState(arg as usize),
        SYM_ARGV_RSP => Sym::ArgvRsp,
        _ => return None,
    })
}

/// Encode a `MachineCode` into bytes.
pub fn encode(mc: &MachineCode) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::new();
    v.extend_from_slice(&MAGIC.to_le_bytes());

    // code
    v.extend_from_slice(&(mc.code.len() as u64).to_le_bytes());
    v.extend_from_slice(&mc.code);
    // data
    v.extend_from_slice(&(mc.data.len() as u64).to_le_bytes());
    v.extend_from_slice(&mc.data);

    // fn_bases
    v.extend_from_slice(&(mc.fn_bases.len() as u32).to_le_bytes());
    for b in &mc.fn_bases {
        v.extend_from_slice(&(*b as u64).to_le_bytes());
    }
    // vtable_offsets
    v.extend_from_slice(&(mc.vtable_offsets.len() as u32).to_le_bytes());
    for o in &mc.vtable_offsets {
        v.extend_from_slice(&(*o as u64).to_le_bytes());
    }
    // static_offsets
    v.extend_from_slice(&(mc.static_offsets.len() as u32).to_le_bytes());
    for o in &mc.static_offsets {
        v.extend_from_slice(&(*o as u64).to_le_bytes());
    }

    // helper flags: bit0=itoa, bit1=itoa_err, bit2=scan_int, bit3=retain, bit4=release, bit5=alloc
    let mut flags: u8 = 0;
    if mc.helper_itoa.is_some() {
        flags |= 1;
    }
    if mc.helper_itoa_err.is_some() {
        flags |= 2;
    }
    if mc.helper_scan_int.is_some() {
        flags |= 4;
    }
    if mc.helper_retain.is_some() {
        flags |= 8;
    }
    if mc.helper_release.is_some() {
        flags |= 16;
    }
    if mc.helper_alloc.is_some() {
        flags |= 32;
    }
    v.push(flags);
    if mc.helper_itoa.is_some() {
        v.extend_from_slice(&(mc.helper_itoa.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_itoa_err.is_some() {
        v.extend_from_slice(&(mc.helper_itoa_err.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_scan_int.is_some() {
        v.extend_from_slice(&(mc.helper_scan_int.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_retain.is_some() {
        v.extend_from_slice(&(mc.helper_retain.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_release.is_some() {
        v.extend_from_slice(&(mc.helper_release.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_alloc.is_some() {
        v.extend_from_slice(&(mc.helper_alloc.unwrap() as u64).to_le_bytes());
    }

    // Exception-runtime helpers: a second flag byte (bit0=enter, bit1=pop,
    // bit2=throw, bit3=rethrow, bit4=load, bit5=clear, bit6=exc_state).
    let mut eflags: u8 = 0;
    if mc.helper_exc_enter.is_some() {
        eflags |= 1;
    }
    if mc.helper_exc_pop.is_some() {
        eflags |= 2;
    }
    if mc.helper_exc_throw.is_some() {
        eflags |= 4;
    }
    if mc.helper_exc_rethrow.is_some() {
        eflags |= 8;
    }
    if mc.helper_exc_load.is_some() {
        eflags |= 16;
    }
    if mc.helper_exc_clear.is_some() {
        eflags |= 32;
    }
    if mc.exc_state_offset.is_some() {
        eflags |= 64;
    }
    if mc.helper_argv.is_some() {
        eflags |= 128;
    }
    v.push(eflags);
    if mc.helper_exc_enter.is_some() {
        v.extend_from_slice(&(mc.helper_exc_enter.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_exc_pop.is_some() {
        v.extend_from_slice(&(mc.helper_exc_pop.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_exc_throw.is_some() {
        v.extend_from_slice(&(mc.helper_exc_throw.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_exc_rethrow.is_some() {
        v.extend_from_slice(&(mc.helper_exc_rethrow.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_exc_load.is_some() {
        v.extend_from_slice(&(mc.helper_exc_load.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_exc_clear.is_some() {
        v.extend_from_slice(&(mc.helper_exc_clear.unwrap() as u64).to_le_bytes());
    }
    if mc.exc_state_offset.is_some() {
        v.extend_from_slice(&(mc.exc_state_offset.unwrap() as u64).to_le_bytes());
    }
    if mc.helper_argv.is_some() {
        v.extend_from_slice(&(mc.helper_argv.unwrap() as u64).to_le_bytes());
    }

    // imports
    v.extend_from_slice(&(mc.import_names.len() as u32).to_le_bytes());
    for (lib, entry) in &mc.import_names {
        push_str(&mut v, lib);
        push_str(&mut v, entry);
    }
    // import_bases
    v.extend_from_slice(&(mc.import_bases.len() as u32).to_le_bytes());
    for b in &mc.import_bases {
        v.extend_from_slice(&(*b as u64).to_le_bytes());
    }

    // exports
    v.extend_from_slice(&(mc.exports.len() as u32).to_le_bytes());
    for (name, fn_idx, arity) in &mc.exports {
        push_str(&mut v, name);
        v.extend_from_slice(&(*fn_idx as u32).to_le_bytes());
        v.extend_from_slice(&(*arity as u32).to_le_bytes());
    }

    // relocs
    v.extend_from_slice(&(mc.relocs.len() as u32).to_le_bytes());
    for r in &mc.relocs {
        let (tag, arg) = sym_tag(&r.sym);
        v.push(match r.kind {
            RelocKind::Abs64 => 0,
            RelocKind::Rel32Call => 1,
        });
        v.push(match r.section {
            Section::Code => 0,
            Section::Data => 1,
        });
        v.extend_from_slice(&(r.site as u64).to_le_bytes());
        v.push(tag);
        v.extend_from_slice(&arg.to_le_bytes());
    }

    // heap_size
    v.extend_from_slice(&mc.heap_size.to_le_bytes());

    // vtable_names (trailing section): class path per vtable row,
    // parallel to vtable_offsets; drives `__vtable` OBJECT exports.
    v.extend_from_slice(&(mc.vtable_names.len() as u32).to_le_bytes());
    for n in &mc.vtable_names {
        push_str(&mut v, n);
    }

    // static_names (trailing section): full `<path>::<field>` symbol per
    // static slot, parallel to static_offsets; drives static OBJECT exports.
    v.extend_from_slice(&(mc.static_names.len() as u32).to_le_bytes());
    for n in &mc.static_names {
        push_str(&mut v, n);
    }

    // export_takes_receiver (trailing section): one byte per export.
    v.extend_from_slice(&(mc.export_takes_receiver.len() as u32).to_le_bytes());
    for t in &mc.export_takes_receiver {
        v.push(u8::from(*t));
    }

    // init_fn (trailing section): opt-u32 like the IR container would,
    // hand-rolled here (u32::MAX = none).
    v.extend_from_slice(&mc.init_fn.map(|i| i as u32).unwrap_or(u32::MAX).to_le_bytes());

    v
}

fn push_str(v: &mut Vec<u8>, s: &str) {
    v.extend_from_slice(&(s.len() as u32).to_le_bytes());
    v.extend_from_slice(s.as_bytes());
}

/// A position-based reader over a byte buffer.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }
    fn peek(&self) -> Result<u8, String> {
        self.bytes.get(self.pos).copied().ok_or_else(|| "machine code: truncated".into())
    }
    fn byte(&mut self) -> Result<u8, String> {
        let b = self.peek()?;
        self.pos += 1;
        Ok(b)
    }
    fn u64(&mut self) -> Result<u64, String> {
        if self.pos + 8 > self.bytes.len() {
            return Err("machine code: truncated u64".into());
        }
        let x = u64::from_le_bytes(self.bytes[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(x)
    }
    fn u32(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.bytes.len() {
            return Err("machine code: truncated u32".into());
        }
        let x = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(x)
    }
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, String> {
        if self.pos + n > self.bytes.len() {
            return Err("machine code: truncated".into());
        }
        let out = self.bytes[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(out)
    }
    fn str(&mut self) -> Result<String, String> {
        let len = self.u32()? as usize;
        let b = self.bytes(len)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
}

/// Decode a `MachineCode` from bytes.
pub fn decode(bytes: &[u8]) -> Result<MachineCode, String> {
    if bytes.len() < 4 || u32::from_le_bytes(bytes[..4].try_into().unwrap()) != MAGIC {
        return Err("machine code: bad magic".into());
    }
    let mut r = Reader::new(&bytes[4..]);

    let code_len = r.u64()? as usize;
    let code = r.bytes(code_len)?;
    let data_len = r.u64()? as usize;
    let data = r.bytes(data_len)?;

    let fn_count = r.u32()? as usize;
    let mut fn_bases = Vec::with_capacity(fn_count);
    for _ in 0..fn_count {
        fn_bases.push(r.u64()? as usize);
    }
    let vt_count = r.u32()? as usize;
    let mut vtable_offsets = Vec::with_capacity(vt_count);
    for _ in 0..vt_count {
        vtable_offsets.push(r.u64()? as usize);
    }
    let st_count = r.u32()? as usize;
    let mut static_offsets = Vec::with_capacity(st_count);
    for _ in 0..st_count {
        static_offsets.push(r.u64()? as usize);
    }

    let flags = r.byte()?;
    let helper_itoa = if flags & 1 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_itoa_err = if flags & 2 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_scan_int = if flags & 4 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_retain = if flags & 8 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_release = if flags & 16 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_alloc = if flags & 32 != 0 { Some(r.u64()? as usize) } else { None };

    let eflags = r.byte()?;
    let helper_exc_enter = if eflags & 1 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_exc_pop = if eflags & 2 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_exc_throw = if eflags & 4 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_exc_rethrow = if eflags & 8 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_exc_load = if eflags & 16 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_exc_clear = if eflags & 32 != 0 { Some(r.u64()? as usize) } else { None };
    let exc_state_offset = if eflags & 64 != 0 { Some(r.u64()? as usize) } else { None };
    let helper_argv = if eflags & 128 != 0 { Some(r.u64()? as usize) } else { None };

    let import_count = r.u32()? as usize;
    let mut import_names = Vec::with_capacity(import_count);
    for _ in 0..import_count {
        let lib = r.str()?;
        let entry = r.str()?;
        import_names.push((lib, entry));
    }
    let ib_count = r.u32()? as usize;
    let mut import_bases = Vec::with_capacity(ib_count);
    for _ in 0..ib_count {
        import_bases.push(r.u64()? as usize);
    }

    let export_count = r.u32()? as usize;
    let mut exports = Vec::with_capacity(export_count);
    for _ in 0..export_count {
        let name = r.str()?;
        let fn_idx = r.u32()? as usize;
        let arity = r.u32()? as usize;
        exports.push((name, fn_idx, arity));
    }

    let reloc_count = r.u32()? as usize;
    let mut relocs = Vec::with_capacity(reloc_count);
    for _ in 0..reloc_count {
        let kind = match r.byte()? {
            0 => RelocKind::Abs64,
            1 => RelocKind::Rel32Call,
            _ => return Err("machine code: bad reloc kind".into()),
        };
        let section = match r.byte()? {
            0 => Section::Code,
            1 => Section::Data,
            _ => return Err("machine code: bad reloc section".into()),
        };
        let site = r.u64()? as usize;
        let tag = r.byte()?;
        let arg = r.u64()?;
        let sym = sym_from(tag, arg).ok_or("machine code: bad reloc sym")?;
        relocs.push(Reloc { kind, section, site, sym });
    }

    let heap_size = r.u64()?;

    // vtable_names (trailing section; absent in older buffers).
    let mut vtable_names = Vec::new();
    if !r.is_empty() {
        let vn_count = r.u32()? as usize;
        vtable_names.reserve(vn_count);
        for _ in 0..vn_count {
            vtable_names.push(r.str()?);
        }
    }

    // static_names (trailing section; absent in older buffers).
    let mut static_names = Vec::new();
    if !r.is_empty() {
        let sn_count = r.u32()? as usize;
        static_names.reserve(sn_count);
        for _ in 0..sn_count {
            static_names.push(r.str()?);
        }
    }

    // export_takes_receiver (trailing section; absent in older buffers:
    // defaults to receiver-taking, the pre-statics convention).
    let mut export_takes_receiver = Vec::new();
    if !r.is_empty() {
        let er_count = r.u32()? as usize;
        export_takes_receiver.reserve(er_count);
        for _ in 0..er_count {
            export_takes_receiver.push(r.byte()? != 0);
        }
    }

    // init_fn (trailing section; absent in older buffers).
    let mut init_fn = None;
    if !r.is_empty() {
        let v = r.u32()?;
        if v != u32::MAX {
            init_fn = Some(v as usize);
        }
    }

    Ok(MachineCode {
        code,
        data,
        fn_bases,
        vtable_offsets,
        vtable_names,
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
        helper_argv,
        import_names,
        import_bases,
        // Runtime-only load-time addresses are never serialized;
        // absolute addresses are meaningless across processes.
        import_abs: Vec::new(),
        exports,
        export_takes_receiver,
        init_fn,
        relocs,
        heap_size,
    })
}
