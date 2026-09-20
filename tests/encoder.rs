//! Tests for the x86-64 instruction encoder via the IR pipeline.

use rubyc::native::ir::{
    BinOp, CmpOp, FBinOp, FCmpOp, Instr, Program, IrFunction, ReceiverKind, Terminator,
};
use rubyc::native::target::CodeGenBackend;
use crate::x86::BACKEND;

fn run_ir(program: Program) -> rubyc::native::target::MachineCode {
    BACKEND.lower(&program).unwrap()
}

fn make_program(f: IrFunction) -> Program {
    Program {
        functions: vec![f],
        statics: vec![],
        vtables: vec![],
        imports: vec![],
        exports: vec![],
        strings: vec![],
        heap_size: 0,
        names: vec![],
        entry: Some(0),
        init: None,
    }
}

fn has_seq(code: &[u8], seq: &[u8]) -> bool {
    code.windows(seq.len()).any(|w| w == seq)
}

fn contains_any_movabs(code: &[u8], imm: u64) -> bool {
    let bytes = imm.to_le_bytes();
    (0..8).any(|id| {
        let mut seq = vec![0x48, 0xB8 | id];
        seq.extend_from_slice(&bytes);
        has_seq(code, &seq)
    })
}

#[test]
fn enc_consts() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    let b = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 1 });
    f.push(Instr::Const { dst: b, imm: -42 });
    f.blocks[0].term = Terminator::Ret(Some(a));
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}



#[test]
fn enc_mov() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    let b = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 7 });
    f.push(Instr::Mov { dst: b, a });
    f.blocks[0].term = Terminator::Ret(Some(b));
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_bin_add() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    let b = f.fresh_vreg();
    let c = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 3 });
    f.push(Instr::Const { dst: b, imm: 4 });
    f.push(Instr::Bin { dst: c, op: BinOp::Add, a, b });
    f.blocks[0].term = Terminator::Ret(Some(c));
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_bin_sub_mul_div_rem() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    let b = f.fresh_vreg();
    let c = f.fresh_vreg();
    let d = f.fresh_vreg();
    let e = f.fresh_vreg();
    let g = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 10 });
    f.push(Instr::Const { dst: b, imm: 3 });
    f.push(Instr::Bin { dst: c, op: BinOp::Sub, a, b });
    f.push(Instr::Bin { dst: d, op: BinOp::Mul, a, b });
    f.push(Instr::Bin { dst: e, op: BinOp::Div, a, b });
    f.push(Instr::Bin { dst: g, op: BinOp::Rem, a, b });
    f.blocks[0].term = Terminator::Ret(Some(g));
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}



#[test]
fn enc_neg() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    let b = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 5 });
    f.push(Instr::Neg { dst: b, a });
    f.blocks[0].term = Terminator::Ret(Some(b));
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_cmp_all_ops() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    let b = f.fresh_vreg();
    let r1 = f.fresh_vreg();
    let r2 = f.fresh_vreg();
    let r3 = f.fresh_vreg();
    let r4 = f.fresh_vreg();
    let r5 = f.fresh_vreg();
    let r6 = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 5 });
    f.push(Instr::Const { dst: b, imm: 3 });
    f.push(Instr::Cmp { dst: r1, op: CmpOp::Eq, a, b });
    f.push(Instr::Cmp { dst: r2, op: CmpOp::Ne, a, b });
    f.push(Instr::Cmp { dst: r3, op: CmpOp::Lt, a, b });
    f.push(Instr::Cmp { dst: r4, op: CmpOp::Gt, a, b });
    f.push(Instr::Cmp { dst: r5, op: CmpOp::Le, a, b });
    f.push(Instr::Cmp { dst: r6, op: CmpOp::Ge, a, b });
    f.blocks[0].term = Terminator::Ret(Some(r6));
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_branch_conditional() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    let b = f.fresh_vreg();
    let cond = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 1 });
    f.push(Instr::Const { dst: b, imm: 2 });
    f.push(Instr::Cmp { dst: cond, op: CmpOp::Lt, a, b });
    f.blocks[0].term = Terminator::Branch { cond: Some(cond), if_true: 1, if_false: 2 };
    f.blocks.push(rubyc::native::ir::Block { instrs: vec![], term: Terminator::Ret(None) });
    f.blocks.push(rubyc::native::ir::Block { instrs: vec![], term: Terminator::Ret(None) });
    let r1 = f.fresh_vreg();
    let r2 = f.fresh_vreg();
    f.blocks[1].instrs.push(Instr::Const { dst: r1, imm: 100 });
    f.blocks[1].term = Terminator::Ret(Some(r1));
    f.blocks[2].instrs.push(Instr::Const { dst: r2, imm: 200 });
    f.blocks[2].term = Terminator::Ret(Some(r2));
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_call_fn() {
    let mut f1 = IrFunction::new("helper", ReceiverKind::None, 0);
    let a1 = f1.fresh_vreg();
    f1.push(Instr::Const { dst: a1, imm: 42 });
    f1.blocks[0].term = Terminator::Ret(Some(a1));

    let mut f2 = IrFunction::new("main", ReceiverKind::None, 0);
    let a2 = f2.fresh_vreg();
    f2.push(Instr::Const { dst: a2, imm: 0 });
    f2.push(Instr::CallFn { dst: a2, fidx: 0, args: vec![] });
    f2.blocks[0].term = Terminator::Ret(Some(a2));

    let p = Program {
        functions: vec![f1, f2],
        statics: vec![],
        vtables: vec![],
        imports: vec![],
        exports: vec![],
        strings: vec![],
        heap_size: 0,
        names: vec![],
        entry: Some(1),
        init: None,
    };
    let mc = run_ir(p);
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_call_import() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 0 });
    f.push(Instr::CallImport { dst: a, import: 0, args: vec![] });
    f.blocks[0].term = Terminator::Ret(Some(a));
    let p = Program {
        functions: vec![f],
        statics: vec![],
        vtables: vec![],
        imports: vec![("libc".into(), "exit".into())],
        exports: vec![],
        strings: vec![],
        heap_size: 0,
        names: vec![],
        entry: Some(0),
        init: None,
    };
    let mc = run_ir(p);
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_alloc() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    f.push(Instr::Alloc { dst: a, bytes: 8 });
    f.blocks[0].term = Terminator::Ret(Some(a));
    let p = Program {
        functions: vec![f],
        statics: vec![],
        vtables: vec![],
        imports: vec![],
        exports: vec![],
        strings: vec![],
        heap_size: 16,
        names: vec![],
        entry: Some(0),
        init: None,
    };
    let mc = run_ir(p);
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_load_store_field() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let base = f.fresh_vreg();
    let val = f.fresh_vreg();
    let loaded = f.fresh_vreg();
    f.push(Instr::Alloc { dst: base, bytes: 16 });
    f.push(Instr::Const { dst: val, imm: 99 });
    f.push(Instr::StoreField { base, off: 0, src: val });
    f.push(Instr::LoadField { dst: loaded, base, off: 0 });
    f.blocks[0].term = Terminator::Ret(Some(loaded));
    let p = Program {
        functions: vec![f],
        statics: vec![],
        vtables: vec![],
        imports: vec![],
        exports: vec![],
        strings: vec![],
        heap_size: 32,
        names: vec![],
        entry: Some(0),
        init: None,
    };
    let mc = run_ir(p);
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_vtable_addr() {
    let mut f1 = IrFunction::new("method_a", ReceiverKind::This, 0);
    let a1 = f1.fresh_vreg();
    f1.push(Instr::Const { dst: a1, imm: 1 });
    f1.blocks[0].term = Terminator::Ret(Some(a1));

    let mut f2 = IrFunction::new("method_b", ReceiverKind::This, 0);
    let a2 = f2.fresh_vreg();
    f2.push(Instr::Const { dst: a2, imm: 2 });
    f2.blocks[0].term = Terminator::Ret(Some(a2));

    let mut f3 = IrFunction::new("main", ReceiverKind::None, 0);
    let vt = f3.fresh_vreg();
    f3.push(Instr::VTableAddr { dst: vt, row: 0 });
    f3.blocks[0].term = Terminator::Ret(Some(vt));

    let p = Program {
        functions: vec![f1, f2, f3],
        statics: vec![],
        vtables: vec![("ClassA".into(), vec![0, 1])],
        imports: vec![],
        exports: vec![],
        strings: vec![],
        heap_size: 0,
        names: vec![],
        entry: Some(2),
        init: None,
    };
    let mc = run_ir(p);
    assert!(!mc.code.is_empty());
    assert!(!mc.data.is_empty());
}

#[test]
fn enc_call_virt() {
    let mut f1 = IrFunction::new("method_a", ReceiverKind::This, 0);
    let a1 = f1.fresh_vreg();
    f1.push(Instr::Const { dst: a1, imm: 1 });
    f1.blocks[0].term = Terminator::Ret(Some(a1));

    let mut f2 = IrFunction::new("main", ReceiverKind::None, 0);
    let recv = f2.fresh_vreg();
    let vt = f2.fresh_vreg();
    let ret = f2.fresh_vreg();
    f2.push(Instr::Alloc { dst: recv, bytes: 16 });
    f2.push(Instr::VTableAddr { dst: vt, row: 0 });
    f2.push(Instr::StoreField { base: recv, off: 0, src: vt });
    f2.push(Instr::LoadField { dst: vt, base: recv, off: 0 });
    f2.push(Instr::CallVirt { dst: ret, slot: 0, recv: vt, args: vec![] });
    f2.blocks[0].term = Terminator::Ret(Some(ret));

    let p = Program {
        functions: vec![f1, f2],
        statics: vec![],
        vtables: vec![("ClassA".into(), vec![0])],
        imports: vec![],
        exports: vec![],
        strings: vec![],
        heap_size: 32,
        names: vec![],
        entry: Some(1),
        init: None,
    };
    let mc = run_ir(p);
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_write_str() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    f.push(Instr::WriteStr { off: 0, len: 5 });
    f.blocks[0].term = Terminator::Exit;
    let p = Program {
        functions: vec![f],
        statics: vec![],
        vtables: vec![],
        imports: vec![],
        exports: vec![],
        strings: b"hello\0".to_vec(),
        heap_size: 0,
        names: vec![],
        entry: Some(0),
        init: None,
    };
    let mc = run_ir(p);
    assert!(!mc.code.is_empty());
    assert!(!mc.data.is_empty());
}

#[test]
fn enc_write_int() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 42 });
    f.push(Instr::WriteInt { src: a });
    f.blocks[0].term = Terminator::Exit;
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_scan_int() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    f.push(Instr::ScanInt { dst: a });
    f.blocks[0].term = Terminator::Ret(Some(a));
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_released() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    f.push(Instr::Alloc { dst: a, bytes: 8 });
    f.push(Instr::Release { src: a, bytes: 8 });
    f.blocks[0].term = Terminator::Exit;
    let p = Program {
        functions: vec![f],
        statics: vec![],
        vtables: vec![],
        imports: vec![],
        exports: vec![],
        strings: vec![],
        heap_size: 16,
        names: vec![],
        entry: Some(0),
        init: None,
    };
    let mc = run_ir(p);
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_retain() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    f.push(Instr::Alloc { dst: a, bytes: 8 });
    f.push(Instr::Retain { src: a });
    f.push(Instr::Release { src: a, bytes: 8 });
    f.blocks[0].term = Terminator::Exit;
    let p = Program {
        functions: vec![f],
        statics: vec![],
        vtables: vec![],
        imports: vec![],
        exports: vec![],
        strings: vec![],
        heap_size: 16,
        names: vec![],
        entry: Some(0),
        init: None,
    };
    let mc = run_ir(p);
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_raw_bytes() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    f.push(Instr::Raw(vec![0x90]));
    let a = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 0 });
    f.blocks[0].term = Terminator::Ret(Some(a));
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_exit_terminator() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    f.blocks[0].term = Terminator::Exit;
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_exit_with() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    f.push(Instr::Const { dst: a, imm: 7 });
    f.blocks[0].term = Terminator::ExitWith(a);
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_many_vregs() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let mut vregs = Vec::new();
    for i in 0..16 {
        let v = f.fresh_vreg();
        f.push(Instr::Const { dst: v, imm: i as i64 });
        vregs.push(v);
    }
    f.push(Instr::Bin { dst: vregs[0], op: BinOp::Add, a: vregs[0], b: vregs[1] });
    f.push(Instr::Bin { dst: vregs[0], op: BinOp::Add, a: vregs[0], b: vregs[2] });
    f.push(Instr::Bin { dst: vregs[0], op: BinOp::Mul, a: vregs[0], b: vregs[3] });
    f.blocks[0].term = Terminator::Ret(Some(vregs[0]));
    let mc = run_ir(make_program(f));
    assert!(!mc.code.is_empty());
}

#[test]
fn enc_fconst() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    f.push(Instr::FConst { dst: a, bits: 1.5f64.to_bits() });
    f.blocks[0].term = Terminator::Ret(Some(a));
    let mc = run_ir(make_program(f));
    assert!(contains_any_movabs(&mc.code, 1.5f64.to_bits()));
}

#[test]
fn enc_fbin() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    let b = f.fresh_vreg();
    let r1 = f.fresh_vreg();
    let r2 = f.fresh_vreg();
    let r3 = f.fresh_vreg();
    let r4 = f.fresh_vreg();
    f.push(Instr::FConst { dst: a, bits: 1.5f64.to_bits() });
    f.push(Instr::FConst { dst: b, bits: 2.25f64.to_bits() });
    f.push(Instr::FBin { dst: r1, op: FBinOp::Add, a, b });
    f.push(Instr::FBin { dst: r2, op: FBinOp::Sub, a, b });
    f.push(Instr::FBin { dst: r3, op: FBinOp::Mul, a, b });
    f.push(Instr::FBin { dst: r4, op: FBinOp::Div, a, b });
    f.blocks[0].term = Terminator::Ret(Some(r4));
    let mc = run_ir(make_program(f));
    assert!(has_seq(&mc.code, &[0x66, 0x48, 0x0F, 0x58, 0xC1]));
    assert!(has_seq(&mc.code, &[0x66, 0x48, 0x0F, 0x5C, 0xC1]));
    assert!(has_seq(&mc.code, &[0x66, 0x48, 0x0F, 0x59, 0xC1]));
    assert!(has_seq(&mc.code, &[0x66, 0x48, 0x0F, 0x5E, 0xC1]));
}

#[test]
fn enc_fcmp() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    let b = f.fresh_vreg();
    let r1 = f.fresh_vreg();
    let r2 = f.fresh_vreg();
    let r3 = f.fresh_vreg();
    let r4 = f.fresh_vreg();
    let r5 = f.fresh_vreg();
    let r6 = f.fresh_vreg();
    f.push(Instr::FConst { dst: a, bits: 1.5f64.to_bits() });
    f.push(Instr::FConst { dst: b, bits: 2.25f64.to_bits() });
    f.push(Instr::FCmp { dst: r1, op: FCmpOp::Eq, a, b });
    f.push(Instr::FCmp { dst: r2, op: FCmpOp::Ne, a, b });
    f.push(Instr::FCmp { dst: r3, op: FCmpOp::Lt, a, b });
    f.push(Instr::FCmp { dst: r4, op: FCmpOp::Gt, a, b });
    f.push(Instr::FCmp { dst: r5, op: FCmpOp::Le, a, b });
    f.push(Instr::FCmp { dst: r6, op: FCmpOp::Ge, a, b });
    f.blocks[0].term = Terminator::Ret(Some(r6));
    let mc = run_ir(make_program(f));
    assert!(has_seq(&mc.code, &[0x66, 0x48, 0x0F, 0x2F, 0xC1]));
    assert!(has_seq(&mc.code, &[0x0F, 0x94, 0xC0]));
    assert!(has_seq(&mc.code, &[0x0F, 0x95, 0xC0]));
    assert!(has_seq(&mc.code, &[0x0F, 0x92, 0xC0]));
    assert!(has_seq(&mc.code, &[0x0F, 0x97, 0xC0]));
    assert!(has_seq(&mc.code, &[0x0F, 0x96, 0xC0]));
    assert!(has_seq(&mc.code, &[0x0F, 0x93, 0xC0]));
}

#[test]
fn enc_fneg() {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let a = f.fresh_vreg();
    let b = f.fresh_vreg();
    f.push(Instr::FConst { dst: a, bits: 1.5f64.to_bits() });
    f.push(Instr::FNeg { dst: b, a });
    f.blocks[0].term = Terminator::Ret(Some(b));
    let mc = run_ir(make_program(f));
    assert!(has_seq(
        &mc.code,
        &[0x48, 0xB9, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x48, 0x31, 0xC8]
    ));
}

#[test]
fn enc_build_argv_rex_bytes() {
    // Hand-written machine code lives or dies by REX bits: REX.R
    // extends the ModRM *reg* field, REX.B the *rm*/base field.
    // Two wrong bits once shipped (`4D 8B 08` reads [r8] instead of
    // [rax]; `48 8B 09` reads [rcx] instead of [r9]) and segfaulted
    // every argv program. Pin the correct encodings.
    let mut f = IrFunction::new("@start", ReceiverKind::None, 0);
    let argv = f.fresh_vreg();
    f.push(Instr::BuildArgv { dst: argv });
    f.blocks[0].term = Terminator::Ret(Some(argv));
    let mc = run_ir(make_program(f));
    // mov r9, [rax]: REX.R set (r9 dest), REX.B clear ([rax] base).
    assert!(
        has_seq(&mc.code, &[0x4C, 0x8B, 0x08]),
        "mov r9,[rax] must use REX.WR, not WRB"
    );
    // mov rcx, [r9]: REX.B set (r9 base), REX.R clear (rcx dest).
    assert!(
        has_seq(&mc.code, &[0x49, 0x8B, 0x09]),
        "mov rcx,[r9] must use REX.WB, not W"
    );
    assert!(
        !has_seq(&mc.code, &[0x4D, 0x8B, 0x08]),
        "stale [r8]-reading encoding must be gone"
    );
    // mov [rdi],rcx needs mod=00 (memory); mod=11 (0xCF) would move
    // rcx into rdi, lose the array, and fault on the Length read.
    assert!(
        has_seq(&mc.code, &[0x48, 0x89, 0x0F]),
        "array length store must address [rdi]"
    );
    // REX rule (burned in by live segfaults): REX.R ⟺ ModRM.reg ≥ 8,
    // REX.X ⟺ SIB.index ≥ 8, REX.B ⟺ rm/base ≥ 8. rsi/rdi/rcx (6/7/1)
    // take no extension bits; r8/r9/r11 (8/9/11) take them.
    for good in [
        &[0x48, 0x8D, 0x34, 0xCD][..], // lea rsi,[rcx*8+8], REX.R clear
        &[0x4B, 0x8B, 0x74, 0xD9][..], // mov rsi,[r9+r11*8+8], R clear
        &[0x49, 0x89, 0xF0][..],       // mov r8,rsi, REX.R clear
        &[0x48, 0x8D, 0x72, 0x10][..], // lea rsi,[rdx+16], REX.R clear
        &[0x41, 0x8A, 0x34, 0x08][..], // mov sil,[r8+rcx]: sil, not al (al is rax's low byte and rax holds strobj live; a char load into al rewrites the pointer)
        &[0x48, 0x88, 0xB4, 0x08][..], // mov [rax+rcx+16],sil, REX.X clear
        &[0x4A, 0x89, 0x44, 0xDF][..], // mov [rdi+r11*8+8],rax, REX.B clear
        &[0x4C, 0x3B, 0x1F][..],       // cmp r11,[rdi], REX.B clear, CMP r64,r/m64
    ] {
        assert!(has_seq(&mc.code, good), "missing correct encoding {good:02x?}");
    }
    // Loop-branch opcode: 0x3B compares r11 AGAINST [rdi] (skip when
    // i >= length); 0x39 would compare backwards and skip the loop
    // whenever length > 0 (shipped that way once: empty argv arrays).
    assert!(
        has_seq(&mc.code, &[0x4C, 0x3B, 0x1F]),
        "loop condition must use CMP r64,r/m64"
    );
    assert!(
        !has_seq(&mc.code, &[0x4C, 0x39, 0x1F]),
        "inverted loop condition must be gone"
    );
    for bad in [
        &[0x4C, 0x8D, 0x34, 0xCD][..],
        &[0x4F, 0x8B, 0x74, 0xD9][..],
        &[0x4D, 0x89, 0xF0][..],
        &[0x4C, 0x8D, 0x72, 0x10][..],
        &[0x42, 0x88, 0x04, 0x08][..],
        &[0x4B, 0x89, 0x44, 0xDF][..],
        &[0x4D, 0x39, 0x3B][..],
        &[0x49, 0x89, 0xCF][..],
        &[0x48, 0x8B, 0x09][..],
        &[0x41, 0x8A, 0x04, 0x08][..], // old copy load into al (clobbers strobj low byte)
        &[0x48, 0x88, 0x84, 0x08][..], // old copy store from al
    ] {
        assert!(!has_seq(&mc.code, bad), "stale mis-encoded {bad:02x?} must be gone");
    }
}
