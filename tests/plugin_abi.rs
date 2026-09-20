//! Tests for the plugin ABI entry points and the full lower/write_image pipeline.

use rubyc::native::ir::{Instr, Program, IrFunction, ReceiverKind, Terminator, VReg};

#[test]
fn run_machine_code_returns_42() {
    let program = minimal_program();
    let ir_bytes = rubyc::native::ir::encode_program(&program);
    let mc = crate::lower_target(&ir_bytes).unwrap();
    let status = crate::run_machine_code(&mc).unwrap();
    assert_eq!(status, 42);
}

/// Build a minimal valid IR program: one function returning 42.
pub fn minimal_program() -> Program {
    let mut f = IrFunction::new("main", ReceiverKind::None, 0);
    let v = f.fresh_vreg();
    f.push(Instr::Const { dst: v, imm: 42 });
    f.blocks[0].term = Terminator::Ret(Some(v));
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

#[test]
fn lower_target_succeeds_on_valid_ir() {
    let program = minimal_program();
    let ir_bytes = rubyc::native::ir::encode_program(&program);
    let mc = crate::lower_target(&ir_bytes).unwrap();
    assert!(!mc.is_empty());
}

#[test]
fn lower_target_fails_on_garbage_ir() {
    let result = crate::lower_target(&[0xFF, 0xFF, 0xFF]);
    assert!(result.is_err());
}

#[test]
fn lower_target_fails_on_empty_ir() {
    let result = crate::lower_target(&[]);
    assert!(result.is_err());
}

#[test]
fn write_image_target_fails_on_garbage_mc() {
    let result = crate::write_image_target(&[0xFF, 0xFF], false);
    assert!(result.is_err());
}

#[test]
fn write_shared_target_succeeds_on_valid_mc() {
    let program = minimal_program();
    let ir_bytes = rubyc::native::ir::encode_program(&program);
    let mc = crate::lower_target(&ir_bytes).unwrap();
    let image = crate::write_shared_target(&mc).unwrap();
    assert!(image.len() > 0x1000);
    assert_eq!(&image[..4], b"\x7fELF");
}

#[test]
fn write_shared_target_fails_on_garbage_mc() {
    let result = crate::write_shared_target(&[0xFF, 0xFF]);
    assert!(result.is_err());
}

#[test]
fn run_machine_code_fails_on_garbage() {
    let result = crate::run_machine_code(&[0xFF, 0xFF]);
    assert!(result.is_err());
}

#[test]
fn plugin_abi_name_and_triple() {
    let name_ptr = crate::rbxt_name();
    let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) }.to_str().unwrap();
    assert_eq!(name, "linux_x86_64");

    let triple_ptr = crate::rbxt_triple();
    let triple = unsafe { std::ffi::CStr::from_ptr(triple_ptr) }.to_str().unwrap();
    assert_eq!(triple, "x86_64-unknown-linux-gnu");

    assert_eq!(crate::rbxt_base_addr(), 0);
}

#[test]
fn full_pipeline_ir_to_shared() {
    let program = minimal_program();
    let ir_bytes = rubyc::native::ir::encode_program(&program);
    let mc = crate::lower_target(&ir_bytes).unwrap();
    let image = crate::write_shared_target(&mc).unwrap();
    assert_eq!(&image[..4], b"\x7fELF");
    assert!(image.len() > 0x1000);
}

#[test]
fn mc_roundtrip_encode_decode() {
    let program = minimal_program();
    let ir_bytes = rubyc::native::ir::encode_program(&program);
    let mc_bytes = crate::lower_target(&ir_bytes).unwrap();
    let decoded = crate::mc_serialize::decode(&mc_bytes).unwrap();
    let re_encoded = crate::mc_serialize::encode(&decoded);
    assert_eq!(re_encoded, mc_bytes);
}
