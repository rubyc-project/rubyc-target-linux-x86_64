use crate::LinuxX86_64Backend;
use rubyc_target::TargetBackend;
use rubyc_target::TargetExtension;

#[test]
fn backend_name_is_linux_x86_64() {
    let b = LinuxX86_64Backend;
    assert_eq!(b.name(), "linux_x86_64");
}

#[test]
fn backend_triple_is_x86_64_linux() {
    let b = LinuxX86_64Backend;
    assert_eq!(TargetExtension::triple(&b), "x86_64-unknown-linux-gnu");
}

#[test]
fn backend_exposes_standard_metadata() {
    let metadata = LinuxX86_64Backend.metadata();
    assert_eq!(metadata.name, "linux_x86_64");
    assert!(!metadata.author.is_empty());
    assert!(!metadata.version.is_empty());
    assert!(!metadata.description.is_empty());
}

#[test]
fn lower_returns_delegation_error() {
    let b = LinuxX86_64Backend;
    let result = b.lower(&[]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "delegated to core pipeline");
}

#[test]
fn write_image_returns_delegation_error() {
    let b = LinuxX86_64Backend;
    let result = b.write_image(&[], false);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "delegated to core pipeline");
}

#[test]
fn write_image_with_entry_preamble_returns_error() {
    let b = LinuxX86_64Backend;
    let result = b.write_image(&[], true);
    assert!(result.is_err());
}

#[test]
fn backend_works_through_dyn_trait() {
    let b: Box<dyn TargetBackend> = Box::new(LinuxX86_64Backend);
    assert_eq!(b.name(), "linux_x86_64");
    assert_eq!(b.triple(), "x86_64-unknown-linux-gnu");
}
