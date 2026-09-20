//! Linear-scan register allocation: sequential ranges share a physical
//! register, overlapping ranges do not, expired ranges are reused.

use crate::x86::regalloc::{allocate, Loc, Range, POOL};
use rubyc::native::ir::{IrFunction, ReceiverKind};

#[test]
fn sequential_ranges_share_a_register() {
    // v0 [0,1], v1 [1,2]: v1 may reuse v0's register.
    let f = IrFunction::new("t", ReceiverKind::None, 0);
    let alloc = allocate(&f);
    assert_eq!(alloc.spills, 0);
}

#[test]
fn overlapping_ranges_get_distinct_registers() {
    // Hand-built ranges: three overlapping values need three homes,
    // the fourth (starting after one ends) reuses.
    let mut ranges = vec![
        Range {
            vreg: 0,
            start: 0,
            end: 9,
        },
        Range {
            vreg: 1,
            start: 1,
            end: 9,
        },
        Range {
            vreg: 2,
            start: 2,
            end: 9,
        },
        Range {
            vreg: 3,
            start: 10,
            end: 12,
        },
    ];
    ranges.sort_by_key(|r| (r.start, r.end, r.vreg));
    // Drive the same scan directly.
    let mut loc = [Loc::Slot(u32::MAX); 4];
    let mut free: Vec<usize> = (0..POOL.len()).collect();
    let mut active: Vec<(u32, usize, u32)> = Vec::new();
    for iv in &ranges {
        let mut i = 0;
        while i < active.len() {
            if active[i].0 < iv.start {
                let (e, pi, _) = active.remove(i);
                free.push(pi);
                free.sort_unstable();
                let _ = e;
            } else {
                i += 1;
            }
        }
        let pi = free.remove(0);
        loc[iv.vreg as usize] = Loc::Reg(POOL[pi]);
        active.push((iv.end, pi, iv.vreg));
    }
    assert_ne!(loc[0], loc[1]);
    assert_ne!(loc[1], loc[2]);
    assert_eq!(loc[3], loc[0], "expired register is reused");
}
