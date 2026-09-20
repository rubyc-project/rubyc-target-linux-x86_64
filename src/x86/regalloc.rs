//! Linear-scan register allocation over callee-saved registers.
//!
//! Only `rbx` and `r12`–`r15` receive allocated values. Caller-saved
//! registers stay reserved for argument passing and emission scratch, so a
//! call can never invalidate an allocated value.
//!
//! Two live ranges conflict when they overlap on at least one instruction
//! position; a range ending exactly where another begins does **not**
//! conflict (`end < next.start` expires).

use rubyc::native::ir::{Instr, IrFunction, Terminator};

/// Callee-saved physical registers available to the allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phys {
    Rbx,
    R12,
    R13,
    R14,
    R15,
}

pub const POOL: [Phys; 5] = [Phys::Rbx, Phys::R12, Phys::R13, Phys::R14, Phys::R15];

impl Phys {
    /// Register identifier for encoding (low 3 bits + REX extension bit).
    pub fn id(self) -> u8 {
        match self {
            Phys::Rbx => 3,
            Phys::R12 => 12,
            Phys::R13 => 13,
            Phys::R14 => 14,
            Phys::R15 => 15,
        }
    }

    pub fn needs_rex(self) -> bool {
        !matches!(self, Phys::Rbx)
    }
}

/// Where a virtual register lives for its whole live range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loc {
    Reg(Phys),
    /// Spill slot k → `[rbp - 8*(k+1)]`.
    Slot(u32),
}

/// vreg → location, plus how many spill slots the frame needs.
#[derive(Debug)]
pub struct Allocation {
    pub loc: Vec<Loc>,
    pub spills: u32,
}

pub(crate) struct Range {
    pub(crate) vreg: u32,
    pub(crate) start: u32,
    pub(crate) end: u32,
}

fn uses_and_defs(instr: &Instr) -> (Option<u32>, Vec<u32>) {
    let (d, mut u): (Option<u32>, Vec<u32>) = match instr {
        Instr::Const { dst, .. } | Instr::FConst { dst, .. } => (Some(dst.0), vec![]),
        Instr::Neg { dst, a } | Instr::FNeg { dst, a } => (Some(dst.0), vec![a.0]),
        Instr::Mov { dst, a } => (Some(dst.0), vec![a.0]),
                Instr::Raw(_) => (None, vec![]),
        Instr::Bin { dst, a, b, .. } => (Some(dst.0), vec![a.0, b.0]),
        Instr::Cmp { dst, a, b, .. } => (Some(dst.0), vec![a.0, b.0]),
        Instr::FBin { dst, a, b, .. } => (Some(dst.0), vec![a.0, b.0]),
        Instr::FCmp { dst, a, b, .. } => (Some(dst.0), vec![a.0, b.0]),
        Instr::CallByName {
            dst,
            recv,
            args,
            ..
        } => {
            let mut u: Vec<u32> = std::iter::once(recv.0)
                .chain(args.iter().map(|v| v.0))
                .collect();
            u.push(dst.0);
            (Some(dst.0), u)
        }
        Instr::CallFn { dst, args, .. }
        | Instr::CallStatic { dst, args, .. }
        | Instr::CallImport { dst, args, .. } => {
            let mut u: Vec<u32> = args.iter().map(|v| v.0).collect();
            u.push(dst.0); // receiver/args are uses; dst counts too
            (Some(dst.0), u)
        }
        // Address import: pure def, no register operands.
        Instr::LoadImport { dst, .. } => (Some(dst.0), vec![dst.0]),
        // The receiver is an operand like any other — omitting it here
        // truncates its live range and lets the allocator hand its home to
        // another value mid-call.
        Instr::CallVirt {
            dst, recv, args, ..
        } => {
            let mut u: Vec<u32> = std::iter::once(recv.0)
                .chain(args.iter().map(|v| v.0))
                .collect();
            u.push(dst.0);
            (Some(dst.0), u)
        }
        Instr::Alloc { dst, .. } | Instr::VTableAddr { dst, .. } | Instr::LoadStaticAddr { dst, .. }
        | Instr::StackAlloc { dst, .. } | Instr::LoadStrAddr { dst, .. } => (Some(dst.0), vec![]),
        // Process argv: defines the array vreg, reads no vregs (the
        // kernel stack comes from the startup slot, not the frame).
        Instr::BuildArgv { dst } => (Some(dst.0), vec![]),
        Instr::AllocDyn { dst, bytes } => (Some(dst.0), vec![bytes.0]),
        Instr::LoadField { dst, base, .. } => (Some(dst.0), vec![base.0]),
        Instr::LoadByte { dst, base, .. } => (Some(dst.0), vec![base.0]),
        Instr::StoreField { base, src, .. } => (None, vec![base.0, src.0]),
        Instr::MemCpy { dst, src, len } => (None, vec![dst.0, src.0, len.0]),
        Instr::MemCmp { dst, a, b, len } => (Some(dst.0), vec![a.0, b.0, len.0]),
        Instr::WriteInt { src } | Instr::EWriteInt { src } => (None, vec![src.0]),
        Instr::WriteString { src } => (None, vec![src.0]),
        Instr::Retain { src } => (None, vec![src.0]),
        Instr::Release { src, .. } => (None, vec![src.0]),
        Instr::Drop { src } => (None, vec![src.0]),
        Instr::HandlerEnter { dst } => (Some(dst.0), vec![]),
        Instr::HandlerPop | Instr::ClearInflight => (None, vec![]),
        Instr::LoadExc { dst } => (Some(dst.0), vec![]),
        Instr::WriteStr { .. } | Instr::EWriteStr { .. } | Instr::Exit | Instr::ScanInt { .. } => {
            (None, vec![])
        }
        Instr::TaskAlloc { dst, .. } => (Some(dst.0), vec![]),
        Instr::Suspend { state } => (None, vec![state.0]),
        Instr::CoroutineResume { task, dst } => (Some(dst.0), vec![task.0]),
    };
    u.sort_unstable();
    u.dedup();
    (d, u)
}

/// Allocate every vreg in the function.
///
/// Liveness is computed by standard backward dataflow over the CFG, then
/// a vreg's interval spans [first touch … last point of liveness]. A vreg
/// live-in to any block has its interval extended through that block's
/// entry position — which, via loop-header live-in sets, keeps values
/// used across back-edges alive for the whole trip around the loop. The
/// old min/max-over-textual-uses scheme missed exactly that case and let
/// a home be reused mid-loop.
pub fn allocate(f: &IrFunction) -> Allocation {
    // ---- global positions: one slot per instruction, one per terminator
    let mut block_entry: Vec<u32> = Vec::with_capacity(f.blocks.len());
    let mut total_slots: u32 = 0;
    for b in &f.blocks {
        block_entry.push(total_slots);
        total_slots += b.instrs.len() as u32 + 1; // +1 = terminator slot
    }

    // ---- per-block gen (upward-exposed uses) and kill (defs)
    let nb = f.blocks.len();
    let mut upward_uses: Vec<std::collections::BTreeSet<u32>> = vec![Default::default(); nb];
    let mut kill: Vec<std::collections::BTreeSet<u32>> = vec![Default::default(); nb];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for (bi, b) in f.blocks.iter().enumerate() {
        for instr in &b.instrs {
            let (d, u) = uses_and_defs(instr);
            for v in u {
                if !kill[bi].contains(&v) {
                    upward_uses[bi].insert(v);
                }
            }
            if let Some(d) = d {
                kill[bi].insert(d);
            }
        }
        if let Terminator::Branch {
            cond,
            if_true,
            if_false,
        } = &b.term
        {
            if let Some(c) = cond
                && !kill[bi].contains(&c.0)
            {
                upward_uses[bi].insert(c.0);
            }
            succs[bi] = vec![*if_true as usize, *if_false as usize];
        } else {
            succs[bi] = vec![]; // Ret / Exit
        }
    }

    // ---- backward live-in fixpoint
    let mut live_in: Vec<std::collections::BTreeSet<u32>> = vec![Default::default(); nb];
    let mut live_out: Vec<std::collections::BTreeSet<u32>> = vec![Default::default(); nb];
    let mut changed = true;
    while changed {
        changed = false;
        for bi in (0..nb).rev() {
            let mut out: std::collections::BTreeSet<u32> = Default::default();
            for s in &succs[bi] {
                out.extend(live_in[*s].iter().copied());
            }
            let mut inn = upward_uses[bi].clone();
            for v in &out {
                if !kill[bi].contains(v) {
                    inn.insert(*v);
                }
            }
            if out != live_out[bi] || inn != live_in[bi] {
                live_out[bi] = out;
                live_in[bi] = inn;
                changed = true;
            }
        }
    }

    // ---- intervals
    let mut ranges: std::collections::BTreeMap<u32, (u32, u32)> = Default::default();
    let touch = |v: u32, pos: u32, m: &mut std::collections::BTreeMap<u32, (u32, u32)>| {
        m.entry(v)
            .and_modify(|r| {
                r.0 = r.0.min(pos);
                r.1 = r.1.max(pos);
            })
            .or_insert((pos, pos));
    };

    let incoming = f.arity + usize::from(f.receiver != rubyc::native::ir::ReceiverKind::None);
    for v in 0..incoming as u32 {
        touch(v, 0, &mut ranges);
    }
    for (bi, b) in f.blocks.iter().enumerate() {
        let entry = block_entry[bi];
        // Anything live into the block stays alive at least to its entry…
        for v in &live_in[bi] {
            touch(*v, entry, &mut ranges);
        }
        // …and anything live-out survives past the whole block.
        for v in &live_out[bi] {
            touch(*v, entry + b.instrs.len() as u32, &mut ranges);
        }
        let mut position = entry;
        for instr in &b.instrs {
            let (d, u) = uses_and_defs(instr);
            for v in u {
                touch(v, position, &mut ranges);
            }
            if let Some(v) = d {
                touch(v, position, &mut ranges);
            }
            position += 1;
        }
        if let Terminator::Branch { cond: Some(c), .. } = &b.term {
            touch(c.0, position, &mut ranges);
        }
    }
    for v in 0..f.vregs {
        ranges.entry(v).or_insert((0, 0));
    }

    let mut ordered: Vec<Range> = ranges
        .into_iter()
        .map(|(vreg, (start, end))| Range { vreg, start, end })
        .collect();
    ordered.sort_by_key(|r| (r.start, r.end, r.vreg));

    let mut loc = vec![Loc::Slot(u32::MAX); f.vregs as usize];
    // Ascending free pool; we always allocate the lowest free register.
    let mut free: Vec<usize> = (0..POOL.len()).collect();
    // (end position, pool index, vreg)
    let mut active: Vec<(u32, usize, u32)> = Vec::new();
    let mut spills = 0u32;

    for iv in ordered {
        // Expire ranges that ended strictly before this one begins.
        let mut i = 0;
        while i < active.len() {
            if active[i].0 < iv.start {
                let (_, pool_idx, _) = active.remove(i);
                free.push(pool_idx);
                free.sort_unstable();
            } else {
                i += 1;
            }
        }

        if let Some(pool_idx) = free.first().copied() {
            free.remove(0);
            loc[iv.vreg as usize] = Loc::Reg(POOL[pool_idx]);
            active.push((iv.end, pool_idx, iv.vreg));
        } else {
            // Spill whichever live range (newcomer included) extends
            // furthest; spilling the longest is linear-scan's classic
            // heuristic and keeps short temps in registers.
            let furthest_active = active.iter().copied().max_by_key(|&(e, _, _)| e);
            match furthest_active {
                Some((end, pool_idx, resident)) if end > iv.end => {
                    loc[iv.vreg as usize] = Loc::Reg(POOL[pool_idx]);
                    loc[resident as usize] = Loc::Slot(spills);
                    spills += 1;
                    if let Some(a) = active.iter_mut().find(|a| a.1 == pool_idx) {
                        a.2 = iv.vreg;
                    }
                }
                _ => {
                    loc[iv.vreg as usize] = Loc::Slot(spills);
                    spills += 1;
                }
            }
        }
    }

    Allocation { loc, spills }
}
