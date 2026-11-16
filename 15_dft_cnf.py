"""
DFT validation of CNF competition (266 Å², Au(111) 6×6×3)

DFT (PBE-D3) single-points on the MLIP structures:
    Au, CNF@Au, THI@Au, TBZ@Au, DQ·2Br@Au  +  gas-phase CNF
Gas-phase THI, TBZ and DQ are recomputed at the same cutoff or reused if available.

Definitions (water-referenced):
    E_ads,aq(X)           = E(Au+X) − E(Au) − [E_gas(X) + G_solv(X)]
    equal-area exchange   = E_ads,aq(X) − E_ads,aq(CNF) × A_X / A_CNF
    CNF → Au transfer     = equal-area exchange − ΔE_bind,aq(X)   (ΔE_bind: xTB, data/results_cnf/cnf_binding.csv)

k-points: Γ-only (gamma trick) by default. Only energy differences within the same cell are used,
          so most of the error cancels. Set DFT_CNF_K=2,2,1 for k-points (about 3–4× slower, more memory).
Cutoff: 45/360 Ry by default (checked against 60/480 Ry in the 07 convergence test).

Input:  data/results_cnf/, data/results_all/, data/results_fix/fix_summary.csv, results/dft_check/
Output: results/dft_cnf/dft_cnf.csv, *.in/.out, cnf_dft_log.txt
"""
import os, csv, shutil, subprocess, tempfile
import numpy as np
import qe_tools as Q
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import BFGS

W = "results/dft_cnf"; os.makedirs(W, exist_ok=True)
LOG = open(f"{W}/cnf_dft_log.txt", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

os.environ.setdefault("DFT_ECUT", "45,360")       # large 6×6 cell: save memory (cutoff checked in 07)
# (Γ-only needs ~14 GB even for 6×6, so the Davidson workspace stays at default; no LOWMEM)
ECUT = Q.ecut_override()
_k = os.environ.get("DFT_CNF_K")
KPTS = tuple(int(x) for x in _k.split(",")) if _k else None   # None = Γ-only (gamma trick): half the memory and time
NAMES = ["THI", "TBZ", "DQ"]
HA2EV = 27.211386

Q.check_programs("pw.x", "mpirun")
SSSP, _ = Q.load_sssp()

# ---------------------------------------------------------------- check inputs
need = [f"data/results_cnf/best_{m}.traj" for m in ["CNF"] + NAMES] + ["data/results_cnf/CNF_gas.xyz",
        "data/results_cnf/cnf_au.csv", "data/results_cnf/cnf_binding.csv"] + [f"results/dft_check/{n}_gas.out" for n in NAMES]
miss = [f for f in need if not os.path.exists(f)]
if miss: raise SystemExit(f"[abort] missing inputs: {miss}  (run 04_dft_singlepoint.py first)")

FP = {r[0]: float(r[3]) for r in csv.reader(open("data/results_cnf/cnf_au.csv")) if r and r[0] != "molecule"}   # footprint area
CB = {r[0]: float(r[2]) for r in csv.reader(open("data/results_cnf/cnf_binding.csv")) if r and r[0] != "molecule"}
GS = {}
for r in csv.reader(open("data/results_fix/fix_summary.csv")):
    if r and r[0].startswith("xTB"):
        for kv in r[1:]:
            k, v = kv.split("="); GS[k] = float(v)

# CNF (neutral cellobiose) solvation energy: xTB optimization, water − gas
def xtb(atoms, solvent):
    with tempfile.TemporaryDirectory() as d:
        write(f"{d}/m.xyz", atoms)
        cmd = [shutil.which("xtb"), "m.xyz", "--gfn", "2", "--chrg", "0", "--opt"] + (["--alpb", "water"] if solvent else [])
        out = subprocess.run(cmd, cwd=d, capture_output=True, text=True,
                             env=dict(os.environ, OMP_NUM_THREADS="4", OMP_STACKSIZE="1G")).stdout
        return float([l for l in out.splitlines() if "TOTAL ENERGY" in l][-1].split()[3]) * HA2EV
cnf_gas = read("data/results_cnf/CNF_gas.xyz")
GS["CNF"] = xtb(cnf_gas, True) - xtb(cnf_gas, False)
log(f"G_solv(xTB): THI {GS['THI']:.2f}  TBZ {GS['TBZ']:.2f}  DQ·2Br {GS['DQ']:.2f}  CNF {GS['CNF']:.2f} eV")

# ---------------------------------------------------------------- structures (MLIP) + bare Au
os.environ.setdefault("OMP_NUM_THREADS", "10")
from mace.calculators import mace_mp
MACE = mace_mp(model="medium", dispersion=True, default_dtype="float64", device="cpu")   # MACE-MP-0 + D3; no silent fallback
def e_ml(a, pbc=True):
    b = a.copy(); b.set_constraint(); b.pbc = pbc; b.calc = MACE; return b.get_potential_energy()

SYS = {f"{m}@Au": read(f"data/results_cnf/best_{m}.traj") for m in ["CNF"] + NAMES}
first = SYS["CNF@Au"]
bare = first[[a.index for a in first if a.symbol == "Au"]].copy()
z = bare.positions[:, 2]
bare.set_constraint(FixAtoms(indices=[i for i in range(len(bare)) if z[i] < z.min() + 0.5]))
bare.calc = MACE; BFGS(bare, logfile=None).run(fmax=0.03, steps=200); bare.set_constraint(); bare.calc = None
SYS = {"Au": bare, **SYS}
for s in SYS.values(): s.set_constraint()
zmin = min(s.positions[[i for i, x in enumerate(s) if x.symbol == "Au"], 2].min() for s in SYS.values())
ztop = max(s.positions[:, 2].max() for s in SYS.values())
for s in SYS.values():                        # common cell height (vacuum ≥ 13 Å), as in 12
    s.positions[:, 2] += 1.0 - zmin
    c = s.cell.copy(); c[2] = [0, 0, (ztop - zmin) + 13.0]; s.set_cell(c); s.pbc = True
g = cnf_gas.copy(); L = max(18.0, np.ptp(g.positions, axis=0).max() + 10.0)
g.set_cell([L, L, L]); g.center(); g.pbc = True

# ---------------------------------------------------------------- DFT
elems = sorted({e for s in list(SYS.values()) + [g] for e in s.get_chemical_symbols()})
PPS, W0, R0 = Q.pp_info(SSSP, elems)
if ECUT: W0, R0 = ECUT
log(f"\n=== DFT single-point (Au(111) 6×6, 266 Å², k {KPTS or 'Γ'}, ecut {W0}/{R0} Ry) ===")
E = {}
out = Q.run_pw(W, "CNF_gas", g, PPS, W0, R0, None, False, log)
if out: E["CNF_gas"] = Q.energy(out)
for key in ["Au", "CNF@Au", "THI@Au", "DQ@Au", "TBZ@Au"]:
    out = Q.run_pw(W, key.replace("@", "_"), SYS[key], PPS, W0, R0, KPTS, True, log)
    if out: E[key] = Q.energy(out)
for n in NAMES: E[f"{n}_gas"] = Q.gas_energy(n, ECUT, f"{W}/gas", PPS, log)   # at the same cutoff
if "Au" not in E or "CNF@Au" not in E or "CNF_gas" not in E:
    raise SystemExit("[abort] bare Au / CNF calculation failed; check the log")

# ---------------------------------------------------------------- summary (DFT vs MLIP, same structures)
gas_ml = {"CNF": e_ml(cnf_gas, pbc=False)}
for n, f in [("THI", "THI_gas.xyz"), ("TBZ", "TBZ_gas.xyz"), ("DQ", "DQ2Br_gas.xyz")]:
    gas_ml[n] = e_ml(read(f"data/results_all/{f}"), pbc=False)
Eml_bare = e_ml(SYS["Au"])
ead = {"DFT": {}, "MLIP": {}}
for m in ["CNF"] + NAMES:
    if f"{m}@Au" not in E: continue
    ead["DFT"][m] = E[f"{m}@Au"] - E["Au"] - E[f"{m}_gas"] - GS[m]
    ead["MLIP"][m] = e_ml(SYS[f"{m}@Au"]) - Eml_bare - gas_ml[m] - GS[m]

rows = []
log("\n" + "=" * 92)
log(" CNF competition: DFT (PBE-D3) vs MLIP, same structures (water-referenced, eV; negative = analyte favoured)")
log("=" * 92)
for meth in ("DFT", "MLIP"):
    d = ead[meth]; gC = d["CNF"]
    log(f"  [{meth}] CNF E_ads,aq {gC:6.2f}")
    for n in NAMES:
        if n not in d: continue
        ex = d[n] - gC * FP[n] / FP["CNF"]
        tr = ex - CB[n]
        rows.append([meth, n, f"{d[n]:.3f}", f"{gC:.3f}", f"{ex:.3f}", f"{CB[n]:.3f}", f"{tr:.3f}"])
        log(f"  [{meth}] {n:<4s} E_ads,aq {d[n]:6.2f} | equal-area exchange {ex:+6.2f} | CNF→Au transfer {tr:+6.2f}")
    order = sorted([r for r in rows if r[0] == meth], key=lambda r: float(r[6]))
    log(f"  [{meth}] transfer order (easiest first): " + " > ".join(r[1] for r in order) + "\n")
with open(f"{W}/dft_cnf.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "molecule", "Eads_aq_X", "Eads_aq_CNF", "dE_replace_CNF_same_area",
                "dE_bind_CNF_xTB", "dE_move_CNF_to_CNFcovered_Au"])
    w.writerows(rows)
    w.writerow([]); w.writerow(["kpts", "x".join(map(str, KPTS)) if KPTS else "gamma", "ecut", f"{W0}/{R0}",
                                "GS_CNF_xTB", f"{GS['CNF']:.3f}"])
same = [r[1] for r in sorted([r for r in rows if r[0] == "DFT"], key=lambda r: float(r[6]))] == \
       [r[1] for r in sorted([r for r in rows if r[0] == "MLIP"], key=lambda r: float(r[6]))]
log("  Verdict: DFT and MLIP transfer order " + ("agree → CNF conclusion does not depend on the method" if same else "differ → use the DFT order"))
LOG.close()
