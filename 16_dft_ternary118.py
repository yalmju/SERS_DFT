"""
DFT re-ranking of 1:1:1 co-adsorption (118 Å², Au(111) 4×4×3)

MLIP adsorption energies can deviate from DFT (see 04), which may change the ranking of
co-adsorbed states. This recomputes the state energies with DFT (PBE-D3) single-points on the
MLIP co-adsorption structures.

    G(S) = E(Au + S) − E(Au) − Σ_{X∈S} [E_gas(X) + G_solv(X)]      (water-referenced)
Gas molecules: reuse 04 results (same 4×4 settings: 60/480 Ry, k 2×2×1)

Input:  data/results_ternary/, data/results_fix/fix_summary.csv, results/dft_check/
Output: results/dft_ternary118/ternary118_dft.csv, *.in/.out, ternary118_log.txt
"""
import os, csv
import numpy as np
import qe_tools as Q
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import BFGS

W = "results/dft_ternary118"; os.makedirs(W, exist_ok=True)
LOG = open(f"{W}/ternary118_log.txt", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

KPTS = (2, 2, 1)
ECUT = Q.ecut_override()
Q.check_programs("pw.x", "mpirun")
SSSP, _ = Q.load_sssp()

# ---------------------------------------------------------------- pick states from the MLIP co-adsorption results (all with values)
T = [r for r in csv.reader(open("data/results_ternary/ternary.csv")) if len(r) >= 7 and r[0] == "4x4" and r[4]]
STATES = {r[2]: float(r[4]) for r in T if os.path.exists(f"data/results_ternary/4x4_{r[2].replace('+', '_')}.traj")}
if not STATES: raise SystemExit("[abort] data/results_ternary/4x4_*.traj not found")
log("MLIP state energies:", ", ".join(f"{k} {v:.2f}" for k, v in sorted(STATES.items(), key=lambda t: t[1])))

GS = {}
for r in csv.reader(open("data/results_fix/fix_summary.csv")):
    if r and r[0].startswith("xTB"):
        for kv in r[1:]:
            k, v = kv.split("="); GS[k] = float(v)

# ---------------------------------------------------------------- structures + bare Au (as in 12)
os.environ.setdefault("OMP_NUM_THREADS", "10")
from mace.calculators import mace_mp
MACE = mace_mp(model="medium", dispersion=True, default_dtype="float64", device="cpu")   # MACE-MP-0 + D3; no silent fallback
SYS = {k: read(f"data/results_ternary/4x4_{k.replace('+', '_')}.traj") for k in STATES}
first = next(iter(SYS.values()))
bare = first[[a.index for a in first if a.symbol == "Au"]].copy()
z = bare.positions[:, 2]
bare.set_constraint(FixAtoms(indices=[i for i in range(len(bare)) if z[i] < z.min() + 0.5]))
bare.calc = MACE; BFGS(bare, logfile=None).run(fmax=0.03, steps=200); bare.set_constraint(); bare.calc = None
SYS = {"Au": bare, **SYS}
for s in SYS.values(): s.set_constraint()
zmin = min(s.positions[[i for i, x in enumerate(s) if x.symbol == "Au"], 2].min() for s in SYS.values())
ztop = max(s.positions[:, 2].max() for s in SYS.values())
for s in SYS.values():
    s.positions[:, 2] += 1.0 - zmin
    c = s.cell.copy(); c[2] = [0, 0, (ztop - zmin) + 13.0]; s.set_cell(c); s.pbc = True

elems = sorted({e for s in SYS.values() for e in s.get_chemical_symbols()})
PPS, W0, R0 = Q.pp_info(SSSP, elems)
if ECUT: W0, R0 = ECUT
log(f"\n=== DFT single-point, Au(111) 4×4 (118 Å²), k {KPTS}, ecut {W0}/{R0} Ry ===")
Egas = {n: Q.gas_energy(n, ECUT, f"{W}/gas", PPS, log) for n in ("THI", "TBZ", "DQ")}

# most important first: bare Au → THI alone → MLIP best (TBZ+DQ) → rest
order = ["Au", "THI"] + [k for k in sorted(STATES, key=STATES.get) if k != "THI"]
E = {}
for key in order:
    out = Q.run_pw(W, f"S_{key.replace('+', '_')}", SYS[key], PPS, W0, R0, KPTS, True, log)
    if out: E[key] = Q.energy(out)
if "Au" not in E: raise SystemExit("[abort] bare Au calculation failed")

rows = []
for k in STATES:
    if k not in E: continue
    mols = k.split("+")
    if any(Egas[m] is None for m in mols): continue
    g = E[k] - E["Au"] - sum(Egas[m] + GS[m] for m in mols)
    rows.append([k, f"{STATES[k]:.3f}", f"{g:.3f}"])
rows.sort(key=lambda r: float(r[2]))
log("\n" + "=" * 70)
log(" 118 Å² 1:1:1 co-adsorption state energies (water-referenced, eV); lower = more stable")
log("=" * 70)
log(f"  {'composition':<14s} {'MLIP':>10s} {'DFT':>10s}")
for r in rows: log(f"  {r[0]:<14s} {float(r[1]):10.2f} {float(r[2]):10.2f}")
if rows:
    mw = min(rows, key=lambda r: float(r[1]))[0]; dw = rows[0][0]
    log(f"\n  most stable composition: MLIP {mw}  /  DFT {dw}"
        + ("  → same" if mw == dw else "  → different: use the DFT result"))
with open(f"{W}/ternary118_dft.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["adsorbed_set", "G_state_aq_MLIP_eV", "G_state_aq_DFT_eV"]); w.writerows(rows)
LOG.close()
