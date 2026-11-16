"""
DFT single-points at other surface densities (in addition to 118 Å² from 04)

  cell  area/molecule  k-points  systems
  3×2   44 Å²          3×4×1     Au, THI(DMDTC×2)@Au, TBZ@Au
  3×3   66 Å²          3×3×1     Au, THI@Au, TBZ@Au, DQ·2Br@Au
  5×5   184 Å²         2×2×1     Au, THI@Au, TBZ@Au, DQ·2Br@Au

Structures: for each molecule, the lowest-energy MLIP structure from the MLIP coverage runs.
Gas molecules: reuse 04 results (same pseudopotentials, cutoff and box). Water-referenced; DQ as DQ·2Br salt.
Run selected cells only: DFT_COV_CELLS=3x2,3x3

Input:  data/results_compete/, data/results_fix/, data/results_aq/, results/dft_check/
Output: results/dft_coverage/dft_coverage.csv, coverage_log.txt
"""
import os, csv, glob
import numpy as np
import qe_tools as Q

W = "results/dft_coverage"; os.makedirs(W, exist_ok=True)
LOG = open(f"{W}/coverage_log.txt", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

CELLS = {"3x2": ((3, 2), (3, 4, 1)), "3x3": ((3, 3), (3, 3, 1)), "5x5": ((5, 5), (2, 2, 1))}
sel = os.environ.get("DFT_COV_CELLS")
if sel: CELLS = {k: v for k, v in CELLS.items() if k in sel.split(",")}
NAMES = ["THI", "TBZ", "DQ"]
ECUT = Q.ecut_override()          # low memory: DFT_ECUT=45,360 DFT_NK=1 DFT_LOWMEM=1 (for 5×5; see 07 for the cutoff check)

Q.check_programs("pw.x", "mpirun")
SSSP, sname = Q.load_sssp()
S04 = Q.read_structures_from_singlepoint()

# ---------------------------------------------------------------- references: gas molecules (04), xTB solvation energies
E_gas = {}
for n in NAMES:
    out = f"results/dft_check/{n}_gas.out"
    if not Q.done(out): raise SystemExit(f"[abort] {out} not found; run 04_dft_singlepoint.py first")
    E_gas[n] = Q.energy(out)
GS = {}
for r in csv.reader(open("data/results_fix/fix_summary.csv")):
    if r and r[0].startswith("xTB"):
        for kv in r[1:]:
            k, v = kv.split("="); GS[k] = float(v)

# ---------------------------------------------------------------- MLIP (structure selection and comparison)
os.environ.setdefault("OMP_NUM_THREADS", "10")
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from mace.calculators import mace_mp
try: MACE = mace_mp(model="medium", dispersion=True, default_dtype="float64", device="cpu")
except Exception: MACE = mace_mp(model="medium", default_dtype="float64", device="cpu")
def e_ml(a, pbc=True):
    b = a.copy(); b.set_constraint(); b.pbc = pbc; b.calc = MACE; return b.get_potential_energy()
E_gas_ml = {n: e_ml(S04[f"{n}_gas"], pbc=False) for n in NAMES}

def candidates(n, cl):
    fs = glob.glob(f"data/results_compete/{n}_*_{cl}.traj")
    if n == "THI": fs += glob.glob(f"data/results_fix/THI_DMDTCx2_{cl}.traj")
    return fs

rows = []
for cl, (size, kpts) in CELLS.items():
    log(f"\n=== {cl} (k {kpts}) ===")
    # pick the lowest-MLIP-energy structure per molecule
    picks = {}
    for n in NAMES:
        fs = candidates(n, cl)
        if not fs: log(f"  {n}: no structure for this cell (does not fit)"); continue
        picks[n] = min(fs, key=lambda f: e_ml(read(f)))
        log(f"  {n}: {os.path.basename(picks[n])}")
    if not picks: continue
    nAu = size[0] * size[1] * 3
    first = read(next(iter(picks.values())))
    bare = first[[a.index for a in first if a.symbol == "Au"]].copy()
    z = bare.positions[:, 2]
    bare.set_constraint(FixAtoms(indices=[i for i in range(len(bare)) if z[i] < z.min() + 0.5]))
    bare.calc = MACE; BFGS(bare, logfile=None).run(fmax=0.03, steps=200); bare.set_constraint(); bare.calc = None
    SYS = {"Au": bare}
    for n, f in picks.items():
        a = read(f); a.set_constraint(); SYS[f"{n}@Au"] = a
    # common cell height (vacuum ≥ 12 Å), as in 04
    zmin = min(s.positions[[i for i, x in enumerate(s) if x.symbol == "Au"], 2].min() for s in SYS.values())
    ztop = max(s.positions[:, 2].max() for s in SYS.values())
    for s in SYS.values():
        s.positions[:, 2] += 1.0 - zmin
        c = s.cell.copy(); c[2] = [0, 0, (ztop - zmin) + 13.0]; s.set_cell(c); s.pbc = True
    elems = sorted({e for s in SYS.values() for e in s.get_chemical_symbols()})
    PPS, W0, R0 = Q.pp_info(SSSP, elems)
    wd = f"{W}/{cl}"
    Egas = dict(E_gas)
    if ECUT:                                       # changed cutoff: recompute gas molecules at the same cutoff (error cancellation)
        W0, R0 = ECUT; wd = f"{W}/{cl}_ecut{int(ECUT[0])}"
        for n in picks:
            Egas[n] = Q.gas_energy(n, ECUT, f"{W}/gas_ecut{int(ECUT[0])}", PPS, log)
    E = {}
    for key, at in SYS.items():
        out = Q.run_pw(wd, f"{key.replace('@', '_')}_{cl}", at, PPS, W0, R0, kpts, True, log)
        if out: E[key] = Q.energy(out)
    if "Au" not in E: log(f"  [skipped] {cl}: bare surface calculation failed"); continue
    area = np.linalg.norm(np.cross(bare.cell[0], bare.cell[1]))
    Eml_bare = e_ml(SYS["Au"])
    ead, eml = {}, {}
    for n in picks:
        if f"{n}@Au" not in E: continue
        if Egas.get(n) is None: continue
        ead[n] = E[f"{n}@Au"] - E["Au"] - Egas[n] - GS[n]
        eml[n] = e_ml(SYS[f"{n}@Au"]) - Eml_bare - E_gas_ml[n] - GS[n]
    for meth, d in (("DFT", ead), ("MLIP", eml)):
        x = lambda a, b: f"{d[a] - d[b]:.3f}" if a in d and b in d else ""
        rows.append([cl, f"{area:.0f}", meth] + [f"{d[n]:.3f}" if n in d else "" for n in NAMES]
                    + [x("THI", "TBZ"), x("THI", "DQ"), x("TBZ", "DQ"), f"{int(W0)}/{int(R0)}"])
        log(f"  {meth:<4s} E_ads,aq " + "  ".join(f"{n} {d[n]:.2f}" for n in d)
            + (f" | THI←TBZ {d['THI'] - d['TBZ']:+.2f}" if "THI" in d and "TBZ" in d else "")
            + (f" | THI←DQ {d['THI'] - d['DQ']:+.2f}" if "THI" in d and "DQ" in d else ""))

# add 118 Å² (04 results) to the same table
if os.path.exists("data/results_aq/aq_summary.csv"):
    sp = {}
    for r in csv.reader(open("data/results_aq/aq_summary.csv")):
        if r and r[0].startswith("single_point_118A2_"): sp.setdefault(r[0].split("_")[-1], {})[r[1]] = r[2]
    for meth, key in (("DFT", "DFT"), ("MLIP", "MACE")):
        d = sp.get(key, {})
        if d:
            rows.append(["4x4", "118", meth, d.get("E_ads,aq THI", ""), d.get("E_ads,aq TBZ", ""), d.get("E_ads,aq DQ", ""),
                         d.get("ΔE_aq THI←TBZ", ""), d.get("ΔE_aq THI←DQ", ""), d.get("ΔE_aq TBZ←DQ", "")])
# keep existing table rows for cells not computed in this run
if os.path.exists(f"{W}/dft_coverage.csv"):
    now = {r[0] for r in rows}
    for r in csv.reader(open(f"{W}/dft_coverage.csv")):
        if r and r[0] not in ("cell",) and r[0] not in now:
            rows.append(r + ["60/480"] * (10 - len(r)) if len(r) < 10 else r)
rows = [r + [""] * (10 - len(r)) for r in rows]
rows.sort(key=lambda r: (-float(r[1]), r[2]))
with open(f"{W}/dft_coverage.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cell", "area_A2", "method", "Eads_aq_THI", "Eads_aq_TBZ", "Eads_aq_DQ2Br",
                "dE_aq_THI<-TBZ", "dE_aq_THI<-DQ", "dE_aq_TBZ<-DQ", "ecut_Ry"])
    w.writerows(rows)
neg = [float(r[6]) < 0 for r in rows if r[2] == "DFT" and r[6]]
log(f"\n  DFT THI←TBZ negative at {sum(neg)}/{len(neg)} densities"
    + (" → THI wins at every density (DFT)" if neg and all(neg) else ""))
LOG.close()
