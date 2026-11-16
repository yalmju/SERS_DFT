"""
DFT convergence test: sensitivity of ΔE(THI←TBZ) to cutoff and k-points (118 Å², 4×4)

  ref   : ecutwfc 60 / ecutrho 480 Ry, k 2×2×1   (reuses 04 results)
  ecut45: ecutwfc 45 / ecutrho 360 Ry, k 2×2×1   (Au, THI@Au, TBZ@Au, THI(g), TBZ(g))
  k3    : ecutwfc 60 / ecutrho 480 Ry, k 3×3×1   (Au, THI@Au, TBZ@Au; gas molecules reuse ref)
  CONV_DQ=1 also includes DQ·2Br

Input:  results/dft_check/ (04), data/results_fix/fix_summary.csv (xTB solvation energies)
Output: results/dft_convergence/convergence.csv, convergence_log.txt
"""
import os, csv
import numpy as np
import qe_tools as Q

W = "results/dft_convergence"; os.makedirs(W, exist_ok=True)
LOG = open(f"{W}/convergence_log.txt", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

Q.check_programs("pw.x", "mpirun")
SSSP, sname = Q.load_sssp()
S = Q.read_structures_from_singlepoint()
MOLS = ["THI", "TBZ"] + (["DQ"] if os.environ.get("CONV_DQ") == "1" else [])
ELEMS = sorted({e for k, a in S.items() for e in a.get_chemical_symbols()})
PPS, W0, R0 = Q.pp_info(SSSP, ELEMS)
log(f"[07] convergence test: {sname}, reference ecut {W0}/{R0} Ry, molecules {MOLS}")

# xTB solvation energies
GS = {}
if os.path.exists("data/results_fix/fix_summary.csv"):
    for r in csv.reader(open("data/results_fix/fix_summary.csv")):
        if r and r[0].startswith("xTB"):
            for kv in r[1:]:
                k, v = kv.split("="); GS[k] = float(v)

SETTINGS = {
    "ref (60/480 Ry, k2)":   dict(ecut=(W0, R0), k=(2, 2, 1), dir="results/dft_check", reuse=True),
    "ecut45 (45/360 Ry, k2)": dict(ecut=(45, 360), k=(2, 2, 1), dir=f"{W}/ecut45", reuse=False),
    "k3 (60/480 Ry, k3)":    dict(ecut=(W0, R0), k=(3, 3, 1), dir=f"{W}/k3", reuse=False),
}
REF_FILES = {"Au": "Au", "THI@Au": "THI_Au", "TBZ@Au": "TBZ_Au", "DQ@Au": "DQ_Au",
             "THI_gas": "THI_gas", "TBZ_gas": "TBZ_gas", "DQ_gas": "DQ_gas"}

E = {}
for sname_, st in SETTINGS.items():
    log(f"\n=== {sname_} ===")
    E[sname_] = {}
    systems = ["Au"] + [f"{m}@Au" for m in MOLS] + [f"{m}_gas" for m in MOLS]
    for key in systems:
        slab = "@" in key or key == "Au"
        if st["reuse"]:
            out = os.path.join("results/dft_check", REF_FILES[key] + ".out")
            if not Q.done(out): log(f"  [warning] missing reference result: {out}"); continue
        elif not slab and sname_.startswith("k3"):
            # gas molecules do not depend on k-points: reuse ref
            out = os.path.join("results/dft_check", REF_FILES[key] + ".out")
        else:
            out = Q.run_pw(st["dir"], REF_FILES[key], S[key], PPS, st["ecut"][0], st["ecut"][1],
                           st["k"] if slab else None, slab, log)
            if out is None: log(f"  [skipped] {key}"); continue
        E[sname_][key] = Q.energy(out)

# ---------------------------------------------------------------- summary
rows = []
log("\n" + "=" * 78)
log(" Convergence test (DFT PBE-D3, Au(111) 4×4, 118 Å²/molecule), eV")
log("=" * 78)
hdr = f" {'setting':<24s}" + "".join(f"{'E_ads ' + m:>12s}" for m in MOLS) + f"{'ΔE THI←TBZ':>13s}{'(water)':>11s}"
log(hdr)
ref = None
for sn, e in E.items():
    if "Au" not in e: continue
    ead = {m: e[f"{m}@Au"] - e["Au"] - e[f"{m}_gas"] for m in MOLS if f"{m}@Au" in e and f"{m}_gas" in e}
    if "THI" not in ead or "TBZ" not in ead: continue
    x = ead["THI"] - ead["TBZ"]
    xa = x + GS.get("TBZ", 0) - GS.get("THI", 0) if GS else None
    if ref is None: ref = (ead, x)
    log(f" {sn:<24s}" + "".join(f"{ead.get(m, np.nan):12.3f}" for m in MOLS) + f"{x:13.3f}"
        + (f"{xa:11.3f}" if xa is not None else ""))
    rows.append([sn] + [f"{ead.get(m, np.nan):.3f}" for m in MOLS] + [f"{x:.3f}", "" if xa is None else f"{xa:.3f}",
                f"{x - ref[1]:+.3f}"])
with open(f"{W}/convergence.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["setting"] + [f"E_ads_{m}_eV" for m in MOLS] + ["dE_THI<-TBZ_gas_eV", "dE_THI<-TBZ_aq_eV", "diff_from_ref_eV"])
    w.writerows(rows)
if len(rows) > 1:
    dev = max(abs(float(r[-1])) for r in rows)
    same = all(float(r[len(MOLS) + 1]) < 0 for r in rows)
    log(f"\n  ΔE THI←TBZ max change {dev:.3f} eV, sign {'negative for all settings → conclusion is insensitive to settings' if same else 'changes → check'}")
LOG.close()
