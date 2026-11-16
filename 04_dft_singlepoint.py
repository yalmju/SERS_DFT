"""
DFT single-point validation (118 Å², Au(111) 4×4×3)

Keeps the MLIP (MACE-MP-0+D3) structures fixed and recomputes only the energy with QE PBE+D3(BJ).
  Systems: Au, THI@Au, TBZ@Au, DQ·2Br@Au  +  gas-phase THI, TBZ, DQ·2Br (ion pair, 18 Å box)
  Pose: for each molecule, the 4×4 pose with the lowest MLIP adsorption energy in compete.csv
One SCF per system (no relaxation). The DFT vs MLIP table is written by aggregate.py.

Input:  data/results_compete/ (compete.csv, *_4x4.traj), data/results_all/ (*_gas.xyz),
        data/results_fix/fix_summary.csv (xTB solvation energies, for the water-referenced columns)
Output: results/dft_check/ (QE inputs/outputs, *.meta.json, record.json, dft_log.txt); tables in results/tables/
"""
import csv
import qe_tools as Q
from ase.io import read

ST = Q.Stage("check", "dft_check", "dft_log.txt")
log = ST.log
Q.check_programs("pw.x", "mpirun")
table = Q.load_sssp(log)
MACE = Q.load_mace(log)
ST.constants["G_solv_eV"] = Q.read_gsolv()
ST.constants["G_solv_source"] = "data/results_fix/fix_summary.csv (GFN2-xTB/ALPB water)"

# ---------------------------------------------------------------- structures
best = {}
for r in csv.reader(open("data/results_compete/compete.csv")):
    if len(r) >= 5 and r[0] == "4x4" and r[2] in ("THI", "TBZ", "DQ") and r[4]:
        e = float(r[4])
        if r[2] not in best or e < best[r[2]][0]: best[r[2]] = (e, r[3])
SYS, SRC = {}, {}
for n in ("THI", "TBZ", "DQ"):
    if n not in best: Q.fail(f"no 4x4 entry for {n} in compete.csv")
    f = f"data/results_compete/{n}_{best[n][1]}_4x4.traj"
    SYS[f"{n}@Au"] = read(f); SRC[f"{n}@Au"] = Q.source(f, pose=best[n][1], mlip_Eads_eV=best[n][0])
    log(f"  {n}@Au(111) 4x4: {best[n][1]} pose (MLIP E_ads {best[n][0]:.2f} eV)")
SYS["Au"] = Q.relax_bare(Q.bare_from(SYS["THI@Au"]), MACE, log)
SRC["Au"] = {"derived_from": SRC["THI@Au"]["path"], "note": "Au atoms only, MLIP-relaxed, bottom layer fixed"}
Q.align_cells(SYS, vacuum=13.0)
for n, f in [("THI", "THI_gas.xyz"), ("TBZ", "TBZ_gas.xyz"), ("DQ", "DQ2Br_gas.xyz")]:
    SYS[f"{n}_gas"] = Q.gas_box(read(f"data/results_all/{f}"), L=18.0)
    SRC[f"{n}_gas"] = Q.source(f"data/results_all/{f}")

# ---------------------------------------------------------------- DFT
ELEMS = sorted({s for a in SYS.values() for s in a.get_chemical_symbols()})
_, ECUTW, ECUTR = Q.pp_info(table, ELEMS)   # one cutoff for all systems (largest SSSP recommendation)
log(f"  QE: PBE + D3(BJ, QE built-in), ecutwfc {ECUTW} Ry, ecutrho {ECUTR} Ry, {Q.SSSP['name']}")
for key in ["Au", "THI_gas", "TBZ_gas", "DQ_gas", "THI@Au", "TBZ@Au", "DQ@Au"]:
    at = SYS[key]; slab = "@" in key or key == "Au"
    pps, _, _ = Q.pp_info(table, sorted(set(at.get_chemical_symbols())))
    meta = Q.run_pw(ST.dir, key.replace("@", "_"), at, pps, ECUTW, ECUTR, (2, 2, 1) if slab else None,
                    slab, log, SRC[key])
    mol = None if key == "Au" else key.split("@")[0].split("_")[0]
    ST.add(key, meta, role="slab" if key == "Au" else "adsorbed" if slab else "gas", molecule=mol,
           cell="4x4", mlip_eV=Q.mlip_energy(at, MACE, periodic=slab), source=SRC[key])
ST.finish()
