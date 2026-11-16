"""
DFT single-points at other surface densities (118 Å² is computed by 04)

  cell  area/molecule  k-points  systems
  3×2   44 Å²          3×4×1     Au, THI(DMDTC×2)@Au, TBZ@Au   (DQ does not fit)
  3×3   66 Å²          3×3×1     Au, THI@Au, TBZ@Au, DQ·2Br@Au
  5×5   184 Å²         2×2×1     Au, THI@Au, TBZ@Au, DQ·2Br@Au

Pose selection rule: for each molecule and cell, the candidate structure with the lowest MLIP total
energy (all candidates and their MLIP energies are stored in record.json).
Gas molecules: 04 structures and box, at the same cutoff as the slabs.
Options: DFT_COV_CELLS=3x2,3x3 (subset), DFT_ECUT=45,360 (cutoff for slabs and gas references).

Input:  data/results_compete/, data/results_fix/, completed 04 results
Output: results/dft_coverage/ (QE files, *.meta.json, record.json, coverage_log.txt); table in results/tables/
"""
import glob, os
import numpy as np
import qe_tools as Q
from ase.io import read

CELLS = {"3x2": ((3, 2), (3, 4, 1)), "3x3": ((3, 3), (3, 3, 1)), "5x5": ((5, 5), (2, 2, 1))}
SEL = Q.env_list("DFT_COV_CELLS", CELLS)
ECUT = Q.env_ecut("DFT_ECUT")
NAMES = ["THI", "TBZ", "DQ"]

ST = Q.Stage("coverage", "dft_coverage", "coverage_log.txt")
log = ST.log
Q.check_programs("pw.x", "mpirun")
table = Q.load_sssp(log)
MACE = Q.load_mace(log)
S04 = Q.structures_from_04()
ST.constants["G_solv_eV"] = Q.read_gsolv()
ST.constants["selection_rule"] = "lowest MLIP total energy among candidates of the same molecule and cell"


def candidates(n, cl):
    fs = sorted(glob.glob(f"data/results_compete/{n}_*_{cl}.traj"))
    if n == "THI": fs += sorted(glob.glob(f"data/results_fix/THI_DMDTCx2_{cl}.traj"))
    return fs


for cl in SEL:
    size, kpts = CELLS[cl]
    log(f"\n=== {cl} (k {kpts}) ===")
    picks, cand = {}, {}
    for n in NAMES:
        fs = candidates(n, cl)
        if not fs:
            ST.add(f"{cl}/{n}@Au", None, "not_available", "no candidate structure (molecule does not fit this cell)",
                   cell=cl, system=f"{n}@Au", molecule=n, role="adsorbed"); continue
        cand[n] = {f: Q.mlip_energy(read(f), MACE) for f in fs}
        picks[n] = min(cand[n], key=cand[n].get)
        log(f"  {n}: {os.path.basename(picks[n])} (lowest MLIP energy of {len(fs)} candidates)")
    if not picks: continue
    SYS = {"Au": Q.relax_bare(Q.bare_from(read(next(iter(picks.values())))), MACE, log)}
    for n, f in picks.items(): SYS[f"{n}@Au"] = read(f)
    Q.align_cells(SYS, vacuum=13.0)
    area = float(np.linalg.norm(np.cross(SYS["Au"].cell[0], SYS["Au"].cell[1])))
    _, W0, R0 = Q.pp_info(table, sorted({e for s in SYS.values() for e in s.get_chemical_symbols()}))
    if ECUT: W0, R0 = ECUT
    wd = f"{ST.dir}/{cl}" + (f"_ecut{int(W0)}" if ECUT else "")
    for key, at in SYS.items():
        n = None if key == "Au" else key.split("@")[0]
        src = ({"derived_from": picks[next(iter(picks))], "note": "Au atoms only, MLIP-relaxed"} if key == "Au"
               else Q.source(picks[n], pose=os.path.basename(picks[n]),
                             candidates={os.path.basename(f): e for f, e in cand[n].items()}))
        pps, _, _ = Q.pp_info(table, sorted(set(at.get_chemical_symbols())))
        meta = Q.run_pw(wd, f"{key.replace('@', '_')}_{cl}", at, pps, W0, R0, kpts, True, log, src)
        ST.add(f"{cl}/{key}", meta, cell=cl, area_A2=area, system=key, molecule=n,
               role="slab" if key == "Au" else "adsorbed", mlip_eV=Q.mlip_energy(at, MACE), source=src, ecut=[W0, R0])
    for n in picks:
        g = Q.gas_reference(n, table, (W0, R0), f"{ST.dir}/gas_ecut{int(W0)}", log)
        ST.add(f"{cl}/{n}_gas", g, cell=cl, system=f"{n}_gas", molecule=n, role="gas",
               mlip_eV=Q.mlip_energy(S04[f"{n}_gas"], MACE, periodic=False) if f"{n}_gas" in S04 else None)
ST.finish()
