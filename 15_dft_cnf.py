"""
DFT validation of CNF competition (266 Å², Au(111) 6×6×3)

DFT (PBE-D3) single-points on the MLIP structures:
    Au, CNF@Au, THI@Au, TBZ@Au, DQ·2Br@Au  +  gas-phase CNF (and THI, TBZ, DQ from 04 at the same cutoff)

Definitions (water-referenced; computed by aggregate.py):
    E_ads,aq(X)           = E(Au+X) − E(Au) − [E_gas(X) + G_solv(X)]
    equal-area exchange   = E_ads,aq(X) − E_ads,aq(CNF) × A_X / A_CNF          (A: data/results_cnf/cnf_au.csv)
    CNF → Au transfer     = equal-area exchange − ΔE_bind,aq(X)               (ΔE_bind: data/results_cnf/cnf_binding.csv)
G_solv(CNF): GFN2-xTB optimisation in ALPB water minus gas phase (run here); others from fix_summary.csv.

Options: DFT_CNF_K=2,2,1 (k-points instead of Γ-only; ~3–4× slower), DFT_CNF_ECUT (default 45,360).

Input:  data/results_cnf/, data/results_all/, data/results_fix/fix_summary.csv, completed 04 results
Output: results/dft_cnf/ (QE files, *.meta.json, record.json, cnf_dft_log.txt); table in results/tables/
"""
import csv, os
import qe_tools as Q
from ase.io import read

ECUT = Q.env_ecut("DFT_CNF_ECUT", default="45,360")
KPTS = Q.env_kpts("DFT_CNF_K")      # None = Γ-only (gamma trick)
NAMES = ["THI", "TBZ", "DQ"]

ST = Q.Stage("cnf", "dft_cnf", "cnf_dft_log.txt")
log = ST.log
Q.check_programs("pw.x", "mpirun", "xtb")
need = [f"data/results_cnf/best_{m}.traj" for m in ["CNF"] + NAMES] + \
       ["data/results_cnf/CNF_gas.xyz", "data/results_cnf/cnf_au.csv", "data/results_cnf/cnf_binding.csv"]
miss = [f for f in need if not os.path.exists(f)]
if miss: Q.fail(f"missing inputs: {miss}")
table = Q.load_sssp(log)
MACE = Q.load_mace(log)

GS = Q.read_gsolv()
cnf_gas = read("data/results_cnf/CNF_gas.xyz")
w, g = Q.xtb_energy(cnf_gas, True, log), Q.xtb_energy(cnf_gas, False, log)
GS["CNF"] = None if w is None else w - g
ST.constants.update({
    "G_solv_eV": GS,
    "G_solv_source": {"THI/TBZ/DQ": "data/results_fix/fix_summary.csv", "CNF": "GFN2-xTB --opt, ALPB water − gas (this run)"},
    "footprint_A2": {r[0]: float(r[3]) for r in csv.reader(open("data/results_cnf/cnf_au.csv")) if r and r[0] != "molecule"},
    "dE_bind_CNF_eV": {r[0]: float(r[2]) for r in csv.reader(open("data/results_cnf/cnf_binding.csv")) if r and r[0] != "molecule"},
    "kpts": list(KPTS) if KPTS else "gamma", "ecut": list(ECUT)})

SYS = {f"{m}@Au": read(f"data/results_cnf/best_{m}.traj") for m in ["CNF"] + NAMES}
SRC = {f"{m}@Au": Q.source(f"data/results_cnf/best_{m}.traj") for m in ["CNF"] + NAMES}
SYS = {"Au": Q.relax_bare(Q.bare_from(SYS["CNF@Au"]), MACE, log), **SYS}
SRC["Au"] = {"derived_from": "data/results_cnf/best_CNF.traj", "note": "Au atoms only, MLIP-relaxed"}
Q.align_cells(SYS, vacuum=13.0)
gas = Q.gas_box(cnf_gas)

W0, R0 = ECUT
log(f"\n=== DFT single-point (Au(111) 6×6, 266 Å², k {KPTS or 'Γ'}, ecut {W0}/{R0} Ry) ===")
pps, _, _ = Q.pp_info(table, sorted(set(gas.get_chemical_symbols())))
ST.add("CNF_gas", Q.run_pw(ST.dir, "CNF_gas", gas, pps, W0, R0, None, False, log, Q.source("data/results_cnf/CNF_gas.xyz")),
       system="CNF_gas", molecule="CNF", role="gas", mlip_eV=Q.mlip_energy(cnf_gas, MACE, periodic=False))
for key in ["Au", "CNF@Au", "THI@Au", "DQ@Au", "TBZ@Au"]:
    at = SYS[key]
    pps, _, _ = Q.pp_info(table, sorted(set(at.get_chemical_symbols())))
    meta = Q.run_pw(ST.dir, key.replace("@", "_"), at, pps, W0, R0, KPTS, True, log, SRC[key])
    ST.add(key, meta, system=key, molecule=None if key == "Au" else key.split("@")[0],
           role="slab" if key == "Au" else "adsorbed", mlip_eV=Q.mlip_energy(at, MACE), source=SRC[key])
for n, f in [("THI", "THI_gas.xyz"), ("TBZ", "TBZ_gas.xyz"), ("DQ", "DQ2Br_gas.xyz")]:
    ST.add(f"{n}_gas", Q.gas_reference(n, table, ECUT, f"{ST.dir}/gas", log), system=f"{n}_gas", molecule=n,
           role="gas", mlip_eV=Q.mlip_energy(read(f"data/results_all/{f}"), MACE, periodic=False))
ST.finish()
