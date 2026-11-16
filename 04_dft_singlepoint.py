"""
DFT single-point validation (118 Å², Au(111) 4×4×3)

Keeps the MLIP (MACE-MP-0+D3) structures fixed and recomputes only the energy with QE PBE+D3(BJ),
then compares against MLIP.
  Systems: Au, THI@Au, TBZ@Au, DQ·2Br@Au  +  gas-phase THI, TBZ, DQ·2Br (ion pair)
  Compared: E_ads(X), ΔE(THI←TBZ) = E_ads(THI) − E_ads(TBZ), ΔE(THI←DQ·2Br)
One SCF per system (no relaxation). Finished calculations are skipped, so a rerun resumes after interruption.
SSSP pseudopotentials are downloaded into pseudo/ if missing.

Input:  data/results_compete/, data/results_all/
Output: results/dft_check/ (QE inputs/outputs, dft_vs_mace.csv, dft_log.txt)
"""
import os, sys, csv, json, glob, time, shutil, subprocess, tarfile, urllib.request, warnings
import numpy as np
warnings.filterwarnings("ignore")
from ase.io import read, write
from ase.build import fcc111, bulk
from ase.constraints import FixAtoms
from ase.optimize import BFGS

NP = int(os.environ.get("DFT_NP", 10))          # number of MPI processes
DRY = os.environ.get("DFT_DRY_RUN") == "1"      # write inputs only, do not run (for testing)
W = "results/dft_check"; PSEUDO = os.path.abspath("pseudo")
TMP = os.path.expanduser("~/qe_tmp")            # QE scratch directory
os.makedirs(TMP, exist_ok=True)
os.makedirs(W, exist_ok=True); os.makedirs(PSEUDO, exist_ok=True)
LOG = open(f"{W}/dft_log.txt", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ---------------------------------------------------------------- 0. check programs
if not DRY:
    for exe in ("pw.x", "mpirun"):
        if shutil.which(exe) is None:
            sys.exit(f"[abort] {exe} not found. Activate the environment and run\n"
                     f"        conda install -c conda-forge -y qe openmpi")

# ---------------------------------------------------------------- 1. SSSP pseudopotential
URLS = ["https://archive.materialscloud.org/records/1732/files/{f}?download=1",
        "https://archive.materialscloud.org/record/files/{f}?record_id=1732"]
def fetch(fname):
    dst = os.path.join(PSEUDO, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000: return dst
    for u in URLS:
        try:
            log(f"  downloading: {u.format(f=fname)}")
            urllib.request.urlretrieve(u.format(f=fname), dst)
            if os.path.getsize(dst) > 1000: return dst
        except Exception as e:
            log(f"    failed: {e.__class__.__name__}")
    return None

for t in glob.glob(f"{PSEUDO}/*.tar.gz"):              # extract archives first
    with tarfile.open(t) as tar: tar.extractall(PSEUDO)
jf = sorted(glob.glob(f"{PSEUDO}/**/SSSP_*PBE*efficiency*.json", recursive=True))   # accepts 1.0 or 1.3
tf = glob.glob(f"{PSEUDO}/**/*.upf", recursive=True) + glob.glob(f"{PSEUDO}/**/*.UPF", recursive=True)
if not jf or not tf:
    log("[1] preparing SSSP PBE efficiency")
    j = fetch("SSSP_1.3.0_PBE_efficiency.json"); t = fetch("SSSP_1.3.0_PBE_efficiency.tar.gz")
    if not j or not t:
        sys.exit("\n[abort] automatic pseudopotential download failed.\n"
                 "  Open https://doi.org/10.24435/materialscloud:f3-ym, download\n"
                 "  SSSP_*_PBE_efficiency.tar.gz and the matching .json,\n"
                 f"  put both in {PSEUDO} and rerun.")
    with tarfile.open(t) as tar: tar.extractall(PSEUDO)
    jf = [j]
log(f"  pseudopotential set: {os.path.basename(jf[-1])}")
SSSP = json.load(open(jf[-1]))

def pp_info(elements):
    pps, wfc, rho = {}, 0, 0
    for el in elements:
        d = SSSP[el]; fn = d["filename"]
        hits = glob.glob(f"{PSEUDO}/**/{fn}", recursive=True)
        if not hits and not DRY: sys.exit(f"[abort] {fn} not found in pseudo/.")
        pps[el] = os.path.relpath(hits[0], PSEUDO) if hits else fn
        cw = d.get("cutoff_wfc", d.get("cutoff")); cr = d.get("cutoff_rho", cw * d.get("dual", 8))
        wfc, rho = max(wfc, cw), max(rho, cr)
    return pps, wfc, rho

# ---------------------------------------------------------------- 2. load structures (MACE results)
log("\n[2] preparing structures (results_compete 4x4, results_all gas)")
best_kind = {}
for r in csv.reader(open("data/results_compete/compete.csv")):
    if len(r) >= 5 and r[0] == "4x4" and r[2] in ("THI", "TBZ", "DQ") and r[4]:
        e = float(r[4])
        if r[2] not in best_kind or e < best_kind[r[2]][0]: best_kind[r[2]] = (e, r[3])
SYS = {}
for n in ("THI", "TBZ", "DQ"):
    kind = best_kind[n][1]
    a = read(f"data/results_compete/{n}_{kind}_4x4.traj"); a.set_constraint(); SYS[f"{n}@Au"] = a
    log(f"  {n}@Au(111) 4x4 : {kind} pose (MACE E_ads {best_kind[n][0]:.2f} eV)")

# bare 4x4: same lattice, quick MACE relaxation
from mace.calculators import mace_mp
try: MACE = mace_mp(model="medium", dispersion=True, default_dtype="float64", device="cpu")
except Exception: MACE = mace_mp(model="medium", default_dtype="float64", device="cpu")
ref = SYS["THI@Au"]
bare = ref[:48].copy()
bare.set_constraint(FixAtoms(indices=[i for i in range(48) if bare.positions[i, 2] < bare.positions[:, 2].min() + 0.5]))
bare.calc = MACE; BFGS(bare, logfile=None).run(fmax=0.03, steps=200); bare.set_constraint(); bare.calc = None
SYS["Au"] = bare

# common cell height (vacuum ≥ 12 Å)
zmin = min(s.positions[:48, 2].min() for s in SYS.values())
ztop = max(s.positions[:, 2].max() for s in SYS.values())
c_new = (ztop - zmin) + 12.0
for k, s in SYS.items():
    s.positions[:, 2] += 1.0 - zmin
    cell = s.cell.copy(); cell[2] = [0, 0, c_new + 1.0]; s.set_cell(cell); s.pbc = True
log(f"  common cell: {np.linalg.norm(ref.cell[0]):.2f} x {np.linalg.norm(ref.cell[1]):.2f} x {c_new+1:.1f} Å")

for n, f in [("THI", "THI_gas.xyz"), ("TBZ", "TBZ_gas.xyz"), ("DQ", "DQ2Br_gas.xyz")]:
    g = read(f"data/results_all/{f}"); g.set_cell([18.0, 18.0, 18.0]); g.center(); g.pbc = True
    SYS[f"{n}_gas"] = g

# ---------------------------------------------------------------- 3. QE input & run
ELEMS = sorted({s for a in SYS.values() for s in a.get_chemical_symbols()})
PPS, ECUTW, ECUTR = pp_info(ELEMS)
log(f"\n[3] QE settings: PBE + D3(BJ), ecutwfc {ECUTW} Ry, ecutrho {ECUTR} Ry, elements {ELEMS}")

ORDER = ["Au", "THI_gas", "TBZ_gas", "DQ_gas", "THI@Au", "TBZ@Au", "DQ@Au"]
# run mode: (diagonalization, use QE built-in D3). The first mode that works is used for all calculations
MODES = [("david", True), ("cg", True), ("david", False)]
MODE_FILE = f"{W}/mode.json"
MODE = tuple(json.load(open(MODE_FILE))) if os.path.exists(MODE_FILE) else None

def qe_input(name, atoms, mode, gamma=True):
    diag, qe_d3 = mode
    slab = "@" in name or name == "Au"
    data = {
        "control": {"calculation": "scf", "prefix": name.replace("@", "_"), "pseudo_dir": PSEUDO,
                    "outdir": os.path.join(TMP, name.replace('@', '_')), "disk_io": "low", "tprnfor": False},
        "system": {"ecutwfc": ECUTW, "ecutrho": ECUTR, "occupations": "smearing", "smearing": "mv",
                   "degauss": 0.01},
        "electrons": {"conv_thr": 1e-6, "diagonalization": diag, "mixing_beta": 0.2 if slab else 0.4,
                      "mixing_mode": "local-TF" if slab else "plain", "electron_maxstep": 300},
    }
    if qe_d3:
        data["system"].update({"vdw_corr": "dft-d3", "dftd3_version": 4})
    fn = f"{W}/{name.replace('@', '_')}.in"
    write(fn, atoms, format="espresso-in", input_data=data, pseudopotentials=PPS,
          kpts=(2, 2, 1) if slab else (None if gamma else (1, 1, 1)))
    txt = open(fn).read()
    for nl in ("&FCP\n/\n", "&RISM\n/\n", "&CELL\n/\n", "&IONS\n/\n"):   # version compatibility: drop empty namelists
        txt = txt.replace(nl, "")
    open(fn, "w").write(txt)
    return fn, slab

def done(out):
    return os.path.exists(out) and "JOB DONE" in open(out, errors="ignore").read()

E_DFT, TIMES = {}, {}
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
           OMPI_MCA_rmaps_base_oversubscribe="1")

def run(name, mode, gamma=True):
    fin, slab = qe_input(name, SYS[name], mode, gamma); fout = fin.replace(".in", ".out")
    tag = name.replace("@", "_")
    shutil.rmtree(os.path.join(TMP, tag), ignore_errors=True); os.makedirs(os.path.join(TMP, tag), exist_ok=True)
    nk = 2 if slab else 1
    cmd = (f"ulimit -s hard 2>/dev/null; mpirun -np {NP} pw.x -nk {nk} -nd 1 "
           f"-in {os.path.basename(fin)} > {os.path.basename(fout)} 2> {tag}.err")
    log(f"  ▶ {name:<8s} start {time.strftime('%H:%M')}  (diag {mode[0]}, D3 {'QE' if mode[1] else 'post-hoc'}"
        f"{'' if gamma else ', k=1x1x1'})")
    t = time.time()
    subprocess.run(cmd, shell=True, cwd=W, env=ENV, executable="/bin/zsh" if os.path.exists("/bin/zsh") else None)
    shutil.rmtree(os.path.join(TMP, tag), ignore_errors=True)
    return done(fout), time.time() - t, fout

for name in ORDER:
    fin, slab = qe_input(name, SYS[name], MODE or MODES[0]); fout = fin.replace(".in", ".out")
    if DRY:
        log(f"  [dry-run] wrote {fin}"); continue
    if not done(fout):
        if MODE:
            # keep the D3 setting, retry with another SCF solver only (same energy)
            d3 = MODE[1]
            modes = [(MODE, True), (("cg", d3), True), ((MODE[0], d3), False), (("cg", d3), False)]
        else:
            modes = [(m, True) for m in MODES]
        for k, (m, gam) in enumerate(modes, 1):
            ok, dt, fout = run(name, m, gam)
            if ok:
                if MODE is None:
                    MODE = m; json.dump(list(m), open(MODE_FILE, "w"))
                    log(f"    → working setup fixed: diag {m[0]}, D3 {'QE built-in' if m[1] else 'torch-dftd post-hoc'}")
                break
            err = open(f"{W}/{name.replace('@','_')}.err", errors="ignore").read()
            outt = open(fout, errors="ignore").read() if os.path.exists(fout) else ""
            why = (" (Segmentation fault)" if "SIGSEGV" in err or "signal 11" in err else
                   " (S matrix not positive definite)" if "not positive definite" in outt else "")
            log(f"    failed{why}"
                f"{' → retrying with next setup' if k < len(modes) else ''}")
        if not done(fout):
            tail = open(fout, errors="ignore").read().splitlines()[-25:] if os.path.exists(fout) else []
            if name == "Au":
                log("\n[abort] bare Au calculation failed. End of output:\n" + "\n".join(tail))
                sys.exit(f"Check {W}/Au.out and {W}/Au.err.")
            log(f"    [skipped] {name}: all setups failed; continuing with the rest")
            continue
        TIMES[name] = dt
        log(f"    done {time.strftime('%H:%M')}  ({dt/60:.1f} min)")
        if name == "Au":
            log(f"    → estimated remaining time ~{dt * (3 * 1.15 + 3 * 0.25) / 3600:.1f} h")
    try:
        E_DFT[name] = read(fout, format="espresso-out").get_potential_energy()
        if MODE is not None and not MODE[1]:      # without QE D3: add the same D3(BJ) post hoc
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
            a = SYS[name].copy()
            if "_gas" in name: a.pbc = False
            a.calc = TorchDFTD3Calculator(device="cpu", damping="bj", xc="pbe")
            E_DFT[name] += a.get_potential_energy()
        log(f"    E_DFT({name}) = {E_DFT[name]:.4f} eV")
    except Exception as e:
        log(f"    [warning] could not read {name} result ({e.__class__.__name__}); check {fout}")
if DRY: sys.exit("dry-run finished")

# ---------------------------------------------------------------- 4. MACE single-points on the same structures
E_ML = {}
for name in ORDER:
    a = SYS[name].copy(); a.calc = MACE
    if "_gas" in name: a.pbc = False
    E_ML[name] = a.get_potential_energy()

def table(E):
    if not all(k in E for k in ORDER): return None
    r = {"E_ads THI": E["THI@Au"] - E["Au"] - E["THI_gas"],
         "E_ads TBZ": E["TBZ@Au"] - E["Au"] - E["TBZ_gas"],
         "E_ads DQ·2Br (salt)": E["DQ@Au"] - E["Au"] - E["DQ_gas"]}
    r["ΔE THI←TBZ"] = r["E_ads THI"] - r["E_ads TBZ"]
    r["ΔE THI←DQ·2Br"] = r["E_ads THI"] - r["E_ads DQ·2Br (salt)"]
    return r

D, M = table(E_DFT), table(E_ML)
log("\n" + "=" * 64)
log(" DFT single-point vs MACE (same structures, Au(111) 4x4, 118 Å²/molecule)")
log("=" * 64)
if D is None:
    log(" Some DFT calculations are unfinished. Rerun to resume.")
else:
    with open(f"{W}/dft_vs_mace.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "DFT_eV", "MACE_eV", "same_sign"])
        for k in D:
            same = np.sign(D[k]) == np.sign(M[k])
            log(f"  {k:<22s} DFT {D[k]:7.2f}   MACE {M[k]:7.2f}   {'same sign' if same else 'SIGN DIFFERS'}")
            w.writerow([k, f"{D[k]:.3f}", f"{M[k]:.3f}", same])
    rank = lambda R: " > ".join(sorted(["THI", "TBZ", "DQ·2Br"], key=lambda n: R[{"THI": "E_ads THI",
                                 "TBZ": "E_ads TBZ", "DQ·2Br": "E_ads DQ·2Br (salt)"}[n]]))
    log(f"\n  adsorption strength  DFT : {rank(D)}")
    log(f"                       MACE: {rank(M)}")
    ok = np.sign(D["ΔE THI←TBZ"]) == np.sign(M["ΔE THI←TBZ"])
    log(f"\n  THI vs TBZ exchange energy sign: {'same → DFT supports the MACE trend' if ok else 'different → revisit MACE results'}")
    log("  (DQ is defined as the DQ·2Br salt, so Br–Au bonding is included; for reference)")
LOG.close()
