"""
Shared helpers for all DFT stages (04, 07, 12, 15, 16) and for aggregate.py.

Calculation settings (identical for every stage):
  PBE + D3(BJ) computed by QE itself (vdw_corr='dft-d3', dftd3_version=4)
  SSSP 1.0 PBE efficiency (pinned, MD5 checksums verified), Marzari–Vanderbilt 0.01 Ry, conv_thr 1e-6 Ry
  Slabs: Davidson + local-TF mixing; gas molecules: CG first (Γ-point Davidson can fail numerically)
  mpirun -np $DFT_NP pw.x -nd 1  (parallel diagonalization off: avoids segfaults in the macOS conda build)

Every pw.x calculation is described by a fingerprint (elements, positions, cell, periodicity, physical
QE settings, k-points, pseudopotential checksums). A finished calculation is reused only when its
<name>.meta.json records status "completed" with the same fingerprint hash and the output passes
check_output(). Anything else is moved to <workdir>/_archive/ and recomputed; nothing is deleted.
"""
import os, sys, re, json, glob, time, shutil, hashlib, platform, subprocess, tarfile, tempfile, urllib.request
import numpy as np
from ase.io import read, write
from ase.units import Ry, Hartree

SCHEMA_VERSION = 1
PSEUDO = os.path.abspath("pseudo")
SCRATCH = os.path.expanduser("~/qe_tmp")

# ---------------------------------------------------------------- pinned external inputs
SSSP = {
    "name": "SSSP_1.0_PBE_efficiency",
    "doi": "10.24435/materialscloud:2018.0001/v4",
    "url": "https://archive.materialscloud.org/records/x7v00-pnh34/files/{f}?download=1",
    "md5": {"SSSP_1.0_PBE_efficiency.json": "685c4e04def7259eaabb385e94fa8d6b",
            "SSSP_1.0_PBE_efficiency.tar.gz": "0fc1ac970ab91c2b58aedccd3adb9699"},
}
MACE_MODEL = {
    "name": "MACE-MP-0 medium (2023-12-03-mace-128-L1_epoch-199)",
    "url": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "sha256": "01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845",
}
QE_PARAMS = {  # physical settings that define the energy; solver settings are recorded per attempt
    "calculation": "scf", "occupations": "smearing", "smearing": "mv", "degauss": 0.01,
    "conv_thr": 1e-6, "vdw_corr": "dft-d3", "dftd3_version": 4,
}

# ---------------------------------------------------------------- options
KNOWN_OPTIONS = {
    "DFT_NP": "number of MPI processes (default 10)",
    "DFT_NK": "k-point pools (default: 2 for slabs with more than one k-point, else 1)",
    "DFT_ECUT": "12/16: cutoff override 'ecutwfc,ecutrho' in Ry",
    "DFT_LOWMEM": "1 = minimal Davidson workspace",
    "DFT_DRY_RUN": "1 = write inputs to results_dryrun/ only; pw.x and xtb are not run",
    "DFT_ADOPT": "1 = adopt verified outputs that have no metadata yet; pw.x is not run",
    "DFT_RECOMPUTE": "1 = archive matching results and recompute",
    "DFT_COV_CELLS": "12: comma-separated cells (3x2, 3x3, 5x5)",
    "DFT_CNF_K": "15: k-points 'n1,n2,n3' instead of Γ-only",
    "DFT_CNF_ECUT": "15: cutoff 'ecutwfc,ecutrho' (default 45,360)",
    "CONV_DQ": "07: 1 = also test DQ·2Br",
}


def fail(msg):
    raise SystemExit(f"[abort] {msg}")


def _env_int(name, default, lo=1):
    v = os.environ.get(name)
    if v in (None, ""): return default
    if not re.fullmatch(r"\d+", v) or int(v) < lo: fail(f"{name}={v!r}: expected an integer ≥ {lo}")
    return int(v)


def env_flag(name):
    v = os.environ.get(name, "")
    if v not in ("", "0", "1"): fail(f"{name}={v!r}: expected 0 or 1")
    return v == "1"


def env_ecut(name="DFT_ECUT", default=None):
    v = os.environ.get(name) or default
    if not v: return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*", v)
    if not m: fail(f"{name}={v!r}: expected 'ecutwfc,ecutrho', e.g. 45,360")
    w, r = float(m.group(1)), float(m.group(2))
    if w <= 0 or r < 4 * w: fail(f"{name}={v!r}: need ecutwfc > 0 and ecutrho ≥ 4·ecutwfc")
    return (w, r)


def env_kpts(name):
    v = os.environ.get(name)
    if not v: return None
    if not re.fullmatch(r"\d+,\d+,\d+", v) or 0 in map(int, v.split(",")):
        fail(f"{name}={v!r}: expected three positive integers, e.g. 2,2,1")
    return tuple(int(x) for x in v.split(","))


def env_list(name, allowed):
    v = os.environ.get(name)
    if not v: return list(allowed)
    items = [x.strip() for x in v.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad or not items: fail(f"{name}={v!r}: unsupported {bad or 'empty list'}; allowed: {', '.join(allowed)}")
    return items


for _k in os.environ:
    if (_k.startswith("DFT_") or _k == "CONV_DQ") and _k not in KNOWN_OPTIONS:
        fail(f"unknown option {_k}; known options: {', '.join(KNOWN_OPTIONS)}")
NP = _env_int("DFT_NP", 10)
NK = _env_int("DFT_NK", None)
LOWMEM = env_flag("DFT_LOWMEM")
DRY = env_flag("DFT_DRY_RUN")
ADOPT = env_flag("DFT_ADOPT")
RECOMPUTE = env_flag("DFT_RECOMPUTE")
if sum((DRY, ADOPT, RECOMPUTE)) > 1: fail("DFT_DRY_RUN, DFT_ADOPT and DFT_RECOMPUTE are mutually exclusive")
RESULTS = "results_dryrun" if DRY else "results"   # dry-run never writes into results/
MODE = "dry-run" if DRY else "adopt" if ADOPT else "recompute" if RECOMPUTE else "run"


def options():
    return {k: os.environ[k] for k in KNOWN_OPTIONS if os.environ.get(k)}


# ---------------------------------------------------------------- stage log and record
class Stage:
    """Output folder, log and record of one stage. The record lists every planned item (also the ones
    that could not be computed), so aggregate.py can rebuild all tables from stored outputs."""

    def __init__(self, stage, folder, logname):
        self.stage, self.dir = stage, os.path.join(RESULTS, folder)
        os.makedirs(self.dir, exist_ok=True)
        self._log = open(os.path.join(self.dir, logname), "a")
        self.items, self.constants = [], {}
        self.log(f"\n[{stage}] {time.strftime('%Y-%m-%d %H:%M:%S')}  mode {MODE}  "
                 f"options {options() or 'defaults'}  (NP={NP}, NK={NK or 'auto'}, LOWMEM={int(LOWMEM)})")

    def log(self, *a):
        s = " ".join(str(x) for x in a); print(s, flush=True); self._log.write(s + "\n"); self._log.flush()

    def add(self, key, calc=None, status=None, reason=None, **info):
        """One planned calculation. calc: metadata from run_pw (or None with status/reason)."""
        it = {"key": key, **info}
        if calc is not None:
            it.update(calc_dir=calc.get("workdir"), calc_name=calc.get("name"), hash=calc.get("hash"),
                      run_status=calc.get("status"), run_reason=calc.get("reason"))
        else:
            it.update(run_status=status or "not_available", run_reason=reason)
        self.items.append(it)
        return calc

    def finish(self):
        """Write record.json, rebuild tables from stored outputs, exit 1 if anything failed or is incomplete."""
        rec = {"schema_version": SCHEMA_VERSION, "stage": self.stage, "mode": MODE,
               "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "options": options(),
               "constants": self.constants, "items": self.items}
        if DRY:
            json.dump(rec, open(os.path.join(self.dir, "record.json"), "w"), indent=1)
            self.log(f"[dry-run] inputs written under {self.dir}; results/ untouched"); self._log.close(); return
        path = os.path.join(self.dir, "record.json")
        if os.path.exists(path):   # partial runs (e.g. DFT_COV_CELLS) keep items of the previous record
            old = json.load(open(path)); new = {it["key"] for it in self.items}
            kept = [it for it in old.get("items", []) if it["key"] not in new]
            rec["items"] = kept + self.items
            rec["constants"] = {**old.get("constants", {}), **self.constants}
            if kept: self.log(f"  record: kept {len(kept)} item(s) from the previous run")
        json.dump(rec, open(path, "w"), indent=1)
        import aggregate
        status = aggregate.build(RESULTS, log=self.log).get(self.stage, {})
        self._log.close()
        bad = {k: v for k, v in status.items() if k in ("failed", "incomplete") and v}
        if bad:
            print(f"[{self.stage}] not all calculations completed: {bad}", file=sys.stderr); sys.exit(1)


# ---------------------------------------------------------------- checksums, versions, host
def _hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def md5(path): return _hash(path, "md5")
def sha256(path): return _hash(path, "sha256")


def source(path, **kw):
    """Provenance of an input structure file."""
    return {"path": path, "sha256": sha256(path), **kw}


def check_programs(*exes):
    if DRY or ADOPT: return
    for exe in exes:
        if shutil.which(exe) is None: fail(f"{exe} not found; create the environment from sers_dft.yml")


_VERSIONS = None
def versions():
    global _VERSIONS
    if _VERSIONS is None:
        from importlib import metadata
        v = {"python": platform.python_version(), "platform": platform.platform()}
        for p in ("ase", "numpy", "torch", "mace-torch", "torch-dftd", "e3nn"):
            try: v[p] = metadata.version(p)
            except metadata.PackageNotFoundError: v[p] = None
        _VERSIONS = v
    return dict(_VERSIONS)


def host():
    h = {"machine": platform.machine(), "cpus": os.cpu_count()}
    try:
        if sys.platform == "darwin":
            q = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
            h["mem_bytes"], h["cpu"] = int(q("hw.memsize")), q("machdep.cpu.brand_string")
        else:
            h["mem_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass
    return h


# ---------------------------------------------------------------- pseudopotentials
def load_sssp(log=print):
    """Pinned SSSP table. Downloads it into pseudo/ if absent and verifies MD5 checksums."""
    os.makedirs(PSEUDO, exist_ok=True)
    files = {f: os.path.join(PSEUDO, f) for f in SSSP["md5"]}
    jf, tf = files[f"{SSSP['name']}.json"], files[f"{SSSP['name']}.tar.gz"]
    for f, p in files.items():
        if not os.path.exists(p):
            if f.endswith(".tar.gz") and os.path.exists(jf):
                table = json.load(open(jf))
                if all(os.path.exists(os.path.join(PSEUDO, d["filename"])) for d in table.values()):
                    continue   # UPF files already extracted; per-file MD5 is checked in pp_info()
            url = SSSP["url"].format(f=f); log(f"  downloading {url}")
            try: urllib.request.urlretrieve(url, p)
            except Exception as e:
                fail(f"download failed ({e.__class__.__name__}). Download {f} from https://doi.org/{SSSP['doi']} "
                     f"into {PSEUDO} and rerun.")
        if os.path.exists(p) and md5(p) != SSSP["md5"][f]:
            fail(f"{p}: MD5 mismatch (expected {SSSP['md5'][f]} for {SSSP['name']})")
    table = json.load(open(jf))
    if not all(os.path.exists(os.path.join(PSEUDO, d["filename"])) for d in table.values()):
        with tarfile.open(tf) as tar:
            for m in tar.getmembers():
                if m.isfile() and m.name.lower().endswith(".upf"):
                    m.name = os.path.basename(m.name); tar.extract(m, PSEUDO)
    return table


def pp_info(table, elements):
    """{element: {"file", "md5"}} after verifying each UPF, plus the SSSP-recommended cutoffs."""
    pps, wfc, rho = {}, 0.0, 0.0
    for el in elements:
        d = table[el]; p = os.path.join(PSEUDO, d["filename"])
        if not os.path.exists(p): fail(f"{d['filename']} not found in pseudo/")
        if md5(p) != d["md5"]: fail(f"{p}: MD5 does not match {SSSP['name']}")
        pps[el] = {"file": d["filename"], "md5": d["md5"]}
        wfc, rho = max(wfc, d["cutoff_wfc"]), max(rho, d["cutoff_rho"])
    return pps, wfc, rho


# ---------------------------------------------------------------- MLIP (structure preparation only)
def load_mace(log=print):
    """MACE-MP-0 medium + D3(BJ). Verifies the model file; never falls back to a model without dispersion."""
    from mace.calculators import mace_mp
    cache = os.path.expanduser("~/.cache/mace")
    path = os.path.join(cache, "".join(c for c in os.path.basename(MACE_MODEL["url"]) if c.isalnum() or c == "_"))
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True); log(f"  downloading {MACE_MODEL['url']}")
        urllib.request.urlretrieve(MACE_MODEL["url"], path)
    if sha256(path) != MACE_MODEL["sha256"]: fail(f"{path}: SHA-256 does not match the pinned MACE model")
    try:
        return mace_mp(model=path, dispersion=True, dispersion_damping="bj", dispersion_xc="pbe",
                       default_dtype="float64", device="cpu")
    except Exception as e:
        fail(f"MACE with D3 could not be initialised ({e.__class__.__name__}: {e}); "
             "check torch-dftd. No fallback without dispersion is used.")


def relax_bare(slab, calc, log):
    """Relax a bare Au slab (bottom layer fixed). Aborts if BFGS does not converge."""
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    bare = slab.copy(); z = bare.positions[:, 2]
    bare.set_constraint(FixAtoms(indices=[i for i in range(len(bare)) if z[i] < z.min() + 0.5]))
    bare.calc = calc
    ok = bool(BFGS(bare, logfile=None).run(fmax=0.03, steps=200))
    bare.set_constraint(); bare.calc = None
    log(f"  bare slab MLIP relaxation (fmax 0.03 eV/Å, ≤200 steps): {'converged' if ok else 'NOT converged'}")
    if not ok: fail("bare slab relaxation did not converge")
    return bare


def bare_from(atoms):
    a = atoms[[x.index for x in atoms if x.symbol == "Au"]].copy(); a.set_constraint(); return a


def align_cells(systems, vacuum=13.0):
    """Common origin and cell height: bottom Au layer at z = 1 Å; cell = Au bottom-to-top-atom span + vacuum."""
    for s in systems.values(): s.set_constraint()
    zmin = min(s.positions[[i for i, x in enumerate(s) if x.symbol == "Au"], 2].min() for s in systems.values())
    ztop = max(s.positions[:, 2].max() for s in systems.values())
    for s in systems.values():
        s.positions[:, 2] += 1.0 - zmin
        c = s.cell.copy(); c[2] = [0, 0, (ztop - zmin) + vacuum]; s.set_cell(c); s.pbc = True


def gas_box(atoms, L=None):
    """Cubic box: side L, or max(18 Å, molecule extent + 10 Å)."""
    g = atoms.copy(); g.set_constraint()
    L = L or max(18.0, float(np.ptp(g.positions, axis=0).max()) + 10.0)
    g.set_cell([L, L, L]); g.center(); g.pbc = True
    return g


def mlip_energy(atoms, calc, periodic=True):
    b = atoms.copy(); b.set_constraint(); b.pbc = periodic; b.calc = calc
    return float(b.get_potential_energy())


def read_gsolv(path="data/results_fix/fix_summary.csv"):
    import csv
    if not os.path.exists(path): fail(f"{path} not found")
    gs = {}
    for r in csv.reader(open(path)):
        if r and r[0].startswith("xTB"):
            for kv in r[1:]:
                k, v = kv.split("="); gs[k] = float(v)
    if not gs: fail(f"no xTB solvation energies in {path}")
    return gs


# ---------------------------------------------------------------- xTB
def xtb_energy(atoms, solvent, log):
    """GFN2-xTB optimisation; final total energy in eV. Returns None in dry-run mode."""
    if DRY:
        log("  [dry-run] xtb not run"); return None
    exe = shutil.which("xtb")
    if not exe: fail("xtb not found")
    with tempfile.TemporaryDirectory() as d:
        write(f"{d}/m.xyz", atoms)
        cmd = [exe, "m.xyz", "--gfn", "2", "--chrg", "0", "--opt"] + (["--alpb", "water"] if solvent else [])
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True,
                           env=dict(os.environ, OMP_NUM_THREADS="4", OMP_STACKSIZE="1G"))
    out = p.stdout + p.stderr
    if p.returncode != 0 or "normal termination of xtb" not in out:
        fail(f"xtb failed (exit {p.returncode}): {out.strip().splitlines()[-1:] }")
    if "GEOMETRY OPTIMIZATION CONVERGED" not in out: fail("xtb geometry optimisation did not converge")
    lines = [l for l in p.stdout.splitlines() if "TOTAL ENERGY" in l]
    if not lines: fail("could not parse the xtb total energy")
    return float(lines[-1].split()[3]) * Hartree


# ---------------------------------------------------------------- QE input / output
def fingerprint(atoms, pps, ecutwfc, ecutrho, kpts):
    def r(x): return (np.round(np.asarray(x, float), 5) + 0.0).tolist()
    pps = {el: pps[el] for el in sorted(set(atoms.get_chemical_symbols()))}   # only species present
    return {"symbols": atoms.get_chemical_symbols(), "positions": r(atoms.positions), "cell": r(atoms.cell.array),
            "pbc": [bool(x) for x in atoms.pbc], "ecutwfc": float(ecutwfc), "ecutrho": float(ecutrho),
            "kpts": "gamma" if kpts is None else [int(x) for x in kpts], "pseudopotentials": pps,
            "qe": QE_PARAMS, "d3": "qe_builtin"}


def fp_hash(fp):
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()


def write_input(path, atoms, pps, ecutwfc, ecutrho, kpts, slab, diag, outdir):
    data = {
        "control": {"calculation": "scf", "prefix": "pw", "pseudo_dir": PSEUDO, "outdir": outdir,
                    "disk_io": "low", "tprnfor": False},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": QE_PARAMS["occupations"],
                   "smearing": QE_PARAMS["smearing"], "degauss": QE_PARAMS["degauss"],
                   "vdw_corr": QE_PARAMS["vdw_corr"], "dftd3_version": QE_PARAMS["dftd3_version"]},
        "electrons": {"conv_thr": QE_PARAMS["conv_thr"], "diagonalization": diag,
                      "mixing_beta": 0.2 if slab else 0.4,
                      "mixing_mode": "local-TF" if slab else "plain", "electron_maxstep": 300},
    }
    if LOWMEM: data["electrons"]["diago_david_ndim"] = 2
    write(path, atoms, format="espresso-in", input_data=data,
          pseudopotentials={el: v["file"] for el, v in pps.items()}, kpts=kpts)
    txt = open(path).read()
    for nl in ("&FCP\n/\n", "&RISM\n/\n", "&CELL\n/\n", "&IONS\n/\n"):   # version compatibility: drop empty namelists
        txt = txt.replace(nl, "")
    open(path, "w").write(txt)


def check_output(path):
    """Parse a pw.x output. "complete" requires JOB DONE, SCF convergence, the D3 term and a parsed energy."""
    r = {"exists": os.path.exists(path), "job_done": False, "converged": False, "d3_present": False,
         "energy_total_eV": None, "energy_d3_eV": None, "qe_version": None, "ram_estimate": None, "complete": False}
    if not r["exists"]: return r
    txt = open(path, errors="ignore").read()
    r["job_done"] = "JOB DONE." in txt
    r["converged"] = "convergence has been achieved" in txt and "convergence NOT achieved" not in txt
    m = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", txt, re.M)
    if m: r["energy_total_eV"] = float(m[-1]) * Ry
    m = re.findall(r"DFT-D3 Dispersion\s+=\s+(-?\d+\.\d+)\s+Ry", txt)
    if m: r["d3_present"], r["energy_d3_eV"] = True, float(m[-1]) * Ry
    m = re.search(r"Program PWSCF v\.(\S+)", txt)
    if m: r["qe_version"] = m.group(1)
    m = re.search(r"Estimated max dynamical RAM per process >\s+([\d.]+ \w+)", txt)
    if m: r["ram_estimate"] = m.group(1) + " per process (QE estimate, not measured)"
    r["complete"] = r["job_done"] and r["converged"] and r["d3_present"] and r["energy_total_eV"] is not None
    return r


def read_meta(workdir, name):
    try: return json.load(open(os.path.join(workdir, f"{name}.meta.json")))
    except Exception: return None


def _archive(workdir, name, why, log):
    files = [f for f in glob.glob(os.path.join(workdir, f"{name}.*")) if os.path.isfile(f)]
    if not files: return None
    dst = os.path.join(workdir, "_archive", name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(dst, exist_ok=True)
    for f in files: shutil.move(f, os.path.join(dst, os.path.basename(f)))
    open(os.path.join(dst, "why.txt"), "w").write(why + "\n")
    log(f"    previous files of {name} moved to {dst} ({why})")
    return dst


def stored_input_matches(path, fp):
    """Compare a stored pw.x input with a fingerprint. Used only by DFT_ADOPT."""
    try:
        txt = open(path).read(); a = read(path, format="espresso-in")
    except Exception as e:
        return False, f"stored input unreadable ({e.__class__.__name__})"
    if a.get_chemical_symbols() != fp["symbols"]: return False, "elements differ"
    if np.abs(a.positions - np.array(fp["positions"])).max() > 1e-3: return False, "positions differ"
    if np.abs(a.cell.array - np.array(fp["cell"])).max() > 1e-3: return False, "cell differs"
    def num(k):
        m = re.search(rf"\b{k}\s*=\s*([\d.eEdD+-]+)", txt)
        return float(m.group(1).lower().replace("d", "e")) if m else None
    if num("ecutwfc") != fp["ecutwfc"] or num("ecutrho") != fp["ecutrho"]: return False, "cutoff differs"
    if "dft-d3" not in txt.lower(): return False, "QE D3 not enabled in stored input"
    kp = re.search(r"K_POINTS\s*\{?(\w+)\}?[^\n]*\n([^\n]*)", txt)
    if not kp: return False, "no K_POINTS card"
    stored = "gamma" if kp.group(1).lower() == "gamma" else [int(x) for x in kp.group(2).split()[:3]]
    want = fp["kpts"]
    if stored != want and not (want == "gamma" and stored == [1, 1, 1]):
        return False, f"k-points differ (stored {stored}, requested {want})"
    for el, v in fp["pseudopotentials"].items():
        if not re.search(rf"^\s*{el}\s+\S+\s+{re.escape(v['file'])}\s*$", txt, re.M):
            return False, f"pseudopotential for {el} differs"
    return True, ""


def _meta(name, workdir, h, fp, src, out, attempts, status):
    e = None
    if status == "completed":
        e = {"total_eV": out["energy_total_eV"], "d3_eV": out["energy_d3_eV"],
             "without_d3_eV": out["energy_total_eV"] - out["energy_d3_eV"], "d3_method": "qe_builtin",
             "note": "total_eV already contains D3; do not add D3 again"}
    return {"schema_version": SCHEMA_VERSION, "name": name, "workdir": workdir, "status": status, "hash": h,
            "fingerprint": fp, "source": src, "requested": {"kpts": fp["kpts"], "np": NP, "nk": NK, "lowmem": LOWMEM},
            "attempts": attempts, "energy": e, "qe_version": out.get("qe_version"),
            "ram_estimate": out.get("ram_estimate"), "versions": versions(), "host": host(),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _dump(meta):
    json.dump(meta, open(os.path.join(meta["workdir"], f"{meta['name']}.meta.json"), "w"), indent=1)
    return meta


def run_pw(workdir, name, atoms, pps, ecutwfc, ecutrho, kpts, slab, log, src=None):
    """Run or reuse one pw.x single-point; returns its metadata. meta["status"] is one of
    completed, failed, dry_run, not_adopted. The energy is energy(meta)."""
    os.makedirs(workdir, exist_ok=True)
    fp = fingerprint(atoms, pps, ecutwfc, ecutrho, kpts); h = fp_hash(fp)
    fin, fout = os.path.join(workdir, f"{name}.in"), os.path.join(workdir, f"{name}.out")
    meta, out = read_meta(workdir, name), check_output(fout)

    if DRY:
        write_input(fin, atoms, pps, ecutwfc, ecutrho, kpts, slab, "david" if slab else "cg", "<scratch>")
        log(f"  [dry-run] {name:<22s} wrote {fin}")
        return {"status": "dry_run", "hash": h, "name": name, "workdir": workdir}

    if meta and meta.get("hash") == h and meta.get("status") == "completed" and out["complete"] and not RECOMPUTE:
        log(f"  ✓ {name:<22s} reusing verified result")
        return meta

    if ADOPT:
        reason = ("metadata exists but does not match" if meta else
                  "no complete output" if not out["complete"] else None)
        if reason is None:
            ok, reason = stored_input_matches(fin, fp)
            if ok:
                meta = _meta(name, workdir, h, fp, src, out, [], "completed")
                meta["adopted"] = {"date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                   "verified": ["elements", "positions", "cell", "cutoff", "k-points",
                                                "pseudopotential file names", "QE D3 enabled", "output complete"],
                                   "unverified": ["pseudopotential checksums at run time",
                                                  "software versions at run time", "solver settings used"]}
                log(f"  ⇢ {name:<22s} adopted (stored input verified; run-time checksums unverified)")
                return _dump(meta)
        log(f"  ✗ {name:<22s} not adopted: {reason}")
        return {"status": "not_adopted", "reason": reason, "hash": h, "name": name, "workdir": workdir}

    why = ("DFT_RECOMPUTE=1" if RECOMPUTE else "no metadata" if meta is None else
           "structure or settings changed" if meta.get("hash") != h else
           f"previous status {meta.get('status')}" if meta.get("status") != "completed" else "output incomplete")
    _archive(workdir, name, why, log)

    if slab:   # Γ-only + Davidson can fail on large slabs: try k=1×1×1 Davidson before CG (very slow)
        tries = [("david", kpts)] + ([("david", (1, 1, 1))] if kpts is None else []) + [("cg", kpts)]
    else:
        tries = [("cg", kpts), ("david", kpts), ("cg", (1, 1, 1))]
    attempts = []
    for k, (diag, kp) in enumerate(tries, 1):
        scratch = os.path.join(SCRATCH, f"{name}_{os.getpid()}_{time.time_ns()}")   # unique per run
        os.makedirs(scratch, exist_ok=True)
        write_input(fin, atoms, pps, ecutwfc, ecutrho, kp, slab, diag, scratch)
        nk = NK or (2 if (slab and kp and kp[0] * kp[1] > 1) else 1)
        cmd = (f"ulimit -s hard 2>/dev/null; mpirun -np {NP} pw.x -nk {nk} -nd 1 "
               f"-in {name}.in > {name}.out 2> {name}.err")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                   OMPI_MCA_rmaps_base_oversubscribe="1")
        log(f"  ▶ {name:<22s} {time.strftime('%H:%M')} ecut {ecutwfc}/{ecutrho} Ry, k {kp or 'Γ'}, {diag}, np {NP}, nk {nk}")
        t = time.time()
        p = subprocess.run(cmd, shell=True, cwd=workdir, env=env,
                           executable="/bin/zsh" if os.path.exists("/bin/zsh") else "/bin/sh")
        shutil.rmtree(scratch, ignore_errors=True)
        out = check_output(fout)
        att = {"attempt": k, "diag": diag, "kpts": "gamma" if kp is None else list(kp), "np": NP, "nk": nk,
               "returncode": p.returncode, "wall_s": round(time.time() - t, 1),
               **{x: out[x] for x in ("job_done", "converged", "d3_present", "ram_estimate")}}
        attempts.append(att)
        if p.returncode == 0 and out["complete"]:
            log(f"    done ({att['wall_s'] / 60:.1f} min)")
            return _dump(_meta(name, workdir, h, fp, src, out, attempts, "completed"))
        cause = (f"exit code {p.returncode}" if p.returncode else "SCF not converged" if out["job_done"] and not out["converged"]
                 else "D3 term missing" if out["job_done"] and not out["d3_present"] else "no JOB DONE")
        att["cause"] = cause
        keep = os.path.join(workdir, "_archive", name, "failed_attempts", f"{time.strftime('%Y%m%d-%H%M%S')}_try{k}")
        os.makedirs(keep, exist_ok=True)
        for ext in ("in", "out", "err"):
            f = os.path.join(workdir, f"{name}.{ext}")
            if os.path.exists(f): shutil.copy(f, keep)
        json.dump(att, open(os.path.join(keep, "attempt.json"), "w"), indent=1)
        log(f"    failed (attempt {k}: {cause}){' → retrying' if k < len(tries) else ''}; kept in {keep}")
    return _dump(_meta(name, workdir, h, fp, src, check_output(fout), attempts, "failed"))


def energy(meta):
    """Total energy in eV (D3 included) of a completed calculation, else None."""
    return meta["energy"]["total_eV"] if meta and meta.get("status") == "completed" else None


def find_completed(h, root="results"):
    """A completed calculation with this fingerprint hash anywhere under root (outside _archive)."""
    for p in glob.glob(os.path.join(root, "**", "*.meta.json"), recursive=True):
        if f"{os.sep}_archive{os.sep}" in p: continue
        try: m = json.load(open(p))
        except Exception: continue
        if m.get("hash") == h and m.get("status") == "completed" and \
                check_output(os.path.join(m["workdir"], f"{m['name']}.out"))["complete"]:
            return m
    return None


def run_or_reuse(workdir, name, atoms, pps, ecutwfc, ecutrho, kpts, slab, log, src=None):
    """run_pw, but first reuse an identical completed calculation from another stage (e.g. 04 gas references)."""
    if not (DRY or RECOMPUTE):
        m = find_completed(fp_hash(fingerprint(atoms, pps, ecutwfc, ecutrho, kpts)))
        if m and not (m["workdir"] == workdir and m["name"] == name):
            log(f"  ✓ {name:<22s} identical completed calculation: {m['workdir']}/{m['name']}")
            return m
    return run_pw(workdir, name, atoms, pps, ecutwfc, ecutrho, kpts, slab, log, src)


def structures_from_04():
    """Structures computed by 04, read from its completed inputs (for 07, 12, 15, 16)."""
    S, d = {}, "results/dft_check"
    for key, f in [("Au", "Au"), ("THI@Au", "THI_Au"), ("TBZ@Au", "TBZ_Au"), ("DQ@Au", "DQ_Au"),
                   ("THI_gas", "THI_gas"), ("TBZ_gas", "TBZ_gas"), ("DQ_gas", "DQ_gas")]:
        m = read_meta(d, f)
        if m and m.get("status") == "completed":
            S[key] = read(os.path.join(d, f"{f}.in"), format="espresso-in")
    if not all(k in S for k in ("Au", "THI_gas", "TBZ_gas", "DQ_gas")) and not DRY:
        fail("completed 04 results (results/dft_check/*.meta.json) not found; run 04_dft_singlepoint.py "
             "(or DFT_ADOPT=1 for existing outputs) first")
    return S


def gas_reference(n, table, ecut, workdir, log):
    """Gas-phase DFT energy of molecule n (THI, TBZ, DQ) in the 04 structure and box. ecut=None uses the
    04 cutoff, so the 04 result itself is reused; other cutoffs reuse any identical completed calculation."""
    S = structures_from_04()
    if f"{n}_gas" not in S: return {"status": "not_available", "reason": "04 gas structure missing", "name": f"{n}_gas"}
    if ecut is None:
        ref = read_meta("results/dft_check", f"{n}_gas")
        if not ref or ref.get("status") != "completed":
            return {"status": "not_available", "reason": "04 gas reference not completed", "name": f"{n}_gas"}
        ecut = (ref["fingerprint"]["ecutwfc"], ref["fingerprint"]["ecutrho"])
    at = S[f"{n}_gas"]
    pps, _, _ = pp_info(table, sorted(set(at.get_chemical_symbols())))
    return run_or_reuse(workdir, f"{n}_gas", at, pps, ecut[0], ecut[1], None, False, log,
                        {"path": "results/dft_check/" + f"{n}_gas.in", "note": "04 gas structure"})
