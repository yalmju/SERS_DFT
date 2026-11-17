"""
Rebuild all result tables from stored outputs only (no calculation is run).

    python aggregate.py            # reads results/*/record.json and the QE outputs they point to

Each stage script writes record.json (every planned calculation, including ones that could not be run)
and calls build() at the end. Status of each calculation:
    completed      metadata "completed", same fingerprint hash as planned, output passes check_output()
    failed         all pw.x attempts failed (attempts and causes are in <name>.meta.json and _archive/)
    incomplete     output present but not complete (e.g. killed), or metadata/hash mismatch
    not_available  not planned to run (reason given), or not adopted, or no files at all
    superseded     an older run kept in _archive/ (listed in calculations.csv only)
Derived quantities are computed only when every input is completed; otherwise the value is left
empty and status/reason name the first missing input. Missing values are never treated as zero.

Output: results/tables/{calculations,check,convergence,coverage,cnf,ternary118,poses}.csv, and with
        results/thermo/thermo.json also thermo.csv and free_energy.csv; schema.json
"""
import os, sys, csv, json, glob
from collections import Counter

SCHEMA_VERSION = 1
COLUMNS = {
    "calculations": ["stage", "key", "status", "reason", "workdir", "name", "hash", "source_path", "source_sha256",
                     "pose", "ecutwfc", "ecutrho", "kpts", "E_total_eV", "E_d3_eV", "E_without_d3_eV",
                     "qe_version", "adopted", "wall_s", "ram_estimate"],
    "check": ["quantity", "method", "value_eV", "status", "reason", "inputs"],
    "convergence": ["setting", "quantity", "method", "value_eV", "diff_from_ref_eV", "status", "reason", "inputs"],
    "coverage": ["cell", "area_A2", "quantity", "method", "value_eV", "pose", "ecut_Ry", "status", "reason", "inputs"],
    "cnf": ["molecule", "quantity", "method", "value_eV", "status", "reason", "inputs"],
    "ternary118": ["state", "quantity", "method", "value_eV", "status", "reason", "inputs"],
    "poses": ["cell", "area_A2", "molecule", "pose", "origin", "structure", "quantity", "method", "value_eV",
              "lowest_in_cell", "ecut_Ry", "kpts", "status", "reason", "inputs"],
    "thermo": ["kind", "name", "molecules", "n_modes", "n_imag", "relax_converged", "relax_max_shift_A",
               "internal_rmsd_A", "frequencies_at",
               "ZPE_eV", "G_corr_harmonic_eV", "G_corr_truhlar100_eV", "G_corr_grimme_eV"],
    "free_energy": ["table", "cell", "system", "pose", "origin", "quantity", "E_aq_DFT_eV", "dG_corr_harmonic_eV",
                    "dG_corr_truhlar100_eV", "dG_corr_grimme_eV", "G_aq_truhlar100_eV", "G_aq_grimme_eV",
                    "G_aq_truhlar100_DQ_dissociated_eV", "frequencies_at", "status", "reason"],
}
SCHEMES = ("harmonic", "truhlar100", "grimme")


def _check_output(path):
    import qe_tools
    return qe_tools.check_output(path)


def resolve(item):
    """Status, reason and DFT energy of one planned calculation, from stored files only."""
    st = item.get("run_status")
    if st in ("not_available", "not_adopted", "dry_run", None) and not item.get("calc_name"):
        return "not_available", item.get("run_reason") or "not run", None, None
    wd, name = item.get("calc_dir"), item.get("calc_name")
    if not wd or not name:
        return "not_available", item.get("run_reason") or "not run", None, None
    try: meta = json.load(open(os.path.join(wd, f"{name}.meta.json")))
    except Exception: meta = None
    out = _check_output(os.path.join(wd, f"{name}.out"))
    if meta is None:
        if out["exists"]:
            why = (f"not adopted: {item['run_reason']}" if item.get("run_status") == "not_adopted"
                   else "output without metadata (verify with DFT_ADOPT=1)")
            return "incomplete", why + ("" if out["complete"] else "; output not complete"), None, None
        return "not_available", item.get("run_reason") or "no files", None, None
    if item.get("hash") and meta.get("hash") != item["hash"]:
        return "incomplete", "stored calculation does not match the planned structure/settings", None, meta
    if meta.get("status") == "failed":
        causes = "; ".join(f"try {a['attempt']}: {a.get('cause', '?')}" for a in meta.get("attempts", []))
        return "failed", causes or "failed", None, meta
    if meta.get("status") == "completed" and out["complete"]:
        return "completed", "", meta["energy"]["total_eV"], meta
    return "incomplete", "output missing or not complete", None, meta


class Calc:
    """Resolved items of one stage, addressable by key."""

    def __init__(self, rec):
        self.rec, self.items, self.c = rec, {}, rec.get("constants", {})
        for it in rec["items"]:
            status, reason, e, meta = resolve(it)
            self.items[it["key"]] = dict(it, status=status, reason=reason, E=e, meta=meta)

    def dft(self, *keys):
        """(sum of DFT energies, status, reason, inputs) with signs given by a leading '-'."""
        tot, inputs = 0.0, []
        for k in keys:
            sgn, kk = (-1, k[1:]) if k.startswith("-") else (1, k)
            it = self.items.get(kk); inputs.append(kk)
            if it is None: return None, "not_available", f"{kk}: not planned", inputs
            if it["status"] != "completed": return None, it["status"], f"{kk}: {it['reason'] or it['status']}", inputs
            tot += sgn * it["E"]
        return tot, "completed", "", inputs

    def mlip(self, *keys):
        tot, inputs = 0.0, []
        for k in keys:
            sgn, kk = (-1, k[1:]) if k.startswith("-") else (1, k)
            it = self.items.get(kk); inputs.append(kk)
            if it is None or it.get("mlip_eV") is None: return None, "not_available", f"{kk}: no MLIP energy", inputs
            tot += sgn * it["mlip_eV"]
        return tot, "completed", "", inputs


def _sub(a, b):
    """Difference of two (value, status, reason, inputs) tuples."""
    if a[0] is None: return a
    if b[0] is None: return b
    return a[0] - b[0], "completed", "", a[3] + b[3]


def _add_const(x, c, label):
    if x[0] is None: return x
    if c is None: return None, "not_available", f"{label} not available", x[3]
    return x[0] + c, x[1], x[2], x[3]


def _row(**kw):
    v = kw.get("value_eV")
    kw["value_eV"] = "" if v is None else f"{v:.4f}"
    kw["inputs"] = " ".join(kw.get("inputs") or [])
    return kw


def table_check(S):
    rows, gs = [], S.c.get("G_solv_eV", {})
    for meth, f in (("DFT", S.dft), ("MLIP", S.mlip)):
        ead = {}
        for m in ("THI", "TBZ", "DQ"):
            vac = f(f"{m}@Au", "-Au", f"-{m}_gas")
            aq = _add_const(vac, -gs[m] if m in gs else None, f"G_solv({m})")
            ead[m] = aq
            for q, x in ((f"E_ads_vac {m}", vac), (f"E_ads_aq {m}", aq)):
                rows.append(_row(quantity=q, method=meth, value_eV=x[0], status=x[1], reason=x[2], inputs=x[3]))
        for a, b in (("THI", "TBZ"), ("THI", "DQ"), ("TBZ", "DQ")):
            x = _sub(ead[a], ead[b])
            rows.append(_row(quantity=f"dE_aq {a}<-{b}", method=meth, value_eV=x[0], status=x[1], reason=x[2], inputs=x[3]))
    return rows


def table_convergence(S):
    rows, gs = [], S.c.get("G_solv_eV", {})
    mols = S.c.get("molecules", ["THI", "TBZ"])
    ref = {}
    for st in S.c.get("settings", {}):
        ead = {m: S.dft(f"{st}/{m}@Au", f"-{st}/Au", f"-{st}/{m}_gas") for m in mols}
        qs = [(f"E_ads_vac {m}", ead[m]) for m in mols]
        d = _sub(ead["THI"], ead["TBZ"])
        qs += [("dE_vac THI<-TBZ", d), ("dE_aq THI<-TBZ", _add_const(d, (gs["TBZ"] - gs["THI"]) if gs else None, "G_solv"))]
        for q, x in qs:
            if st == "ref": ref[q] = x[0]
            diff = "" if x[0] is None or ref.get(q) is None else f"{x[0] - ref[q]:+.4f}"
            rows.append(_row(setting=st, quantity=q, method="DFT", value_eV=x[0], diff_from_ref_eV=diff,
                             status=x[1], reason=x[2], inputs=x[3]))
    return rows


def table_coverage(S, check=None):
    rows, gs = [], S.c.get("G_solv_eV", {})
    cells = sorted({it["cell"] for it in S.items.values() if it.get("cell")})
    blocks = [(cl, S, lambda m, cl=cl: f"{cl}/{m}") for cl in cells]
    if check is not None: blocks.append(("4x4", check, lambda m: m))   # 118 Å² from 04, aggregated the same way
    for cl, X, k in blocks:
        au = X.items.get(k("Au"), {})
        area = au.get("area_A2")
        if area is None and au.get("meta"):
            import numpy as np
            c = np.array(au["meta"]["fingerprint"]["cell"]); area = float(np.linalg.norm(np.cross(c[0], c[1])))
        ec = au.get("meta", {}) and au["meta"]["fingerprint"] if au.get("meta") else None
        ecut = f"{ec['ecutwfc']:g}/{ec['ecutrho']:g}" if ec else ""
        for meth in ("DFT", "MLIP"):
            f = X.dft if meth == "DFT" else X.mlip
            ead = {}
            for m in ("THI", "TBZ", "DQ"):
                if k(f"{m}@Au") not in X.items: continue
                x = _add_const(f(k(f"{m}@Au"), f"-{k('Au')}", f"-{k(f'{m}_gas')}"), -gs[m] if m in gs else None, f"G_solv({m})")
                ead[m] = x
                it = X.items[k(f"{m}@Au")]
                pose = (it.get("source") or {}).get("pose", "")
                rows.append(_row(cell=cl, area_A2=f"{area:.1f}" if area else "", quantity=f"E_ads_aq {m}", method=meth,
                                 value_eV=x[0], pose=pose, ecut_Ry=ecut, status=it["status"] if it["status"] != "completed" else x[1],
                                 reason=it["reason"] or x[2], inputs=x[3]))
            for a, b in (("THI", "TBZ"), ("THI", "DQ"), ("TBZ", "DQ")):
                if a in ead and b in ead:
                    x = _sub(ead[a], ead[b])
                    rows.append(_row(cell=cl, area_A2=f"{area:.1f}" if area else "", quantity=f"dE_aq {a}<-{b}", method=meth,
                                     value_eV=x[0], pose="", ecut_Ry=ecut, status=x[1], reason=x[2], inputs=x[3]))
    return rows


def table_cnf(S):
    rows, c = [], S.c
    gs, fp, cb = c.get("G_solv_eV", {}), c.get("footprint_A2", {}), c.get("dE_bind_CNF_eV", {})
    for meth, f in (("DFT", S.dft), ("MLIP", S.mlip)):
        e = {m: _add_const(f(f"{m}@Au", "-Au", f"-{m}_gas"), -gs[m] if gs.get(m) is not None else None, f"G_solv({m})")
             for m in ("CNF", "THI", "TBZ", "DQ")}
        path = lambda m: ((S.items.get(f"{m}@Au") or {}).get("source") or {}).get("path")
        rows.append(_row(molecule="CNF", quantity="E_ads_aq", method=meth, value_eV=e["CNF"][0], status=e["CNF"][1],
                         reason=e["CNF"][2], inputs=e["CNF"][3], _path=path("CNF"), _mols=["CNF"]))
        for m in ("THI", "TBZ", "DQ"):
            x = e[m]
            ex = _sub(x, (e["CNF"][0] * fp[m] / fp["CNF"], *e["CNF"][1:]) if e["CNF"][0] is not None else e["CNF"])
            tr = _add_const(ex, -cb[m] if m in cb else None, f"dE_bind({m})")
            for q, y in (("E_ads_aq", x), ("equal_area_exchange", ex), ("transfer_CNF_to_Au", tr)):
                rows.append(_row(molecule=m, quantity=q, method=meth, value_eV=y[0], status=y[1], reason=y[2], inputs=y[3],
                                 **({"_path": path(m), "_mols": [m]} if q == "E_ads_aq" else {})))
    return rows


def table_ternary(S):
    rows, gs, ml = [], S.c.get("G_solv_eV", {}), S.c.get("mlip_state_eV", {})
    for k, it in S.items.items():
        if it.get("role") != "state": continue
        mols = it.get("molecules") or k.split("+")
        x = S.dft(k, "-Au", *[f"-{m}_gas" for m in mols])
        x = _add_const(x, -sum(gs[m] for m in mols) if all(m in gs for m in mols) else None, "G_solv")
        src = it.get("source") or (it.get("meta") or {}).get("source") or {}
        rows.append(_row(state=k, quantity="G_state_aq", method="DFT", value_eV=x[0], status=x[1], reason=x[2], inputs=x[3],
                         _path=src.get("path"), _mols=list(mols)))
        rows.append(_row(state=k, quantity="G_state_aq", method="MLIP", value_eV=ml.get(k),
                         status="completed" if k in ml else "not_available", reason="" if k in ml else "no MLIP value",
                         inputs=["record.constants.mlip_state_eV"]))
    return rows


def table_poses(S):
    """E_ads,aq per pose; lowest_in_cell marks the lowest completed value per cell, molecule and method."""
    rows, gs = [], S.c.get("G_solv_eV", {})
    for meth, f in (("DFT", S.dft), ("MLIP", S.mlip)):
        block = []
        for k, it in S.items.items():
            if it.get("role") != "pose": continue
            m = it.get("molecule")
            if it.get("ref") and it.get("gas"):
                x = _add_const(f(k, f"-{it['ref']}", f"-{it['gas']}"), -gs[m] if m in gs else None, f"G_solv({m})")
            else:
                x = (None, it["status"], it["reason"], [k])
            ref = S.items.get(it.get("ref") or "", {})
            src = it.get("source") or {}
            area = ref.get("area_A2")
            block.append(_row(cell=it.get("cell"), area_A2=f"{area:.1f}" if area else "", molecule=m, pose=it.get("pose"),
                              origin=src.get("origin", ""), structure=it.get("structure", ""), quantity="E_ads_aq",
                              method=meth, value_eV=x[0], lowest_in_cell="",
                              ecut_Ry="/".join(f"{v:g}" for v in it["ecut"]) if it.get("ecut") else "",
                              kpts="x".join(str(v) for v in ref.get("kpts", [])),
                              status=it["status"] if it["status"] != "completed" else x[1],
                              reason=it["reason"] or x[2], inputs=x[3], _path=src.get("path"), _mols=[m]))
        best = {}
        for r in block:
            if r["value_eV"] != "":
                kk = (r["cell"], r["molecule"])
                if kk not in best or float(r["value_eV"]) < float(best[kk]["value_eV"]): best[kk] = r
        for r in best.values(): r["lowest_in_cell"] = "yes"
        rows += block
    return rows


def _dgcorr(th, path, mols, scheme, dq_diss=False):
    """G_corr(adsorbed structure) − Σ G_corr(gas, 1 M), or None with a reason."""
    st = th.get("structures", {}).get(path)
    if st is None: return None, f"no thermo result for {path}"
    if st.get("sha256") and os.path.exists(path):
        import qe_tools
        if qe_tools.sha256(path) != st["sha256"]: return None, f"{path} changed since its thermo result"
    g = st["corr"][scheme]["G"]
    for m in mols:
        refs = ["DQ2+", "Br-", "Br-"] if (dq_diss and m == "DQ") else [m]
        for r in refs:
            if r not in th.get("gas", {}): return None, f"no gas thermo result for {r}"
            g -= th["gas"][r]["corr"][scheme]["G_1M"]
    return g, ""


def table_thermo(th):
    rows = []
    for kind, block in (("gas", th.get("gas", {})), ("adsorbed", th.get("structures", {}))):
        for name, v in block.items():
            c = v["corr"]
            key = "G_1M" if kind == "gas" else "G"
            rows.append({"kind": kind, "name": name, "molecules": " ".join(v.get("molecules", [])),
                         "n_modes": len(v.get("freqs_cm1", [])), "n_imag": v.get("n_imag", ""),
                         "relax_converged": (v.get("relax") or {}).get("converged", ""),
                         "relax_max_shift_A": f"{v['relax']['max_shift_A']:.3f}" if v.get("relax") else "",
                         "internal_rmsd_A": f"{v['relax']['internal_rmsd_A']:.3f}" if "internal_rmsd_A" in (v.get("relax") or {}) else "",
                         "frequencies_at": (v.get("relax") or {}).get("frequencies_at", ""),
                         "ZPE_eV": f"{c['harmonic']['ZPE']:.4f}" if "ZPE" in c["harmonic"] else "",
                         **{f"G_corr_{s}_eV": f"{c[s][key]:.4f}" for s in SCHEMES}})
    return rows


def table_free_energy(th, blocks):
    """DFT water-referenced energies plus MLIP thermal corrections, one row per DFT row with a structure.
    frequencies_at = "input geometry" means the tight MLIP relaxation changed the pose (see 18_thermo.py)."""
    rows = []
    for table, rs, cell, syskey in blocks:
        for r in rs:
            if r.get("method") != "DFT" or not r.get("_path"): continue
            relax = (th.get("structures", {}).get(r["_path"]) or {}).get("relax") or {}
            out = dict(table=table, cell=r.get("cell") or cell, system=r.get(syskey, ""), pose=r.get("pose", ""),
                       origin=r.get("origin", ""), quantity=r.get("quantity", ""), E_aq_DFT_eV=r["value_eV"],
                       frequencies_at=relax.get("frequencies_at", ""), status=r["status"], reason=r["reason"])
            for sc in SCHEMES:
                d, why = _dgcorr(th, r["_path"], r["_mols"], sc)
                out[f"dG_corr_{sc}_eV"] = "" if d is None else f"{d:.4f}"
                if d is None and out["status"] == "completed": out.update(status="not_available", reason=why)
                if sc != "harmonic":
                    out[f"G_aq_{sc}_eV"] = f"{float(r['value_eV']) + d:.4f}" if d is not None and r["value_eV"] != "" else ""
            if "DQ" in r["_mols"]:
                d, _ = _dgcorr(th, r["_path"], r["_mols"], "truhlar100", dq_diss=True)
                out["G_aq_truhlar100_DQ_dissociated_eV"] = f"{float(r['value_eV']) + d:.4f}" if d is not None and r["value_eV"] != "" else ""
            rows.append(out)
    return rows


def _calc_rows(stage, S, root):
    rows = []
    for k, it in S.items.items():
        m = it.get("meta") or {}
        fpr, e, src = m.get("fingerprint") or {}, m.get("energy") or {}, it.get("source") or m.get("source") or {}
        rows.append({"stage": stage, "key": k, "status": it["status"], "reason": it["reason"],
                     "workdir": it.get("calc_dir", ""), "name": it.get("calc_name", ""), "hash": m.get("hash", ""),
                     "source_path": src.get("path", src.get("derived_from", "")), "source_sha256": src.get("sha256", ""),
                     "pose": src.get("pose", ""), "ecutwfc": fpr.get("ecutwfc", ""), "ecutrho": fpr.get("ecutrho", ""),
                     "kpts": json.dumps(fpr.get("kpts", "")), "E_total_eV": e.get("total_eV", ""),
                     "E_d3_eV": e.get("d3_eV", ""), "E_without_d3_eV": e.get("without_d3_eV", ""),
                     "qe_version": m.get("qe_version", ""), "adopted": "yes" if m.get("adopted") else "",
                     "wall_s": sum(a.get("wall_s", 0) for a in m.get("attempts", [])) or "", "ram_estimate": m.get("ram_estimate", "")})
    return rows


def _write(root, name, rows):
    os.makedirs(os.path.join(root, "tables"), exist_ok=True)
    with open(os.path.join(root, "tables", f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS[name], extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def build(root="results", log=print):
    recs = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "record.json"))):
        r = json.load(open(p)); r["folder"] = os.path.dirname(p); recs[r["stage"]] = Calc(r)
    calc_rows, summary = [], {}
    for stage, S in recs.items():
        summary[stage] = dict(Counter(it["status"] for it in S.items.values()))
        calc_rows += _calc_rows(stage, S, root)
    archived = []
    for p in sorted(glob.glob(os.path.join(root, "**", "_archive", "**", "*.meta.json"), recursive=True)):
        try: m = json.load(open(p))
        except Exception: continue
        archived.append({"stage": os.path.relpath(p, root).split(os.sep)[0], "key": "", "status": "superseded",
                         "reason": "older run kept in _archive", "workdir": os.path.dirname(p), "name": m.get("name", ""),
                         "hash": m.get("hash", ""), "E_total_eV": (m.get("energy") or {}).get("total_eV", "")})
    calc_rows += archived
    _write(root, "calculations", calc_rows)
    if "check" in recs: _write(root, "check", table_check(recs["check"]))
    if "convergence" in recs: _write(root, "convergence", table_convergence(recs["convergence"]))
    if "coverage" in recs or "check" in recs:
        cov = recs.get("coverage") or Calc({"items": [], "constants": recs["check"].c})
        _write(root, "coverage", table_coverage(cov, recs.get("check")))
    blocks = []   # (table, rows, default cell, system column) for free_energy.csv
    if "cnf" in recs:
        cr = table_cnf(recs["cnf"]); _write(root, "cnf", cr)
        blocks.append(("cnf", [r for r in cr if r["quantity"] == "E_ads_aq"], "6x6", "molecule"))
    if "ternary118" in recs:
        tr = table_ternary(recs["ternary118"]); _write(root, "ternary118", tr); blocks.append(("ternary118", tr, "4x4", "state"))
    if "poses" in recs:
        pr = table_poses(recs["poses"]); _write(root, "poses", pr); blocks.append(("poses", pr, "", "molecule"))
    tp = os.path.join(root, "thermo", "thermo.json")
    if os.path.exists(tp):
        th = json.load(open(tp))
        _write(root, "thermo", table_thermo(th)); _write(root, "free_energy", table_free_energy(th, blocks))
    json.dump({"schema_version": SCHEMA_VERSION, "tables": COLUMNS, "status_values":
               ["completed", "failed", "incomplete", "not_available", "superseded"], "summary": summary},
              open(os.path.join(root, "tables", "schema.json"), "w"), indent=1)
    for stage, s in summary.items(): log(f"  [aggregate] {stage}: {s}")
    return summary


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    s = build(root)
    sys.exit(1 if any(v.get("failed") or v.get("incomplete") for v in s.values()) else 0)
