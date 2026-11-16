"""
Checks of the execution and aggregation logic with a stand-in pw.x (tests/bin). No DFT is run.

    python -m unittest discover -s tests -v

Covers the minimum checklist: result reuse and rejection, dry-run, D3 handling, failure classification,
aggregation against raw outputs, completion status, and option validation. test_env_smoke.py runs one
tiny real pw.x calculation when Quantum ESPRESSO is installed.
"""
import csv, json, os, subprocess, sys, tempfile, unittest
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_BIN = os.path.join(REPO, "tests", "bin")
sys.path.insert(0, REPO)
for k in list(os.environ):
    if k.startswith("DFT_") or k == "CONV_DQ": del os.environ[k]
os.environ["PATH"] = FAKE_BIN + os.pathsep + os.environ["PATH"]
import qe_tools as Q          # noqa: E402
import aggregate              # noqa: E402
from ase import Atoms         # noqa: E402
from ase.units import Ry      # noqa: E402

PPS = {"H": {"file": "H.upf", "md5": "a" * 32}, "Au": {"file": "Au.upf", "md5": "b" * 32}}


def h2(dz=0.0):
    a = Atoms("H2", positions=[[5, 5, 5], [5, 5, 5.74 + dz]], cell=[10, 10, 10], pbc=True)
    return a


def slab():
    return Atoms("Au2", positions=[[0, 0, 1], [1.44, 1.44, 3.36]], cell=[2.88, 2.88, 20], pbc=True)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.cwd = os.getcwd(); os.chdir(self.tmp.name)
        self.calls = os.path.join(self.tmp.name, "calls.log")
        os.environ["FAKE_PW_LOG"] = self.calls; os.environ["FAKE_PW_MODE"] = "ok"
        Q.SCRATCH = os.path.join(self.tmp.name, "scratch")
        self.logs = []

    def tearDown(self):
        os.chdir(self.cwd); self.tmp.cleanup()

    def log(self, *a): self.logs.append(" ".join(map(str, a)))

    def ncalls(self):
        return len(open(self.calls).read().splitlines()) if os.path.exists(self.calls) else 0

    def run_pw(self, atoms=None, name="H2", ecut=(30, 240), kpts=None, pps=PPS, slab_=False, wd="results/t"):
        return Q.run_pw(wd, name, atoms if atoms is not None else h2(), pps, ecut[0], ecut[1], kpts, slab_, self.log)


class TestCache(Base):
    def test_same_conditions_reused(self):
        m1 = self.run_pw(); m2 = self.run_pw()
        self.assertEqual(m1["status"], "completed"); self.assertEqual(self.ncalls(), 1)
        self.assertEqual(m1["hash"], m2["hash"])

    def test_any_change_triggers_recompute(self):
        self.run_pw()
        changes = {
            "position": dict(atoms=h2(dz=0.01)),
            "element": dict(atoms=Atoms("HAu", positions=h2().positions, cell=[10, 10, 10], pbc=True)),
            "cell": dict(atoms=(lambda a: (a.set_cell([11, 10, 10]), a)[1])(h2())),
            "pbc": dict(atoms=(lambda a: (setattr(a, "pbc", [True, True, False]), a)[1])(h2())),
            "cutoff": dict(ecut=(35, 280)),
            "kpoints": dict(kpts=(2, 2, 1)),
            "upf": dict(pps={"H": {"file": "H.upf", "md5": "c" * 32}}),
        }
        n = 1
        for what, kw in changes.items():
            self.run_pw(**kw); n += 1
            self.assertEqual(self.ncalls(), n, f"{what} change did not trigger a recompute")
        self.assertTrue(os.path.isdir("results/t/_archive/H2"), "previous runs must be archived, not deleted")

    def test_missing_or_corrupt_metadata_not_reused(self):
        self.run_pw()
        os.remove("results/t/H2.meta.json")
        self.run_pw(); self.assertEqual(self.ncalls(), 2)
        open("results/t/H2.meta.json", "w").write("{not json")
        self.run_pw(); self.assertEqual(self.ncalls(), 3)

    def test_unreadable_input_not_reused_by_adopt(self):
        self.run_pw(); os.remove("results/t/H2.meta.json")
        open("results/t/H2.in", "w").write("garbage")
        ok, why = Q.stored_input_matches("results/t/H2.in", Q.fingerprint(h2(), PPS, 30, 240, None))
        self.assertFalse(ok); self.assertIn("unreadable", why)

    def test_adopt_accepts_only_matching_input(self):
        self.run_pw(); os.remove("results/t/H2.meta.json")
        fp = Q.fingerprint(h2(), PPS, 30, 240, None)
        self.assertTrue(Q.stored_input_matches("results/t/H2.in", fp)[0])
        self.assertFalse(Q.stored_input_matches("results/t/H2.in", Q.fingerprint(h2(), PPS, 35, 280, None))[0])
        self.assertFalse(Q.stored_input_matches("results/t/H2.in", Q.fingerprint(h2(0.01), PPS, 30, 240, None))[0])

    def test_identical_calculation_found_across_stages(self):
        self.run_pw(wd="results/a")
        m = Q.run_or_reuse("results/b", "H2", h2(), PPS, 30, 240, None, False, self.log)
        self.assertEqual(self.ncalls(), 1); self.assertEqual(m["workdir"], "results/a")


class TestD3AndFailures(Base):
    def test_energy_includes_d3_once(self):
        m = self.run_pw()
        e = m["energy"]
        expect = (-2 - 0.001 * (5 + 5.74)) * Ry
        self.assertAlmostEqual(e["total_eV"], expect, places=6)
        self.assertAlmostEqual(e["d3_eV"], -0.02 * Ry, places=6)
        self.assertAlmostEqual(e["without_d3_eV"], e["total_eV"] - e["d3_eV"], places=9)
        self.assertEqual(Q.energy(m), e["total_eV"])
        self.assertEqual(e["d3_method"], "qe_builtin")

    def test_failures_are_not_completed(self):
        for mode, cause in (("noconv", "SCF not converged"), ("crash", "no JOB DONE"),
                            ("nod3", "D3 term missing"), ("exit1", "exit code 1")):
            os.environ["FAKE_PW_MODE"] = mode
            m = self.run_pw(name=f"H2_{mode}")
            self.assertEqual(m["status"], "failed", mode); self.assertIsNone(Q.energy(m))
            self.assertEqual(m["attempts"][0]["cause"], cause)
            self.assertTrue(os.listdir(f"results/t/_archive/H2_{mode}/failed_attempts"))
        os.environ["FAKE_PW_MODE"] = "ok"
        m = self.run_pw(name="H2_noconv")      # failed result is never reused
        self.assertEqual(m["status"], "completed")

    def test_slab_retry_order(self):
        os.environ["FAKE_PW_MODE"] = "noconv"
        m = self.run_pw(atoms=slab(), name="Au", slab_=True, pps={"Au": PPS["Au"]})
        self.assertEqual([(a["diag"], a["kpts"]) for a in m["attempts"]],
                         [("david", "gamma"), ("david", [1, 1, 1]), ("cg", "gamma")])


class TestDryRunAndOptions(Base):
    def py(self, code, **env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("DFT_")}
        e.update(env); e["PYTHONPATH"] = REPO
        return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True)

    def test_dry_run_calls_nothing_and_leaves_results(self):
        code = ("import qe_tools as Q; from ase import Atoms\n"
                "a=Atoms('H2',positions=[[5,5,5],[5,5,5.74]],cell=[10,10,10],pbc=True)\n"
                "m=Q.run_pw(Q.RESULTS+'/t','H2',a,{'H':{'file':'H.upf','md5':'a'*32}},30,240,None,False,print)\n"
                "assert m['status']=='dry_run'; assert Q.xtb_energy(a,True,print) is None\n"
                "Q.check_programs('definitely-missing-program')\n")
        p = self.py(code, DFT_DRY_RUN="1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.ncalls(), 0)
        self.assertTrue(os.path.exists("results_dryrun/t/H2.in"))
        self.assertFalse(os.path.exists("results"))

    def test_option_validation(self):
        for env in ({"DFT_NP": "0"}, {"DFT_NP": "x"}, {"DFT_NK": "-1"}, {"DFT_LOWMEM": "yes"},
                    {"DFT_TYPO": "1"}, {"DFT_DRY_RUN": "1", "DFT_ADOPT": "1"}):
            p = self.py("import qe_tools", **env)
            self.assertNotEqual(p.returncode, 0, env); self.assertIn("[abort]", p.stderr + p.stdout)
        for env, call in (({"DFT_ECUT": "45"}, "Q.env_ecut('DFT_ECUT')"), ({"DFT_ECUT": "45,100"}, "Q.env_ecut('DFT_ECUT')"),
                          ({"DFT_CNF_K": "2,2"}, "Q.env_kpts('DFT_CNF_K')"),
                          ({"DFT_COV_CELLS": "3x3,7x7"}, "Q.env_list('DFT_COV_CELLS',['3x2','3x3','5x5'])")):
            p = self.py(f"import qe_tools as Q; {call}", **env)
            self.assertNotEqual(p.returncode, 0, env); self.assertIn("[abort]", p.stderr + p.stdout)
        p = self.py("import qe_tools as Q; assert Q.env_ecut('DFT_ECUT')==(45.0,360.0)", DFT_ECUT="45,360")
        self.assertEqual(p.returncode, 0, p.stderr)


class TestAggregation(Base):
    def make_check_stage(self, skip_dq=False, break_tbz=False):
        st = Q.Stage("check", "dft_check", "log.txt"); st.constants["G_solv_eV"] = {"THI": -0.4, "TBZ": -0.2, "DQ": -0.8}
        systems = {"Au": slab(), "THI@Au": slab() + Atoms("H", [[0, 0, 6]]), "TBZ@Au": slab() + Atoms("H2", [[0, 0, 6], [0, 0, 6.7]]),
                   "DQ@Au": slab() + Atoms("H3", [[0, 0, 6], [0, 0, 6.7], [0, 0, 7.4]]),
                   "THI_gas": h2(), "TBZ_gas": h2(0.1), "DQ_gas": h2(0.2)}
        for i, (k, a) in enumerate(systems.items()):
            mol = None if k == "Au" else k.split("@")[0].split("_")[0]
            if skip_dq and k == "DQ@Au":
                st.add(k, None, "not_available", "skipped for test", molecule=mol); continue
            os.environ["FAKE_PW_MODE"] = "crash" if (break_tbz and k == "TBZ@Au") else "ok"
            pps = {el: PPS[el] for el in set(a.get_chemical_symbols())}
            m = Q.run_pw(st.dir, k.replace("@", "_"), a, pps, 30, 240, (2, 2, 1) if "_gas" not in k else None,
                         "_gas" not in k, self.log)
            st.add(k, m, molecule=mol, mlip_eV=-float(i))
        os.environ["FAKE_PW_MODE"] = "ok"
        return st

    def raw(self, name):
        return Q.check_output(f"results/dft_check/{name}.out")["energy_total_eV"]

    def test_tables_match_raw_outputs(self):
        st = self.make_check_stage(); st.finish()
        rows = {(r["quantity"], r["method"]): r for r in csv.DictReader(open("results/tables/check.csv"))}
        for m, gs in (("THI", -0.4), ("TBZ", -0.2), ("DQ", -0.8)):
            vac = self.raw(f"{m}_Au") - self.raw("Au") - self.raw(f"{m}_gas")
            self.assertAlmostEqual(float(rows[(f"E_ads_vac {m}", "DFT")]["value_eV"]), vac, places=3)
            self.assertAlmostEqual(float(rows[(f"E_ads_aq {m}", "DFT")]["value_eV"]), vac - gs, places=3)
        self.assertTrue(all(r["status"] == "completed" for r in rows.values()))
        calc = list(csv.DictReader(open("results/tables/calculations.csv")))
        self.assertEqual(len([r for r in calc if r["status"] == "completed"]), 7)
        self.assertEqual(json.load(open("results/tables/schema.json"))["schema_version"], aggregate.SCHEMA_VERSION)

    def test_missing_and_incomplete_are_not_zero(self):
        st = self.make_check_stage(skip_dq=True, break_tbz=True)
        with self.assertRaises(SystemExit) as cm: st.finish()          # failed run → non-zero exit
        self.assertEqual(cm.exception.code, 1)
        rows = {(r["quantity"], r["method"]): r for r in csv.DictReader(open("results/tables/check.csv"))}
        dq, tbz = rows[("E_ads_aq DQ", "DFT")], rows[("E_ads_aq TBZ", "DFT")]
        self.assertEqual((dq["value_eV"], dq["status"]), ("", "not_available"))
        self.assertEqual((tbz["value_eV"], tbz["status"]), ("", "failed"))
        self.assertEqual(rows[("dE_aq THI<-TBZ", "DFT")]["value_eV"], "")
        self.assertNotEqual(rows[("E_ads_aq THI", "DFT")]["value_eV"], "")

    def test_killed_output_is_incomplete(self):
        st = self.make_check_stage(); st.finish()
        txt = open("results/dft_check/Au.out").read().replace("JOB DONE.", "")
        open("results/dft_check/Au.out", "w").write(txt)
        s = aggregate.build("results", log=self.log)
        self.assertEqual(s["check"].get("incomplete"), 1)

    def test_partial_run_keeps_previous_items(self):
        st = Q.Stage("coverage", "dft_coverage", "log.txt"); st.add("3x3/Au", None, "not_available", "x", cell="3x3"); st.finish()
        st = Q.Stage("coverage", "dft_coverage", "log.txt"); st.add("5x5/Au", None, "not_available", "y", cell="5x5"); st.finish()
        keys = [it["key"] for it in json.load(open("results/dft_coverage/record.json"))["items"]]
        self.assertEqual(sorted(keys), ["3x3/Au", "5x5/Au"])


if __name__ == "__main__":
    unittest.main()
