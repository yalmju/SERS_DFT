"""
Environment check: one tiny real pw.x calculation (H2 in a 6 Å box, Γ-point, low cutoff) with the pinned
SSSP H pseudopotential and QE built-in D3. Skipped if pw.x/mpirun or pseudo/ with the SSSP set is missing.

    python -m unittest discover -s tests -p test_env_smoke.py -v      (from the repository root)
"""
import os, shutil, sys, tempfile, unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


@unittest.skipUnless(shutil.which("pw.x") and shutil.which("mpirun") and
                     os.path.exists(os.path.join(REPO, "pseudo", "SSSP_1.0_PBE_efficiency.json")),
                     "Quantum ESPRESSO or pseudo/SSSP_1.0_PBE_efficiency.json not available")
class TestSmoke(unittest.TestCase):
    def test_tiny_pw_run(self):
        os.environ["DFT_NP"] = os.environ.get("DFT_NP", "1")
        cwd = os.getcwd(); os.chdir(REPO)
        try:
            import qe_tools as Q
            from ase import Atoms
            table = Q.load_sssp()
            pps, _, _ = Q.pp_info(table, ["H"])
            a = Atoms("H2", positions=[[3, 3, 3], [3, 3, 3.74]], cell=[6, 6, 6], pbc=True)
            with tempfile.TemporaryDirectory() as d:
                Q.SCRATCH = os.path.join(d, "scratch")
                m = Q.run_pw(os.path.join(d, "smoke"), "H2", a, pps, 25, 200, None, False, print)
                self.assertEqual(m["status"], "completed", m.get("attempts"))
                self.assertIsNotNone(m["energy"]["d3_eV"])
                print("QE", m["qe_version"], "| versions", m["versions"], "| E", m["energy"]["total_eV"], "eV")
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
