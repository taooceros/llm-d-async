"""Mid-band (W7) job model, separate full/partial switch costs, a0H override, W7 rate-table reader."""
import gzip
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_long_job_model import ROOT, bcm, lrt  # noqa: E402

R = {"a0S": 200.0, "a0H": 300.0, "t7S": 500.0, "t7H": 350.0, "t6H": 400.0}
C, O = 100.0, 30.0
TABLE = str(ROOT / "results/bulk_cost_model/throughput_table_fullbf16.json")
LTABLE = str(ROOT / "results/bulk_cost_model/throughput_table_long_fullbf16.json")


def run(*args):
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "o.json"
        buf = io.StringIO()
        with redirect_stdout(buf):
            bcm.main(["--table", TABLE, "--long-table", LTABLE, "--json", str(out), *args])
        return json.loads(out.read_text()), buf.getvalue()


class Arms(unittest.TestCase):
    def test_formulas(self):
        m = bcm.mid_arms(R, 2.0, 3.0, 0.0, C, O)
        self.assertAlmostEqual(m["S"], 2 * 200 + 3 * 500 - O)
        self.assertAlmostEqual(m["H"], 2 * 300 + 3 * 350 - O)
        self.assertAlmostEqual(m["h7"], 2 * 200 + C + 3 * 350)
        self.assertIsNone(m["h6"])
        m = bcm.mid_arms(R, 2.0, 3.0, 1.0, C, O)
        self.assertIsNone(m["S"])  # W6 does not fit a TP8
        self.assertAlmostEqual(m["H"], 2 * 300 + 3 * 350 + 400 - O)
        self.assertAlmostEqual(m["h7"], 2 * 200 + C + 3 * 350 + 400)
        self.assertAlmostEqual(m["h6"], 2 * 200 + 3 * 500 + C + 400)

    def test_overlap_only_with_two_phases(self):
        self.assertAlmostEqual(bcm.mid_arms(R, 0.0, 2.0, 0.0, C, O)["S"], 2 * 500)

    def test_k_m_star(self):
        be = bcm.mid_breakevens(R, C, O)
        self.assertAlmostEqual(be["k_m_star"], (C + O) / (500 - 350))
        km = be["k_m_star"]
        for ks, sign in ((1.0, 1), (1.0, -1)):
            k = km + sign * 0.01
            m = bcm.mid_arms(R, ks, k, 0.0, C, O)
            self.assertEqual(m["h7"] < m["S"], sign > 0)
        self.assertAlmostEqual(be["k_s_star"], (C + O) / (300 - 200))
        self.assertIsNone(bcm.mid_breakevens({**R, "t7H": 600.0}, C, O)["k_m_star"])

    def test_winner_never_infeasible_static_s(self):
        for ks in (0.0, 1.0, 5.0):
            for km in (0.0, 1.0, 5.0):
                for kl in (1.0, 2.0):
                    self.assertNotEqual(bcm.mid_winner(R, ks, km, kl, C, O)[0], "S")

    def test_placeholders_tagged(self):
        tab = bcm.load_table(TABLE)
        r, src, ph = bcm.mid_rates(tab, bcm.load_table(LTABLE), None, True)
        self.assertEqual((r["t7S"], r["t7H"]), (300.0, 240.0))
        self.assertEqual(ph, ["t7S", "t7H"])
        self.assertIn("PLACEHOLDER", src["t7S"])
        with self.assertRaises(ValueError):
            bcm.mid_rates(tab, None, None, False)


class SwitchCosts(unittest.TestCase):
    def test_long_arms_route_c_partial_to_s_to_m_only(self):
        r = {"a0S": 200.0, "a0H": 300.0, "a0M": 400.0, "t6H": 400.0, "t6M": 800.0}
        base = bcm.long_arms(r, 3.0, 2.0, 150.0, O)
        split = bcm.long_arms(r, 3.0, 2.0, 150.0, O, 120.0)
        self.assertEqual(split["h"], base["h"])
        self.assertEqual(split["H"], base["H"])
        self.assertAlmostEqual(split["m"], bcm.s_to_m(r, 3.0, 2.0, 120.0)["makespan"])
        self.assertNotEqual(split["m"], base["m"])

    def test_cli_c_full_c_partial_recorded(self):
        out, _ = run("--mid-placeholder", "--c-full", "150", "--c-partial", "120",
                     "--c-full-src", "full src", "--c-partial-src", "part src")
        for job in ("long_job", "mid_job"):
            self.assertEqual(out[job]["measured_C"]["c_full_s"], 150.0)
            self.assertEqual(out[job]["measured_C"]["c_partial_s"], 120.0)
            self.assertEqual(out[job]["measured_C"]["c_partial_source"], "part src")
            self.assertEqual(list(out[job]["C_values_s"]), ["measured_full"])
        cell = out["mid_job"]["results"]["0"]["measured_full"]
        self.assertEqual(cell["C_s"], 150.0)
        self.assertAlmostEqual(cell["break_even"]["k_m_star"], 150.0 / (300.0 - 240.0))
        self.assertEqual(out["long_job"]["results"]["0"]["measured_full"]["C_partial_s"], 120.0)

    def test_a0h_override_applied_to_both_jobs(self):
        out, _ = run("--mid-placeholder", "--a0h-override", "250", "--a0h-src", "P4")
        self.assertEqual(out["long_job"]["inputs"]["a0H"]["value"], 250.0)
        self.assertEqual(out["mid_job"]["inputs"]["a0H"]["value"], 250.0)
        self.assertIn("P4", out["mid_job"]["inputs"]["a0H"]["source"])
        self.assertEqual(out["a0h_override"]["applied_to"], ["long_job", "mid_job"])

    def test_default_output_unchanged(self):
        plain, _ = run()
        self.assertNotIn("mid_job", plain)
        self.assertNotIn("a0h_override", plain)
        again, _ = run("--c-full", "160.5")
        for k in plain:
            if k != "long_job":
                self.assertEqual(plain[k], again[k], k)
        withlong, _ = run()
        self.assertEqual(plain, withlong)
        committed = json.loads((ROOT / "results/bulk_cost_model/model_long_fullbf16.json").read_text())
        self.assertEqual(plain["long_job"], committed["long_job"])
        self.assertNotIn("measured_C", plain["long_job"])


class W7Table(unittest.TestCase):
    def test_reads_runs_and_writes_mid_table(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "data/manifests").mkdir(parents=True)
            (d / "pilot").mkdir()
            for name, tok, req in (("W7", 1000, 4), ("W0", 500, 2)):
                (d / f"data/manifests/{name}.json").write_text(
                    json.dumps({"useful_output_tokens": tok, "num_requests": req}))

            def result(rid, wl, arm, span, tok, req, gz=False):
                body = json.dumps({"measurements": {"result": {
                    "all_requests_complete": True, "useful_output_tokens": tok, "num_requests": req,
                    "makespan_s": span}}})
                path = d / "pilot" / f"{rid}_{wl}_{arm}.json"
                if gz:
                    gzip.open(str(path) + ".gz", "wt").write(body)
                else:
                    path.write_text(body)
            result("w7-abc-plain-r1", "W7", "S", 300.0, 1000, 4)
            result("w7-abc-plain-r2", "W7", "S", 310.0, 1000, 4, gz=True)
            result("w7-abc-plain-r1", "W7", "H", 240.0, 1000, 4)
            result("w0h131k-abc-plain-r1", "W0", "H", 320.0, 500, 2)
            out = d / "mid.json"
            args = ["--repo-root", str(d), "--pilot-dir", str(d / "pilot"), "--out", str(out),
                    "--w7-s-runs", "w7-abc-plain-r1,w7-abc-plain-r2", "--w7-h-runs", "w7-abc-plain-r1",
                    "--w0h131k-runs", "w0h131k-abc-plain-r1"]
            self.assertEqual(lrt.main(args), 0)
            t = json.loads(out.read_text())
            self.assertEqual(t["W7"]["makespan_s"]["S"]["value"], 305.0)
            self.assertEqual(t["W7"]["makespan_s"]["H"]["value"], 240.0)
            self.assertEqual(t["W0"]["makespan_s"]["H_131k"]["value"], 320.0)
            tab = bcm.load_table(str(out))
            self.assertEqual(tab["W7"]["reps"]["S"], [300.0, 310.0])
            # token mismatch is refused
            result("w7-abc-plain-r3", "W7", "S", 1.0, 999, 4)
            out2 = d / "bad.json"
            self.assertEqual(lrt.main(["--repo-root", str(d), "--pilot-dir", str(d / "pilot"), "--out", str(out2),
                                       "--w7-s-runs", "w7-abc-plain-r3"]), 1)
            self.assertFalse(out2.exists())


if __name__ == "__main__":
    unittest.main()
