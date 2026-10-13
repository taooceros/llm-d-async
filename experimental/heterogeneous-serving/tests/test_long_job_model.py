"""Long-context job model (scripts/bulk_cost_model.py) and the long-table record join."""
import gzip
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bcm = load("bulk_cost_model")
lrt = load("long_rate_table")

R = {"a0S": 200.0, "a0H": 300.0, "a0M": 400.0, "t6H": 400.0, "t6M": 800.0}
C, O = 100.0, 30.0


class Arms(unittest.TestCase):
    def test_formulas(self):
        ks, kl = 3.0, 2.0
        m = bcm.long_arms(R, ks, kl, C, O)
        self.assertAlmostEqual(m["H"], 3 * 300 + 2 * 400 - O)
        self.assertAlmostEqual(m["M"], max(3 * 400, 2 * 800))
        self.assertAlmostEqual(m["h"], 3 * 200 + C + 2 * 400)
        tau = (3 * 400 - C - 2 * 800) / 2  # (ks a0M - C - kl t6M)/rho < 0 -> clipped
        self.assertLess(tau, 0)
        self.assertAlmostEqual(m["m"], C + 2 * 800)

    def test_no_overlap_credit_without_both_phases(self):
        m = bcm.long_arms(R, 0.0, 2.0, C, O)
        self.assertAlmostEqual(m["H"], 2 * 400)
        self.assertAlmostEqual(m["h"], C + 2 * 400)
        self.assertAlmostEqual(m["M"], 2 * 800)


class SToM(unittest.TestCase):
    def test_closed_form_vs_bruteforce(self):
        rates = [R, {**R, "a0M": 380.0}, {**R, "a0M": 150.0}, {**R, "t6M": 300.0}]
        for r in rates:
            for ks in (0.0, 0.25, 1.0, 2.5, 8.0):
                for kl in (0.25, 1.0, 4.0):
                    for c in (45.0, 186.0):
                        got = bcm.s_to_m(r, ks, kl, c)["makespan"]
                        want = bcm.s_to_m_numeric(r, ks, kl, c) if ks > 0 else c + kl * r["t6M"]
                        self.assertAlmostEqual(got, want, delta=max(1e-6, ks * r["a0S"] * 2e-4 + 1e-3),
                                               msg=f"{r} ks={ks} kl={kl} c={c}")

    def test_tau_star_and_interior_value(self):
        ks, kl = 6.0, 1.0
        d = bcm.s_to_m(R, ks, kl, C)
        tau_star = (ks - (C + kl * R["t6M"]) / R["a0M"]) * R["a0S"]
        self.assertAlmostEqual(d["tau"], tau_star)
        self.assertAlmostEqual(d["t0"], d["t6"])
        want = ks * R["a0S"] + (C + kl * R["t6M"]) * (1 - R["a0S"] / R["a0M"])
        self.assertAlmostEqual(d["makespan"], want)
        self.assertLess(d["tau"], ks * R["a0S"])

    def test_beats_static_m_iff_threshold(self):
        kl = 2.0
        thr = (C + kl * R["t6M"]) / R["a0M"]
        for ks, expect in ((thr - 0.05, False), (thr + 0.05, True)):
            m = bcm.long_arms(R, ks, kl, C, O)
            self.assertEqual(m["m"] < m["M"], expect)

    def test_slower_pair_not_slower_than_s_never_helps(self):
        r = {**R, "a0M": 150.0}
        self.assertEqual(bcm.s_to_m(r, 4.0, 1.0, C)["tau"], 0.0)


class Regions(unittest.TestCase):
    def test_winners_hand_made(self):
        # tiny W0 -> static H; big W0, tiny W6 -> static M ... check against explicit arithmetic
        w, margin, ms = bcm.long_winner(R, 0.25, 0.25, C, O)
        self.assertEqual(w, "H")
        self.assertAlmostEqual(margin, sorted(ms.values())[1] - sorted(ms.values())[0])
        # W6-heavy long job with big W0: S->H wins (cheap W0 on S, then H)
        w, _, _ = bcm.long_winner(R, 8.0, 8.0, C, O)
        self.assertEqual(w, "h")
        # W0-heavy, little W6: S->M
        w, _, _ = bcm.long_winner(R, 8.0, 0.25, C, O)
        self.assertEqual(w, "m")

    def test_region_map_shape_and_letters(self):
        kss, kls = bcm.frange("0:2:0.5"), bcm.frange("0.5:2:0.5")
        mp = bcm.region_map(R, kss, kls, C, O)
        self.assertEqual(len(mp["ascii"]), 4)
        for row in mp["ascii"]:
            self.assertEqual(len(row.split()[-1]), 5)
            self.assertTrue(set(row.split()[-1]) <= set("HMhm"))
        self.assertEqual(mp["cells"]["0,0.5"]["winner"], "static H")

    def test_k_s_star_formula(self):
        res = bcm.long_breakevens(R, C, O, (1, 2), 8.0)
        k = res["S->H_vs_static_H_k_s_star"]
        self.assertAlmostEqual(k, (C + O) / (R["a0H"] - R["a0S"]))
        self.assertNotIn("S->H_vs_static_H_k_s_star_scenario_convention", res)
        for kl in (0.5, 4.0):  # independent of k_l: S->H - H changes sign exactly at k_s*
            lo = bcm.long_arms(R, k - 0.01, kl, C, O)
            hi = bcm.long_arms(R, k + 0.01, kl, C, O)
            self.assertGreater(lo["h"], lo["H"])
            self.assertLess(hi["h"], hi["H"])

    def test_static_m_vs_h_boundary(self):
        kl = 2.0
        res = bcm.long_breakevens(R, C, O, (kl,), 8.0)["static_M_vs_static_H_by_k_l"]["2"]
        piece = res["pieces"][0]
        self.assertEqual(piece["regime"], "W6-bound")
        x = piece["M_better_above_k_s"]
        self.assertAlmostEqual(x, (kl * (R["t6M"] - R["t6H"]) + O) / R["a0H"])
        for ks, expect in ((x - 0.01, False), (x + 0.01, True)):
            m = bcm.long_arms(R, ks, kl, C, O)
            self.assertEqual(m["M"] < m["H"], expect)

    def test_ranges_cover_axis(self):
        rg = bcm.winner_ranges(R, 2.0, C, O, 8.0)
        self.assertEqual(rg[0]["k_s"][0], 0.0)
        self.assertEqual(rg[-1]["k_s"][1], 8.0)
        for a, b in zip(rg, rg[1:]):
            self.assertAlmostEqual(a["k_s"][1], b["k_s"][0], places=3)


class Placeholders(unittest.TestCase):
    def tab(self):
        return {"W0": {"T": {"S": 190.8, "H": 306.4}, "sources": {"S": "s", "H": "h"}}}

    def test_placeholder_values_and_flags(self):
        r, src, ph = bcm.long_rates(self.tab(), None, True)
        self.assertEqual((r["t6H"], r["a0M"], r["t6M"]), (420.0, 2 * 190.8, 840.0))
        self.assertEqual(set(ph), {"t6H", "a0M", "t6M"})
        self.assertTrue(all("PLACEHOLDER, not measured" in src[k] for k in ph))

    def test_missing_without_flag_errors(self):
        with self.assertRaises(ValueError):
            bcm.long_rates(self.tab(), None, False)


class ExistingOutputs(unittest.TestCase):
    def test_validation_rows_and_keys(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "m.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                bcm.main(["--table", str(ROOT / "results/bulk_cost_model/throughput_table_fullbf16.json"),
                          "--json", str(out)])
            j = json.loads(out.read_text())
            self.assertNotIn("long_job", j)
            self.assertEqual(len(j["breakeven_validation"]), 10)
            self.assertLess(max(abs(r["res_k"]) for r in j["breakeven_validation"]), 0.01)
            for key in ("mix_validation", "calibration", "kv_capacity", "scenario",
                        "example_job_4W0_2W5_1W4", "regimes", "table", "validation_table"):
                self.assertIn(key, j)
            self.assertIn("break-even rows: 10; max |k residual| = 0.009", buf.getvalue())

    def test_long_job_key_with_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "m.json"
            with redirect_stdout(io.StringIO()):
                bcm.main(["--table", str(ROOT / "results/bulk_cost_model/throughput_table_fullbf16.json"),
                          "--long-placeholder", "--json", str(out)])
            lj = json.loads(out.read_text())["long_job"]
            self.assertTrue(lj["any_placeholder"])
            self.assertEqual(sorted(lj["placeholder_inputs"]), ["a0M", "t6H", "t6M"])
            self.assertEqual(set(lj["C_values_s"].values()), {160.5, 186.0, 80.0, 45.0})


class RecordJoin(unittest.TestCase):
    def make_root(self, d):
        m = pathlib.Path(d) / "data/manifests"
        m.mkdir(parents=True)
        part = lambda n, cls: {"workload": n, "requests": [
            {"request_id": f"{n}-{i}", "max_tokens": 5, **({"request_class": c} if c else {})}
            for i, c in enumerate(cls)]}
        (m / "A.json").write_text(json.dumps(part("A", ["short", "long"])))
        (m / "B.json").write_text(json.dumps(part("B", [None])))
        (m / "MIX.json").write_text(json.dumps({"workload": "MIX", "composite": [["A", 1], ["B", 2]],
                                                "num_requests": 4}))
        return d

    def test_join_composite_part_and_request_class(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_root(d)
            _, rows = lrt.expand_manifest("MIX", d)
            self.assertEqual([r["request_id"] for r in rows],
                             ["MIX-p0r0-A-0", "MIX-p0r0-A-1", "MIX-p1r0-B-0", "MIX-p1r1-B-0"])
            recs = [{"request_id": f"ns:{r['request_id']}", "canonical_request_id": r["request_id"],
                     "client_started_at_s": 100.0 + i, "client_finished_at_s": 110.0 + 10 * i}
                    for i, r in enumerate(rows)]
            fin, unl = lrt.class_finish_times(recs, lrt.LABELLERS["composite_part"](rows))
            self.assertEqual(unl, 0)
            self.assertEqual(fin["A"]["finish_s"], 20.0)   # max(110, 120) - 100
            self.assertEqual(fin["B"]["finish_s"], 40.0)   # 140 - 100
            fin, unl = lrt.class_finish_times(recs, lrt.LABELLERS["request_class"](rows))
            self.assertEqual((fin["short"]["finish_s"], fin["long"]["finish_s"], unl), (10.0, 20.0, 2))

    def test_fallback_to_prefixed_request_id(self):
        self.assertEqual(lrt.canonical_id({"request_id": "abc:MIX-p0r0-A-0"}), "MIX-p0r0-A-0")


if __name__ == "__main__":
    unittest.main()
