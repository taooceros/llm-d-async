"""Error magnitudes of the Triton RPA vs the float32 reference, plus a negative control
(wrong sliding window) showing the test tolerances discriminate. Run like the tests."""
import numpy as np, sys
sys.path.insert(0, "tests")
import test_rpa_triton as t
res = {}
for shape, dims, win in (("sliding", t.GEMMA4_SLIDING, 1024), ("full", t.GEMMA4_FULL, None)):
    for batch in t.BATCHES:
        case = t.build(0, t.BATCHES[batch], **dims, page_size=16, max_num_seqs=16, max_tokens=1536, max_len=4096, garbage=False)
        out, _ = t.run_case(case, sliding_window=win)
        ref, _ = t.run_ref(case, sliding_window=win)
        n = case["total"]; e = np.abs(out[:n] - ref)
        res[f"{shape}/{batch}"] = (float(e.max()), float(e.mean()), float(np.abs(ref).mean()))
case = t.build(0, t.BATCHES["decode"], **t.GEMMA4_SLIDING, page_size=16, max_num_seqs=16, max_tokens=1536, max_len=4096, garbage=False)
out, _ = t.run_case(case, sliding_window=None)          # wrong window on purpose
ref, _ = t.run_ref(case, sliding_window=1024)
e = np.abs(out[:case["total"]] - ref)
res["NEGATIVE window None vs 1024 (must be large)"] = (float(e.max()), float(e.mean()), float(np.abs(ref).mean()))
for k, v in res.items(): print(f"ERR {k}: max {v[0]:.4f} mean {v[1]:.5f} |ref| mean {v[2]:.4f}")
