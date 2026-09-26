"""Check stage accounting and overlapping GPU interval handling without a model."""
import runpy
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
parse = runpy.run_path(str(ROOT / "scripts/benchmark-klein.py"))["parse_log"]
merge = runpy.run_path(str(ROOT / "scripts/analyze-metal-trace.py"))["merged"]


class AccountingTest(unittest.TestCase):
    def test_stages_do_not_double_count_nested_logs(self):
        rows = parse("""BENCH_BEGIN -1 1700000000.0
get_learned_condition completed, taking 4.10s
sampling completed, taking 10.00s
computing vae decode graph completed, taking 2.00s
latent 1 decoded, taking 2.00s
decode_first_stage completed, taking 2.00s
BENCH_END -1 16.20
BENCH_BEGIN 0 1700000017.0
encode_first_stage completed, taking 1.20s
get_learned_condition completed, taking 4.00s
sampling completed, taking 9.00s
decode_first_stage completed, taking 1.90s
BENCH_END 0 16.30""")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["decode"], 2)
        self.assertEqual(rows[1]["reference_encode"], 1.2)
        self.assertEqual(rows[1]["started_unix"], 1700000017)

    def test_mlx_stage_accounting(self):
        rows = parse("""BENCH_BEGIN 0
MLX text encoding: 1.500s
MLX reference encoding: 0.100s
MLX sampling: 3.000s
MLX VAE decoding: 0.200s
BENCH_END 0 4.810""")
        self.assertEqual(rows[0]["text"],1.5)
        self.assertEqual(rows[0]["decode"],0.2)
        self.assertEqual(rows[0]["reference_encode"],0.1)

    def test_incomplete_run_rejected(self):
        with self.assertRaises(ValueError):
            parse("BENCH_BEGIN 0\nsampling completed, taking 1.00s")

    def test_gpu_overlap_is_not_counted_twice(self):
        self.assertEqual(merge([(2, 4), (1, 3), (1, 2), (4, 5), (7, 8)]), [[1, 5], [7, 8]])


if __name__ == "__main__":
    unittest.main()
