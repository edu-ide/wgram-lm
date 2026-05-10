from pathlib import Path
import importlib.util
import json
import tempfile
import unittest


def load_module():
    path = Path("scripts/313_build_scalar_arithmetic_codec_gate.py")
    spec = importlib.util.spec_from_file_location("scalar_codec_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScalarArithmeticCodecGateBuilderTests(unittest.TestCase):
    def test_make_case_records_scalar_affine_trace_without_evidence(self):
        module = load_module()

        row = module.make_case(index=0, base_start=50000, variant=6)

        self.assertEqual(row["task_family"], "scalar_affine_arithmetic")
        self.assertEqual(row["base_value"], 50000)
        self.assertEqual(row["scalar_coeff"], 4)
        self.assertEqual(row["depth_targets"]["1"], str(4 * 50000 + row["scalar_initial_residual"]))
        self.assertEqual(row["depth_targets"]["2"], row["answer"])
        self.assertEqual(row["transition_finality_targets"]["2"], 1)
        self.assertEqual(row["evidence"], [])

    def test_main_writes_train_eval_splits(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train.jsonl"
            eval_path = Path(tmp) / "eval.jsonl"

            args = module.build_arg_parser().parse_args(
                [
                    "--train-out",
                    str(train),
                    "--eval-out",
                    str(eval_path),
                    "--train-cases",
                    "2",
                    "--eval-cases",
                    "1",
                    "--train-variants",
                    "0,1",
                    "--eval-variants",
                    "6",
                ]
            )
            module.write_jsonl(
                args.train_out,
                module.build_rows(
                    cases=args.train_cases,
                    base_start=args.train_base_start,
                    variants=module.parse_variants(args.train_variants),
                ),
            )
            module.write_jsonl(
                args.eval_out,
                module.build_rows(
                    cases=args.eval_cases,
                    base_start=args.eval_base_start,
                    variants=module.parse_variants(args.eval_variants),
                ),
            )

            train_rows = [json.loads(line) for line in train.read_text().splitlines()]
            eval_rows = [json.loads(line) for line in eval_path.read_text().splitlines()]

        self.assertEqual(len(train_rows), 4)
        self.assertEqual(len(eval_rows), 1)
        self.assertEqual({row["surface_variant_index"] for row in train_rows}, {0, 1})
        self.assertEqual(eval_rows[0]["surface_variant_index"], 6)


if __name__ == "__main__":
    unittest.main()
