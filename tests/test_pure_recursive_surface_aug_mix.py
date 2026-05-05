from pathlib import Path
import importlib.util
import unittest


def _load_module():
    path = Path("scripts/226_build_pure_recursive_surface_aug_mix.py")
    spec = importlib.util.spec_from_file_location("pure_recursive_surface_aug_mix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PureRecursiveSurfaceAugMixTests(unittest.TestCase):
    def test_build_mix_contains_original_and_ood_surface_rows(self):
        module = _load_module()

        rows = module.build_surface_aug_mix(
            cases_per_family=1,
            original_start_index=5000,
            surface_start_index=9000,
        )

        self.assertEqual(len(rows), 8)
        distributions = {row["surface_distribution"] for row in rows}
        self.assertEqual(distributions, {"canonical_surface", "ood_surface_paraphrase_v1"})
        self.assertTrue(all(row.get("chosen") or row.get("answer") for row in rows))
        self.assertTrue(all(row.get("solver_trace") for row in rows))

    def test_build_mix_has_no_eval_ood_id_overlap_by_default(self):
        module = _load_module()

        rows = module.build_surface_aug_mix(cases_per_family=2)

        self.assertFalse(any("-8000" in row["id"] or "-8001" in row["id"] for row in rows))


if __name__ == "__main__":
    unittest.main()
