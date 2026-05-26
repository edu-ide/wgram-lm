from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "579_audit_decision_archive_ordering.py"


def load_module():
    spec = importlib.util.spec_from_file_location("decision_archive_ordering_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DecisionArchiveOrderingAuditTests(unittest.TestCase):
    def test_audit_finds_unordered_files_broken_links_and_dry_run_renames(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decisions = root / "docs" / "wiki" / "decisions"
            decisions.mkdir(parents=True)
            (decisions / "0001-active-decision-index.md").write_text(
                "active: 2026-05-25-stage101-active.md\n",
                encoding="utf-8",
            )
            (decisions / "2026-05-25-stage101-active.md").write_text("# Active\n", encoding="utf-8")
            (decisions / "asi-sufficiency-gate-2026-05-02.md").write_text("# Old Dated\n", encoding="utf-8")
            (decisions / "legacy-reject.md").write_text("# Legacy\n", encoding="utf-8")
            index = root / "docs" / "wiki" / "index.md"
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(
                "\n".join(
                    [
                        "[ok](decisions/2026-05-25-stage101-active.md)",
                        "[missing](decisions/missing.md)",
                    ]
                ),
                encoding="utf-8",
            )

            report = module.audit_decision_archive(
                root=root,
                decisions_dir=decisions,
                scan_dirs=[root / "docs"],
            )

        self.assertEqual(report["decision_file_count"], 4)
        self.assertIn("legacy-reject.md", report["unordered_files"])
        self.assertEqual(
            report["rename_candidates"]["asi-sufficiency-gate-2026-05-02.md"],
            "2026-05-02-asi-sufficiency-gate.md",
        )
        self.assertEqual(
            report["rename_candidates"]["legacy-reject.md"],
            "archive-unknown-legacy-reject.md",
        )
        self.assertEqual(report["broken_links"][0]["target"], "decisions/missing.md")

    def test_main_writes_json_report(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decisions = root / "docs" / "wiki" / "decisions"
            decisions.mkdir(parents=True)
            (decisions / "legacy.md").write_text("# Legacy\n", encoding="utf-8")
            out = root / "report.json"
            args = module.build_arg_parser().parse_args(
                [
                    "--root",
                    str(root),
                    "--decisions-dir",
                    str(decisions),
                    "--scan-dir",
                    str(root / "docs"),
                    "--out-json",
                    str(out),
                ]
            )
            report = module.run(args)
            self.assertTrue(out.exists())
            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(loaded["decision_file_count"], report["decision_file_count"])


if __name__ == "__main__":
    unittest.main()
