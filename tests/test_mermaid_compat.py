from __future__ import annotations

import re
from pathlib import Path
import unittest


class MermaidCompatTests(unittest.TestCase):
    def test_sequence_participant_ids_avoid_mermaid_8_8_reserved_words(self) -> None:
        reserved = {
            "alt",
            "and",
            "critical",
            "else",
            "end",
            "loop",
            "opt",
            "par",
            "rect",
        }
        failures: list[str] = []
        for path in Path("docs").rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            for block_match in re.finditer(r"```mermaid\n(.*?)```", text, re.DOTALL):
                block = block_match.group(1)
                if not block.lstrip().startswith("sequenceDiagram"):
                    continue
                start_line = text[: block_match.start()].count("\n") + 1
                for line_offset, line in enumerate(block.splitlines(), start=0):
                    participant_match = re.match(r"\s*participant\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
                    if not participant_match:
                        continue
                    participant_id = participant_match.group(1)
                    if participant_id.lower() in reserved:
                        failures.append(
                            f"{path}:{start_line + line_offset}: participant id "
                            f"{participant_id!r} conflicts with Mermaid 8.8.0 syntax"
                        )

        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
