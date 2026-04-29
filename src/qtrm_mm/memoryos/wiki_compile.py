from __future__ import annotations
import argparse
from pathlib import Path
from .chunk import iter_text_files


def compile_stub(input_dir: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in iter_text_files(input_dir):
        text = p.read_text(errors="ignore")[:2000]
        page = out / (p.stem.replace(" ", "_") + ".md")
        page.write_text(
            f"# {p.stem}\n\n"
            f"Source: `{p}`\n\n"
            "## Summary Stub\n\n"
            f"{text}\n\n"
            "## Claims\n\n"
            "- [ ] TODO: extract source-backed claims.\n",
            encoding="utf-8",
        )
    print(f"compiled wiki stubs to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    compile_stub(args.input_dir, args.out_dir)


if __name__ == "__main__":
    main()
