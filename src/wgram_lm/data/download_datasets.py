from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "\n".join(_safe_text(v) for v in x)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _truncate(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def _iter_limited(ds, limit: int):
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        yield i, ex


def _load_streaming(repo: str, config: Optional[str], split: str):
    if config:
        return load_dataset(repo, config, split=split, streaming=True)
    return load_dataset(repo, split=split, streaming=True)


def _text_from_smollm(ex: Dict[str, Any]) -> str:
    if "text" in ex:
        return _safe_text(ex["text"])
    if "prompt" in ex and "text" in ex:
        return _safe_text(ex["prompt"]) + "\n" + _safe_text(ex["text"])
    return _safe_text(ex)


def _math_text(ex: Dict[str, Any], source: str) -> str:
    # Works across NuminaMath and OpenR1-style rows.
    problem = ex.get("problem") or ex.get("question") or ex.get("prompt") or ""
    solution = ex.get("solution") or ex.get("answer") or ""
    if not solution and isinstance(ex.get("messages"), list):
        parts = []
        for m in ex["messages"]:
            role = m.get("role") or m.get("from") or ""
            content = m.get("content") or m.get("value") or ""
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    if not solution and isinstance(ex.get("generations"), list) and ex["generations"]:
        solution = _safe_text(ex["generations"][0])
    return f"Problem:\n{_safe_text(problem)}\n\nSolution:\n{_safe_text(solution)}"


def download_text(out_dir: Path, max_samples: int, max_chars: int, profile: str):
    rows: List[Dict[str, Any]] = []
    sources = []
    if profile in {"smoke", "4090", "dgx", "full"}:
        sources.append(("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", max_samples // 2))
    if profile in {"4090", "dgx", "full"}:
        sources.append(("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "train", max_samples // 2))
    if profile == "full":
        sources.append(("HuggingFaceTB/smollm-corpus", "python-edu", "train", max_samples // 4))

    for repo, cfg, split, n in sources:
        n = max(1, n)
        print(f"[text] {repo}/{cfg} split={split} n={n}")
        try:
            ds = _load_streaming(repo, cfg, split)
            for _, ex in tqdm(_iter_limited(ds, n), total=n):
                text = _truncate(_text_from_smollm(ex), max_chars)
                if len(text) >= 40:
                    rows.append({"type": "text", "source": f"{repo}:{cfg}", "text": text})
        except Exception as e:
            print(f"[warn] failed {repo}/{cfg}: {e}")
    _write_jsonl(out_dir / "text_train.jsonl", rows)
    print(f"[text] wrote {len(rows)} rows -> {out_dir/'text_train.jsonl'}")


def download_math(out_dir: Path, max_samples: int, max_chars: int, profile: str):
    rows: List[Dict[str, Any]] = []
    sources = [
        ("AI-MO/NuminaMath-CoT", None, "train", max_samples // 2),
        ("open-r1/OpenR1-Math-220k", "default", "train", max_samples // 2),
    ]
    for repo, cfg, split, n in sources:
        n = max(1, n)
        print(f"[math] {repo}/{cfg or ''} split={split} n={n}")
        try:
            ds = _load_streaming(repo, cfg, split)
            for _, ex in tqdm(_iter_limited(ds, n), total=n):
                text = _truncate(_math_text(ex, repo), max_chars)
                if len(text) >= 40:
                    rows.append({"type": "math", "source": f"{repo}:{cfg or 'default'}", "text": text})
        except Exception as e:
            print(f"[warn] failed {repo}/{cfg}: {e}")
    _write_jsonl(out_dir / "math_train.jsonl", rows)
    print(f"[math] wrote {len(rows)} rows -> {out_dir/'math_train.jsonl'}")


def _save_image(img: Any, path: Path) -> Optional[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(img, Image.Image):
            im = img.convert("RGB")
        else:
            im = Image.open(img).convert("RGB")
        im.save(path, format="JPEG", quality=90)
        return str(path)
    except Exception as e:
        print(f"[warn] image save failed {path}: {e}")
        return None


def _cauldron_prompt_answer(ex: Dict[str, Any]) -> tuple[str, str]:
    texts = ex.get("texts") or []
    if isinstance(texts, list) and texts:
        # The Cauldron format is list of turn dicts with user/assistant/source.
        user_parts, assistant_parts = [], []
        for t in texts:
            if not isinstance(t, dict):
                continue
            if t.get("user"):
                user_parts.append(_safe_text(t.get("user")))
            if t.get("assistant"):
                assistant_parts.append(_safe_text(t.get("assistant")))
        return "\n".join(user_parts), "\n".join(assistant_parts)
    return _safe_text(ex.get("question") or ex.get("prompt") or ex), _safe_text(ex.get("answer") or ex.get("solution") or "")


def download_cauldron(out_dir: Path, configs: List[str], per_config: int, max_chars: int):
    rows: List[Dict[str, Any]] = []
    img_root = out_dir / "images" / "cauldron"
    for cfg in configs:
        cfg = cfg.strip()
        if not cfg:
            continue
        print(f"[mm:cauldron] {cfg} n={per_config}")
        try:
            ds = load_dataset("HuggingFaceM4/the_cauldron", cfg, split="train", streaming=True)
            for i, ex in tqdm(_iter_limited(ds, per_config), total=per_config):
                image_paths = []
                imgs = ex.get("images") or []
                if not isinstance(imgs, list):
                    imgs = [imgs]
                for j, img in enumerate(imgs[:4]):
                    p = _save_image(img, img_root / cfg / f"{i:07d}_{j}.jpg")
                    if p:
                        image_paths.append(p)
                prompt, answer = _cauldron_prompt_answer(ex)
                prompt = _truncate(prompt, max_chars)
                answer = _truncate(answer, max_chars)
                if prompt and answer and image_paths:
                    rows.append({
                        "type": "multimodal_sft",
                        "source": f"HuggingFaceM4/the_cauldron:{cfg}",
                        "images": image_paths,
                        "prompt": prompt,
                        "answer": answer,
                        "text": f"<image>\n{prompt}\n\n{answer}",
                    })
        except Exception as e:
            print(f"[warn] failed Cauldron/{cfg}: {e}")
    _write_jsonl(out_dir / "mm_train.jsonl", rows)
    print(f"[mm] appended/wrote {len(rows)} Cauldron rows -> {out_dir/'mm_train.jsonl'}")


def download_fallback_mm(out_dir: Path, per_dataset: int, max_chars: int):
    rows: List[Dict[str, Any]] = []
    img_root = out_dir / "images" / "fallback"
    fallback = [
        ("lmms-lab/ScienceQA-IMG", None, "train"),
        ("lmms-lab/ChartQA", None, "test"),
    ]
    for repo, cfg, split in fallback:
        print(f"[mm:fallback] {repo} split={split} n={per_dataset}")
        try:
            ds = _load_streaming(repo, cfg, split)
            for i, ex in tqdm(_iter_limited(ds, per_dataset), total=per_dataset):
                image = ex.get("image")
                if image is None:
                    continue
                p = _save_image(image, img_root / repo.replace('/', '_') / f"{i:07d}.jpg")
                if not p:
                    continue
                if "choices" in ex and isinstance(ex.get("choices"), list):
                    answer_idx = ex.get("answer")
                    choices = ex.get("choices") or []
                    answer = choices[answer_idx] if isinstance(answer_idx, int) and answer_idx < len(choices) else _safe_text(answer_idx)
                    prompt = f"Question: {ex.get('question','')}\nChoices: {', '.join(map(str, choices))}\nHint: {ex.get('hint','')}"
                else:
                    prompt = _safe_text(ex.get("question") or ex.get("query") or "")
                    answer = _safe_text(ex.get("answer") or ex.get("label") or "")
                prompt = _truncate(prompt, max_chars)
                answer = _truncate(answer, max_chars)
                if prompt and answer:
                    rows.append({
                        "type": "multimodal_sft",
                        "source": repo,
                        "images": [p],
                        "prompt": prompt,
                        "answer": answer,
                        "text": f"<image>\n{prompt}\n\n{answer}",
                    })
        except Exception as e:
            print(f"[warn] failed fallback {repo}: {e}")
    _write_jsonl(out_dir / "mm_train.jsonl", rows)
    print(f"[mm] appended/wrote {len(rows)} fallback rows -> {out_dir/'mm_train.jsonl'}")


def build_docs(out_dir: Path):
    docs = out_dir.parent / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    # Create a tiny seed corpus from downloaded jsonl so MemoryOS can be built immediately.
    corpus_md = docs / "downloaded_dataset_seed_corpus.md"
    parts = ["# Downloaded Dataset Seed Corpus\n"]
    for name in ["text_train.jsonl", "math_train.jsonl", "mm_train.jsonl"]:
        p = out_dir / name
        if not p.exists():
            continue
        parts.append(f"\n## {name}\n")
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 25:
                    break
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                txt = row.get("text") or row.get("prompt") or ""
                parts.append(f"- Source: {row.get('source','unknown')}\n  Excerpt: {_truncate(txt, 500)}\n")
    corpus_md.write_text("\n".join(parts), encoding="utf-8")
    print(f"[docs] wrote {corpus_md}")


def main():
    ap = argparse.ArgumentParser(description="Download and normalize text/math/multimodal datasets for QTRM-MemoryOS.")
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--profile", default=os.environ.get("PROFILE", "smoke"), choices=["smoke", "4090", "dgx", "full"])
    ap.add_argument("--text-samples", type=int, default=int(os.environ.get("TEXT_SAMPLES", "0")))
    ap.add_argument("--math-samples", type=int, default=int(os.environ.get("MATH_SAMPLES", "0")))
    ap.add_argument("--mm-samples-per-config", type=int, default=int(os.environ.get("MM_SAMPLES_PER_CONFIG", "0")))
    ap.add_argument("--cauldron-configs", default=os.environ.get("CAULDRON_CONFIGS", "scienceqa,ai2d,chartqa,docvqa,textvqa"))
    ap.add_argument("--include-fallbacks", action="store_true", default=os.environ.get("INCLUDE_FALLBACKS", "1") == "1")
    ap.add_argument("--max-chars", type=int, default=int(os.environ.get("MAX_CHARS", "6000")))
    args = ap.parse_args()

    defaults = {
        "smoke": (400, 200, 40),
        "4090": (20000, 4000, 1000),
        "dgx": (100000, 20000, 5000),
        "full": (500000, 100000, 20000),
    }
    t_def, m_def, mm_def = defaults[args.profile]
    text_samples = args.text_samples or t_def
    math_samples = args.math_samples or m_def
    mm_samples = args.mm_samples_per_config or mm_def

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Start fresh unless APPEND_DATA=1.
    if os.environ.get("APPEND_DATA", "0") != "1":
        for name in ["text_train.jsonl", "math_train.jsonl", "mm_train.jsonl"]:
            try:
                (out_dir / name).unlink()
            except FileNotFoundError:
                pass

    print(f"[profile] {args.profile}; text={text_samples} math={math_samples} mm/config={mm_samples}")
    download_text(out_dir, text_samples, args.max_chars, args.profile)
    download_math(out_dir, math_samples, args.max_chars, args.profile)
    configs = [c.strip() for c in args.cauldron_configs.split(",") if c.strip()]
    download_cauldron(out_dir, configs, mm_samples, args.max_chars)
    if args.include_fallbacks:
        download_fallback_mm(out_dir, max(10, mm_samples // 2), args.max_chars)
    build_docs(out_dir)
    print("[done] dataset download/normalization complete")


if __name__ == "__main__":
    main()
