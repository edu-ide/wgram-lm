from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def iter_images(root: str | Path):
    root = Path(root)
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield p


class ReferenceVisualEmbedder:
    """Small CLIP/SigLIP-style adapter using Transformers.

    Default model is intentionally configurable. For production, replace with
    Qwen donor vision embeddings or SigLIP2-class embeddings.
    """

    def __init__(self, model_id: str = "google/siglip-base-patch16-224"):
        from transformers import AutoProcessor, AutoModel
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        if __import__("torch").cuda.is_available():
            self.model.cuda()

    def encode_images(self, paths: list[Path]):
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                emb = self.model.get_image_features(**inputs)
            else:
                out = self.model(**inputs)
                emb = coerce_embedding_tensor(out)
            emb = coerce_embedding_tensor(emb)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        return emb.cpu().numpy().astype("float32")


def coerce_embedding_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = value[-1]
        if torch.is_tensor(value):
            if attr == "last_hidden_state":
                return value.mean(dim=1)
            return value
    if isinstance(output, (list, tuple)):
        for value in output:
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Could not extract tensor embedding from {type(output).__name__}")


def build_visual_index(input_dir: str, out_dir: str, model_id: str = "google/siglip-base-patch16-224"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = list(iter_images(input_dir))
    if not paths:
        raise RuntimeError(f"No image files found in {input_dir}")
    embedder = ReferenceVisualEmbedder(model_id)
    all_emb = []
    batch = 8
    for i in range(0, len(paths), batch):
        all_emb.append(embedder.encode_images(paths[i:i+batch]))
    emb = np.concatenate(all_emb, axis=0)
    try:
        import faiss
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(out / "index.faiss"))
    except Exception:
        np.save(out / "embeddings.npy", emb)
    records = [{"source": str(p), "kind": "image"} for p in paths]
    (out / "records.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"model_id": model_id, "num_records": len(records)}, indent=2), encoding="utf-8")
    print(f"built visual index: {out}, images={len(records)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model-id", default="google/siglip-base-patch16-224")
    args = ap.parse_args()
    build_visual_index(args.input_dir, args.out_dir, args.model_id)


if __name__ == "__main__":
    main()
