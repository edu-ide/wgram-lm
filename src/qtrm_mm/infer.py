from __future__ import annotations
import argparse
import pickle
import torch

from .config import load_config
from .qtrm_model import QTRMMultimodalModel


def format_memory_context(results, max_chars: int = 4000) -> str:
    lines = ["MemoryOS evidence"]
    for score, rec in results:
        header = f"SOURCE={rec.get('source', '?')} CHUNK={rec.get('chunk_id', '?')} SCORE={float(score):.4f}"
        text = str(rec.get("text", "")).replace("\n", " ").strip()
        block = f"{header}\n{text}"
        current = "\n".join(lines)
        tentative = f"{current}\n{block}"
        if len(tentative) <= max_chars:
            lines.append(block)
            continue
        remaining = max_chars - len(current) - len(header) - 2
        if remaining > 0:
            lines.append(f"{header}\n{text[:remaining]}")
        break
    return "\n".join(lines)[:max_chars]


def build_prompt_with_memory(prompt: str, memory_context: str) -> str:
    if not memory_context:
        return prompt
    return (
        f"{memory_context}\n\n"
        "Use the evidence above when it is relevant. If it is not relevant, answer from the prompt.\n\n"
        f"User prompt:\n{prompt}"
    )


def load_checkpoint_state(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        return torch.load(path, map_location=map_location, weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt", default="Explain QTRM in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--memory-index")
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--memory-max-chars", type=int, default=4000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QTRMMultimodalModel(cfg.model).to(device)
    if args.checkpoint:
        state = load_checkpoint_state(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    prompt = args.prompt
    if args.memory_index:
        from .memoryos.retrieve import retrieve
        results = retrieve(args.memory_index, args.prompt, top_k=args.memory_top_k)
        memory_context = format_memory_context(results, max_chars=args.memory_max_chars)
        prompt = build_prompt_with_memory(args.prompt, memory_context)

    # Simple byte-ish prompt encoding for smoke mode only.
    ids = [min(ord(c), cfg.model.vocab_size - 1) for c in prompt]
    if not ids:
        ids = [1]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(args.max_new_tokens):
        out = model(input_ids)
        next_id = int(out["logits"][:, -1].argmax(dim=-1).item())
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        if input_ids.shape[1] > cfg.model.max_seq_len:
            input_ids = input_ids[:, -cfg.model.max_seq_len:]
    text = ''.join(chr(int(i)) if 32 <= int(i) < 127 else '?' for i in input_ids[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
