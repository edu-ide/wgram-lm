"""
Final TRM-llm Training Script (Strict TRM + Data IO Language Healing).
Version 7: Full-Spectrum Monitoring (Split Losses, Grad Norm, LR).
"""

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from qtrm_mm.qwen_backbone_state_transition import build_qwen_state_transition_model

# --- Synthetic Data Generation ---

@dataclass
class SyntheticCase:
    input_ids: List[int]
    attention_mask: List[int]
    operation_ids: List[int]
    answer_label: int
    state_labels: List[int]
    family: str

def build_synthetic_cases(count=1024, seed=42):
    rng = random.Random(seed)
    families = ("chain5", "chain5", "checksum4", "select_pair")
    cases = []
    for _ in range(count):
        family = rng.choice(families)
        if family == "chain5":
            start, add_a, mul, sub, add_b = [rng.randint(0, 9) for _ in range(5)]
            s1 = (start + add_a) % 10
            s2 = (s1 * mul) % 10
            s3 = (s2 - sub) % 10
            s4 = (s3 + add_b) % 10
            cases.append(SyntheticCase([0]*10, [1]*10, [0, 1, 2, 0], s4, [s1, s2, s3, s4], "chain5"))
        elif family == "checksum4":
            digits = [rng.randint(0, 9) for _ in range(4)]
            res = sum(digits) % 10
            cases.append(SyntheticCase([0]*10, [1]*10, [0, 0, 0, 0], res, [digits[0], (digits[0]+digits[1])%10, (digits[0]+digits[1]+digits[2])%10, res], "checksum4"))
        else:
            a, b = rng.randint(0, 9), rng.randint(0, 9)
            res = (a + b) % 10
            cases.append(SyntheticCase([0]*10, [1]*10, [0, 0, 0, 0], res, [a, b, res, res], "select_pair"))
    return cases

# --- Datasets ---

class SyntheticDataset(Dataset):
    def __init__(self, tokenizer, count=4096, seed=42):
        self.tokenizer = tokenizer
        self.cases = build_synthetic_cases(count=count, seed=seed)
        self.data = []
        for c in self.cases:
            txt = f"Reason: {c.family}. Result: "
            t = tokenizer(txt, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
            self.data.append({
                "input_ids": t["input_ids"].squeeze(0),
                "attention_mask": t["attention_mask"].squeeze(0),
                "operation_ids": torch.tensor(c.operation_ids, dtype=torch.long),
                "reasoning_labels": torch.tensor(c.answer_label, dtype=torch.long),
                "state_labels": torch.tensor(c.state_labels, dtype=torch.long),
                "is_healing": False
            })
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class HealingDataset(Dataset):
    def __init__(self, tokenizer, count=1024):
        self.tokenizer = tokenizer
        self.texts = []
        if load_dataset:
            try:
                ds = load_dataset("databricks/databricks-dolly-15k", split="train", trust_remote_code=True)
                for item in ds.shuffle(seed=42).select(range(min(count, len(ds)))):
                    self.texts.append(f"Instruction: {item['instruction']}\nResponse: {item['response']}")
            except Exception: pass
        if not self.texts: self.texts = ["Instruction: Hello\nResponse: Hi!"] * count
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        t = self.tokenizer(self.texts[idx], truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        return {
            "input_ids": t["input_ids"].squeeze(0),
            "attention_mask": t["attention_mask"].squeeze(0),
            "healing_labels": t["input_ids"].squeeze(0),
            "is_healing": True
        }

def collate_fn(batch):
    res = {}
    keys = ["input_ids", "attention_mask", "is_healing"]
    for k in keys:
        res[k] = torch.stack([torch.tensor(d[k]) if isinstance(d[k], bool) else d[k] for d in batch])
    if any(not d["is_healing"] for d in batch):
        r_indices = [i for i, d in enumerate(batch) if not d["is_healing"]]
        res["operation_ids"] = torch.stack([batch[i]["operation_ids"] for i in r_indices])
        res["reasoning_labels"] = torch.stack([batch[i]["reasoning_labels"] for i in r_indices])
        res["state_labels"] = torch.stack([batch[i]["state_labels"] for i in r_indices])
        res["r_indices"] = torch.tensor(r_indices)
    if any(d["is_healing"] for d in batch):
        h_indices = [i for i, d in enumerate(batch) if d["is_healing"]]
        res["healing_labels"] = torch.stack([batch[i]["healing_labels"] for i in h_indices])
        res["h_indices"] = torch.tensor(h_indices)
    return res

# --- Training ---

def train_one_epoch(model, loader, optimizer, device, args, writer, epoch):
    model.train()
    total_loss, total_r_loss, total_h_loss = 0, 0, 0
    total_acc, n_r = 0, 0
    
    for batch_idx, batch in enumerate(loader):
        b = batch["input_ids"].size(0)
        batch_loss = torch.tensor(0.0, device=device)
        r_loss_val, h_loss_val = 0.0, 0.0
        
        if "r_indices" in batch:
            idx = batch["r_indices"]
            r_in = batch["input_ids"][idx].to(device)
            r_mask = batch["attention_mask"][idx].to(device)
            r_ops = batch["operation_ids"].to(device)
            out = model(input_ids=r_in, attention_mask=r_mask, operation_ids=r_ops)
            ans_l = F.cross_entropy(out["answer_logits"], batch["reasoning_labels"].to(device))
            state_l = F.cross_entropy(out["state_digit_logits"][:, 1:, :].reshape(-1, 10), batch["state_labels"].to(device).reshape(-1))
            curr_r_loss = (ans_l + state_l) * (len(idx) / b)
            batch_loss += curr_r_loss
            r_loss_val = curr_r_loss.item()
            total_acc += (out["answer_logits"].argmax(-1) == batch["reasoning_labels"].to(device)).float().sum().item()
            n_r += len(idx)

        if "h_indices" in batch:
            idx = batch["h_indices"]
            h_in = batch["input_ids"][idx].to(device)
            h_mask = batch["attention_mask"][idx].to(device)
            out = model(input_ids=h_in, attention_mask=h_mask)
            final_states = out["qtrm_core_step_states"][:, -1, :].to(model.qwen.lm_head.weight.dtype)
            h_logits = model.qwen.lm_head(final_states.unsqueeze(1)).squeeze(1)
            last_idx = h_mask.sum(1).cpu() - 1
            targets = batch["healing_labels"].to(device).gather(1, last_idx.to(device).unsqueeze(1)).squeeze(1)
            curr_h_loss = F.cross_entropy(h_logits, targets) * (len(idx) / b)
            batch_loss += curr_h_loss
            h_loss_val = curr_h_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        
        grad_norm = 0.0
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        total_loss += batch_loss.item() * b
        total_r_loss += r_loss_val * b
        total_h_loss += h_loss_val * b

        if batch_idx % 10 == 0:
            it = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Step/Loss_Total", batch_loss.item(), it)
            writer.add_scalar("Step/Loss_Reasoning", r_loss_val, it)
            writer.add_scalar("Step/Loss_Healing", h_loss_val, it)
            writer.add_scalar("Step/Grad_Norm", grad_norm, it)

    avg_acc = (total_acc / n_r if n_r > 0 else 0)
    writer.add_scalar("Epoch/Loss_Total", total_loss / len(loader.dataset), epoch)
    writer.add_scalar("Epoch/Loss_Reasoning", total_r_loss / len(loader.dataset), epoch)
    writer.add_scalar("Epoch/Loss_Healing", total_h_loss / len(loader.dataset), epoch)
    writer.add_scalar("Epoch/Accuracy_Reasoning", avg_acc, epoch)
    writer.add_scalar("Epoch/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
    
    return total_loss / len(loader.dataset), avg_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-model-id", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_qwen_state_transition_model(args.qwen_model_id, freeze_qwen=False, device=device)
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
    
    train_ds = ConcatDataset([SyntheticDataset(tokenizer, 2800), HealingDataset(tokenizer, 1200)])
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss, acc = train_one_epoch(model, loader, optimizer, device, args, writer, epoch)
        duration = time.time() - start_time
        print(f"Epoch {epoch:3d} | loss={loss:.4f} | acc={acc:.4f} | time={duration:.1f}s")
        
        # Save checkpoints
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"epoch_{epoch}.pt"))
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))
        
    writer.close()

if __name__ == "__main__": main()
