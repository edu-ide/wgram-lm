#!/usr/bin/env bash
set -euo pipefail
cd ~/qtrm-workspace/wgram-lm
source .venv/bin/activate
export PYTHONPATH=$PWD/src

python -c "
import torch
from wgram_lm.backends import HAS_FLASH_ATTN, HAS_FLA
from wgram_lm.config import load_config
from wgram_lm.wgram_model import QTRMMultimodalModel
from wgram_lm.qwen_donor import QwenDonorAdapter
from wgram_lm.losses import qtrm_smoke_loss

print('=' * 60)
print('Integration Test: Backend + Donor + Model (4090)')
print('=' * 60)

cfg = load_config('configs/qwen35_2b_4090.yaml')
print(f'\nflash-attn: {HAS_FLASH_ATTN}, fla: {HAS_FLA}')
print(f'config: d_model={cfg.model.d_model}, workspace={cfg.model.workspace_tokens}')
print(f'attention={cfg.model.attention_backend}, delta={cfg.model.delta_backend}')

print('\nLoading donor (4bit frozen)...')
donor = QwenDonorAdapter(cfg.donor)
result = donor.encode('What is the capital of France?')
print(f'  text_states: {result[\"text_states\"].shape}')

print('Building QTRM model...')
model = QTRMMultimodalModel(cfg.model)
total = sum(p.numel() for p in model.parameters())
print(f'  params: {total:,}')

model = model.to('cuda')
model.train()
text_st = result['text_states'].to('cuda')
bs, sl, _ = text_st.shape
ids = torch.randint(32, cfg.model.vocab_size, (bs, sl), dtype=torch.long, device='cuda')
mask = result['attention_mask'].to('cuda')

print('Forward pass...')
with torch.autocast('cuda', torch.bfloat16):
    loss, m, _ = qtrm_smoke_loss(model, ids, attention_mask=mask, text_states=text_st)
print(f'\n  loss={float(loss):.4f}')
print(f'  lm={float(m[\"lm\"]):.4f}')
print(f'  jepa={float(m[\"jepa\"]):.4f}')
print(f'  aux={float(m[\"aux\"]):.4f}')
print(f'  VRAM peak: {torch.cuda.max_memory_allocated(\"cuda\")/1e9:.1f} GB')
print('\n' + '=' * 60)
print('ALL INTEGRATION TESTS PASSED')
print('=' * 60)
"
