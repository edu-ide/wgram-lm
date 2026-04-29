import torch
from qtrm_mm import QTRMConfig, QTRMMultimodalModel
from qtrm_mm.losses import qtrm_smoke_loss


def main():
    cfg = QTRMConfig(
        vocab_size=1024,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        d_ff=384,
        max_seq_len=128,
        n_prelude_layers=1,
        n_core_layers=1,
        n_coda_layers=1,
        workspace_tokens=16,
        h_cycles=1,
        l_cycles=1,
        outer_steps=1,
        visual_dim=64,
        max_visual_tokens=8,
    )
    model = QTRMMultimodalModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    visual = torch.randn(2, 8, cfg.visual_dim)
    loss, metrics, out = qtrm_smoke_loss(model, ids, visual_features=visual)
    loss.backward()
    print("ok", {k: float(v) for k, v in metrics.items()}, out["logits"].shape)


if __name__ == "__main__":
    main()
