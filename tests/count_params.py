import argparse
from wgram_lm.config import load_config
from wgram_lm.wgram_model import QTRMMultimodalModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke_multimodal.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = QTRMMultimodalModel(cfg.model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_params={total:,}")
    print(f"trainable_params={trainable:,}")


if __name__ == "__main__":
    main()
