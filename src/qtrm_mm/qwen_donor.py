from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import torch
from torch import nn

from .config import DonorConfig


class QwenDonorAdapter(nn.Module):
    """Lazy Hugging Face adapter for Qwen3.5-style multimodal donor models.

    The adapter is intentionally thin. It does not merge donor weights into QTRM;
    it exposes hidden states and optional visual features so QTRM can train
    projectors/core/heads around a frozen donor.
    """

    def __init__(self, cfg: DonorConfig):
        super().__init__()
        if cfg.model_id is None:
            raise ValueError("DonorConfig.model_id is required")
        self.cfg = cfg
        self.processor = None
        self.model = None
        self._load()

    def _load(self):
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except Exception as exc:
            raise RuntimeError(
                "transformers with AutoModelForImageTextToText support is required"
            ) from exc

        quantization_config = None
        if self.cfg.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except ImportError:
                print("[warn] bitsandbytes not found; loading without 4bit quantization")
                quantization_config = None

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=self.cfg.trust_remote_code,
        )

        kwargs: dict[str, Any] = {
            "trust_remote_code": self.cfg.trust_remote_code,
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config

        print(f"[donor] Loading {self.cfg.model_id} "
              f"({'4bit' if self.cfg.load_in_4bit else 'full'}), "
              f"device_map={kwargs['device_map']}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.model_id, **kwargs
        )

        if self.cfg.freeze_donor:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            print("[donor] Frozen (all params requires_grad=False)")
        else:
            print("[warn] donor NOT frozen")

        param_count = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[donor] params: {param_count:,} total, {trainable:,} trainable")

    def _extract_visual_features(self, out) -> Optional[torch.Tensor]:
        # Qwen-specific vision internals are version dependent.
        for attr in ("image_embeds", "vision_hidden_states", "visual_hidden_states"):
            if hasattr(out, attr):
                visual_features = getattr(out, attr)
                if isinstance(visual_features, (list, tuple)):
                    visual_features = visual_features[-1]
                return visual_features
        return None

    @torch.no_grad()
    def encode_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        if self.model is None:
            raise RuntimeError("Donor not loaded")
        device = next(self.model.parameters()).device
        inputs = {"input_ids": input_ids.to(device)}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask.to(device)
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        result = {
            "text_states": out.hidden_states[-1].detach(),
            "attention_mask": inputs.get("attention_mask"),
            "visual_features": self._extract_visual_features(out),
        }
        if return_logits:
            result["logits"] = out.logits.detach()
        return result

    @torch.no_grad()
    def encode(
        self,
        text: str | list[str],
        images: Optional[Any] = None,
    ) -> dict[str, torch.Tensor]:
        if self.processor is None or self.model is None:
            raise RuntimeError("Donor not loaded")
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True
        )
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        out = self.model(**inputs, output_hidden_states=True)
        text_states = out.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")
        return {
            "text_states": text_states,
            "attention_mask": attention_mask,
            "visual_features": self._extract_visual_features(out),
        }
