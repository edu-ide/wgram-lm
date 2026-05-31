from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional
import torch
from torch import nn

from .config import DonorConfig


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def _build_4bit_quantization_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    if not _bitsandbytes_available():
        print("[warn] bitsandbytes not found; loading without 4bit quantization")
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("[warn] BitsAndBytesConfig unavailable; loading without 4bit quantization")
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


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

        quantization_config = _build_4bit_quantization_config(self.cfg.load_in_4bit)

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

        self._configure_trainability()

        param_count = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[donor] params: {param_count:,} total, {trainable:,} trainable")

    def _configure_trainability(self) -> None:
        if self.model is None:
            raise RuntimeError("Donor not loaded")

        if self.cfg.load_in_4bit and not self.cfg.train_lora and not self.cfg.freeze_donor:
            raise ValueError(
                "4bit donor training requires donor.train_lora=true. "
                "Set load_in_4bit=false for full/partial unfreeze."
            )

        if self.cfg.train_lora:
            self._enable_lora_training()
            self.model.train()
            print("[donor] LoRA trainable; base donor remains adapter-tuned")
            return

        if self.cfg.freeze_donor:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            print("[donor] Frozen (all params requires_grad=False)")
            return

        if int(self.cfg.train_last_n_layers) > 0:
            self._enable_last_n_layers(int(self.cfg.train_last_n_layers))
            self.model.train()
            return

        if bool(self.cfg.gradient_checkpointing) and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
        self.model.train()
        print("[warn] donor fully trainable")

    def _enable_lora_training(self) -> None:
        if self.model is None:
            raise RuntimeError("Donor not loaded")
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except Exception as exc:
            raise RuntimeError(
                "donor.train_lora=true requires the optional 'peft' package. "
                "Install it with `pip install peft`, or set donor.train_lora=false."
            ) from exc

        if bool(self.cfg.gradient_checkpointing) and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
        if bool(self.cfg.load_in_4bit):
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=bool(self.cfg.gradient_checkpointing),
            )
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        target_modules: str | list[str]
        target_modules = (
            list(self.cfg.lora_target_modules)
            if self.cfg.lora_target_modules
            else "all-linear"
        )
        lora_cfg = LoraConfig(
            r=max(1, int(self.cfg.lora_rank)),
            lora_alpha=max(1, int(self.cfg.lora_alpha)),
            lora_dropout=max(0.0, float(self.cfg.lora_dropout)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, lora_cfg)

    def _enable_last_n_layers(self, n_layers: int) -> None:
        if self.model is None:
            raise RuntimeError("Donor not loaded")
        layers = self._find_decoder_layers()
        if not layers:
            raise ValueError(
                "donor.train_last_n_layers was set, but decoder layers could "
                "not be located on this donor model."
            )
        for p in self.model.parameters():
            p.requires_grad_(False)
        for layer in layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        if bool(self.cfg.gradient_checkpointing) and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
        print(f"[donor] Partially trainable: last {min(n_layers, len(layers))} decoder layers")

    def _find_decoder_layers(self) -> list[nn.Module]:
        if self.model is None:
            return []
        roots: list[Any] = [
            self.model,
            getattr(self.model, "model", None),
            getattr(self.model, "language_model", None),
            getattr(getattr(self.model, "model", None), "language_model", None),
            getattr(getattr(self.model, "language_model", None), "model", None),
        ]
        for root in roots:
            if root is None:
                continue
            for attr in ("layers", "h", "blocks"):
                layers = getattr(root, attr, None)
                if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                    return list(layers)
        return []

    def has_trainable_parameters(self) -> bool:
        return any(p.requires_grad for p in self.parameters())

    def save_trainable(self, path: str | Any) -> None:
        if self.model is None or not self.has_trainable_parameters():
            return
        path = str(path)
        if self.cfg.train_lora and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
            return
        state = {
            name: p.detach().cpu()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        torch.save(state, path)

    def _extract_visual_features(self, out) -> Optional[torch.Tensor]:
        # Qwen-specific vision internals are version dependent.
        for attr in ("image_embeds", "vision_hidden_states", "visual_hidden_states"):
            if hasattr(out, attr):
                visual_features = getattr(out, attr)
                if isinstance(visual_features, (list, tuple)):
                    visual_features = visual_features[-1]
                return visual_features
        return None

    def encode_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        detach: bool = True,
        detach_logits: Optional[bool] = None,
    ) -> dict[str, torch.Tensor | None]:
        if self.model is None:
            raise RuntimeError("Donor not loaded")
        detach_logits = detach if detach_logits is None else bool(detach_logits)
        device = next(self.model.parameters()).device
        inputs = {"input_ids": input_ids.to(device)}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask.to(device)
        context = torch.no_grad() if detach else nullcontext()
        with context:
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        text_states = out.hidden_states[-1]
        visual_features = self._extract_visual_features(out)
        result = {
            "text_states": text_states.detach() if detach else text_states,
            "attention_mask": inputs.get("attention_mask"),
            "visual_features": (
                visual_features.detach()
                if detach and visual_features is not None
                else visual_features
            ),
        }
        if return_logits:
            result["logits"] = out.logits.detach() if detach_logits else out.logits
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
