from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .chunk_encoder import CausalByteChunkEncoder
from .config import WGRAMV2Config
from .contracts import validate_v2_contract
from .imta import InternalMultiTrajectoryAdapter
from .latent_prediction import OwnLatentPredictor
from .recurrent_core import build_v2_recurrent_core
from .speaker import CausalByteSpeaker


class WGRAMReasoningLMV2(nn.Module):
    """Clean W-GRAM V2 path: bytes -> causal chunks -> recurrent IMTA -> one speaker."""

    def __init__(self, config: WGRAMV2Config) -> None:
        super().__init__()
        validate_v2_contract(config, require_promotion_ready=False)
        self.config = config
        self.byte_embed = nn.Embedding(int(config.vocab_size), int(config.d_model))
        self.chunk_encoder = CausalByteChunkEncoder(config)
        self.core = build_v2_recurrent_core(config)
        self.imta = InternalMultiTrajectoryAdapter(config)
        self.latent_predictor = OwnLatentPredictor(config)
        self.speaker = CausalByteSpeaker(config)
        if bool(config.tie_input_output_embeddings):
            self.speaker.tie_output_weight(self.byte_embed.weight)
        self.last_metrics: dict[str, float | int | str] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.byte_embed.weight, mean=0.0, std=0.02)
        if not bool(self.config.tie_input_output_embeddings):
            nn.init.normal_(self.speaker.head.weight, mean=0.0, std=0.02)

    def _core_forward(self, chunk_states: torch.Tensor, chunk_valid: torch.Tensor, *, think_steps: int) -> torch.Tensor:
        return self.core(chunk_states, chunk_valid, think_steps=int(think_steps))

    def forward_logits_and_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        think_steps: int = 1,
        response_prediction_mask: torch.Tensor | None = None,
        answer_memory_injection_scale: float | None = None,
        answer_memory_commitment_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        safe_ids = input_ids.clamp(min=0, max=int(self.config.vocab_size) - 1)
        byte_embeddings = self.byte_embed(safe_ids)
        chunk_encoding = self.chunk_encoder(byte_embeddings, safe_ids, attention_mask)
        imta_out = self.imta(
            chunk_encoding.chunk_states,
            chunk_encoding.chunk_valid,
            core_fn=lambda states, valid: self._core_forward(states, valid, think_steps=int(think_steps)),
            speaker_norm=self.speaker.norm,
        )
        bos_context = imta_out.chunk_hidden.new_zeros(
            (input_ids.shape[0], 1, int(self.config.d_model)),
        )
        chunk_hidden_with_bos = torch.cat([bos_context, imta_out.chunk_hidden], dim=1)
        dechunked = torch.gather(
            chunk_hidden_with_bos,
            dim=1,
            index=chunk_encoding.dechunk_indices.unsqueeze(-1).expand(
                input_ids.shape[0],
                input_ids.shape[1],
                int(self.config.d_model),
            ),
        )
        logits, hidden, speaker_metrics = self.speaker(
            byte_embeddings,
            dechunked,
            attention_mask,
            response_prediction_mask=response_prediction_mask,
            answer_memory_injection_scale=answer_memory_injection_scale,
            answer_memory_commitment_scale=answer_memory_commitment_scale,
        )
        latent_out = self.latent_predictor(
            imta_out.chunk_hidden,
            chunk_encoding.chunk_valid,
            target_hidden=chunk_encoding.chunk_states,
        )
        metrics: dict[str, float | int | str] = {
            **chunk_encoding.metrics,
            **imta_out.metrics,
            **speaker_metrics,
            "runtime_profile": str(self.config.runtime_profile),
            "core_implementation": str(getattr(self.core, "core_implementation", self.config.core_implementation)),
            "core_attention_causal": bool(getattr(self.core, "core_attention_causal", self.config.core_attention_causal)),
            "official_gdn2_force_chunk_eval": bool(
                getattr(self.core, "official_gdn2_force_chunk_eval", self.config.official_gdn2_force_chunk_eval)
            ),
            "think_steps": int(think_steps),
            "own_latent_prediction_loss": float(latent_out.loss.detach().float().cpu().item()),
            "own_latent_prediction_targets": int(latent_out.targets),
            "own_latent_prediction_cosine": float(latent_out.cosine.detach().float().cpu().item()),
            "own_latent_prediction_target_source": str(latent_out.target_source),
        }
        self.last_metrics = metrics
        self._last_imta_diversity_loss = imta_out.diversity_loss
        self._last_imta_route_entropy_loss = imta_out.route_entropy_loss
        self._last_imta_route_balance_loss = imta_out.route_balance_loss
        self._last_own_latent_prediction_loss = latent_out.loss
        return logits, hidden, metrics

    def _repeat_unlikelihood_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if float(self.config.repeat_unlikelihood_weight) <= 0.0 or int(input_ids.shape[1]) <= 1:
            return logits.new_zeros(())
        valid = labels != -100
        previous_valid = torch.zeros_like(valid)
        previous_valid[:, 1:] = attention_mask[:, :-1].bool()
        previous_ids = torch.zeros_like(input_ids)
        previous_ids[:, 1:] = input_ids[:, :-1]
        negative_mask = valid & previous_valid & (labels != previous_ids)
        if not bool(negative_mask.any()):
            return logits.new_zeros(())
        safe_previous = previous_ids.clamp(min=0, max=int(self.config.vocab_size) - 1)
        logits_f = logits.float()
        previous_logits = logits_f.gather(dim=-1, index=safe_previous.unsqueeze(-1)).squeeze(-1)
        previous_log_probs = previous_logits - logits_f.logsumexp(dim=-1)
        previous_probs = previous_log_probs.exp().clamp(max=1.0 - 1.0e-6)
        return (-torch.log1p(-previous_probs[negative_mask])).mean().to(logits.dtype)

    def _premature_stop_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        stop_token_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        if float(self.config.premature_stop_loss_weight) <= 0.0 or not stop_token_ids:
            return logits.new_zeros(()), {
                "premature_stop_positions": 0,
                "premature_stop_mean_probability": 0.0,
            }
        valid = labels != -100
        stop_ids = torch.tensor(tuple(int(token_id) for token_id in stop_token_ids), dtype=labels.dtype, device=labels.device)
        stop_target = torch.isin(labels, stop_ids)
        positions = valid & ~stop_target
        if not bool(positions.any()):
            return logits.new_zeros(()), {
                "premature_stop_positions": 0,
                "premature_stop_mean_probability": 0.0,
            }
        selected_logits = logits.float()[positions]
        stop_logits = selected_logits[:, stop_ids.to(device=selected_logits.device)]
        loss = F.binary_cross_entropy_with_logits(stop_logits, torch.zeros_like(stop_logits), reduction="mean")
        with torch.no_grad():
            mean_probability = float(stop_logits.sigmoid().detach().float().mean().cpu().item())
        return loss.to(logits.dtype), {
            "premature_stop_positions": int(positions.detach().sum().cpu().item()),
            "premature_stop_mean_probability": mean_probability,
        }

    def _response_start_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_start_mask: torch.Tensor | None,
        *,
        stop_token_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        empty = {
            "response_start_positions": 0,
            "response_start_accuracy": 0.0,
            "response_start_gold_probability": 0.0,
            "response_start_stop_probability": 0.0,
        }
        if float(self.config.response_start_loss_weight) <= 0.0 or response_start_mask is None:
            return logits.new_zeros(()), empty
        positions = response_start_mask.bool() & (labels != -100)
        if not bool(positions.any()):
            return logits.new_zeros(()), empty
        selected_logits = logits.float()[positions]
        selected_labels = labels[positions]
        loss = F.cross_entropy(selected_logits, selected_labels, reduction="mean")
        with torch.no_grad():
            probs = selected_logits.softmax(dim=-1)
            gold_prob = probs.gather(1, selected_labels.unsqueeze(1)).mean()
            accuracy = selected_logits.argmax(dim=-1).eq(selected_labels).float().mean()
            stop_probability = selected_logits.new_zeros(())
            if stop_token_ids:
                stop_ids = torch.tensor(
                    tuple(int(token_id) for token_id in stop_token_ids),
                    dtype=selected_labels.dtype,
                    device=selected_labels.device,
                )
                stop_probability = probs[:, stop_ids.to(device=probs.device)].sum(dim=-1).mean()
        return loss.to(logits.dtype), {
            "response_start_positions": int(positions.detach().sum().cpu().item()),
            "response_start_accuracy": float(accuracy.detach().cpu().item()),
            "response_start_gold_probability": float(gold_prob.detach().cpu().item()),
            "response_start_stop_probability": float(stop_probability.detach().cpu().item()),
        }

    def _response_start_stop_margin_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_start_mask: torch.Tensor | None,
        *,
        stop_token_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        empty = {
            "response_start_stop_margin_positions": 0,
            "response_start_gold_minus_best_stop_logit": 0.0,
            "response_start_stop_margin_violation_fraction": 0.0,
            "response_start_best_stop_probability": 0.0,
        }
        if (
            float(self.config.response_start_stop_margin_weight) <= 0.0
            or response_start_mask is None
            or not stop_token_ids
        ):
            return logits.new_zeros(()), empty
        stop_ids = torch.tensor(
            tuple(int(token_id) for token_id in stop_token_ids),
            dtype=labels.dtype,
            device=labels.device,
        )
        positions = response_start_mask.bool() & (labels != -100) & ~torch.isin(labels, stop_ids)
        if not bool(positions.any()):
            return logits.new_zeros(()), empty
        selected_logits = logits.float()[positions]
        selected_labels = labels[positions]
        gold_logits = selected_logits.gather(dim=1, index=selected_labels.unsqueeze(1)).squeeze(1)
        selected_stop_logits = selected_logits[:, stop_ids.to(device=selected_logits.device)]
        best_stop_logits = selected_stop_logits.max(dim=1).values
        margin = float(self.config.response_start_stop_margin)
        violations = best_stop_logits - gold_logits + margin
        loss = F.relu(violations).mean()
        with torch.no_grad():
            probs = selected_logits.softmax(dim=-1)
            stop_probs = probs[:, stop_ids.to(device=probs.device)]
            gold_minus_stop = gold_logits - best_stop_logits
            violation_fraction = (violations > 0.0).float().mean()
            best_stop_probability = stop_probs.max(dim=1).values.mean()
        return loss.to(logits.dtype), {
            "response_start_stop_margin_positions": int(positions.detach().sum().cpu().item()),
            "response_start_gold_minus_best_stop_logit": float(gold_minus_stop.detach().float().mean().cpu().item()),
            "response_start_stop_margin_violation_fraction": float(
                violation_fraction.detach().float().cpu().item()
            ),
            "response_start_best_stop_probability": float(best_stop_probability.detach().float().cpu().item()),
        }

    def _response_body_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_start_mask: torch.Tensor | None,
        *,
        stop_token_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        empty = {
            "response_body_positions": 0,
            "response_body_accuracy": 0.0,
            "response_body_gold_probability": 0.0,
            "response_body_stop_probability": 0.0,
        }
        if float(self.config.response_body_loss_weight) <= 0.0:
            return logits.new_zeros(()), empty
        positions = labels != -100
        if response_start_mask is not None:
            positions = positions & ~response_start_mask.bool()
        if stop_token_ids:
            stop_ids = torch.tensor(tuple(int(token_id) for token_id in stop_token_ids), dtype=labels.dtype, device=labels.device)
            positions = positions & ~torch.isin(labels, stop_ids)
        else:
            stop_ids = None
        if not bool(positions.any()):
            return logits.new_zeros(()), empty
        selected_logits = logits.float()[positions]
        selected_labels = labels[positions]
        loss = F.cross_entropy(selected_logits, selected_labels, reduction="mean")
        with torch.no_grad():
            probs = selected_logits.softmax(dim=-1)
            gold_prob = probs.gather(1, selected_labels.unsqueeze(1)).mean()
            accuracy = selected_logits.argmax(dim=-1).eq(selected_labels).float().mean()
            stop_probability = selected_logits.new_zeros(())
            if stop_ids is not None:
                stop_probability = probs[:, stop_ids.to(device=probs.device)].sum(dim=-1).mean()
        return loss.to(logits.dtype), {
            "response_body_positions": int(positions.detach().sum().cpu().item()),
            "response_body_accuracy": float(accuracy.detach().cpu().item()),
            "response_body_gold_probability": float(gold_prob.detach().cpu().item()),
            "response_body_stop_probability": float(stop_probability.detach().cpu().item()),
        }

    def _response_continue_stop_margin_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_start_mask: torch.Tensor | None,
        *,
        stop_token_ids: tuple[int, ...],
        active_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        empty = {
            "response_continue_stop_margin_positions": 0,
            "response_continue_gold_minus_best_stop_logit": 0.0,
            "response_continue_stop_margin_violation_fraction": 0.0,
            "response_continue_best_stop_probability": 0.0,
        }
        if float(active_weight) <= 0.0 or not stop_token_ids:
            return logits.new_zeros(()), empty
        stop_ids = torch.tensor(
            tuple(int(token_id) for token_id in stop_token_ids),
            dtype=labels.dtype,
            device=labels.device,
        )
        positions = (labels != -100) & ~torch.isin(labels, stop_ids)
        if response_start_mask is not None:
            positions = positions & ~response_start_mask.bool()
        if not bool(positions.any()):
            return logits.new_zeros(()), empty
        selected_logits = logits.float()[positions]
        selected_labels = labels[positions]
        gold_logits = selected_logits.gather(dim=1, index=selected_labels.unsqueeze(1)).squeeze(1)
        selected_stop_logits = selected_logits[:, stop_ids.to(device=selected_logits.device)]
        best_stop_logits = selected_stop_logits.max(dim=1).values
        margin = float(self.config.response_continue_stop_margin)
        violations = best_stop_logits - gold_logits + margin
        loss = F.relu(violations).mean()
        with torch.no_grad():
            probs = selected_logits.softmax(dim=-1)
            stop_probs = probs[:, stop_ids.to(device=probs.device)]
            gold_minus_stop = gold_logits - best_stop_logits
            violation_fraction = (violations > 0.0).float().mean()
            best_stop_probability = stop_probs.max(dim=1).values.mean()
        return loss.to(logits.dtype), {
            "response_continue_stop_margin_positions": int(positions.detach().sum().cpu().item()),
            "response_continue_gold_minus_best_stop_logit": float(gold_minus_stop.detach().float().mean().cpu().item()),
            "response_continue_stop_margin_violation_fraction": float(
                violation_fraction.detach().float().cpu().item()
            ),
            "response_continue_best_stop_probability": float(best_stop_probability.detach().float().cpu().item()),
        }

    def _response_stop_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        stop_token_ids: tuple[int, ...],
        active_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        empty = {
            "response_stop_positions": 0,
            "response_stop_accuracy": 0.0,
            "response_stop_probability": 0.0,
        }
        if float(active_weight) <= 0.0 or not stop_token_ids:
            return logits.new_zeros(()), empty
        stop_ids = torch.tensor(tuple(int(token_id) for token_id in stop_token_ids), dtype=labels.dtype, device=labels.device)
        positions = (labels != -100) & torch.isin(labels, stop_ids)
        if not bool(positions.any()):
            return logits.new_zeros(()), empty
        selected_logits = logits.float()[positions]
        selected_labels = labels[positions]
        loss = F.cross_entropy(selected_logits, selected_labels, reduction="mean")
        with torch.no_grad():
            probs = selected_logits.softmax(dim=-1)
            accuracy = selected_logits.argmax(dim=-1).eq(selected_labels).float().mean()
            stop_probability = probs[:, stop_ids.to(device=probs.device)].sum(dim=-1).mean()
        return loss.to(logits.dtype), {
            "response_stop_positions": int(positions.detach().sum().cpu().item()),
            "response_stop_accuracy": float(accuracy.detach().cpu().item()),
            "response_stop_probability": float(stop_probability.detach().cpu().item()),
        }

    def _token_maturation_aux_loss(
        self,
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        empty = {
            "token_maturation_aux_depths": 0,
            "token_maturation_aux_loss": 0.0,
        }
        aux_logits = list(getattr(self.speaker, "last_maturation_logits", []))
        if float(self.config.token_maturation_aux_loss_weight) <= 0.0 or not aux_logits:
            return labels.new_zeros((), dtype=torch.float32), empty
        if not bool(valid.any()):
            return aux_logits[0].sum() * 0.0, {
                "token_maturation_aux_depths": int(len(aux_logits)),
                "token_maturation_aux_loss": 0.0,
            }
        flat_labels = labels.reshape(-1)
        flat_valid = valid.reshape(-1)
        losses = [
            F.cross_entropy(
                logits.reshape(-1, int(self.config.vocab_size))[flat_valid],
                flat_labels[flat_valid],
            )
            for logits in aux_logits
        ]
        aux_loss = torch.stack([loss.to(aux_logits[-1].dtype) for loss in losses]).mean()
        return aux_loss, {
            "token_maturation_aux_depths": int(len(aux_logits)),
            "token_maturation_aux_loss": float(aux_loss.detach().float().cpu().item()),
        }

    def _answer_memory_aux_loss(
        self,
        labels: torch.Tensor,
        response_start_mask: torch.Tensor | None,
        *,
        stop_token_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        empty = {
            "answer_memory_aux_rows": 0,
            "answer_memory_aux_tokens": 0,
            "answer_memory_aux_loss": 0.0,
            "answer_memory_aux_accuracy": 0.0,
            "answer_memory_aux_gold_probability": 0.0,
            "answer_memory_stop_margin_loss": 0.0,
            "answer_memory_stop_margin_weight": float(self.config.answer_memory_stop_margin_loss_weight),
            "answer_memory_stop_margin": float(self.config.answer_memory_stop_margin),
            "answer_memory_stop_margin_positions": 0,
            "answer_memory_stop_margin_violation_fraction": 0.0,
            "answer_memory_gold_minus_best_stop_logit": 0.0,
            "answer_memory_best_stop_probability": 0.0,
        }
        memory_logits = getattr(self.speaker, "last_answer_memory_plan_logits", None)
        if (
            float(self.config.answer_memory_aux_loss_weight) <= 0.0
            or memory_logits is None
            or response_start_mask is None
        ):
            return labels.new_zeros((), dtype=torch.float32), empty
        positions = response_start_mask.bool() & (labels != -100)
        row_mask = positions.any(dim=1)
        if not bool(row_mask.any()):
            return memory_logits.sum() * 0.0, empty
        start_positions = positions.float().argmax(dim=1).to(torch.long)
        plan_tokens = int(memory_logits.shape[1])
        target = torch.full(
            (labels.shape[0], plan_tokens),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        for row_idx in torch.nonzero(row_mask, as_tuple=False).flatten().tolist():
            start = int(start_positions[int(row_idx)].detach().cpu().item())
            end = min(int(labels.shape[1]), start + plan_tokens)
            if end > start:
                values = labels[int(row_idx), start:end]
                target[int(row_idx), : int(end - start)] = values
        valid = target != -100
        if not bool(valid.any()):
            return memory_logits.sum() * 0.0, empty
        selected_logits = memory_logits.float()[valid]
        selected_labels = target[valid]
        loss = F.cross_entropy(selected_logits, selected_labels, reduction="mean")
        stop_margin_loss = memory_logits.sum() * 0.0
        stop_margin_positions = 0
        stop_margin_violation_fraction = 0.0
        gold_minus_best_stop_mean = 0.0
        best_stop_probability_mean = 0.0
        if float(self.config.answer_memory_stop_margin_loss_weight) > 0.0 and stop_token_ids:
            stop_ids = torch.tensor(
                tuple(int(token_id) for token_id in stop_token_ids),
                dtype=target.dtype,
                device=target.device,
            )
            safe_stop_ids = stop_ids[(stop_ids >= 0) & (stop_ids < int(self.config.vocab_size))]
            plan_nonstop = valid & ~torch.isin(target, safe_stop_ids) if bool(safe_stop_ids.numel()) else valid
            if bool(plan_nonstop.any()) and bool(safe_stop_ids.numel()):
                margin_logits = memory_logits.float()[plan_nonstop]
                margin_labels = target[plan_nonstop]
                gold_logits = margin_logits.gather(dim=1, index=margin_labels.unsqueeze(1)).squeeze(1)
                stop_logits = margin_logits[:, safe_stop_ids.to(device=margin_logits.device)]
                best_stop_logits = stop_logits.max(dim=1).values
                violations = best_stop_logits - gold_logits + float(self.config.answer_memory_stop_margin)
                stop_margin_loss = F.relu(violations).mean().to(memory_logits.dtype)
                with torch.no_grad():
                    margin_probs = margin_logits.softmax(dim=-1)
                    stop_probs = margin_probs[:, safe_stop_ids.to(device=margin_probs.device)]
                    stop_margin_positions = int(plan_nonstop.detach().sum().cpu().item())
                    stop_margin_violation_fraction = float(
                        (violations > 0.0).detach().float().mean().cpu().item()
                    )
                    gold_minus_best_stop_mean = float(
                        (gold_logits - best_stop_logits).detach().float().mean().cpu().item()
                    )
                    best_stop_probability_mean = float(stop_probs.max(dim=1).values.detach().float().mean().cpu().item())
        with torch.no_grad():
            probs = selected_logits.softmax(dim=-1)
            gold_prob = probs.gather(1, selected_labels.unsqueeze(1)).mean()
            accuracy = selected_logits.argmax(dim=-1).eq(selected_labels).float().mean()
        total_loss = loss.to(memory_logits.dtype) + float(self.config.answer_memory_stop_margin_loss_weight) * stop_margin_loss.to(memory_logits.dtype)
        return total_loss, {
            "answer_memory_aux_rows": int(row_mask.detach().sum().cpu().item()),
            "answer_memory_aux_tokens": int(valid.detach().sum().cpu().item()),
            "answer_memory_aux_loss": float(loss.detach().float().cpu().item()),
            "answer_memory_aux_accuracy": float(accuracy.detach().float().cpu().item()),
            "answer_memory_aux_gold_probability": float(gold_prob.detach().float().cpu().item()),
            "answer_memory_stop_margin_loss": float(stop_margin_loss.detach().float().cpu().item()),
            "answer_memory_stop_margin_weight": float(self.config.answer_memory_stop_margin_loss_weight),
            "answer_memory_stop_margin": float(self.config.answer_memory_stop_margin),
            "answer_memory_stop_margin_positions": int(stop_margin_positions),
            "answer_memory_stop_margin_violation_fraction": float(stop_margin_violation_fraction),
            "answer_memory_gold_minus_best_stop_logit": float(gold_minus_best_stop_mean),
            "answer_memory_best_stop_probability": float(best_stop_probability_mean),
        }

    def _answer_prefix_commitment_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        empty = {
            "answer_prefix_commitment_tokens": 0,
            "answer_prefix_commitment_loss": 0.0,
            "answer_prefix_commitment_accuracy": 0.0,
            "answer_prefix_commitment_gold_probability": 0.0,
        }
        if float(self.config.answer_prefix_commitment_loss_weight) <= 0.0:
            return logits.new_zeros(()), empty
        valid = labels != -100
        if not bool(valid.any()):
            return logits.new_zeros(()), empty
        plan_tokens = max(1, int(self.config.answer_memory_plan_tokens))
        response_positions = valid.to(torch.long).cumsum(dim=1) - 1
        positions = valid & (response_positions >= 0) & (response_positions < plan_tokens)
        if not bool(positions.any()):
            return logits.new_zeros(()), empty
        selected_logits = logits.float()[positions]
        selected_labels = labels[positions]
        loss = F.cross_entropy(selected_logits, selected_labels, reduction="mean")
        with torch.no_grad():
            probs = selected_logits.softmax(dim=-1)
            gold_probability = probs.gather(1, selected_labels.unsqueeze(1)).mean()
            accuracy = selected_logits.argmax(dim=-1).eq(selected_labels).float().mean()
        return loss.to(logits.dtype), {
            "answer_prefix_commitment_tokens": int(positions.detach().sum().cpu().item()),
            "answer_prefix_commitment_loss": float(loss.detach().float().cpu().item()),
            "answer_prefix_commitment_accuracy": float(accuracy.detach().float().cpu().item()),
            "answer_prefix_commitment_gold_probability": float(gold_probability.detach().float().cpu().item()),
        }

    def forward_losses(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        think_steps: int = 1,
        response_start_mask: torch.Tensor | None = None,
        stop_token_ids: tuple[int, ...] = (),
        response_stop_loss_weight: float | None = None,
        response_continue_stop_margin_weight: float | None = None,
        answer_memory_injection_scale: float | None = None,
        answer_memory_commitment_scale: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int | str]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        response_prediction_mask = (labels != -100).to(attention_mask.dtype)
        logits, _, metrics = self.forward_logits_and_hidden(
            input_ids,
            attention_mask,
            think_steps=int(think_steps),
            response_prediction_mask=response_prediction_mask,
            answer_memory_injection_scale=answer_memory_injection_scale,
            answer_memory_commitment_scale=answer_memory_commitment_scale,
        )
        valid = labels != -100
        if bool(valid.any()):
            clean_loss = F.cross_entropy(logits.reshape(-1, int(self.config.vocab_size))[valid.reshape(-1)], labels.reshape(-1)[valid.reshape(-1)])
        else:
            clean_loss = logits.sum() * 0.0
        token_maturation_aux_loss, token_maturation_aux_metrics = self._token_maturation_aux_loss(labels, valid)
        answer_memory_aux_loss, answer_memory_aux_metrics = self._answer_memory_aux_loss(
            labels,
            response_start_mask,
            stop_token_ids=tuple(int(token_id) for token_id in stop_token_ids),
        )
        answer_prefix_commitment_loss, answer_prefix_commitment_metrics = self._answer_prefix_commitment_loss(
            logits,
            labels,
        )
        imta_diversity_loss = getattr(self, "_last_imta_diversity_loss", clean_loss.new_zeros(()))
        imta_route_entropy_loss = getattr(self, "_last_imta_route_entropy_loss", clean_loss.new_zeros(()))
        imta_route_balance_loss = getattr(self, "_last_imta_route_balance_loss", clean_loss.new_zeros(()))
        own_latent_loss = getattr(self, "_last_own_latent_prediction_loss", clean_loss.new_zeros(()))
        repeat_unlikelihood_loss = self._repeat_unlikelihood_loss(logits, input_ids, labels, attention_mask)
        premature_stop_loss, premature_stop_metrics = self._premature_stop_loss(
            logits,
            labels,
            stop_token_ids=tuple(int(token_id) for token_id in stop_token_ids),
        )
        response_start_loss, response_start_metrics = self._response_start_loss(
            logits,
            labels,
            response_start_mask,
            stop_token_ids=tuple(int(token_id) for token_id in stop_token_ids),
        )
        response_start_stop_margin_loss, response_start_stop_margin_metrics = self._response_start_stop_margin_loss(
            logits,
            labels,
            response_start_mask,
            stop_token_ids=tuple(int(token_id) for token_id in stop_token_ids),
        )
        response_body_loss, response_body_metrics = self._response_body_loss(
            logits,
            labels,
            response_start_mask,
            stop_token_ids=tuple(int(token_id) for token_id in stop_token_ids),
        )
        effective_response_continue_stop_margin_weight = (
            float(self.config.response_continue_stop_margin_weight)
            if response_continue_stop_margin_weight is None
            else max(0.0, float(response_continue_stop_margin_weight))
        )
        (
            response_continue_stop_margin_loss,
            response_continue_stop_margin_metrics,
        ) = self._response_continue_stop_margin_loss(
            logits,
            labels,
            response_start_mask,
            stop_token_ids=tuple(int(token_id) for token_id in stop_token_ids),
            active_weight=float(effective_response_continue_stop_margin_weight),
        )
        effective_response_stop_loss_weight = (
            float(self.config.response_stop_loss_weight)
            if response_stop_loss_weight is None
            else max(0.0, float(response_stop_loss_weight))
        )
        response_stop_loss, response_stop_metrics = self._response_stop_loss(
            logits,
            labels,
            stop_token_ids=tuple(int(token_id) for token_id in stop_token_ids),
            active_weight=float(effective_response_stop_loss_weight),
        )
        loss = (
            clean_loss
            + float(self.config.token_maturation_aux_loss_weight)
            * token_maturation_aux_loss.to(clean_loss.dtype)
            + float(self.config.answer_memory_aux_loss_weight)
            * answer_memory_aux_loss.to(clean_loss.dtype)
            + float(self.config.answer_prefix_commitment_loss_weight)
            * answer_prefix_commitment_loss.to(clean_loss.dtype)
            + float(self.config.imta_diversity_weight) * imta_diversity_loss.to(clean_loss.dtype)
            + float(self.config.imta_route_entropy_weight) * imta_route_entropy_loss.to(clean_loss.dtype)
            + float(self.config.imta_route_balance_weight) * imta_route_balance_loss.to(clean_loss.dtype)
            + float(self.config.own_latent_prediction_weight) * own_latent_loss.to(clean_loss.dtype)
            + float(self.config.repeat_unlikelihood_weight) * repeat_unlikelihood_loss.to(clean_loss.dtype)
            + float(self.config.premature_stop_loss_weight) * premature_stop_loss.to(clean_loss.dtype)
            + float(self.config.response_start_loss_weight) * response_start_loss.to(clean_loss.dtype)
            + float(self.config.response_start_stop_margin_weight)
            * response_start_stop_margin_loss.to(clean_loss.dtype)
            + float(self.config.response_body_loss_weight) * response_body_loss.to(clean_loss.dtype)
            + float(effective_response_continue_stop_margin_weight)
            * response_continue_stop_margin_loss.to(clean_loss.dtype)
            + float(effective_response_stop_loss_weight) * response_stop_loss.to(clean_loss.dtype)
        )
        loss_metrics = {
            **metrics,
            "loss": float(loss.detach().float().cpu().item()),
            "clean_loss": float(clean_loss.detach().float().cpu().item()),
            "token_maturation_aux_loss_weight": float(self.config.token_maturation_aux_loss_weight),
            **token_maturation_aux_metrics,
            "answer_memory_aux_loss_weight": float(self.config.answer_memory_aux_loss_weight),
            **answer_memory_aux_metrics,
            "answer_prefix_commitment_loss_weight": float(self.config.answer_prefix_commitment_loss_weight),
            **answer_prefix_commitment_metrics,
            "imta_diversity_weight": float(self.config.imta_diversity_weight),
            "imta_route_entropy_weight": float(self.config.imta_route_entropy_weight),
            "imta_route_balance_weight": float(self.config.imta_route_balance_weight),
            "own_latent_prediction_weight": float(self.config.own_latent_prediction_weight),
            "repeat_unlikelihood_weight": float(self.config.repeat_unlikelihood_weight),
            "premature_stop_loss_weight": float(self.config.premature_stop_loss_weight),
            "response_start_loss_weight": float(self.config.response_start_loss_weight),
            "response_start_stop_margin_weight": float(self.config.response_start_stop_margin_weight),
            "response_start_stop_margin": float(self.config.response_start_stop_margin),
            "response_body_loss_weight": float(self.config.response_body_loss_weight),
            "response_continue_stop_margin_weight": float(effective_response_continue_stop_margin_weight),
            "response_continue_stop_margin_target_weight": float(
                self.config.response_continue_stop_margin_weight
            ),
            "response_continue_stop_margin": float(self.config.response_continue_stop_margin),
            "response_stop_loss_weight": float(effective_response_stop_loss_weight),
            "response_stop_loss_target_weight": float(self.config.response_stop_loss_weight),
            "imta_diversity_loss": float(imta_diversity_loss.detach().float().cpu().item()),
            "imta_route_entropy_loss": float(imta_route_entropy_loss.detach().float().cpu().item()),
            "imta_route_balance_loss": float(imta_route_balance_loss.detach().float().cpu().item()),
            "own_latent_prediction_loss": float(own_latent_loss.detach().float().cpu().item()),
            "repeat_unlikelihood_loss": float(repeat_unlikelihood_loss.detach().float().cpu().item()),
            "premature_stop_loss": float(premature_stop_loss.detach().float().cpu().item()),
            "response_start_loss": float(response_start_loss.detach().float().cpu().item()),
            "response_start_stop_margin_loss": float(
                response_start_stop_margin_loss.detach().float().cpu().item()
            ),
            "response_body_loss": float(response_body_loss.detach().float().cpu().item()),
            "response_continue_stop_margin_loss": float(
                response_continue_stop_margin_loss.detach().float().cpu().item()
            ),
            "response_stop_loss": float(response_stop_loss.detach().float().cpu().item()),
            **premature_stop_metrics,
            **response_start_metrics,
            **response_start_stop_margin_metrics,
            **response_body_metrics,
            **response_continue_stop_margin_metrics,
            **response_stop_metrics,
        }
        return loss, loss_metrics


# Backward-compatible implementation alias.  Older checkpoints and historical
# scripts may still refer to QTRMReasoningLMV2 after the public W-GRAM rebrand.
QTRMReasoningLMV2 = WGRAMReasoningLMV2
