import os
import types
from typing import Dict

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaForCausalLM, repeat_kv
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from base.h2o_kv_cache import (
    BaseSampler,
    llama_h2o_attention_forward,
    qwen3_h2o_attention_forward,
)
from base.tuple_kv_cache import (
    enable_tuple_kv_cache_for_llama,
    enable_tuple_kv_cache_for_qwen,
    enable_tuple_kv_cache_for_qwen3,
)


def _resolve_learned_loki_path(load_path: str) -> str:
    if os.path.isdir(load_path):
        load_path = os.path.join(load_path, "learned_loki.pt")
    if not os.path.exists(load_path):
        raise ValueError(f"Learned-Loki checkpoint {load_path} does not exist")
    return load_path


def load_learned_loki_checkpoint(load_path: str) -> Dict:
    load_path = _resolve_learned_loki_path(load_path)
    checkpoint = torch.load(load_path, map_location="cpu")
    if "projection_weight_dict" not in checkpoint:
        raise ValueError(f"Invalid Learned-Loki checkpoint format: {load_path}")
    return checkpoint


class LearnedLokiKV(BaseSampler):
    def __init__(
        self,
        budget_ratio: float,
        sink_size: int,
        recent_size: int,
        num_key_value_groups: int,
        projection_weight: torch.Tensor,
    ):
        super().__init__(budget_ratio=budget_ratio, window_size=sink_size + recent_size)
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.num_key_value_groups = num_key_value_groups
        self.projection_weight = projection_weight.float()

    @property
    def budget(self):
        protected = self.sink_size + self.recent_size
        middle_tokens = max(self.seq_len - protected, 0)
        middle_budget = min(int(middle_tokens * self.budget_ratio), middle_tokens)
        return protected + middle_budget

    def _approximate_scores(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        projected_query = F.linear(
            query_states[:, :, -1, :].float(),
            self.projection_weight,
        )
        projected_keys = F.linear(
            key_states.transpose(1, 2).float(),
            self.projection_weight,
        )
        expanded_projected_keys = repeat_kv(
            projected_keys.transpose(1, 2),
            self.num_key_value_groups,
        ).transpose(1, 2)
        approx_scores = (
            torch.einsum(
                "bhr,bhsr->bhs",
                projected_query.float(),
                expanded_projected_keys.float(),
            )
            / (self.projection_weight.shape[0] ** 0.5)
        )
        return approx_scores.mean(dim=1)

    def update_kv(self, key_states, query_states, value_states, attention_mask=None):
        del attention_mask
        self.seq_len += query_states.shape[2]
        bsz, kv_heads, kv_len, head_dim = key_states.shape
        cache_budget = min(self.budget, kv_len)
        if kv_len <= cache_budget:
            return (key_states, value_states)

        keep_sink = min(self.sink_size, cache_budget, kv_len)
        remaining_after_sink = max(cache_budget - keep_sink, 0)
        keep_recent = min(self.recent_size, remaining_after_sink, kv_len - keep_sink)
        middle_end = kv_len - keep_recent
        num_select = max(cache_budget - keep_sink - keep_recent, 0)

        protected_indices = []
        if keep_sink > 0:
            protected_indices.append(
                torch.arange(keep_sink, device=key_states.device, dtype=torch.long)
            )
        if keep_recent > 0:
            protected_indices.append(
                torch.arange(middle_end, kv_len, device=key_states.device, dtype=torch.long)
            )
        protected_indices = (
            torch.cat(protected_indices, dim=0)
            if protected_indices
            else torch.empty(0, device=key_states.device, dtype=torch.long)
        )

        if num_select > 0 and middle_end > keep_sink:
            approx_scores = self._approximate_scores(key_states, query_states)
            candidate_indices = torch.arange(
                keep_sink, middle_end, device=key_states.device, dtype=torch.long
            )
            selected_middle = torch.topk(
                approx_scores[:, candidate_indices],
                k=min(num_select, candidate_indices.numel()),
                dim=-1,
            ).indices
            selected_middle = candidate_indices[selected_middle]
        else:
            selected_middle = torch.empty(
                bsz, 0, dtype=torch.long, device=key_states.device
            )

        if protected_indices.numel() > 0:
            keep_idx = torch.cat(
                [protected_indices.unsqueeze(0).expand(bsz, -1), selected_middle], dim=-1
            )
        else:
            keep_idx = selected_middle

        keep_idx = keep_idx.sort(dim=-1).values
        gather_idx = keep_idx.unsqueeze(1).unsqueeze(-1).expand(-1, kv_heads, -1, head_dim)
        key_states = key_states.gather(2, gather_idx)
        value_states = value_states.gather(2, gather_idx)
        return (key_states, value_states)


def _extract_projection_weights(checkpoint: Dict, num_layers: int) -> list[torch.Tensor]:
    projection_weight_dict = checkpoint["projection_weight_dict"]
    weights = []
    for layer_idx in range(num_layers):
        layer_key = layer_idx if layer_idx in projection_weight_dict else str(layer_idx)
        if layer_key not in projection_weight_dict:
            raise ValueError(f"Missing Learned-Loki projection for layer {layer_idx}")
        weights.append(projection_weight_dict[layer_key].float())
    return weights


def enable_llama_learned_loki_eval(
    model: LlamaForCausalLM,
    checkpoint: Dict,
    budget_ratio: float,
    sink_size: int,
    recent_size: int,
):
    enable_tuple_kv_cache_for_llama(model)
    projection_weights = _extract_projection_weights(
        checkpoint, len(model.model.layers)
    )

    for layer_idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        module.kv_sampler = LearnedLokiKV(
            budget_ratio=budget_ratio,
            sink_size=sink_size,
            recent_size=recent_size,
            num_key_value_groups=module.num_key_value_groups,
            projection_weight=projection_weights[layer_idx].to(
                device=next(module.parameters()).device,
                dtype=torch.float32,
            ),
        )
        module.forward = types.MethodType(llama_h2o_attention_forward, module)


def enable_qwen_learned_loki_eval(
    model: Qwen2ForCausalLM,
    checkpoint: Dict,
    budget_ratio: float,
    sink_size: int,
    recent_size: int,
):
    enable_tuple_kv_cache_for_qwen(model)
    projection_weights = _extract_projection_weights(
        checkpoint, len(model.model.layers)
    )

    for layer_idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        module.kv_sampler = LearnedLokiKV(
            budget_ratio=budget_ratio,
            sink_size=sink_size,
            recent_size=recent_size,
            num_key_value_groups=module.num_key_value_groups,
            projection_weight=projection_weights[layer_idx].to(
                device=next(module.parameters()).device,
                dtype=torch.float32,
            ),
        )
        module.forward = types.MethodType(llama_h2o_attention_forward, module)


def enable_qwen3_learned_loki_eval(
    model: Qwen3ForCausalLM,
    checkpoint: Dict,
    budget_ratio: float,
    sink_size: int,
    recent_size: int,
):
    enable_tuple_kv_cache_for_qwen3(model)
    projection_weights = _extract_projection_weights(
        checkpoint, len(model.model.layers)
    )

    for layer_idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        module.kv_sampler = LearnedLokiKV(
            budget_ratio=budget_ratio,
            sink_size=sink_size,
            recent_size=recent_size,
            num_key_value_groups=module.num_key_value_groups,
            projection_weight=projection_weights[layer_idx].to(
                device=next(module.parameters()).device,
                dtype=torch.float32,
            ),
        )
        module.forward = types.MethodType(qwen3_h2o_attention_forward, module)


def enable_learned_loki_eval(
    model,
    checkpoint: Dict,
    budget_ratio: float,
    sink_size: int,
    recent_size: int,
):
    if "llama" in model.config.model_type:
        enable_llama_learned_loki_eval(
            model, checkpoint, budget_ratio, sink_size, recent_size
        )
    elif "qwen2" in model.config.model_type:
        enable_qwen_learned_loki_eval(
            model, checkpoint, budget_ratio, sink_size, recent_size
        )
    elif "qwen3" in model.config.model_type:
        enable_qwen3_learned_loki_eval(
            model, checkpoint, budget_ratio, sink_size, recent_size
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")
