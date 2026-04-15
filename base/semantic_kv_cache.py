import os
import types
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
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


def _resolve_semantic_kv_path(load_path: str) -> str:
    if os.path.isdir(load_path):
        load_path = os.path.join(load_path, "semantic_kv.pt")
    if not os.path.exists(load_path):
        raise ValueError(f"Semantic KV checkpoint {load_path} does not exist")
    return load_path


def load_semantic_kv_checkpoint(load_path: str) -> Dict:
    load_path = _resolve_semantic_kv_path(load_path)
    checkpoint = torch.load(load_path, map_location="cpu")
    if "projection_weight_dict" not in checkpoint:
        raise ValueError(f"Invalid semantic KV checkpoint format: {load_path}")
    return checkpoint


def _compute_attention_importance(
    key_states: torch.Tensor,
    query_states: torch.Tensor,
    num_key_value_groups: int,
    scaling: float,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    key_states_rep = repeat_kv(key_states, num_key_value_groups)
    attn_weights = (
        torch.matmul(query_states, key_states_rep.transpose(2, 3)) * scaling
    )
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)

    # Aggregate over all query heads and query positions to get token importance.
    return attn_weights.sum(dim=(1, 2))


def _greedy_farthest_point_select(
    candidate_features: torch.Tensor,
    candidate_scores: torch.Tensor,
    num_select: int,
    retained_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, num_candidates, _ = candidate_features.shape
    if num_select <= 0 or num_candidates == 0:
        return torch.empty(
            batch_size, 0, dtype=torch.long, device=candidate_features.device
        )

    selected_indices = []
    for batch_idx in range(batch_size):
        feats = candidate_features[batch_idx]
        scores = candidate_scores[batch_idx]

        current_retained = (
            retained_features[batch_idx]
            if retained_features is not None
            else feats.new_empty((0, feats.shape[-1]))
        )
        remaining_mask = torch.ones(num_candidates, dtype=torch.bool, device=feats.device)
        chosen = []

        if current_retained.numel() == 0:
            seed_idx = scores.argmax()
            chosen.append(seed_idx.item())
            remaining_mask[seed_idx] = False
            current_retained = feats[seed_idx : seed_idx + 1]

        target_count = min(num_select, num_candidates)
        while len(chosen) < target_count:
            remaining_idx = remaining_mask.nonzero(as_tuple=False).flatten()
            if remaining_idx.numel() == 0:
                break

            remaining_feats = feats[remaining_idx]
            min_distance = torch.cdist(remaining_feats, current_retained, p=2).min(
                dim=-1
            ).values
            joint_score = scores[remaining_idx] * min_distance
            best_idx = remaining_idx[joint_score.argmax()]

            chosen.append(best_idx.item())
            remaining_mask[best_idx] = False
            current_retained = torch.cat(
                [current_retained, feats[best_idx : best_idx + 1]], dim=0
            )

        selected_indices.append(torch.tensor(chosen, device=feats.device, dtype=torch.long))

    return torch.stack(selected_indices, dim=0)


def _assign_clusters(
    token_features: torch.Tensor,
    selected_features: torch.Tensor,
) -> torch.Tensor:
    if selected_features.numel() == 0:
        return torch.zeros(
            token_features.shape[:2], dtype=torch.long, device=token_features.device
        )

    distances = torch.cdist(token_features, selected_features, p=2)
    return distances.argmin(dim=-1)


class SemanticKV(BaseSampler):
    def __init__(
        self,
        budget_ratio: float,
        sink_size: int,
        recent_size: int,
        num_key_value_groups: int,
        scaling: float,
        projection_weight: torch.Tensor,
    ):
        super().__init__(budget_ratio=budget_ratio, window_size=sink_size + recent_size)
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.num_key_value_groups = num_key_value_groups
        self.scaling = scaling
        self.projection_weight = projection_weight.float()
        self.importance_scores = None
        self.last_cluster_assignment = None

    def _projection_weight_for(self, device: torch.device) -> torch.Tensor:
        if self.projection_weight.device != device:
            self.projection_weight = self.projection_weight.to(
                device=device, dtype=torch.float32
            )
        return self.projection_weight

    def reset(self):
        super().reset()
        self.importance_scores = None
        self.last_cluster_assignment = None

    def _project_tokens(self, key_states: torch.Tensor) -> torch.Tensor:
        token_keys = key_states.mean(dim=1)
        return F.linear(
            token_keys.float(), self._projection_weight_for(key_states.device)
        )

    def update_kv(self, key_states, query_states, value_states, attention_mask=None):
        bsz, kv_heads, kv_len, head_dim = key_states.shape
        q_len = query_states.shape[2]

        self.seq_len += q_len
        cache_budget = min(self.budget, kv_len)
        if cache_budget <= 0:
            return (key_states, value_states)

        importance = _compute_attention_importance(
            key_states=key_states,
            query_states=query_states,
            num_key_value_groups=self.num_key_value_groups,
            scaling=self.scaling,
            attention_mask=attention_mask,
        )

        if self.importance_scores is None:
            self.importance_scores = importance
        else:
            importance[..., :-q_len] += self.importance_scores
            self.importance_scores = importance

        if kv_len <= cache_budget:
            return (key_states, value_states)

        keep_sink = min(self.sink_size, cache_budget, kv_len)
        remaining_after_sink = max(cache_budget - keep_sink, 0)
        keep_recent = min(self.recent_size, remaining_after_sink, kv_len - keep_sink)
        middle_end = kv_len - keep_recent
        num_select = max(cache_budget - keep_sink - keep_recent, 0)

        projected_keys = self._project_tokens(key_states)

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

        if protected_indices.numel() > 0:
            protected_features = projected_keys[:, protected_indices, :]
        else:
            protected_features = None

        candidate_indices = torch.arange(
            keep_sink, middle_end, device=key_states.device, dtype=torch.long
        )
        candidate_features = projected_keys[:, candidate_indices, :]
        candidate_scores = self.importance_scores[:, candidate_indices]

        selected_middle = _greedy_farthest_point_select(
            candidate_features=candidate_features,
            candidate_scores=candidate_scores,
            num_select=num_select,
            retained_features=protected_features,
        )

        if selected_middle.numel() > 0:
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

        selected_features = projected_keys.gather(
            1, keep_idx.unsqueeze(-1).expand(-1, -1, projected_keys.shape[-1])
        )
        self.last_cluster_assignment = _assign_clusters(projected_keys, selected_features)

        key_states = key_states.gather(2, gather_idx)
        value_states = value_states.gather(2, gather_idx)
        self.importance_scores = self.importance_scores.gather(1, keep_idx)
        return (key_states, value_states)


def _extract_projection_weights(checkpoint: Dict, num_layers: int) -> list[torch.Tensor]:
    projection_weight_dict = checkpoint["projection_weight_dict"]
    weights = []
    for layer_idx in range(num_layers):
        layer_key = layer_idx if layer_idx in projection_weight_dict else str(layer_idx)
        if layer_key not in projection_weight_dict:
            raise ValueError(f"Missing semantic KV projection for layer {layer_idx}")
        weights.append(projection_weight_dict[layer_key].float())
    return weights


def enable_llama_semantic_kv_eval(
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
        module.kv_sampler = SemanticKV(
            budget_ratio=budget_ratio,
            sink_size=sink_size,
            recent_size=recent_size,
            num_key_value_groups=module.num_key_value_groups,
            scaling=module.scaling,
            projection_weight=projection_weights[layer_idx].to(
                device=next(module.parameters()).device,
                dtype=torch.float32,
            ),
        )
        module.forward = types.MethodType(llama_h2o_attention_forward, module)


def enable_qwen_semantic_kv_eval(
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
        module.kv_sampler = SemanticKV(
            budget_ratio=budget_ratio,
            sink_size=sink_size,
            recent_size=recent_size,
            num_key_value_groups=module.num_key_value_groups,
            scaling=module.scaling,
            projection_weight=projection_weights[layer_idx].to(
                device=next(module.parameters()).device,
                dtype=torch.float32,
            ),
        )
        module.forward = types.MethodType(llama_h2o_attention_forward, module)


def enable_qwen3_semantic_kv_eval(
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
        module.kv_sampler = SemanticKV(
            budget_ratio=budget_ratio,
            sink_size=sink_size,
            recent_size=recent_size,
            num_key_value_groups=module.num_key_value_groups,
            scaling=module.scaling,
            projection_weight=projection_weights[layer_idx].to(
                device=next(module.parameters()).device,
                dtype=torch.float32,
            ),
        )
        module.forward = types.MethodType(qwen3_h2o_attention_forward, module)


def enable_semantic_kv_eval(
    model,
    checkpoint: Dict,
    budget_ratio: float,
    sink_size: int,
    recent_size: int,
):
    if "llama" in model.config.model_type:
        enable_llama_semantic_kv_eval(
            model, checkpoint, budget_ratio, sink_size, recent_size
        )
    elif "qwen2" in model.config.model_type:
        enable_qwen_semantic_kv_eval(
            model, checkpoint, budget_ratio, sink_size, recent_size
        )
    elif "qwen3" in model.config.model_type:
        enable_qwen3_semantic_kv_eval(
            model, checkpoint, budget_ratio, sink_size, recent_size
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")
