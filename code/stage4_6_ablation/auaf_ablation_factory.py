from __future__ import annotations

from typing import Mapping
import torch
from torch import nn

from auaf_model_factory import (
    AdaptiveUncertaintyAwareFusion,
    build_model,
    require_at_least_one_modality,
)

ABLATION_VARIANTS = [
    "AUAF",
    "AUAF_NoMaskTokens",
    "AUAF_NoContentGate",
    "AUAF_NoStructuralGate",
    "AUAF_UniformGate",
    "AUAF_NoTransformer",
    "AUAF_NoModalityDropout",
]

VARIANT_DEFINITIONS = {
    "AUAF": "Original AUAF implementation; both gates, mask-token inputs, Transformer blocks and structured modality dropout retained.",
    "AUAF_NoMaskTokens": "For unavailable modalities, replace the learned mask-token vector with zeros. The mask-token parameters remain instantiated but are disconnected to preserve matched initialization. The original key-padding mask and availability-aware gate remain unchanged.",
    "AUAF_NoContentGate": "Remove the content-derived gate score; retain only the structural availability-derived gate score.",
    "AUAF_NoStructuralGate": "Remove the structural availability-derived gate score; retain only the content-derived gate score, with unavailable modalities still masked before softmax.",
    "AUAF_UniformGate": "Remove both learned gate scores and assign equal weights across the modalities available for each participant.",
    "AUAF_NoTransformer": "Bypass all cross-modal Transformer blocks and the CLS aggregation branch; classify the availability-weighted modality-token summary directly.",
    "AUAF_NoModalityDropout": "Disable structured modality dropout during training; retain the original inference-time availability mask, Transformer and both gates.",
}

class AblationAUAF(AdaptiveUncertaintyAwareFusion):
    """AUAF with one prespecified component function disabled.

    All modules/parameters are instantiated in the same order as original AUAF
    so common parameters start from matched random initialization for a given
    seed. Disabled component parameters remain present but receive no gradient.
    """
    def __init__(self, dimensions: Mapping[str, int], n_classes: int, variant: str):
        if variant not in ABLATION_VARIANTS or variant == "AUAF":
            raise ValueError(f"Unsupported ablation variant: {variant}")
        super().__init__(dimensions, n_classes)
        self.ablation_variant = variant
        # Remove the selected component after parent initialization. This keeps
        # common layers identically initialized while changing active capacity.
        if variant == "AUAF_NoMaskTokens":
            self.mask_tokens = nn.ParameterDict()
        elif variant == "AUAF_NoContentGate":
            self.gating.content_gate = None
        elif variant == "AUAF_NoStructuralGate":
            self.gating.struct_gate = None
        elif variant == "AUAF_UniformGate":
            self.gating = None
        elif variant == "AUAF_NoTransformer":
            self.transformer_blocks = nn.ModuleList([])
            self.cls_token = None
        elif variant == "AUAF_NoModalityDropout":
            self.modality_dropout = 0.0

    def forward(self, inputs, mask, return_aux=False):
        require_at_least_one_modality(mask)
        effective_mask = self._apply_modality_dropout(mask)
        batch_size = effective_mask.shape[0]

        tokens = torch.stack(
            [self.encoders[m](inputs[m]) for m in self.modalities], dim=1
        )
        encoded_tokens = []
        for j, modality in enumerate(self.modalities):
            present = effective_mask[:, j].unsqueeze(1)
            if self.ablation_variant == "AUAF_NoMaskTokens":
                missing_representation = torch.zeros(
                    batch_size, tokens.shape[-1], dtype=tokens.dtype, device=tokens.device
                )
            else:
                missing_representation = self.mask_tokens[modality].expand(batch_size, -1)
            encoded_tokens.append(
                present * tokens[:, j, :] + (1.0 - present) * missing_representation
            )
        tokens = torch.stack(encoded_tokens, dim=1)

        if self.ablation_variant == "AUAF_NoTransformer":
            modality_tokens = tokens + self.pos_emb[:, 1:, :]
            attention_maps = []
            sequence = None
        else:
            cls = self.cls_token.expand(batch_size, -1, -1)
            sequence = torch.cat([cls, tokens], dim=1) + self.pos_emb
            padding_mask = torch.cat([
                torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device),
                effective_mask.eq(0)
            ], dim=1)
            attention_maps = []
            for block in self.transformer_blocks:
                sequence, attn = block(sequence, padding_mask)
                attention_maps.append(attn)
            modality_tokens = sequence[:, 1:, :]

        if self.ablation_variant == "AUAF_NoContentGate":
            scores = self.gating.struct_gate(effective_mask)
            scores = scores.masked_fill(effective_mask <= 0, -1e9)
            gates = torch.softmax(scores, dim=1)
        elif self.ablation_variant == "AUAF_NoStructuralGate":
            scores = self.gating.content_gate(modality_tokens).squeeze(-1)
            scores = scores.masked_fill(effective_mask <= 0, -1e9)
            gates = torch.softmax(scores, dim=1)
        elif self.ablation_variant == "AUAF_UniformGate":
            gates = effective_mask / effective_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            gates = self.gating(modality_tokens, effective_mask)

        gated = (modality_tokens * gates.unsqueeze(-1)).sum(dim=1)
        if self.ablation_variant == "AUAF_NoTransformer":
            fused = gated
        else:
            cls_output = sequence[:, 0, :]
            fused = 0.5 * gated + 0.5 * cls_output

        logits = self.classifier(self.mc_dropout(fused))
        aux = {
            "gates": gates,
            "attention": (
                torch.stack(attention_maps)
                if attention_maps else torch.empty(0, device=logits.device)
            ),
        }
        return (logits, aux) if return_aux else logits


def build_ablation_model(variant: str, dimensions: Mapping[str, int], n_classes: int):
    if variant == "AUAF":
        return build_model("AUAF", dimensions, n_classes)
    return AblationAUAF(dimensions, n_classes, variant)
