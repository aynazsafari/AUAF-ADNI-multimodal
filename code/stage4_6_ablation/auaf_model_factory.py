"""Canonical neural model factory for the AUAF Stage 4.5 project.

Architecture constants are versioned with this file and included in every run hash.
"""
from typing import Mapping
import numpy as np
import torch
import torch.nn as nn

LATENT_DIM = 128
MODALITY_DROPOUT = 0.20
MC_PASSES = 30

def require_at_least_one_modality(mask: torch.Tensor):
    if mask.ndim != 2:
        raise ValueError("Modality mask must have shape (batch, n_modalities).")
    if torch.any(mask.sum(dim=1) <= 0):
        raise ValueError("At least one modality must be available for every evaluated participant.")


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.20):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 2, width), nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM, dropout: float = 0.20):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, latent_dim), nn.LayerNorm(latent_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks = nn.Sequential(
            ResidualMLPBlock(latent_dim, dropout),
            ResidualMLPBlock(latent_dim, dropout)
        )

    def forward(self, x):
        return self.blocks(self.input(x))


class EarlyFusionMLP(nn.Module):
    def __init__(self, dimensions: Mapping[str, int], n_classes: int):
        super().__init__()
        self.modalities = list(dimensions)
        total = sum(dimensions.values()) + len(dimensions)
        self.net = nn.Sequential(
            nn.Linear(total, LATENT_DIM * 2),
            nn.LayerNorm(LATENT_DIM * 2),
            nn.GELU(),
            nn.Dropout(0.25),
            ResidualMLPBlock(LATENT_DIM * 2, 0.25),
            nn.Linear(LATENT_DIM * 2, n_classes)
        )

    def forward(self, inputs, mask, return_aux=False):
        require_at_least_one_modality(mask)
        x = torch.cat([inputs[m] for m in self.modalities] + [mask], dim=1)
        logits = self.net(x)
        return (logits, {}) if return_aux else logits


class LateFusionNetwork(nn.Module):
    def __init__(self, dimensions: Mapping[str, int], n_classes: int):
        super().__init__()
        self.modalities = list(dimensions)
        self.encoders = nn.ModuleDict({m: ModalityEncoder(d) for m, d in dimensions.items()})
        self.heads = nn.ModuleDict({m: nn.Linear(LATENT_DIM, n_classes) for m in dimensions})
        self.gate = nn.Sequential(
            nn.Linear(len(dimensions), LATENT_DIM // 2),
            nn.GELU(),
            nn.Linear(LATENT_DIM // 2, len(dimensions))
        )

    def forward(self, inputs, mask, return_aux=False):
        require_at_least_one_modality(mask)
        tokens = [self.encoders[m](inputs[m]) for m in self.modalities]
        logits_by_modality = torch.stack(
            [self.heads[m](z) for m, z in zip(self.modalities, tokens)], dim=1
        )
        gate_logits = self.gate(mask).masked_fill(mask <= 0, -1e9)
        gates = torch.softmax(gate_logits, dim=1)
        logits = (logits_by_modality * gates.unsqueeze(-1)).sum(dim=1)
        return (logits, {"gates": gates}) if return_aux else logits


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, width: int = LATENT_DIM, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 2, width), nn.Dropout(dropout)
        )

    def forward(self, x, unavailable):
        normalized = self.norm1(x)
        attended, weights = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=unavailable,
            need_weights=True,
            average_attn_weights=True
        )
        x = x + attended
        x = x + self.feedforward(self.norm2(x))
        return x, weights


class MissingAwareAttentionFusion(nn.Module):
    def __init__(self, dimensions: Mapping[str, int], n_classes: int,
                 modality_dropout: float = MODALITY_DROPOUT):
        super().__init__()
        self.modalities = list(dimensions)
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleDict({m: ModalityEncoder(d) for m, d in dimensions.items()})
        self.modality_embedding = nn.Parameter(torch.randn(1, len(dimensions), LATENT_DIM) * 0.02)
        self.blocks = nn.ModuleList([CrossModalAttentionBlock(), CrossModalAttentionBlock()])
        self.gate = nn.Sequential(nn.LayerNorm(LATENT_DIM), nn.Linear(LATENT_DIM, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(LATENT_DIM), nn.Dropout(0.20), nn.Linear(LATENT_DIM, n_classes)
        )

    def _drop_modalities(self, mask):
        if not self.training or self.modality_dropout <= 0:
            return mask
        keep = (torch.rand_like(mask) > self.modality_dropout).float() * mask
        empty = keep.sum(dim=1) == 0
        if empty.any():
            chosen = torch.multinomial(mask[empty], 1).squeeze(1)
            keep[empty] = 0
            keep[empty, chosen] = 1
        return keep

    def forward(self, inputs, mask, return_aux=False):
        require_at_least_one_modality(mask)
        effective_mask = self._drop_modalities(mask)
        tokens = torch.stack([self.encoders[m](inputs[m]) for m in self.modalities], dim=1)
        tokens = (tokens + self.modality_embedding) * effective_mask.unsqueeze(-1)
        unavailable = effective_mask <= 0
        attention_maps = []
        for block in self.blocks:
            tokens, attention = block(tokens, unavailable)
            tokens = tokens * effective_mask.unsqueeze(-1)
            attention_maps.append(attention)
        gate_logits = self.gate(tokens).squeeze(-1).masked_fill(unavailable, -1e9)
        gates = torch.softmax(gate_logits, dim=1)
        fused = (tokens * gates.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused)
        aux = {"gates": gates, "attention": torch.stack(attention_maps)}
        return (logits, aux) if return_aux else logits


class ModalityGatingNetwork(nn.Module):
    def __init__(self, latent_dim: int, n_modalities: int, dropout: float = 0.20):
        super().__init__()
        self.content_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(latent_dim // 2, 1)
        )
        self.struct_gate = nn.Sequential(
            nn.Linear(n_modalities, n_modalities * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(n_modalities * 2, n_modalities)
        )

    def forward(self, tokens, mask):
        scores = self.content_gate(tokens).squeeze(-1) + self.struct_gate(mask)
        scores = scores.masked_fill(mask <= 0, -1e9)
        return torch.softmax(scores, dim=1)


class CrossModalTransformerBlock(nn.Module):
    def __init__(self, latent_dim: int, n_heads: int = 4, dropout: float = 0.20):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim), nn.Dropout(dropout)
        )

    def forward(self, x, key_padding_mask):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


class AdaptiveUncertaintyAwareFusion(nn.Module):
    def __init__(self, dimensions: Mapping[str, int], n_classes: int,
                 latent_dim: int = LATENT_DIM, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.25, modality_dropout: float = MODALITY_DROPOUT):
        super().__init__()
        self.modalities = list(dimensions)
        self.n_modalities = len(self.modalities)
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(d, latent_dim, dropout=dropout)
            for m, d in dimensions.items()
        })
        # Preserve the original per-modality mask-token parameterization.
        self.mask_tokens = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, latent_dim) * 0.02)
            for m in self.modalities
        })
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_modalities + 1, latent_dim) * 0.02)
        self.transformer_blocks = nn.ModuleList([
            CrossModalTransformerBlock(latent_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.gating = ModalityGatingNetwork(latent_dim, self.n_modalities, dropout)
        self.mc_dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(latent_dim // 2, n_classes)
        )

    def _apply_modality_dropout(self, mask):
        if not self.training or self.modality_dropout <= 0:
            return mask
        keep = (torch.rand_like(mask) > self.modality_dropout).float() * mask
        empty = keep.sum(dim=1) == 0
        if empty.any():
            chosen = torch.multinomial(mask[empty], 1).squeeze(1)
            keep[empty] = 0
            keep[empty, chosen] = 1
        return keep

    def forward(self, inputs, mask, return_aux=False):
        require_at_least_one_modality(mask)
        effective_mask = self._apply_modality_dropout(mask)
        batch_size = effective_mask.shape[0]
        tokens = torch.stack([self.encoders[m](inputs[m]) for m in self.modalities], dim=1)
        encoded_tokens = []
        for j, modality in enumerate(self.modalities):
            present = effective_mask[:, j].unsqueeze(1)
            mask_token = self.mask_tokens[modality].expand(batch_size, -1)
            encoded_tokens.append(
                present * tokens[:, j, :] + (1.0 - present) * mask_token
            )
        tokens = torch.stack(encoded_tokens, dim=1)
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
        cls_output = sequence[:, 0, :]
        gates = self.gating(modality_tokens, effective_mask)
        gated = (modality_tokens * gates.unsqueeze(-1)).sum(dim=1)
        fused = 0.5 * gated + 0.5 * cls_output
        logits = self.classifier(self.mc_dropout(fused))
        aux = {"gates": gates, "attention": torch.stack(attention_maps)}
        return (logits, aux) if return_aux else logits

    @torch.no_grad()
    def mc_forward(self, inputs, mask, n_passes=MC_PASSES):
        # Keep the parent model in eval mode (thus disabling structured
        # modality dropout), while selectively enabling nn.Dropout layers.
        was_training = self.training
        self.eval()
        dropout_layers = [
            module for module in self.modules()
            if isinstance(module, nn.Dropout)
        ]
        for module in dropout_layers:
            module.train(True)
        try:
            probabilities, gates = [], []
            for _ in range(n_passes):
                logits, aux = self.forward(inputs, mask, return_aux=True)
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
                gates.append(aux["gates"].cpu().numpy())
            probs = np.stack(probabilities)
            mean_prob = probs.mean(axis=0)
            return {
                "mean_prob": mean_prob,
                "std_prob": probs.std(axis=0),
                "entropy": -(mean_prob * np.log(mean_prob + 1e-8)).sum(axis=1),
                "mean_gates": np.stack(gates).mean(axis=0),
            }
        finally:
            self.train(was_training)


def build_model(variant: str, dimensions: Mapping[str, int], n_classes: int):
    if variant == "EarlyFusion":
        return EarlyFusionMLP(dimensions, n_classes)
    if variant == "LateFusion":
        return LateFusionNetwork(dimensions, n_classes)
    if variant == "AttentionFusion":
        return MissingAwareAttentionFusion(dimensions, n_classes)
    if variant == "AUAF":
        return AdaptiveUncertaintyAwareFusion(dimensions, n_classes)
    raise KeyError(f"Unknown model variant: {variant}")
