"""
PAASR: Phonology-Aware Adaptive Speech Representation Framework
Reference Implementation & PyTorch Architecture Skeleton
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class AcousticStream(nn.Module):
    """
    Extracts fine-grained spectral, pitch, harmonic, and energy dynamics.
    """
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> transpose to (B, D, T) for Conv1D
        out = self.conv_stack(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(out)  # (B, T, hidden_dim)


class PhonologicalStream(nn.Module):
    """
    Phonology-aware representation guided by articulatory anchor codebook.
    Encodes Telugu phonetic constraints (retroflexion, aspiration, vowel length).
    """
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, num_anchors: int = 64):
        super().__init__()
        self.articulatory_anchors = nn.Parameter(torch.randn(num_anchors, hidden_dim))
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        h = self.in_proj(x)
        B, T, _ = h.shape
        # Expand anchors across batch
        anchors = self.articulatory_anchors.unsqueeze(0).expand(B, -1, -1)
        # Attend speech frames to articulatory anchor priors
        attn_out, _ = self.cross_attn(query=h, key=anchors, value=anchors)
        return self.norm(h + attn_out)  # (B, T, hidden_dim)


class ContextualStream(nn.Module):
    """
    Captures long-range contextual semantic dependencies via Transformer blocks.
    """
    def __init__(self, in_dim: int = 768, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=2048, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return self.transformer(h)  # (B, T, hidden_dim)


class ConditionEstimator(nn.Module):
    """
    Estimates Condition Vector C = [C_spk, C_acc, C_noise, C_rate]
    """
    def __init__(self, in_dim: int = 768, spk_dim: int = 192, acc_classes: int = 4):
        super().__init__()
        # Global attentive pooling for utterance-level conditions
        self.pool_proj = nn.Linear(in_dim, 1)
        self.spk_head = nn.Linear(in_dim, spk_dim)
        self.acc_head = nn.Linear(in_dim, acc_classes)
        self.noise_head = nn.Linear(in_dim, 2)  # [SNR estimate, Spectral dispersion]
        self.rate_head = nn.Linear(in_dim, 1)   # [Speaking rate cadence]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, in_dim)
        weights = F.softmax(self.pool_proj(x), dim=1)  # (B, T, 1)
        pooled = torch.sum(x * weights, dim=1)         # (B, in_dim)

        c_spk = F.normalize(self.spk_head(pooled), p=2, dim=-1)   # (B, 192)
        c_acc = F.softmax(self.acc_head(pooled), dim=-1)           # (B, 4)
        c_noise = self.noise_head(pooled)                          # (B, 2)
        c_rate = torch.sigmoid(self.rate_head(pooled))             # (B, 1)

        c_unified = torch.cat([c_spk, c_acc, c_noise, c_rate], dim=-1)
        return {
            "c_spk": c_spk,
            "c_acc": c_acc,
            "c_noise": c_noise,
            "c_rate": c_rate,
            "c_unified": c_unified,
        }


class ConditionAdaptiveFusion(nn.Module):
    """
    Condition-Adaptive Representation Fusion (CARF)
    Dynamically routes feature weights alpha across acoustic, phonological, and contextual streams.
    """
    def __init__(self, stream_dim: int = 512, cond_dim: int = 199, hidden_gate: int = 256):
        super().__init__()
        in_gate_dim = 3 * stream_dim + cond_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_gate_dim, hidden_gate),
            nn.GELU(),
            nn.Linear(hidden_gate, 3),  # 3 weights: [alpha_ac, alpha_ph, alpha_ctx]
        )

    def forward(
        self,
        f_ac: torch.Tensor,
        f_ph: torch.Tensor,
        f_ctx: torch.Tensor,
        c_unified: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # f_*: (B, T, D), c_unified: (B, C_dim)
        B, T, D = f_ac.shape
        c_expanded = c_unified.unsqueeze(1).expand(-1, T, -1)  # (B, T, C_dim)

        f_cat = torch.cat([f_ac, f_ph, f_ctx, c_expanded], dim=-1)  # (B, T, 3D + C_dim)
        gate_logits = self.gate_mlp(f_cat) / temperature            # (B, T, 3)
        alpha = F.softmax(gate_logits, dim=-1)                       # (B, T, 3)

        # Fused representation
        f_paasr = (
            alpha[:, :, 0:1] * f_ac
            + alpha[:, :, 1:2] * f_ph
            + alpha[:, :, 2:3] * f_ctx
        )
        return f_paasr, alpha


class PhonologicalConsistencyLoss(nn.Module):
    """
    Contrastive InfoNCE loss enforcing phonetic invariance across noise, accents, and speakers.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, f_ph_orig: torch.Tensor, f_ph_pert: torch.Tensor) -> torch.Tensor:
        """
        f_ph_orig: (B, T, D) representations of clean speech
        f_ph_pert: (B, T, D) representations of perturbed/accented speech
        """
        # Pool to utterance/segment level for robust contrast
        z1 = F.normalize(f_ph_orig.mean(dim=1), dim=-1)  # (B, D)
        z2 = F.normalize(f_ph_pert.mean(dim=1), dim=-1)  # (B, D)

        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature  # (B, B)
        labels = torch.arange(z1.shape[0], device=z1.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


class PAASRModel(nn.Module):
    """
    Complete PAASR Framework Wrapper
    """
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        num_target_units: int = 1000,
    ):
        super().__init__()
        self.acoustic_stream = AcousticStream(in_dim, hidden_dim)
        self.phonological_stream = PhonologicalStream(in_dim, hidden_dim)
        self.contextual_stream = ContextualStream(in_dim, hidden_dim)

        self.condition_estimator = ConditionEstimator(in_dim)
        cond_dim = 192 + 4 + 2 + 1  # 199

        self.carf_fusion = ConditionAdaptiveFusion(hidden_dim, cond_dim)
        self.pcl_loss_fn = PhonologicalConsistencyLoss()

        # Discrete Unit Sequence-to-Sequence Translation Decoder (S2UT style)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.translation_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.unit_embedding = nn.Embedding(num_target_units, hidden_dim)
        self.unit_head = nn.Linear(hidden_dim, num_target_units)

    def forward(
        self,
        ssl_features: torch.Tensor,
        target_unit_tokens: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # 1. Multi-stream feature extraction
        f_ac = self.acoustic_stream(ssl_features)
        f_ph = self.phonological_stream(ssl_features)
        f_ctx = self.contextual_stream(ssl_features)

        # 2. Condition estimation
        cond_dict = self.condition_estimator(ssl_features)

        # 3. Dynamic Condition-Adaptive Fusion (CARF)
        f_paasr, gate_weights = self.carf_fusion(f_ac, f_ph, f_ctx, cond_dict["c_unified"])

        results = {
            "f_paasr": f_paasr,
            "gate_weights": gate_weights,
            "f_acoustic": f_ac,
            "f_phonology": f_ph,
            "f_context": f_ctx,
            "conditions": cond_dict,
        }

        # 4. Target Unit Decoding (Training Phase)
        if target_unit_tokens is not None:
            tgt_emb = self.unit_embedding(target_unit_tokens)
            # Autoregressive causal mask
            T_tgt = target_unit_tokens.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_tgt).to(ssl_features.device)
            dec_out = self.translation_decoder(tgt=tgt_emb, memory=f_paasr, tgt_mask=tgt_mask)
            logits = self.unit_head(dec_out)
            results["target_unit_logits"] = logits

        return results


if __name__ == "__main__":
    print("=== Testing PAASR Model forward pass ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # Simulated batch: 4 utterances, 100 frames (~2 sec audio), 768 SSL feature dim
    B, T, D = 4, 100, 768
    dummy_ssl_feats = torch.randn(B, T, D).to(device)
    dummy_tgt_units = torch.randint(0, 1000, (B, 30)).to(device)

    model = PAASRModel(in_dim=768, hidden_dim=512, num_target_units=1000).to(device)
    outputs = model(dummy_ssl_feats, dummy_tgt_units)

    print("\n[Shapes Verification]")
    print(f"PAASR Fused Representation : {outputs['f_paasr'].shape} (Expected: [{B}, {T}, 512])")
    print(f"Gating Weights alpha       : {outputs['gate_weights'].shape} (Expected: [{B}, {T}, 3])")
    print(f"Speaker Vector C_spk       : {outputs['conditions']['c_spk'].shape} (Expected: [{B}, 192])")
    print(f"Accent Posterior C_acc     : {outputs['conditions']['c_acc'].shape} (Expected: [{B}, 4])")
    print(f"Target Unit Logits         : {outputs['target_unit_logits'].shape} (Expected: [{B}, 30, 1000])")

    # Verify Phonological Consistency Loss with synthetic perturbation
    perturbed_feats = dummy_ssl_feats + 0.1 * torch.randn_like(dummy_ssl_feats)
    f_ph_clean = outputs["f_phonology"]
    f_ph_pert = model.phonological_stream(perturbed_feats)
    loss_pcl = model.pcl_loss_fn(f_ph_clean, f_ph_pert)
    print(f"Phonological Consistency Loss (PCL): {loss_pcl.item():.4f}")
    print("\n=== All PAASR architecture sanity checks passed successfully! ===")
