"""
model/mcat.py — Multi-Modal Cross-Attention Transformer (MCAT).
================================================================
The core model architecture fusing 4 financial data modalities through
Transformer encoders and cross-attention.

Architecture:
  Stage 1: Input Tokenization (Linear projections + CausalConv + positional + stock embeddings)
  Stage 2: Modality-Specific Transformer Encoders (self-attention within each modality)
  Stage 3: Cross-Modal Attention Fusion (price attends to sentiment, then macro)
  Stage 4: Prediction Head (mean pool + fundamentals → α-scaled regression output)

Output scaling: A learnable scalar α multiplies the raw
prediction, producing conservative initial predictions near the mean. This
prevents magnitude overshoots while preserving ranking (IC), since scaling
is a monotonic transform. The gradient learns the optimal confidence level.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_CONFIG, N_STOCKS


# ─────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution: output at time t depends only on times ≤ t.

    Inspired by Wang (2023) who showed BiLSTM preprocessing before
    Transformer dramatically improves performance. We use a lightweight
    causal convolution instead (~20K params vs ~170K for BiLSTM),
    providing local sequential context before attention provides global context.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, channels) — note: channels-last format
        Returns:
            (batch, seq_len, channels) — same shape, causally convolved
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Remove future-looking padding (keep only causal part)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional info added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer encoder block: MHSA → Add&Norm → FFN → Add&Norm.

    Uses pre-norm (LayerNorm before attention/FFN) for more stable training.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out

        # Pre-norm FFN
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block: Q from one modality, K/V from another.

    This is the core fusion mechanism. Price tokens attend to sentiment
    (or macro) tokens, learning which cross-modal information is relevant
    for each timestep.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> tuple:
        """
        Args:
            query: (batch, seq_q, d_model) — e.g., price tokens
            key_value: (batch, seq_kv, d_model) — e.g., sentiment tokens

        Returns:
            Tuple of (enriched_query, attention_weights)
            enriched_query: (batch, seq_q, d_model)
            attention_weights: (batch, n_heads, seq_q, seq_kv)
        """
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, attn_weights = self.cross_attn(q, kv, kv)
        x = query + attn_out

        x_norm = self.norm_ff(x)
        x = x + self.ffn(x_norm)
        return x, attn_weights


# ─────────────────────────────────────────────────────────────
# FULL MCAT MODEL
# ─────────────────────────────────────────────────────────────

class MCAT(nn.Module):
    """
    Multi-Modal Cross-Attention Transformer for equity return forecasting.

    Stages:
        1. Tokenize each modality into d_model-dimensional tokens
        2. Self-attention within each modality (Transformer encoder)
        3. Cross-attention: price ← sentiment, then price ← macro
        4. Mean-pool + fundamentals → regression prediction

    Args:
        n_price_features: Number of price/technical features (39)
        n_sent_features: Number of sentiment features (769 = 768 FinBERT + 1 count)
        n_fund_features: Number of fundamental features (7 or 10)
        n_macro_features: Number of macro features (~20)
        config: Model configuration dict (from config.py)
        disable_modalities: List of modalities to disable for ablation
    """

    def __init__(
        self,
        n_price_features: int = 39,
        n_sent_features: int = 769,
        n_fund_features: int = 7,
        n_macro_features: int = 20,
        config: dict = None,
        disable_modalities: list = None,
    ):
        super().__init__()

        if config is None:
            config = MODEL_CONFIG

        # Track which modalities are active (for ablation studies)
        self.disable_modalities = set(disable_modalities or [])
        self.use_sentiment = "sentiment" not in self.disable_modalities
        self.use_macro = "macro" not in self.disable_modalities
        self.use_fundamentals = "fundamentals" not in self.disable_modalities

        d = config["d_model"]
        h = config["n_heads"]
        ff = config["d_ff"]
        drop = config["dropout"]

        self.d_model = d

        # ─── Stage 1: Input Tokenization ───

        # Price branch: always active
        self.price_proj = nn.Linear(n_price_features, d)
        self.causal_conv = CausalConv1d(d, d, config["causal_conv_kernel"])
        self.stock_embedding = nn.Embedding(N_STOCKS, d)
        self.price_pos_enc = PositionalEncoding(d, config["max_seq_len"], drop)

        # Sentiment branch (conditional)
        if self.use_sentiment:
            self.sent_proj = nn.Linear(n_sent_features, d)
            self.sent_pos_enc = PositionalEncoding(d, config["max_seq_len"], drop)
            self.no_news_embedding = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Macro branch (conditional)
        if self.use_macro:
            self.macro_proj = nn.Linear(n_macro_features, d)
            self.macro_pos_enc = PositionalEncoding(d, config["max_seq_len"], drop)

        # Fundamental branch (conditional)
        if self.use_fundamentals:
            self.fund_mlp = nn.Sequential(
                nn.Linear(n_fund_features, d),
                nn.GELU(),
                nn.LayerNorm(d),
                nn.Dropout(drop),
                nn.Linear(d, d),
            )

        # ─── Stage 2: Modality-Specific Encoders ───

        self.price_encoder = nn.ModuleList([
            TransformerEncoderBlock(d, h, ff, drop)
            for _ in range(config["price_encoder_layers"])
        ])

        if self.use_sentiment:
            self.sent_encoder = nn.ModuleList([
                TransformerEncoderBlock(d, h, ff, drop)
                for _ in range(config["sentiment_encoder_layers"])
            ])

        if self.use_macro:
            self.macro_encoder = nn.ModuleList([
                TransformerEncoderBlock(d, h, ff, drop)
                for _ in range(config["macro_encoder_layers"])
            ])

        # ─── Stage 3: Cross-Attention Fusion ───

        if self.use_sentiment:
            self.cross_attn_sent = CrossAttentionBlock(d, h, ff, drop)
        if self.use_macro:
            self.cross_attn_macro = CrossAttentionBlock(d, h, ff, drop)

        # ─── Stage 4: Prediction Head ───

        self.final_norm = nn.LayerNorm(d)
        head_input_dim = d * 2 if self.use_fundamentals else d
        self.pred_head = nn.Sequential(
            nn.Linear(head_input_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(drop),
            nn.Linear(d, 1),
        )

        # Output scaling: learnable scalar α that multiplies raw predictions.
        # Initialized small (default 0.3) → conservative predictions near mean.
        # Preserves IC (monotonic transform) while preventing R²-destroying
        # magnitude overshoots. The gradient learns the optimal confidence.
        init_alpha = config.get("output_scale_init", 1.0)
        self.output_scale = nn.Parameter(torch.tensor(float(init_alpha)))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        price: torch.Tensor,        # (B, 60, n_price_features)
        sentiment: torch.Tensor,     # (B, 60, n_sent_features)
        fundamentals: torch.Tensor,  # (B, n_fund_features)
        macro: torch.Tensor,         # (B, 60, n_macro_features)
        stock_id: torch.Tensor,      # (B,) int
        return_attention: bool = False,
    ) -> dict:
        """
        Forward pass through all 4 stages.

        Args:
            price: Price/technical features (batch, 60, 39)
            sentiment: FinBERT embeddings + count (batch, 60, 769)
            fundamentals: Static fundamental features (batch, d_f)
            macro: Macro features (batch, 60, d_m)
            stock_id: Stock index for embedding lookup (batch,)
            return_attention: If True, return cross-attention weights

        Returns:
            Dict with:
                "prediction": (batch,) predicted return in percentage points
                "attn_sent": cross-attention weights (price→sentiment) if requested
                "attn_macro": cross-attention weights (price→macro) if requested
        """
        B = price.size(0)

        # ─── Stage 1: Tokenize ───

        # Price: project → causal conv → prepend [STOCK] → positional enc
        p = self.price_proj(price)                          # (B, 60, d)
        p = self.causal_conv(p)                             # (B, 60, d)
        stock_tok = self.stock_embedding(stock_id)          # (B, d)
        stock_tok = stock_tok.unsqueeze(1)                  # (B, 1, d)
        p = torch.cat([stock_tok, p], dim=1)                # (B, 61, d)
        p = self.price_pos_enc(p)                           # (B, 61, d)

        # Sentiment: tokenize if enabled
        if self.use_sentiment:
            s = self.sent_proj(sentiment)                   # (B, 60, d)
            s = self.sent_pos_enc(s)                        # (B, 60, d)

        # Macro: tokenize if enabled
        if self.use_macro:
            m = self.macro_proj(macro)                      # (B, 60, d)
            m = self.macro_pos_enc(m)                       # (B, 60, d)

        # Fundamentals: MLP if enabled
        if self.use_fundamentals:
            f = self.fund_mlp(fundamentals)                 # (B, d)

        # ─── Stage 2: Self-Attention Encoders ───

        for layer in self.price_encoder:
            p = layer(p)                                    # (B, 61, d)

        if self.use_sentiment:
            for layer in self.sent_encoder:
                s = layer(s)                                # (B, 60, d)

        if self.use_macro:
            for layer in self.macro_encoder:
                m = layer(m)                                # (B, 60, d)

        # ─── Stage 3: Cross-Attention Fusion ───

        attn_sent = attn_macro = None

        if self.use_sentiment:
            p, attn_sent = self.cross_attn_sent(p, s)      # (B, 61, d)

        if self.use_macro:
            p, attn_macro = self.cross_attn_macro(p, m)    # (B, 61, d)

        # ─── Stage 4: Prediction Head ───

        p = self.final_norm(p)

        # Mean pool across all 61 tokens (including [STOCK])
        p_pooled = p.mean(dim=1)                                # (B, d)

        # Concatenate with fundamentals if enabled
        if self.use_fundamentals:
            combined = torch.cat([p_pooled, f], dim=1)          # (B, 2d)
        else:
            combined = p_pooled                                  # (B, d)

        prediction = self.pred_head(combined).squeeze(-1)       # (B,)

        # Apply learned output scaling (preserves IC, improves R²)
        prediction = prediction * self.output_scale

        result = {"prediction": prediction}
        if return_attention:
            result["attn_sent"] = attn_sent
            result["attn_macro"] = attn_macro

        return result


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameter_breakdown(model: MCAT):
    """Print detailed parameter breakdown by component."""
    components = {
        "Price projection": model.price_proj,
        "Causal Conv1d": model.causal_conv,
        "Stock embedding": model.stock_embedding,
        "Price pos encoding": model.price_pos_enc,
        "Price encoder": model.price_encoder,
        "Final norm": model.final_norm,
        "Prediction head": model.pred_head,
        "Output scale (α)": [model.output_scale],
    }

    # Conditionally add modality-specific components
    if model.use_sentiment:
        components["Sentiment projection"] = model.sent_proj
        components["Sent pos encoding"] = model.sent_pos_enc
        components["No-news embedding"] = [model.no_news_embedding]
        components["Sentiment encoder"] = model.sent_encoder
        components["Cross-attn (sent)"] = model.cross_attn_sent

    if model.use_macro:
        components["Macro projection"] = model.macro_proj
        components["Macro pos encoding"] = model.macro_pos_enc
        components["Macro encoder"] = model.macro_encoder
        components["Cross-attn (macro)"] = model.cross_attn_macro

    if model.use_fundamentals:
        components["Fundamental MLP"] = model.fund_mlp

    total = 0
    print(f"\n{'Component':<30} {'Parameters':>12} {'%':>6}")
    print("─" * 52)
    grand_total = count_parameters(model)

    for name, module in components.items():
        if isinstance(module, list):
            params = sum(p.numel() for p in module if isinstance(p, nn.Parameter))
        elif isinstance(module, nn.Module):
            params = sum(p.numel() for p in module.parameters())
        else:
            params = 0
        pct = 100 * params / grand_total if grand_total > 0 else 0
        print(f"  {name:<28} {params:>12,} {pct:>5.1f}%")
        total += params

    print("─" * 52)
    disabled = list(model.disable_modalities)
    disabled_str = f"  (disabled: {disabled})" if disabled else ""
    print(f"  {'TOTAL':<28} {grand_total:>12,}{disabled_str}")


if __name__ == "__main__":
    # Build model with default config and test forward pass
    model = MCAT(
        n_price_features=39,
        n_sent_features=769,
        n_fund_features=7,
        n_macro_features=20,
    )

    print(f"Total parameters: {count_parameters(model):,}")
    print_parameter_breakdown(model)

    # Test forward pass with random data
    B = 4  # batch size
    x_price = torch.randn(B, 60, 39)
    x_sent = torch.randn(B, 60, 769)
    x_fund = torch.randn(B, 7)
    x_macro = torch.randn(B, 60, 20)
    x_stock = torch.randint(0, 15, (B,))

    output = model(x_price, x_sent, x_fund, x_macro, x_stock, return_attention=True)

    print(f"\nForward pass test:")
    print(f"  Prediction shape: {output['prediction'].shape}")
    print(f"  Prediction values: {output['prediction'].detach().numpy()}")
    print(f"  Attn (sent) shape: {output['attn_sent'].shape}")
    print(f"  Attn (macro) shape: {output['attn_macro'].shape}")

    # Verify gradient flow
    loss = output["prediction"].sum()
    loss.backward()
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    print(f"\n  Gradient flow: {len(grad_norms)}/{count_parameters(model)} params have gradients")
    print(f"  Min grad norm: {min(grad_norms.values()):.6f}")
    print(f"  Max grad norm: {max(grad_norms.values()):.6f}")
    print(f"  All gradients non-zero: {all(v > 0 for v in grad_norms.values())}")
