"""
baselines/nn_baselines.py — Neural network baseline architectures.
===================================================================
B4: PriceOnlyTransformer — single-layer Transformer on price features only
B5: LSTMConcatFusion — BiLSTM per modality + concatenation fusion (our v1 arch)

Both use the same training infrastructure (Trainer, walk-forward, Huber loss).
"""

import torch
import torch.nn as nn
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_CONFIG, N_STOCKS, LOOKBACK_WINDOW
from model.mcat import CausalConv1d, PositionalEncoding, TransformerEncoderBlock


# ═════════════════════════════════════════════════════════════
# B4: PRICE-ONLY TRANSFORMER
# ═════════════════════════════════════════════════════════════

class PriceOnlyTransformer(nn.Module):
    """
    Baseline B4: Single-layer Transformer encoder on price features only.
    No cross-attention, no multi-modal fusion.

    Tests whether the Transformer architecture itself provides value
    beyond what simpler models (Ridge, LightGBM) achieve.

    Architecture:
      Price (45, n_feat) → Linear → CausalConv → [STOCK] → PosEnc
      → 1-layer Transformer Encoder → Mean pool → Dense → prediction
    """

    def __init__(
        self,
        n_price_features: int = 39,
        n_fund_features: int = 7,
        config: dict = None,
    ):
        super().__init__()
        if config is None:
            config = MODEL_CONFIG

        d = config["d_model"]
        h = config["n_heads"]
        ff = config["d_ff"]
        drop = config["dropout"]

        # Input tokenization (same as MCAT price branch)
        self.price_proj = nn.Linear(n_price_features, d)
        self.causal_conv = CausalConv1d(d, d, config["causal_conv_kernel"])
        self.stock_embedding = nn.Embedding(N_STOCKS, d)
        self.pos_enc = PositionalEncoding(d, config["max_seq_len"], drop)

        # Single Transformer encoder layer
        self.encoder = TransformerEncoderBlock(d, h, ff, drop)

        # Prediction head (no fundamentals concatenation — price only)
        self.final_norm = nn.LayerNorm(d)
        self.pred_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(drop),
            nn.Linear(d, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, price, sentiment, fundamentals, macro, stock_id,
                return_attention=False):
        """
        Forward pass — only uses price and stock_id.
        Other inputs are accepted but ignored (for API compatibility with Trainer).
        """
        p = self.price_proj(price)
        p = self.causal_conv(p)
        stock_tok = self.stock_embedding(stock_id).unsqueeze(1)
        p = torch.cat([stock_tok, p], dim=1)
        p = self.pos_enc(p)

        p = self.encoder(p)
        p = self.final_norm(p)
        p_pooled = p.mean(dim=1)
        prediction = self.pred_head(p_pooled).squeeze(-1)

        result = {"prediction": prediction}
        if return_attention:
            result["attn_sent"] = None
            result["attn_macro"] = None
        return result


# ═════════════════════════════════════════════════════════════
# B5: LSTM + CONCATENATION FUSION
# ═════════════════════════════════════════════════════════════

class LSTMConcatFusion(nn.Module):
    """
    Baseline B5: BiLSTM per modality + concatenation fusion.
    This is our "v1 architecture" — the previous approach that
    the cross-attention Transformer is designed to improve upon.

    Architecture:
      Each modality → BiLSTM → take final hidden state
      Concatenate all hidden states + fundamentals
      → MLP prediction head

    This tests concatenation vs cross-attention fusion.
    The LSTM backbone also tests recurrence vs attention.
    """

    def __init__(
        self,
        n_price_features: int = 39,
        n_sent_features: int = 769,
        n_fund_features: int = 7,
        n_macro_features: int = 20,
        hidden_size: int = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # BiLSTM for each temporal modality
        self.price_lstm = nn.LSTM(
            input_size=n_price_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.sent_lstm = nn.LSTM(
            input_size=n_sent_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.macro_lstm = nn.LSTM(
            input_size=n_macro_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Fundamental MLP
        self.fund_mlp = nn.Sequential(
            nn.Linear(n_fund_features, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stock embedding
        self.stock_embedding = nn.Embedding(N_STOCKS, hidden_size)

        # Concatenation: 3 BiLSTM outputs (2*hidden each) + fund + stock
        concat_dim = 3 * (2 * hidden_size) + hidden_size + hidden_size

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(concat_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def forward(self, price, sentiment, fundamentals, macro, stock_id,
                return_attention=False):
        """
        Forward pass with concatenation fusion.
        All modalities processed independently by LSTMs, then concatenated.
        """
        # BiLSTM: take final hidden state (concatenation of forward and backward)
        price_out, _ = self.price_lstm(price)
        price_h = price_out[:, -1, :]       # (B, 2*hidden)

        sent_out, _ = self.sent_lstm(sentiment)
        sent_h = sent_out[:, -1, :]          # (B, 2*hidden)

        macro_out, _ = self.macro_lstm(macro)
        macro_h = macro_out[:, -1, :]        # (B, 2*hidden)

        # Fundamentals
        fund_h = self.fund_mlp(fundamentals)  # (B, hidden)

        # Stock embedding
        stock_h = self.stock_embedding(stock_id)  # (B, hidden)

        # Concatenate everything
        combined = torch.cat([price_h, sent_h, macro_h, fund_h, stock_h], dim=1)

        prediction = self.pred_head(combined).squeeze(-1)

        result = {"prediction": prediction}
        if return_attention:
            result["attn_sent"] = None
            result["attn_macro"] = None
        return result


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test both baselines
    B = 4

    # B4: Price-only Transformer
    model_b4 = PriceOnlyTransformer(n_price_features=39, n_fund_features=7)
    print(f"B4 PriceOnlyTransformer: {count_parameters(model_b4):,} params")

    out = model_b4(
        price=torch.randn(B, 60, 39),
        sentiment=torch.randn(B, 60, 6),
        fundamentals=torch.randn(B, 7),
        macro=torch.randn(B, 60, 5),
        stock_id=torch.randint(0, 15, (B,)),
    )
    print(f"  Output shape: {out['prediction'].shape}")
    out["prediction"].sum().backward()
    print(f"  Gradient flow: OK")

    # B5: LSTM + Concat
    model_b5 = LSTMConcatFusion(
        n_price_features=39, n_sent_features=6,
        n_fund_features=7, n_macro_features=5,
    )
    print(f"\nB5 LSTMConcatFusion: {count_parameters(model_b5):,} params")

    out = model_b5(
        price=torch.randn(B, 60, 39),
        sentiment=torch.randn(B, 60, 6),
        fundamentals=torch.randn(B, 7),
        macro=torch.randn(B, 60, 5),
        stock_id=torch.randint(0, 15, (B,)),
    )
    print(f"  Output shape: {out['prediction'].shape}")
    out["prediction"].sum().backward()
    print(f"  Gradient flow: OK")
