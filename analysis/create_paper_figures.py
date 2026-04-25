"""
analysis/create_paper_figures.py — Generate publication figures for the MCAT paper.
====================================================================================
Reads aggregated metrics from results/tables/*_results.json and produces five
figures (PDF + PNG) under results/figures/:

  Figure 1: Baseline comparison — 3 side-by-side grouped bar charts
            (IC | R^2 | DA) across MCAT + all 5 baselines. Different metric
            scales are kept on separate axes for readability.
  Figure 2: Modality ablation — horizontal bar chart of IC (panel a) and DA
            (panel b) by variant. Ablated variants are ordered by IC
            degradation magnitude (most damaged at the bottom); each bar is
            annotated with its absolute IC, dIC vs Full MCAT, and % drop.
  Figure 3: Per-fold regime stability — 2x3 grid. Top row: IC per fold
            across MCAT + 3 strongest baselines. Bottom row: R^2 per fold,
            same models. Columns correspond to bear recovery / AI rally /
            consolidation regimes. R^2 downside is clipped; off-scale
            baseline bars are annotated with their true value.
  Figure 4: Architecture diagram — 4-stage MCAT (tokenization ->
            per-modality encoders -> cross-attention fusion -> prediction),
            with Q/K/V badges and dimension annotations.
  Figure 5: Multi-metric radar — MCAT vs LightGBM vs Single Transformer on
            IC / R^2 / DA / Skill / MAE (MAE inverted so higher = better
            everywhere on the chart). Per-metric min-max normalization.

Usage:
    python analysis/create_paper_figures.py
    python analysis/create_paper_figures.py --results-dir results/tables
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR
from utils import setup_logger

logger = setup_logger(__name__)

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    # "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

COLORS = {
    "mcat": "#2166AC", "lightgbm": "#B2182B", "ridge": "#EF8A62",
    "hist_mean": "#999999", "single_tf": "#67A9CF", "lstm": "#D1E5F0",
    "full": "#2166AC", "no_macro": "#E08214", "no_fund": "#FDB863",
    "no_sent": "#B2ABD2", "price_only": "#999999",
    "fold0": "#D73027", "fold1": "#4575B4", "fold2": "#FEE090",
}


# ---- DATA LOADING ----

def load_result(results_dir, name):
    path = Path(results_dir) / f"{name}_results.json"
    if not path.exists():
        logger.warning(f"  Not found: {path}")
        return None
    with open(path) as f:
        raw = json.load(f)
    agg = raw["aggregated"]
    return {
        "ic": agg["ic"]["mean"], "ic_std": agg["ic"]["std"],
        "r2": agg["r2"]["mean"], "r2_std": agg["r2"]["std"],
        "da": agg["directional_accuracy"]["mean"],
        "da_std": agg["directional_accuracy"]["std"],
        "skill": agg["skill_score"]["mean"],
        "mae": agg["mae"]["mean"], "rmse": agg["rmse"]["mean"],
        "folds": [{"fold": f["fold"], "ic": f["ic"], "r2": f["r2"],
                   "da": f["directional_accuracy"], "skill": f["skill_score"]}
                  for f in raw["folds"]],
    }


def load_all_results(results_dir):
    results_dir = Path(results_dir)
    mcat = load_result(results_dir, "full_mcat")
    if mcat is None:
        logger.error("  full_mcat_results.json not found!")
        return None

    baseline_map = {
        "historical_mean": "Historical\nMean", "ridge": "Ridge",
        "lightgbm": "LightGBM", "single_transformer": "Single\nTransformer",
        "lstm_concat": "LSTM\nConcat",
    }
    baselines = {}
    for fname, dname in baseline_map.items():
        r = load_result(results_dir, fname)
        if r: baselines[dname] = r

    ablation_map = {
        "ablate_no_sentiment": "- Sentiment",
        "ablate_no_fundamentals": "- Fundamentals",
        "ablate_no_macro": "- Macro",
        "ablate_price_only": "Price Only",
    }
    ablations = {"Full MCAT": mcat}
    for fname, dname in ablation_map.items():
        r = load_result(results_dir, fname)
        if r: ablations[dname] = r

    logger.info(f"  Loaded: MCAT + {len(baselines)} baselines + {len(ablations)-1} ablations")
    return {"mcat": mcat, "baselines": baselines, "ablations": ablations}


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", format="png", bbox_inches="tight", dpi=300)
    logger.info(f"  Saved: {name}.pdf + {name}.png")
    plt.close(fig)


# ---- FIGURE 1: BASELINE COMPARISON ----

def fig1_baseline_comparison(data):
    mcat = data["mcat"]
    baselines = data["baselines"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    models = ["MCAT\n(Ours)"] + list(baselines.keys())
    cmap = {"MCAT\n(Ours)": COLORS["mcat"], "Historical\nMean": COLORS["hist_mean"],
            "Ridge": COLORS["ridge"], "LightGBM": COLORS["lightgbm"],
            "Single\nTransformer": COLORS["single_tf"], "LSTM\nConcat": COLORS["lstm"]}
    bar_colors = [cmap.get(m, "#AAA") for m in models]
    all_r = [mcat] + [baselines[m] for m in list(baselines.keys())]
    ics = [r["ic"] for r in all_r]
    r2s = [r["r2"] for r in all_r]
    das = [r["da"] for r in all_r]
    x = np.arange(len(models))

    # (a) IC
    ax = axes[0]
    bars = ax.bar(x, ics, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax.set_ylabel("Information Coefficient (IC)")
    ax.set_title("(a) Ranking Ability", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars, ics)):
        yp = val + 0.003 if val >= 0 else val - 0.010
        ax.text(bar.get_x()+bar.get_width()/2, yp, f"{val:+.3f}",
                ha="center", va="bottom" if val>=0 else "top", fontsize=7.5,
                fontweight="bold" if i==0 else "normal")

    # (b) R2 — zoomed
    ax = axes[1]
    r2c = r2s.copy()
    clip = -0.065
    for i in range(len(r2c)):
        if r2c[i] < clip: r2c[i] = clip
    bars = ax.bar(x, r2c, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax.set_ylabel("R2 (Variance Explained)")
    ax.set_title("(b) Calibration Quality", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.axhline(y=0, color="black", linewidth=1.0, zorder=5)
    ax.set_ylim(clip - 0.005, max(r2s) + 0.012)
    for i, (bar, vc, va) in enumerate(zip(bars, r2c, r2s)):
        if va < clip:
            ax.text(bar.get_x()+bar.get_width()/2, clip-0.003,
                    f"v {va:+.3f}", ha="center", va="top", fontsize=7,
                    color=bar_colors[i], fontweight="bold")
        else:
            yp = vc+0.002 if vc>=0 else vc-0.004
            ax.text(bar.get_x()+bar.get_width()/2, yp, f"{va:+.3f}",
                    ha="center", va="bottom" if vc>=0 else "top", fontsize=7.5,
                    fontweight="bold" if i==0 else "normal")

    # (c) DA
    ax = axes[2]
    bars = ax.bar(x, [d*100 for d in das], color=bar_colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax.set_ylabel("Directional Accuracy (%)")
    ax.set_title("(c) Direction Prediction", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.axhline(y=50, color="black", linewidth=0.5, linestyle="--", label="Random (50%)")
    ax.set_ylim(48, max(d*100 for d in das)+2.5)
    for i, (bar, val) in enumerate(zip(bars, das)):
        ax.text(bar.get_x()+bar.get_width()/2, val*100+0.3, f"{val:.1%}",
                ha="center", va="bottom", fontsize=7.5,
                fontweight="bold" if i==0 else "normal")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("MCAT vs Baselines -- Out-of-Sample Performance (Jul 2022 - Dec 2023)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig1_baseline_comparison")


# ---- FIGURE 2: ABLATION STUDY ----

def fig2_ablation_study(data):
    """Horizontal bar chart of IC (and DA) by ablation variant.

    Bars are ordered by IC degradation magnitude: Full MCAT stays on top as
    the reference, then ablated variants appear in order of increasing IC
    (worst-performing at the bottom). Each ablated bar is annotated with its
    absolute IC, the delta vs Full MCAT (dIC), and the relative drop (%).
    """
    ablations = data["ablations"]
    COLOR_BY_VARIANT = {
        "Full MCAT": COLORS["full"],
        "- Sentiment": COLORS["no_sent"],
        "- Fundamentals": COLORS["no_fund"],
        "- Macro": COLORS["no_macro"],
        "Price Only": COLORS["price_only"],
    }

    # Order: Full MCAT on top, then ablations sorted by IC descending
    # (least damaged first, most damaged last) so the catastrophic drop is
    # immediately visible at the bottom of the chart.
    full_ic = ablations["Full MCAT"]["ic"]
    ablated_only = [(name, d) for name, d in ablations.items() if name != "Full MCAT"]
    ablated_sorted = sorted(ablated_only, key=lambda kv: -kv[1]["ic"])
    ordered = [("Full MCAT", ablations["Full MCAT"])] + ablated_sorted

    variants = [name for name, _ in ordered]
    ics = [d["ic"] for _, d in ordered]
    das = [d["da"] for _, d in ordered]
    colors = [COLOR_BY_VARIANT.get(v, "#AAA") for v in variants]
    yp = np.arange(len(variants))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # ---- Panel (a): IC by variant with degradation annotations ----
    ax = axes[0]
    bars = ax.barh(yp, ics, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.6)
    ax.set_yticks(yp); ax.set_yticklabels(variants, fontsize=10)
    ax.set_xlabel("Information Coefficient (IC)")
    ax.set_title("(a) Ranking Ability by Variant", fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.axvline(x=full_ic, color=COLORS["full"], linewidth=0.8,
               linestyle=":", alpha=0.6, label=f"Full MCAT IC = {full_ic:+.3f}")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    # Reserve a right-side "annotation column" for the dIC / % drop text so
    # it never overlaps the per-bar IC value labels.
    x_left = min(min(ics) - 0.015, -0.08)
    x_right = max(ics) + 0.14
    ax.set_xlim(x_left, x_right)
    annot_col_x = max(ics) + 0.03  # fixed column for all dIC annotations

    for i, (bar, val) in enumerate(zip(bars, ics)):
        # Absolute IC label just past the bar's right/left edge
        x_label = val + 0.002 if val >= 0 else val - 0.002
        ha = "left" if val >= 0 else "right"
        ax.text(x_label, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8.5,
                fontweight="bold" if i == 0 else "normal")
        # Degradation annotation for ablations (anchored at the annotation column).
        if i > 0:
            delta = val - full_ic
            pct_drop = delta / full_ic * 100 if abs(full_ic) > 1e-10 else float("nan")
            ax.text(annot_col_x, bar.get_y() + bar.get_height() / 2,
                    f"dIC = {delta:+.3f}  ({pct_drop:+.0f}%)",
                    va="center", ha="left", fontsize=7.5,
                    color="#B2182B", fontstyle="italic")

    # ---- Panel (b): DA by variant (same ordering as panel a) ----
    ax = axes[1]
    bars = ax.barh(yp, [d * 100 for d in das], color=colors,
                   edgecolor="white", linewidth=0.5, height=0.6)
    ax.set_yticks(yp); ax.set_yticklabels(variants, fontsize=10)
    ax.set_xlabel("Directional Accuracy (%)")
    ax.set_title("(b) Direction Prediction by Variant", fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(x=50, color="black", linewidth=0.5, linestyle="--",
               label="Random (50%)")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.set_xlim(50, max(d * 100 for d in das) + 1.8)
    for bar, val in zip(bars, das):
        ax.text(val * 100 + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8.5)

    fig.suptitle("Modality Ablation Study -- Contribution of Each Data Source",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig2_ablation_study")


# ---- FIGURE 3: PER-FOLD REGIME ANALYSIS (IC + R2 over 3 folds) ----

def fig3_regime_analysis(data):
    """2x3 grid: IC (top row) and R2 (bottom row) across the three focused folds.

    Columns are color-coded by market regime (bear recovery / AI rally /
    consolidation). Each cell compares MCAT against the three strongest
    reference models (LightGBM, Single Transformer, Ridge) so the reader can
    see (a) MCAT's absolute per-fold stability and (b) its relative advantage
    in every regime. The R2 axis is clipped on the downside to keep the small
    positive MCAT values visible; off-scale baseline bars are annotated with
    their true value beneath the axis.
    """
    mcat = data["mcat"]
    baselines = data["baselines"]

    regime_info = [
        ("Fold 0: Bear Recovery\n(Jul-Dec 2022)", COLORS["fold0"]),
        ("Fold 1: AI Rally\n(Jan-Jun 2023)", COLORS["fold1"]),
        ("Fold 2: Consolidation\n(Jul-Dec 2023)", COLORS["fold2"]),
    ]

    model_entries = [("MCAT\n(Ours)", mcat, COLORS["mcat"])]
    for dname, ckey in [("LightGBM", "lightgbm"),
                        ("LSTM\nConcat", "lstm"),
                        ("Single\nTransformer", "single_tf"),
                        ("Ridge", "ridge")]:
        if dname in baselines and len(baselines[dname].get("folds", [])) >= 3:
            model_entries.append((dname, baselines[dname], COLORS[ckey]))

    model_names = [m[0] for m in model_entries]
    model_colors = [m[2] for m in model_entries]
    n_models = len(model_names)

    # Gather per-fold metrics
    ics_per_fold = [[e[1]["folds"][f]["ic"] for e in model_entries] for f in range(3)]
    r2s_per_fold = [[e[1]["folds"][f]["r2"] for e in model_entries] for f in range(3)]

    ic_flat = [v for row in ics_per_fold for v in row]
    r2_flat = [v for row in r2s_per_fold for v in row]
    ic_lo, ic_hi = min(ic_flat), max(ic_flat)
    ic_pad = max((ic_hi - ic_lo) * 0.25, 0.03)

    # R2 clipping: baselines (esp. Ridge) can hit -0.3; clip to keep MCAT's
    # small positive values visually readable.
    r2_hi = max(r2_flat)
    r2_clip_lo = max(-0.12, min(r2_flat) - 0.02)
    r2_span = r2_hi - r2_clip_lo
    r2_pad = max(r2_span * 0.20, 0.01)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), sharey="row")

    for fold_idx, (title, bg_color) in enumerate(regime_info):
        x = np.arange(n_models)

        # ---- TOP ROW: IC ----
        ax = axes[0, fold_idx]
        fold_ics = ics_per_fold[fold_idx]
        bars = ax.bar(x, fold_ics, color=model_colors, edgecolor="white",
                      linewidth=0.5, width=0.7)
        ax.set_title(f"({chr(97 + fold_idx)}) {title}",
                     fontweight="bold", fontsize=10)
        ax.set_facecolor(matplotlib.colors.to_rgba(bg_color, 0.06))
        ax.axhline(y=0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        if fold_idx == 0:
            ax.set_ylabel("Information Coefficient (IC)", fontweight="bold")
        for i, (bar, val) in enumerate(zip(bars, fold_ics)):
            yp = val + ic_pad * 0.12 if val >= 0 else val - ic_pad * 0.12
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, yp, f"{val:+.3f}",
                    ha="center", va=va, fontsize=8.5,
                    fontweight="bold" if i == 0 else "normal")
        bars[0].set_edgecolor(COLORS["mcat"])
        bars[0].set_linewidth(2.0)

        # ---- BOTTOM ROW: R^2 (with clipping + off-scale annotations) ----
        ax = axes[1, fold_idx]
        fold_r2s_raw = r2s_per_fold[fold_idx]
        fold_r2s_clip = [max(v, r2_clip_lo) for v in fold_r2s_raw]
        bars = ax.bar(x, fold_r2s_clip, color=model_colors, edgecolor="white",
                      linewidth=0.5, width=0.7)
        ax.set_facecolor(matplotlib.colors.to_rgba(bg_color, 0.06))
        ax.axhline(y=0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=8)
        if fold_idx == 0:
            ax.set_ylabel("R$^2$ (Variance Explained)", fontweight="bold")
        for i, (bar, raw, clip) in enumerate(zip(bars, fold_r2s_raw, fold_r2s_clip)):
            if raw < r2_clip_lo:
                # Off-scale baseline: annotate true value below the axis floor.
                ax.text(bar.get_x() + bar.get_width() / 2,
                        r2_clip_lo - r2_pad * 0.35,
                        f"v {raw:+.3f}", ha="center", va="top", fontsize=7,
                        color=model_colors[i], fontweight="bold")
            else:
                yp = clip + r2_pad * 0.08 if clip >= 0 else clip - r2_pad * 0.08
                va = "bottom" if clip >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, yp, f"{raw:+.3f}",
                        ha="center", va=va, fontsize=8.5,
                        fontweight="bold" if i == 0 else "normal")
        bars[0].set_edgecolor(COLORS["mcat"])
        bars[0].set_linewidth(2.0)

    axes[0, 0].set_ylim(ic_lo - ic_pad, ic_hi + ic_pad * 1.5)
    axes[1, 0].set_ylim(r2_clip_lo - r2_pad * 0.9, r2_hi + r2_pad * 1.4)

    fig.suptitle(
        "Per-Fold Stability Across Market Regimes -- IC (top) and R$^2$ (bottom)",
        fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout()
    save_fig(fig, "fig3_regime_analysis")


# ---- FIGURE 4: ARCHITECTURE DIAGRAM ----

def fig4_architecture(data):
    """Publication-grade MCAT architecture diagram.

    Four stages (tokenise -> self-attention encoders -> sequential
    cross-attention fusion -> prediction head) rendered as a left-to-right
    flow with a row per modality, explicit Q/K,V colour coding on attention
    edges, post-encoder shape annotations, and a curved fundamentals skip
    connection into Concatenate. Matches the forward pass in model/mcat.py
    line-for-line: Linear+CausalConv1d + [STOCK] + PosEnc -> TransformerEncoder
    x{2,1,1} -> CA(Price<-Sent) -> CA(Price'<-Macro) -> LayerNorm -> MeanPool
    -> Concat(fund) -> MLP(128->64->1) -> * alpha."""

    # ---- Colour-blind-friendly palette (one colour per modality) ----
    CP = {
        "price":        "#2E5C8A",   # deep blue
        "sent":         "#B84A56",   # muted red
        "macro":        "#C89632",   # warm amber
        "fund":         "#4B8C5E",   # forest green
        "fusion":       "#6B4C9A",   # purple (cross-attn)
        "pred":         "#374151",   # slate
        "output":       "#1A4D8F",   # navy
        "stage_bg":     "#FAFBFC",
        "stage_border": "#CFD6DE",
        "arrow":        "#6B7280",
    }

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(-0.3, 16.3)
    ax.set_ylim(-0.35, 9.35)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ------------------------- helpers -------------------------
    def add_box(x, y, w, h, title, subtitle=None, color="#333",
                text_color="white", title_size=10.0, sub_size=8.0,
                edge="white", zbase=2):
        sh = FancyBboxPatch((x + 0.03, y - 0.035), w, h,
                            boxstyle="round,pad=0.08",
                            facecolor="#00000022", edgecolor="none",
                            zorder=zbase - 1)
        ax.add_patch(sh)
        bp = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.08",
                            facecolor=color, edgecolor=edge,
                            linewidth=1.1, alpha=0.97, zorder=zbase)
        ax.add_patch(bp)
        if subtitle:
            ax.text(x + w/2, y + h*0.63, title,
                    ha="center", va="center",
                    fontsize=title_size, fontweight="bold",
                    color=text_color, zorder=zbase + 1)
            ax.text(x + w/2, y + h*0.27, subtitle,
                    ha="center", va="center",
                    fontsize=sub_size, fontstyle="italic",
                    color=text_color, alpha=0.92, zorder=zbase + 1)
        else:
            ax.text(x + w/2, y + h/2, title,
                    ha="center", va="center",
                    fontsize=title_size, fontweight="bold",
                    color=text_color, linespacing=1.25,
                    zorder=zbase + 1)

    def arrow(x1, y1, x2, y2, color=None, lw=1.3, rad=0.0,
              style="-", mut=14, alpha=1.0, z=5):
        c = color if color else CP["arrow"]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=c,
                                    lw=lw, linestyle=style,
                                    connectionstyle=f"arc3,rad={rad}",
                                    shrinkA=3, shrinkB=3,
                                    mutation_scale=mut,
                                    alpha=alpha),
                    zorder=z)

    def dim_label(x, y, text, size=7.8):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=size, color="#374151", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.20",
                          fc="white", ec="#9CA3AF",
                          alpha=0.95, linewidth=0.6),
                zorder=7)

    def qkv_tag(x, y, text, color):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8.8, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="white", ec=color,
                          alpha=0.98, linewidth=1.1),
                zorder=8)

    # ------------------------- stage panels -------------------------
    stage_bot, stage_top = 0.30, 8.40
    stages = [
        (0.00, 4.50, "Stage 1", "Input  \u2192  Tokenise"),
        (4.75, 2.40, "Stage 2", "Self-Attention\nEncoders"),
        (7.40, 3.75, "Stage 3", "Cross-Attention\nFusion"),
        (11.40, 4.55, "Stage 4", "Prediction Head"),
    ]
    for sx, sw, slabel, ssub in stages:
        bg = FancyBboxPatch((sx, stage_bot), sw, stage_top - stage_bot,
                            boxstyle="round,pad=0.10",
                            facecolor=CP["stage_bg"],
                            edgecolor=CP["stage_border"],
                            linewidth=1.0, alpha=0.55,
                            linestyle=(0, (4, 3)),
                            zorder=0)
        ax.add_patch(bg)
        ax.text(sx + sw/2, 8.12, slabel,
                ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#1F2937",
                bbox=dict(boxstyle="round,pad=0.28",
                          fc="white", ec="#9AA3AD", lw=1.0),
                zorder=1)
        ax.text(sx + sw/2, 7.68, ssub,
                ha="center", va="top",
                fontsize=9.3, fontstyle="italic", color="#4B5563",
                linespacing=1.2, zorder=1)

    # Stage 1 sub-column mini-labels
    ax.text(0.925, 7.25, "Input Tensors", ha="center", va="center",
            fontsize=9.0, color="#6B7280", fontstyle="italic", zorder=1)
    ax.text(3.35, 7.25, "Tokenise", ha="center", va="center",
            fontsize=9.0, color="#6B7280", fontstyle="italic", zorder=1)

    # ------------------------- row geometry -------------------------
    row = {"price": 6.25, "sent": 4.95, "macro": 3.65, "fund": 1.60}

    # ------------------------- Stage 1.a: raw input tensors -------------------------
    ix, iw, ih = 0.15, 1.55, 0.90
    add_box(ix, row["price"] - ih/2, iw, ih,
            "Price / Tech", "(B, L, 39)", CP["price"])
    add_box(ix, row["sent"]  - ih/2, iw, ih,
            "Sentiment",    "(B, L, 769)", CP["sent"])
    add_box(ix, row["macro"] - ih/2, iw, ih,
            "Macro",        "(B, L, 18)",  CP["macro"])
    add_box(ix, row["fund"]  - ih/2, iw, ih,
            "Fundamentals", "(B, d_f = 10)", CP["fund"])

    # ------------------------- Stage 1.b: tokenise -------------------------
    tx, tw, th = 2.55, 1.60, 1.00
    add_box(tx, row["price"] - th/2, tw, th,
            "Linear + Conv1d", "+ [STOCK] + PosEnc",
            CP["price"], title_size=9.0, sub_size=7.5)
    add_box(tx, row["sent"]  - th/2, tw, th,
            "Linear 769\u219264", "+ PosEnc",
            CP["sent"], title_size=9.0, sub_size=7.5)
    add_box(tx, row["macro"] - th/2, tw, th,
            "Linear 18\u219264", "+ PosEnc",
            CP["macro"], title_size=9.0, sub_size=7.5)
    add_box(tx, row["fund"]  - th/2, tw, th,
            "2-Layer MLP", "d_f \u2192 64 \u2192 64",
            CP["fund"], title_size=9.0, sub_size=7.5)

    for rk in row:
        arrow(ix + iw, row[rk], tx, row[rk], lw=1.1)

    # ------------------------- Stage 2: self-attention encoders -------------------------
    ex, ew, eh = 4.90, 2.00, 1.00
    add_box(ex, row["price"] - eh/2, ew, eh,
            "Transformer \u00d7 2", "Self-Attention (MHSA)",
            CP["price"], title_size=9.3, sub_size=7.7)
    add_box(ex, row["sent"] - eh/2, ew, eh,
            "Transformer \u00d7 1", "Self-Attention (MHSA)",
            CP["sent"], title_size=9.3, sub_size=7.7)
    add_box(ex, row["macro"] - eh/2, ew, eh,
            "Transformer \u00d7 1", "Self-Attention (MHSA)",
            CP["macro"], title_size=9.3, sub_size=7.7)

    for rk in ("price", "sent", "macro"):
        arrow(tx + tw, row[rk], ex, row[rk], lw=1.1)

    # Post-encoder shape annotations (above each encoder box)
    dim_label(ex + ew/2, row["price"] + 0.72, "(B, L+1, 64)")
    dim_label(ex + ew/2, row["sent"]  + 0.72, "(B, L, 64)")
    dim_label(ex + ew/2, row["macro"] + 0.72, "(B, L, 64)")

    # ------------------------- Stage 3: sequential cross-attention -------------------------
    cx, cw, ch = 7.55, 3.00, 1.10
    ca1_cy, ca2_cy = 6.00, 4.30
    add_box(cx, ca1_cy - ch/2, cw, ch,
            "Cross-Attention  (1)",
            "Q: price   \u00b7   K, V: sentiment",
            CP["fusion"], title_size=9.9, sub_size=8.3)
    add_box(cx, ca2_cy - ch/2, cw, ch,
            "Cross-Attention  (2)",
            "Q: price\u2032   \u00b7   K, V: macro",
            CP["fusion"], title_size=9.9, sub_size=8.3)

    # CA1 inputs: price Q (top-left), sent K/V (bottom-left)
    ca1_q_y  = ca1_cy + 0.24
    ca1_kv_y = ca1_cy - 0.24
    arrow(ex + ew, row["price"], cx, ca1_q_y,
          color=CP["price"], lw=2.1, rad=-0.10, mut=15)
    qkv_tag(cx - 0.24, ca1_q_y + 0.26, "Q", CP["price"])
    arrow(ex + ew, row["sent"], cx, ca1_kv_y,
          color=CP["sent"], lw=1.9, rad=0.22,
          style=(0, (5, 2.5)), mut=15)
    qkv_tag(cx - 0.28, ca1_kv_y - 0.26, "K, V", CP["sent"])

    # CA1 -> CA2 Q passthrough (enriched price)
    pass_x = cx + cw * 0.32
    arrow(pass_x, ca1_cy - ch/2, pass_x, ca2_cy + ch/2,
          color=CP["fusion"], lw=2.1, mut=15)
    qkv_tag(pass_x - 0.50, (ca1_cy - ch/2 + ca2_cy + ch/2) / 2,
            "Q (price\u2032)", CP["fusion"])

    # CA2 K,V from macro (dashed amber)
    ca2_kv_y = ca2_cy - 0.20
    arrow(ex + ew, row["macro"], cx, ca2_kv_y,
          color=CP["macro"], lw=1.9, rad=0.22,
          style=(0, (5, 2.5)), mut=15)
    qkv_tag(cx - 0.28, ca2_kv_y - 0.26, "K, V", CP["macro"])

    # ------------------------- Stage 4: prediction head (vertical stack) -------------------------
    px, pw = 11.75, 2.35
    pred_boxes = [
        (4.30, 0.80, "LayerNorm",      "Mean-Pool over L+1 tokens"),
        (3.30, 0.80, "Concatenate",    "with Fundamentals"),
        (2.30, 0.80, "Prediction MLP", "128 \u2192 64 \u2192 1"),
        (1.40, 0.55, "\u00d7  \u03b1", "learnable output scale"),
    ]
    for cy, h_, ttl, sub in pred_boxes:
        add_box(px, cy - h_/2, pw, h_, ttl, sub,
                CP["pred"], title_size=9.5, sub_size=7.7)

    yhat_cy, yhat_h = 0.72, 0.55
    add_box(px + 0.12, yhat_cy - yhat_h/2, pw - 0.24, yhat_h,
            "\u0177  \u2248  3-day return (pp)", None,
            CP["output"], title_size=10.8)

    # CA2 -> Stage 4 entry (horizontal with gentle curve)
    arrow(cx + cw, ca2_cy, px, pred_boxes[0][0], lw=1.7, rad=-0.12, mut=14)
    dim_label((cx + cw + px) / 2, pred_boxes[0][0] + 0.40, "(B, L+1, 64)")

    # Intra-pred-head vertical arrows
    for i in range(len(pred_boxes) - 1):
        y_top_box = pred_boxes[i][0] - pred_boxes[i][1] / 2
        y_bot_box = pred_boxes[i + 1][0] + pred_boxes[i + 1][1] / 2
        arrow(px + pw/2, y_top_box, px + pw/2, y_bot_box,
              lw=1.45, mut=13)
    # x alpha -> yhat
    arrow(px + pw/2,
          pred_boxes[-1][0] - pred_boxes[-1][1] / 2,
          px + pw/2,
          yhat_cy + yhat_h/2,
          lw=1.45, mut=13)

    # Shape annotations inside the pred pipeline
    dim_label(px + pw + 0.20, (pred_boxes[0][0] + pred_boxes[1][0]) / 2,
              "(B, 64)", size=7.5)
    dim_label(px + pw + 0.20, (pred_boxes[1][0] + pred_boxes[2][0]) / 2,
              "(B, 128)", size=7.5)
    dim_label(px + pw + 0.20, (pred_boxes[2][0] + pred_boxes[3][0]) / 2,
              "(B, 1)", size=7.5)

    # ------------------------- Fundamentals skip connection (valley) -------------------------
    # Cubic Bezier routed as a deep U-valley beneath every Stage 2/3 box.
    # The curve exits the Fund-MLP box straight down, traverses beneath the
    # encoders and cross-attention layers, then rises straight up into the
    # Concat box from directly below. Control points are placed directly
    # underneath the endpoints so both terminal tangents are vertical —
    # giving a pronounced flat-bottomed U, not a gentle arc. The curve's
    # minimum is ~y=0.44, which clears the Fund-MLP box bottom (y=1.10).
    skip_start = (tx + tw, row["fund"] - 0.25)              # (4.15, 1.35)
    concat_bot_y = pred_boxes[1][0] - pred_boxes[1][1] / 2  # 2.90
    skip_end   = (px + 0.25, concat_bot_y)                  # (12.00, 2.90)
    valley_y = -0.05                                        # control-point depth

    verts = [
        skip_start,
        (skip_start[0], valley_y),   # CP1 pulls curve straight down from start
        (skip_end[0],   valley_y),   # CP2 pulls curve straight up into end
        skip_end,
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    skip_path = MplPath(verts, codes)
    skip_patch = PathPatch(skip_path, fc="none", ec=CP["fund"],
                           lw=2.2, linestyle=(0, (5, 3)),
                           alpha=0.95, zorder=3)
    ax.add_patch(skip_patch)
    # Terminal tangent is vertical (pointing up), so a tiny upward annotate
    # gives the arrowhead the right orientation without disturbing the path.
    ax.annotate("",
                xy=skip_end,
                xytext=(skip_end[0], skip_end[1] - 0.01),
                arrowprops=dict(arrowstyle="-|>", color=CP["fund"],
                                lw=2.2, mutation_scale=15, alpha=0.95),
                zorder=4)
    ax.text(7.0, 0.55, "Fundamentals (skip connection)", fontsize=8,
            color=CP["fund"], fontstyle="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=CP["fund"],
                      alpha=0.85, linewidth=0.8))

    # ------------------------- Inline arrow legend -------------------------
    # Placed in the empty upper region of the Stage 4 panel (above pred head).
    lx, ly_top = 11.78, 7.30
    rowgap = 0.30
    entries = [
        (CP["price"], "-",            2.0, "Query  (Q)"),
        (CP["sent"],  (0, (4.5, 2.5)), 1.9, "Key, Value  (K, V)"),
        (CP["fund"],  (0, (4.5, 3.0)), 1.9, "Skip connection"),
        (CP["arrow"], "-",            1.5, "Forward flow"),
    ]
    leg_h = len(entries) * rowgap + 0.20
    leg_bg = FancyBboxPatch((lx - 0.10, ly_top - leg_h + 0.10),
                            4.25, leg_h,
                            boxstyle="round,pad=0.14",
                            facecolor="white", edgecolor="#9CA3AF",
                            linewidth=0.8, alpha=0.94, zorder=1)
    ax.add_patch(leg_bg)
    ax.text(lx + 0.10, ly_top + 0.06, "Arrow conventions",
            ha="left", va="bottom", fontsize=8.8,
            color="#374151", fontweight="bold", zorder=2)
    for i, (col, ls, lw_, lbl) in enumerate(entries):
        yy = ly_top - (i + 0.5) * rowgap - 0.05
        ax.plot([lx + 0.10, lx + 0.70], [yy, yy],
                color=col, lw=lw_, linestyle=ls,
                solid_capstyle="butt", zorder=2)
        ax.text(lx + 0.82, yy, lbl, ha="left", va="center",
                fontsize=8.4, color="#374151", zorder=2)

    # ------------------------- Title & footer -------------------------
    ax.text(8.0, 8.96,
            "MCAT \u2014 Multi-Modal Cross-Attention Transformer",
            ha="center", va="center", fontsize=16.5, fontweight="bold",
            color="#1F2937")
    ax.text(8.0, 8.58,
            "Tokenise per-modality inputs  \u2192  "
            "self-attention encoders  \u2192  "
            "sequential cross-attention fusion  \u2192  prediction head",
            ha="center", va="center", fontsize=10.8, fontstyle="italic",
            color="#4B5563")

    footer = ("~280K parameters   \u00b7   d_model = 64   "
              "\u00b7   n_heads = 4   \u00b7   d_ff = 128   "
              "\u00b7   L = 45   \u00b7   Huber (\u03b4 = 1.1)   "
              "\u00b7   AdamW + cosine warmup")
    ax.text(8.0, -0.10, footer, ha="center", va="center",
            fontsize=9.2, color="#4B5563",
            bbox=dict(boxstyle="round,pad=0.32",
                      fc="#F3F4F6", ec="#CBD5E0",
                      lw=0.9))

    fig.tight_layout(pad=0.3)
    save_fig(fig, "fig4_architecture")


# ---- FIGURE 5: RADAR CHART ----

def fig5_radar_chart(data):
    mcat = data["mcat"]
    baselines = data["baselines"]
    cats = ["IC", "R2", "DA", "Skill", "MAE\n(lower=better)"]
    nc = len(cats)

    mr = [mcat["ic"], mcat["r2"], mcat["da"], mcat["skill"], mcat["mae"]]
    lgbm = baselines.get("LightGBM", {})
    stf = baselines.get("Single\nTransformer", {})
    lr = [lgbm.get("ic",0), lgbm.get("r2",0), lgbm.get("da",0.5),
          lgbm.get("skill",0), lgbm.get("mae",3.0)]
    sr = [stf.get("ic",0), stf.get("r2",0), stf.get("da",0.5),
          stf.get("skill",0), stf.get("mae",3.0)]

    norm = []
    for vals in zip(mr, lr, sr):
        mn = min(vals)-abs(min(vals))*0.3
        mx = max(vals)+abs(max(vals))*0.3
        if abs(mx-mn)<1e-10: norm.append([0.5]*3)
        else: norm.append([(v-mn)/(mx-mn) for v in vals])
    norm[-1] = [1-v for v in norm[-1]]

    mn_ = [norm[i][0] for i in range(nc)]
    ln_ = [norm[i][1] for i in range(nc)]
    sn_ = [norm[i][2] for i in range(nc)]

    angles = np.linspace(0, 2*np.pi, nc, endpoint=False).tolist()
    angles += angles[:1]
    mn_ += mn_[:1]; ln_ += ln_[:1]; sn_ += sn_[:1]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    ax.plot(angles, mn_, "o-", lw=2.5, color=COLORS["mcat"], label="MCAT (Ours)", ms=7)
    ax.fill(angles, mn_, alpha=0.15, color=COLORS["mcat"])
    ax.plot(angles, ln_, "s--", lw=2, color=COLORS["lightgbm"], label="LightGBM", ms=6)
    ax.fill(angles, ln_, alpha=0.08, color=COLORS["lightgbm"])
    ax.plot(angles, sn_, "^:", lw=1.5, color=COLORS["single_tf"], label="Single Transformer", ms=5)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylim(0,1); ax.set_yticks([0.25,0.5,0.75]); ax.set_yticklabels(["","",""])
    ax.legend(loc="lower right", bbox_to_anchor=(1.25,0), fontsize=10)

    for i, mval in enumerate(mr):
        label = f"{mval:+.3f}" if i < 4 else f"{mval:.3f}"
        ax.annotate(label, xy=(angles[i], mn_[i]), fontsize=7,
                    color=COLORS["mcat"], fontweight="bold", ha="center")

    ax.set_title("Multi-Metric Comparison -- MCAT vs Top Baselines",
                 fontsize=12, fontweight="bold", pad=25)
    save_fig(fig, "fig5_radar_chart")


# ---- MAIN ----

def generate_all(results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR / "tables"
    logger.info("="*60)
    logger.info("  GENERATING PAPER FIGURES")
    logger.info(f"  Reading results from: {results_dir}")
    logger.info("="*60)

    data = load_all_results(results_dir)
    if data is None: return

    fig1_baseline_comparison(data)
    fig2_ablation_study(data)
    fig3_regime_analysis(data)
    fig4_architecture(data)
    fig5_radar_chart(data)

    logger.info(f"\n  All figures saved to: {FIG_DIR}/")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory with *_results.json (default: results/tables/)")
    args = parser.parse_args()
    generate_all(results_dir=args.results_dir)
