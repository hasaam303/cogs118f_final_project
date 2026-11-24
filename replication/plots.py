"""Figures for the notebook. The generating data remain available in outputs/."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from .analysis import CELL_NAMES, COHORTS, FAMILIES, PAPER_WEIGHTS, bootstrap_means
from .models import PARAMS


TEAL, ORANGE, INK = "#147a79", "#ad501f", "#172e40"
LABELS = {"MF": "Model-free", "MB": "Model-based", "Hybrid": "Hybrid",
          "Unlucky": "Unlucky symbol", "TDLR": "Transition rates"}


def set_style():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#a4afb6",
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
        "savefig.dpi": 160, "savefig.bbox": "tight", "axes.axisbelow": True,
    })


def _stay_axes(ax, title):
    ax.set(xticks=[0, 1], xticklabels=["No reward", "Reward"], xlim=(-.23, 1.23),
           ylim=(0, 1), title=title)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", alpha=.23)


def signature_plot(frame, group_col, groups, seed):
    fig, axes = plt.subplots(1, len(groups), figsize=(2.5*len(groups)+.8, 3.25),
                             sharey=True, layout="constrained")
    for i, (ax, name) in enumerate(zip(axes, groups)):
        group = frame.loc[frame[group_col].eq(name), list(CELL_NAMES)]
        mean, lo, hi = bootstrap_means(group, seed+i)
        for pos, color, marker, label in [([2, 0], TEAL, "o", "Common"), ([3, 1], ORANGE, "s", "Rare")]:
            ax.errorbar([0, 1], mean[pos], yerr=[mean[pos]-lo[pos], hi[pos]-mean[pos]],
                        color=color, marker=marker, capsize=3, label=label, lw=2.2)
        _stay_axes(ax, LABELS.get(name, name))
    axes[0].set_ylabel("Repeat first choice")
    axes[-1].legend(loc="lower right", frameon=False, fontsize=9)
    return fig


def weight_plot(agents):
    fig, ax = plt.subplots(figsize=(7.7, 3.4), layout="constrained")
    for (family, target), color, marker in zip(PAPER_WEIGHTS.items(),
            [TEAL, ORANGE, "#69578a"], ["o", "s", "^"]):
        weights = np.sort(agents.loc[agents.generator.eq(family), "w"].dropna())
        levels = np.arange(1, len(weights)+1) / len(weights)
        ax.step(weights, levels, where="post", color=color, lw=2, label=LABELS[family])
        ax.plot(target, .5, marker=marker, markerfacecolor="white", color=color, ms=8)
    ax.axhline(.5, color="#adb3b6", lw=.8, zorder=0)
    ax.set(xlim=(-.02, 1.02), ylim=(0, 1.02), xlabel="Hybrid model's estimated planning weight (w)",
           ylabel="Fraction of agents at or below w")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.2)
    return fig


def parameter_plot(fits):
    family = "Hybrid"
    fitted = fits[fits.generator.eq(family) & fits.model.eq(family)]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5), layout="constrained")
    for ax, param in zip(axes.flat, PARAMS[family]):
        true, estimate = fitted["true_"+param], fitted[param]
        low, high = min(true.min(), estimate.min()), max(true.max(), estimate.max())
        margin = .05 * (high-low)
        ax.plot([low, high], [low, high], lw=1, ls="--", color="#75838a")
        ax.scatter(true, estimate, s=15, alpha=.65, c=TEAL, edgecolor="none")
        ax.set(title=param, xlim=(low-margin, high+margin), ylim=(low-margin, high+margin))
    axes.flat[-1].axis("off")
    fig.supxlabel("Generating parameter")
    fig.supylabel("Recovered parameter")
    return fig


def confusion_plot(counts):
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.9), layout="constrained")
    ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row in range(len(counts)):
        for col in range(len(counts.columns)):
            ax.text(col, row, f"{counts.iloc[row,col]}\n{fractions.iloc[row,col]:.0%}",
                    ha="center", va="center", fontsize=9,
                    color="white" if fractions.iloc[row,col]>.55 else INK)
    ax.set(xticks=range(len(counts.columns)), xticklabels=counts.columns,
           yticks=range(len(counts)), yticklabels=counts.index,
           xlabel="Family selected by BIC", ylabel="Generating family")
    return fig


def predictive_plot(cells):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10.5, 3.6), layout="constrained")
    for ax, cohort in zip(axes, COHORTS):
        part = cells[cells.cohort.eq(cohort)]
        for common, color, marker in [(1, TEAL, "o"), (0, ORANGE, "s")]:
            row = part[part.common.eq(common)].sort_values("reward")
            x = np.array([0., 1.]) + (.045 if common else -.045)
            ax.fill_between(x, row.lower, row.upper, color=color, alpha=.19, linewidth=0)
            ax.plot(x, row.pred_mean, ls="--", color=color, lw=1.7)
            ax.plot(x, row.observed, marker=marker, color=color,
                    label="Common" if common else "Rare", lw=1.7)
        _stay_axes(ax, f"{cohort} · {part.model.iloc[0]}")
    axes[0].set_ylabel("Repeat first choice")
    axes[-1].legend(loc="lower right", frameon=False)
    return fig
