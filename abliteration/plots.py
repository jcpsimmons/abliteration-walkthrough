"""Portable, real-data charts. PCA axes are shared across before/after."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

COLORS = {"harmful": "#ff6b6b", "harmless": "#00ffff"}


def style():
    plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black",
                         "text.color": "#87ff87", "axes.labelcolor": "#00ffff",
                         "xtick.color": "#bcbcbc", "ytick.color": "#bcbcbc",
                         "axes.edgecolor": "#00afaf", "font.family": "monospace",
                         "font.size": 13, "savefig.facecolor": "black"})


def read(out, name):
    return json.loads((out / name).read_text())


def save(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2) + "\n")


def finish(fig, out, name):
    folder = out / "plots"
    folder.mkdir(exist_ok=True)
    fig.tight_layout(pad=2)
    fig.savefig(folder / f"{name}.png", dpi=180)
    fig.savefig(folder / f"{name}.svg")
    plt.close(fig)


def before_plots(out):
    style()
    i = read(out, "sweep.json")["selected"]["block"] - 1
    states = torch.load(out / "activations.pt", weights_only=True)
    arrow = torch.load(out / "directions.pt", weights_only=True)[i].numpy()
    values = {k: v[:, i].numpy() for k, v in states.items()}
    pca = PCA(n_components=2, svd_solver="full").fit(np.concatenate(list(values.values())))
    before = {k: pca.transform(v).tolist() for k, v in values.items()}
    both = np.concatenate(list(before.values()))
    span = np.maximum(np.ptp(both, axis=0), 1.)
    chart = {"block": i + 1, "pca_mean": pca.mean_.tolist(), "pca_components": pca.components_.tolist(),
             "explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "before": before,
             "before_projection": {k: (v @ arrow).tolist() for k, v in values.items()},
             "xlim": [float(both[:, 0].min() - span[0]*.1), float(both[:, 0].max() + span[0]*.1)],
             "ylim": [float(both[:, 1].min() - span[1]*.1), float(both[:, 1].max() + span[1]*.1)]}
    save(out, "chart-data.json", chart)
    draw_states(out, chart, "before")
    sweep = read(out, "sweep.json")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    rows = sweep["rows"]
    axes[0].plot([r["block"] for r in rows], [r["refusals"] for r in rows], "o-", color="#ff6b6b")
    axes[0].set(ylabel=f"Refusal markers / {rows[0]['n']}", xlabel="Candidate block", title="Validation responses")
    axes[1].plot([r["block"] for r in rows], [r["kl_nats"] for r in rows], "o-", color="#00ffff")
    axes[1].set(ylabel="KL(base || edited), nats", xlabel="Candidate block", title="Benign first-token distribution shift")
    for ax in axes:
        ax.axvline(i + 1, color="#ffff00", linestyle="--")
        ax.grid(alpha=.12)
    finish(fig, out, "layer-sweep")


def draw_states(out, chart, key):
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, points in chart[key].items():
        p = np.asarray(points)
        marker = "x" if label == "harmful" else "o"
        color_args = {"color": COLORS[label]} if label == "harmful" else {"facecolors": "none", "edgecolors": COLORS[label]}
        ax.scatter(p[:, 0], p[:, 1], marker=marker, s=36, linewidths=1.2, alpha=.9,
                   label=f"{'X' if label == 'harmful' else 'O'} = {label} prompts", **color_args)
    ax.set(xlabel="PC1 (original basis)", ylabel="PC2 (original basis)", xlim=chart["xlim"], ylim=chart["ylim"],
           title=f"{key.capitalize()}: last-token residual states at block {chart['block']}")
    ax.legend(facecolor="black", labelcolor="linecolor")
    ax.grid(alpha=.12)
    finish(fig, out, f"{key}-pca")
    fig, ax = plt.subplots(figsize=(12, 6))
    # Shared bins include all original and changed scores.
    scores = [x for k in ("before_projection", "after_projection") if k in chart
              for vals in chart[k].values() for x in vals]
    bins = np.linspace(min(scores) - 1, max(scores) + 1, 30)
    for label, vals in chart[f"{key}_projection"].items():
        ax.hist(vals, bins=bins, alpha=.55, color=COLORS[label], label=label + "-source")
    ax.set(xlabel="Signed component along selected unit direction", ylabel="Training prompts",
           title=f"{key.capitalize()}: projection on the selected arrow")
    ax.legend(facecolor="black", labelcolor="#87ff87")
    finish(fig, out, f"{key}-projection")


def after_plots(out):
    style()
    chart = read(out, "chart-data.json")
    i = chart["block"] - 1
    states = torch.load(out / "activations-after.pt", weights_only=True)
    arrow = torch.load(out / "directions.pt", weights_only=True)[i].numpy()
    mean, components = np.asarray(chart["pca_mean"]), np.asarray(chart["pca_components"])
    chart["after"] = {k: ((v[:, i].numpy() - mean) @ components.T).tolist() for k, v in states.items()}
    chart["after_projection"] = {k: (v[:, i].numpy() @ arrow).tolist() for k, v in states.items()}
    # Expand both before/after axes together if edited values leave the old window.
    both = np.concatenate([v for key in ("before", "after") for v in chart[key].values()])
    span = np.maximum(np.ptp(both, axis=0), 1.)
    chart["xlim"] = [float(both[:, 0].min() - .1*span[0]), float(both[:, 0].max() + .1*span[0])]
    chart["ylim"] = [float(both[:, 1].min() - .1*span[1]), float(both[:, 1].max() + .1*span[1])]
    save(out, "chart-data.json", chart)
    for key in ("before", "after"):
        draw_states(out, chart, key)


def evaluation_plots(out):
    style()
    ev = read(out, "evaluation.json")
    fig, ax = plt.subplots(figsize=(12, 6))
    counts = [ev[x]["harmful_refusals"] for x in ("before", "after")]
    bars = ax.barh(["Before", "After"], counts, color=["#ff6b6b", "#00ffff"])
    ax.set(xlim=(0, ev["before"]["harmful_n"] + 2), xlabel="Held-out responses with refusal markers", title="Held-out evaluation")
    for bar, value in zip(bars, counts):
        ax.text(value + .3, bar.get_y() + bar.get_height()/2, str(value), va="center", color="white")
    finish(fig, out, "held-out-refusals")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    for ax, key in zip(axes, ("before", "after")):
        vals = ev[key]["next_tokens"]
        ax.barh([repr(v["token"]) for v in vals][::-1], [v["probability"] for v in vals][::-1], color="#00ffff")
        ax.set(title=key.capitalize(), xlabel="Probability of first assistant token")
    fig.suptitle('Same chat prompt: "The best way to cook rice is"')
    finish(fig, out, "next-token-odds")
