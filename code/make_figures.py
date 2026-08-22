"""Every figure in the README, from the artefacts on disk.

Charts follow one palette in fixed slot order (blue, orange, aqua, yellow),
carry direct labels rather than relying on colour alone, and never put two
scales on one pair of axes -- AUC and IoU get their own panels because they are
different measures, and edge-correlation gets its own figure rather than a
second y-axis on the loss plot.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e6e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def fig(w, h):
    f = plt.figure(figsize=(w, h), facecolor=SURFACE, dpi=150)
    return f


def save(f, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    p = C.FIGURES / name
    f.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    plt.close(f)
    print(f"  wrote figures/{name}")


# ------------------------------------------------------------------ training

def training_curves(run=None):
    run = Path(run or C.ROOT / "runs" / "cut")
    log = run / "log.csv"
    if not log.exists():
        print("  (no training log yet)")
        return
    rows = list(csv.DictReader(open(log)))
    if len(rows) < 2:
        return
    it = np.array([int(r["iter"]) for r in rows])

    f = fig(7.2, 4.0)
    ax = f.add_subplot(111)
    for i, (k, lab) in enumerate([("d_loss", "critic"), ("g_gan", "generator (adversarial)"),
                                  ("g_nce", "PatchNCE (content)"), ("g_idt", "identity NCE")]):
        v = np.array([float(r[k]) for r in rows])
        ax.plot(it, v, color=SERIES[i], linewidth=2, label=lab, zorder=3)
        ax.annotate(lab, (it[-1], v[-1]), xytext=(6, 0), textcoords="offset points",
                    color=SERIES[i], fontsize=9, va="center")
    style(ax, "iteration", "loss", "Training losses")
    ax.set_xlim(it.min(), it.max() * 1.28)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK2)
    save(f, "training_losses.png")

    f = fig(7.2, 3.2)
    ax = f.add_subplot(111)
    ec = np.array([float(r["edge_corr"]) for r in rows])
    ax.plot(it, ec, color=SERIES[0], linewidth=2, zorder=3)
    ax.annotate(f"{ec[-1]:.3f}", (it[-1], ec[-1]), xytext=(6, 0),
                textcoords="offset points", color=SERIES[0], fontsize=10,
                va="center", fontweight="bold")
    style(ax, "iteration", "edge-map correlation",
          "Structure retention: input edges vs output edges")
    ax.set_xlim(it.min(), it.max() * 1.12)
    ax.set_ylim(0, max(1.0, float(ec.max()) * 1.1))
    save(f, "structure_retention.png")


# ------------------------------------------------------------------ spectra

def spectra():
    p = C.OUT / "eval_translation.json"
    if not p.exists():
        print("  (no eval_translation.json yet)")
        return
    d = json.loads(p.read_text())
    bands = d.get("power_spectrum_bands")
    if not bands:
        return
    order = [("sem_mean", "SEM input"), ("translated_mean", "translated"),
             ("txm_mean", "real TXM")]
    f = fig(6.8, 4.0)
    ax = f.add_subplot(111)
    x = np.arange(1, len(next(iter(bands.values()))) + 1)
    for i, (k, lab) in enumerate(order):
        if k not in bands:
            continue
        y = np.array(bands[k])
        ax.plot(x, y, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=lab, zorder=3)
        ax.annotate(lab, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    color=SERIES[i], fontsize=9, va="center")
    ax.set_yscale("log")
    style(ax, "spatial frequency band  (coarse -> fine)", "fraction of power",
          "Radially averaged power spectrum")
    ax.set_xlim(x.min(), x.max() * 1.22)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK2)
    save(f, "power_spectrum.png")


# ------------------------------------------------------------------ transfer

ARM_LABEL = {
    "A_real_txm_only": "A  real TXM only",
    "B_txm_plus_translated_sem": "B  + translated SEM",
    "C_txm_plus_raw_sem": "C  + raw SEM",
    "D_translated_sem_only": "D  translated SEM only",
}


def label_transfer():
    p = C.OUT / "label_transfer.json"
    if not p.exists():
        print("  (no label_transfer.json yet)")
        return
    d = json.loads(p.read_text())
    arms = [(k, v) for k, v in d["arms"].items() if isinstance(v, dict)]
    if not arms:
        return
    names = [ARM_LABEL.get(k, k) for k, _ in arms]
    f = fig(9.0, 4.2)
    for pi, (metric, lo_pad) in enumerate([("auc", 0.02), ("iou", 0.02)]):
        ax = f.add_subplot(1, 2, pi + 1)
        mu = np.array([v[f"{metric}_mean"] for _, v in arms])
        sd = np.array([v[f"{metric}_sd"] for _, v in arms])
        xs = np.arange(len(arms))
        for i, (m, s) in enumerate(zip(mu, sd)):
            ax.bar(xs[i], m, width=0.62, color=SERIES[i], zorder=3,
                   edgecolor=SURFACE, linewidth=2)
            ax.errorbar(xs[i], m, yerr=s, color=INK2, linewidth=1.4,
                        capsize=4, zorder=4)
            ax.annotate(f"{m:.3f}", (xs[i], m + s), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        color=INK, fontsize=9, fontweight="bold")
        style(ax, "", metric.upper(),
              "Pixel AUC on held-out TXM" if metric == "auc" else "IoU at best threshold")
        ax.set_xticks(xs)
        ax.set_xticklabels([n.split("  ")[0] for n in names], color=INK2)
        top = float((mu + sd).max())
        ax.set_ylim(max(0.0, float((mu - sd).min()) - lo_pad * 4), top + lo_pad * 3)
    f.text(0.5, -0.04, "   ".join(names), ha="center", color=INK2, fontsize=9)
    save(f, "label_transfer.png")


# ------------------------------------------------------------------ contrast

def contrast_scatter():
    p = C.OUT / "eval_translation.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    rows = d.get("contrast") or []
    if not rows:
        print("  (no contrast rows)")
        return
    b = np.array([r["sem_mean_contrast"] for r in rows])
    a = np.array([r["translated_mean_contrast"] for r in rows])
    f = fig(5.4, 5.0)
    ax = f.add_subplot(111)
    lim = float(max(abs(b).max(), abs(a).max())) * 1.25
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=1,
            linestyle=(0, (4, 3)), zorder=2)
    ax.scatter(b, a, s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for r, xx, yy in zip(rows, b, a):
        ax.annotate(r["image"][:16], (xx, yy), xytext=(6, 4),
                    textcoords="offset points", color=INK2, fontsize=7)
    style(ax, "crack contrast in SEM input", "crack contrast after translation",
          "Marked cracks stay darker than their surroundings")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    save(f, "crack_contrast.png")


def main():
    print("figures:")
    training_curves()
    spectra()
    label_transfer()
    contrast_scatter()


if __name__ == "__main__":
    main()
