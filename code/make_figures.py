"""Every figure the README points at, regenerated from the artefacts on disk.

Driven by a RUNS registry rather than hardcoded paths, because this project ended up
with five training runs and figures that silently referred to whichever one happened
to have written `out/eval_translation.json` last. Each figure now carries the run it
came from in its title, so a stale figure is visible rather than misleading.

Charts follow one palette in fixed slot order, carry direct labels or a legend
rather than relying on colour alone, and never put two scales on one pair of axes.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a85"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# (label, run dir, identity json, translation json)
RUNS = [
    ("original", "cut", "eval_identity.json", "eval_translation.json"),
    ("+pixel identity", "cut_idtfix", "idtfix_identity.json", "idtfix_translation.json"),
    ("nce=0.25", "cut_rebal", "rebal_identity.json", "rebal_translation.json"),
    ("high-pass critic", "cut_hp", "hp_identity.json", "hp_translation.json"),
    ("scale-matched 2.9x", "cut_s29", "s29_identity.json", "s29_translation.json"),
]
# The 10-seed transfer run on the best model by intrinsic metrics.
TRANSFER_10 = "idtfix_transfer_10seeds.json"
TRANSFER_LABEL = "+pixel identity model, 10 seeds"

ARM_LABEL = {"A_real_txm_only": "A  real TXM only",
             "B_txm_plus_translated_sem": "B  + translated SEM",
             "C_txm_plus_raw_sem": "C  + raw SEM",
             "D_translated_sem_only": "D  translated SEM only"}


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e6e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def save(f, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    f.savefig(C.FIGURES / name, facecolor=SURFACE, bbox_inches="tight")
    plt.close(f)
    print(f"  figures/{name}")


def jload(name):
    p = C.OUT / name
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- transfer

def transfer_figures():
    d = jload(TRANSFER_10)
    if not d:
        print("  (no 10-seed transfer json)")
        return
    arms = [(k, v) for k, v in d["arms"].items() if isinstance(v, dict) and "runs" in v]
    if not arms:
        return
    seeds = [r["seed"] for r in arms[0][1]["runs"]]
    n = len(seeds)

    # grouped bars, AUC and IoU in separate panels (never one shared axis)
    f = plt.figure(figsize=(9.0, 4.3), facecolor=SURFACE, dpi=150)
    for pi, metric in enumerate(("auc", "iou")):
        ax = f.add_subplot(1, 2, pi + 1)
        mu = np.array([v[f"{metric}_mean"] for _, v in arms])
        sd = np.array([v[f"{metric}_sd"] for _, v in arms])
        xs = np.arange(len(arms))
        for i, (m, sdv) in enumerate(zip(mu, sd)):
            ax.bar(xs[i], m, width=0.62, color=SERIES[i], zorder=3,
                   edgecolor=SURFACE, linewidth=2)
            ax.errorbar(xs[i], m, yerr=sdv, color=INK2, linewidth=1.4, capsize=4,
                        zorder=4)
            ax.annotate(f"{m:.3f}", (xs[i], m + sdv), xytext=(0, 5),
                        textcoords="offset points", ha="center", color=INK,
                        fontsize=9, fontweight="bold")
        style(ax, "", metric.upper(),
              "Pixel AUC on held-out TXM" if metric == "auc"
              else "IoU* (threshold tuned on test)")
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABEL.get(k, k).split("  ")[0] for k, _ in arms],
                           color=INK2)
        lo, hi = float((mu - sd).min()), float((mu + sd).max())
        ax.set_ylim(max(0, lo - 0.06), hi + 0.06)
    f.suptitle(f"Label transfer -- {TRANSFER_LABEL}", color=INK, fontsize=11,
               x=0.02, ha="left")
    f.text(0.5, -0.04, "   ".join(ARM_LABEL[k] for k, _ in arms), ha="center",
           color=INK2, fontsize=9)
    save(f, "label_transfer.png")

    # paired deltas with sign-test p-values
    base = "A_real_txm_only"
    bl = {r["seed"]: r["mean_auc"] for r in dict(arms)[base]["runs"]}
    order = [k for k in ("B_txm_plus_translated_sem", "C_txm_plus_raw_sem",
                         "D_translated_sem_only") if k in dict(arms)]
    from math import comb
    f = plt.figure(figsize=(7.8, 3.6), facecolor=SURFACE, dpi=150)
    ax = f.add_subplot(111)
    ax.axvline(0, color=MUTED, linewidth=1.2, zorder=2)
    for i, k in enumerate(order):
        runs = {r["seed"]: r["mean_auc"] for r in dict(arms)[k]["runs"]}
        dl = np.array([runs[s] - bl[s] for s in seeds])
        nz = dl[dl != 0]
        kpos = int((nz > 0).sum())
        m = len(nz)
        tail = sum(comb(m, j) for j in range(max(kpos, m - kpos), m + 1))
        pv = min(1.0, 2.0 * tail / (2 ** m)) if m else 1.0
        y = len(order) - 1 - i
        col = SERIES[(i + 1) % len(SERIES)]
        ax.barh(y, dl.mean(), height=0.44, color=col, zorder=3,
                edgecolor=SURFACE, linewidth=2)
        ax.errorbar(dl.mean(), y, xerr=dl.std(), color=INK2, linewidth=1.4,
                    capsize=4, zorder=4)
        ax.scatter(dl, np.full_like(dl, y, dtype=float), s=34, color=INK, zorder=5,
                   edgecolor=SURFACE, linewidth=1.0)
        sig = pv < 0.05
        ax.annotate(f"{dl.mean():+.4f}   sign test {kpos}/{m}, p={pv:.3f}"
                    + ("  significant" if sig else "  not significant"),
                    (dl.mean(), y), xytext=(0, 17), textcoords="offset points",
                    ha="center", color=INK, fontsize=9,
                    fontweight="bold" if sig else "normal")
    style(ax, "change in pixel AUC vs arm A  (paired, one dot per seed)", "",
          f"Does adding SEM labels help?  {TRANSFER_LABEL}")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([ARM_LABEL[k] for k in reversed(order)], color=INK2)
    ax.set_xlim(-0.30, 0.16)
    ax.set_ylim(-0.6, len(order) - 0.2)
    save(f, "paired_deltas.png")


# ---------------------------------------------------------------- spectra

def spectra_figure():
    f = plt.figure(figsize=(7.4, 4.2), facecolor=SURFACE, dpi=150)
    ax = f.add_subplot(111)
    drew_ref = False
    for i, (lab, _, _, tj) in enumerate(RUNS):
        d = jload(tj)
        if not d or "power_spectrum_bands" not in d:
            continue
        ps = d["power_spectrum_bands"]
        x = np.arange(1, len(ps["txm_mean"]) + 1)
        if not drew_ref:
            ax.plot(x, ps["sem_mean"], color=MUTED, linewidth=2,
                    linestyle=(0, (4, 3)), label="SEM input", zorder=3)
            ax.plot(x, ps["txm_mean"], color=INK, linewidth=2.6,
                    label="real TXM (target)", zorder=4)
            drew_ref = True
        ax.plot(x, ps["translated_mean"], color=SERIES[i % len(SERIES)],
                linewidth=1.8, marker="o", markersize=4, markeredgecolor=SURFACE,
                label=lab, zorder=3)
    ax.set_yscale("log")
    style(ax, "spatial frequency band  (coarse -> fine)", "fraction of power",
          "Radially averaged power spectrum, all runs")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(f, "power_spectrum.png")


# ---------------------------------------------------------------- identity

def identity_figure():
    labs, vals = [], []
    for lab, _, ij, _ in RUNS:
        d = jload(ij)
        if d:
            labs.append(lab)
            vals.append(d["mean_pearson"])
    if not vals:
        return
    # calibration ladder, measured in eval_identity.py
    ladder = [("blur s1", 0.9064), ("blur s2", 0.8607), ("blur s4", 0.7988),
              ("blur s8", 0.7021)]
    f = plt.figure(figsize=(7.6, 3.8), facecolor=SURFACE, dpi=150)
    ax = f.add_subplot(111)
    xs = np.arange(len(labs))
    for i, v in enumerate(vals):
        ax.bar(xs[i], v, width=0.6, color=SERIES[i % len(SERIES)], zorder=3,
               edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{v:.3f}", (xs[i], v), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK, fontsize=9,
                    fontweight="bold")
    for nm, y in ladder:
        ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
        ax.annotate(nm, (len(labs) - 0.45, y), xytext=(6, -3),
                    textcoords="offset points", color=INK2, fontsize=8)
    style(ax, "", "pearson, G(y) vs y on held-out TXM",
          "Distortion the generator inflicts on input already in its target domain")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, color=INK2, fontsize=8.5, rotation=12, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(-0.6, len(labs) + 0.5)
    save(f, "identity_distortion.png")


# ---------------------------------------------------------------- training

def training_figures(run="cut_idtfix", label="+pixel identity"):
    lg = C.ROOT / "runs" / run / "log.csv"
    if not lg.exists():
        return
    rows = list(csv.DictReader(open(lg)))
    if len(rows) < 3:
        return
    it = np.array([int(r["iter"]) for r in rows])
    f = plt.figure(figsize=(7.4, 4.0), facecolor=SURFACE, dpi=150)
    ax = f.add_subplot(111)
    for i, (k, nm) in enumerate([("d_loss", "critic"),
                                 ("g_gan", "generator (adversarial)"),
                                 ("g_nce", "PatchNCE (content)"),
                                 ("g_idt", "identity NCE")]):
        if k not in rows[0]:
            continue
        ax.plot(it, [float(r[k]) for r in rows], color=SERIES[i], linewidth=2,
                label=nm, zorder=3)
    ax.set_yscale("log")
    style(ax, "iteration", "loss  (log scale)", f"Training losses -- {label} run")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(f, "training_losses.png")


# ---------------------------------------------------------------- contrast

def contrast_figure(tj="idtfix_translation.json", label="+pixel identity"):
    d = jload(tj)
    rows = (d or {}).get("contrast") or []
    if not rows:
        return
    b = np.array([r["sem_mean_contrast"] for r in rows])
    a = np.array([r["translated_mean_contrast"] for r in rows])
    f = plt.figure(figsize=(5.2, 5.0), facecolor=SURFACE, dpi=150)
    ax = f.add_subplot(111)
    lim = float(max(abs(b).max(), abs(a).max())) * 1.25
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=1,
            linestyle=(0, (4, 3)), zorder=2)
    ax.scatter(b, a, s=64, color=SERIES[0], edgecolor=SURFACE, linewidth=1.4,
               zorder=3)
    style(ax, "crack contrast in SEM input", "after translation",
          f"Marked cracks keep their contrast -- {label}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    save(f, "crack_contrast.png")


def main():
    print("figures:")
    transfer_figures()
    spectra_figure()
    identity_figure()
    training_figures()
    contrast_figure()


if __name__ == "__main__":
    main()
