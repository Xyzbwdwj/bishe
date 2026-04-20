#!/usr/bin/env python3
"""
Python replacement for Figure3.m.

Inputs:
- <path>/<model_name>.mat
- <path>/<model_name_pred>.mat

Outputs:
- ElmanSNN_RateCCG12_relu_fixio3.png (+ optional .pdf)
- ElmanSNN_MI12_hist_fixio3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import ttest_1samp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Figure 3 without MATLAB")
    p.add_argument("--path", default="Elman_SGD/Sigmoid", help="Directory containing .mat files")
    p.add_argument("--model-name", default="SeqN1T100_relu_fixio3", help="Current model .mat basename")
    p.add_argument(
        "--model-name-pred",
        default="SeqN1T100_pred_relu_fixio3",
        help="Predictive model .mat basename",
    )
    p.add_argument("--batch-idx", type=int, default=1, help="1-based batch index (MATLAB style)")
    p.add_argument("--repeats", type=int, default=1000)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--maxtau", type=float, default=0.1)
    p.add_argument("--nsubsets", type=int, default=100)
    p.add_argument("--ncode", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-pdf", action="store_true", help="Also export RateCCG panel as PDF")
    p.add_argument("--save-eps", dest="save_pdf", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def load_model(mat_path: Path, batch_idx_1based: int) -> dict[str, np.ndarray]:
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing .mat file: {mat_path}")
    d = loadmat(mat_path)
    b = batch_idx_1based - 1

    hidden = np.squeeze(d["hidden"][-1, b, :, :])  # (T, H)
    y_hat = np.squeeze(d["y_hat"][-1, b, :, :])  # (T, N)
    x_raw = np.squeeze(d["X_mini"][b, :, :])  # (T,) or (T, N)
    if x_raw.ndim == 1:
        x_mini = x_raw[np.newaxis, :]  # (1, T)
    else:
        x_mini = x_raw.T  # (N, T)

    return {
        "W_hh": d["rnn_weight_hh_l0"],
        "W_ih": d["rnn_weight_ih_l0"],
        "b_ih": d["rnn_bias_ih_l0"],
        "hidden": hidden,
        "y_hat": y_hat,
        "x_mini": x_mini,
        "loss": np.squeeze(d["loss"]),
    }


def flatten_trials(x: np.ndarray) -> np.ndarray:
    # MATLAB column-major flattening to mimic reshape behavior in .m workflow.
    return np.asarray(x).reshape(-1, order="F")


def lagged_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    if lag > 0:
        a_l, b_l = a[:-lag], b[lag:]
    elif lag < 0:
        k = -lag
        a_l, b_l = a[k:], b[:-k]
    else:
        a_l, b_l = a, b
    if a_l.size < 2:
        return np.nan
    sa, sb = np.std(a_l), np.std(b_l)
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(a_l, b_l)[0, 1])


def xcorr_pairwise(a: np.ndarray, b: np.ndarray, maxtau: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    max_lag = int(round(maxtau / dt))
    lags = np.arange(-max_lag, max_lag + 1)
    a_vec = flatten_trials(a)
    out = np.zeros((b.shape[0], lags.size), dtype=float)
    for i in range(b.shape[0]):
        b_vec = b[i]
        n = min(a_vec.size, b_vec.size)
        av = a_vec[:n]
        bv = b_vec[:n]
        out[i, :] = [lagged_corr(av, bv, int(l)) for l in lags]
    return lags, out


def binary_mi(x: np.ndarray, y: np.ndarray) -> float:
    # x,y are binary vectors in {0,1}
    n = x.size
    if n == 0:
        return np.nan
    pxy = np.zeros((2, 2), dtype=float)
    for xv in (0, 1):
        for yv in (0, 1):
            pxy[xv, yv] = np.mean((x == xv) & (y == yv))
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mi = 0.0
    for xv in (0, 1):
        for yv in (0, 1):
            p = pxy[xv, yv]
            if p > 0 and px[xv, 0] > 0 and py[0, yv] > 0:
                mi += p * np.log2(p / (px[xv, 0] * py[0, yv]))
    return float(mi)


def shift_for_lag(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return a[:-lag], b[lag:]
    if lag < 0:
        k = -lag
        return a[k:], b[:-k]
    return a, b


def xinfo(
    spk_in: np.ndarray,
    spk_other: np.ndarray,
    maxtau: float,
    dt: float,
    nsubsets: int,
    ncode: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    max_lag = int(round(maxtau / dt))
    lags = np.arange(-max_lag, max_lag + 1)
    in_vec = flatten_trials(spk_in).astype(np.int8)
    n_cells = spk_other.shape[0]
    n_pick = min(ncode, n_cells)
    info = np.zeros((nsubsets, lags.size), dtype=float)

    for s in range(nsubsets):
        pick = rng.choice(n_cells, size=n_pick, replace=False)
        for li, lag in enumerate(lags):
            mi_vals = []
            for c in pick:
                other = spk_other[c, : in_vec.size].astype(np.int8)
                x, y = shift_for_lag(in_vec, other, int(lag))
                mi_vals.append(binary_mi(x, y))
            info[s, li] = np.nanmean(mi_vals)
    return lags, info


def normalize_rows(x: np.ndarray) -> np.ndarray:
    rmin = np.nanmin(x, axis=1, keepdims=True)
    rmax = np.nanmax(x, axis=1, keepdims=True)
    den = np.where((rmax - rmin) == 0, 1.0, (rmax - rmin))
    return (x - rmin) / den


def plot_rate_ccg(
    out_dir: Path,
    lags: np.ndarray,
    xrr_rate: np.ndarray,
    xrr_rate_pred: np.ndarray,
    maxtau: float,
    dt: float,
    save_pdf: bool,
) -> None:
    color2 = np.array([244, 165, 130]) / 255
    color3 = np.array([146, 197, 222]) / 255
    color_combine = (color2 + color3) / 2

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, arry, title in [
        (0, xrr_rate, "Non-predictive"),
        (1, xrr_rate_pred, "Predictive"),
    ]:
        arry = np.flip(arry, axis=1)
        arry_norm = normalize_rows(arry)
        idx = np.nanargmax(arry_norm, axis=1)
        peak_lags = lags[idx]

        ax = axes[row, 0]
        for i in range(arry_norm.shape[0]):
            ax.plot(lags, arry_norm[i], linewidth=0.7, color=(0.7, 0.7, 0.7))
        ax.plot(lags, np.nanmean(arry_norm, axis=0), color="k", linewidth=1)
        ax.axvline(np.nanmean(peak_lags), linewidth=0.7, color="k")
        ax.axvline(0, linestyle="--", linewidth=0.7, color="k")
        ax.set_ylabel("Cross correlation (rate)")
        if row == 1:
            ax.set_xlabel("Input rightward shift (a.u.)")
        ax.set_title(title)

        ax = axes[row, 1]
        t = ttest_1samp(peak_lags, popmean=0.0, nan_policy="omit")
        bins = np.arange(-(maxtau / dt) - 0.5, (maxtau / dt) + 1.5, 1.0)
        ax.hist(peak_lags, bins=bins, color=color_combine, alpha=1.0, edgecolor="white")
        ax.axvline(np.nanmean(peak_lags), linewidth=0.7, color="k")
        ax.axvline(0, linestyle="--", linewidth=0.7, color="k")
        ytop = ax.get_ylim()[1]
        ax.text(2, ytop * 0.75, f"mean={np.nanmean(peak_lags):.2f}", fontsize=12)
        ax.text(2, ytop * 0.60, f"pvalue={t.pvalue:.2g}", fontsize=12)
        ax.set_ylabel("Counts")
        if row == 1:
            ax.set_xlabel("Optimal shift (a.u.)")

    fig.tight_layout()
    png = out_dir / "ElmanSNN_RateCCG12_relu_fixio3.png"
    fig.savefig(png, dpi=200)
    if save_pdf:
        fig.savefig(out_dir / "ElmanSNN_RateCCG12_relu_fixio3.pdf")
    plt.close(fig)


def plot_spike_mi(out_dir: Path, lags: np.ndarray, infor: np.ndarray, infor_pred: np.ndarray, dt: float, maxtau: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    lags_sec = lags * dt

    axes[0, 0].plot(lags_sec, infor.T, color=(0.5, 0.5, 0.5), linewidth=0.3)
    axes[0, 0].plot(lags_sec, np.nanmean(infor, axis=0), color="k", linewidth=1)
    axes[0, 0].axvline(0, linestyle="--", color="k", linewidth=0.8)
    axes[0, 0].set_ylabel("MI")
    axes[0, 0].set_xlabel("Region2 rightward shift (sec)")
    axes[0, 0].set_title("CurrentMdl")

    idx = np.nanargmax(infor, axis=1)
    bins = np.arange(-maxtau - 0.5 * dt, maxtau + 0.5 * dt + dt, dt)
    axes[0, 1].hist(lags_sec[idx], bins=bins)
    axes[0, 1].set_xlabel("Peaked time (sec)")
    axes[0, 1].set_title("CurrentMdl")

    axes[1, 0].plot(lags_sec, infor_pred.T, color=(0.5, 0.5, 0.5), linewidth=0.3)
    axes[1, 0].plot(lags_sec, np.nanmean(infor_pred, axis=0), color="k", linewidth=1)
    axes[1, 0].axvline(dt, linestyle="--", color="k", linewidth=0.8)
    axes[1, 0].set_ylabel("MI")
    axes[1, 0].set_xlabel("Region2 rightward shift (sec)")
    axes[1, 0].set_title("PredMdl")

    idx = np.nanargmax(infor_pred, axis=1)
    axes[1, 1].hist(lags_sec[idx], bins=bins)
    axes[1, 1].set_xlabel("Peaked time (sec)")
    axes[1, 1].set_title("PredMdl")

    fig.tight_layout()
    fig.savefig(out_dir / "ElmanSNN_MI12_hist_fixio3.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.path)
    out_dir.mkdir(parents=True, exist_ok=True)

    cur = load_model(out_dir / f"{args.model_name}.mat", args.batch_idx)
    pred = load_model(out_dir / f"{args.model_name_pred}.mat", args.batch_idx)
    print(f"{args.model_name} final loss: {np.ravel(cur['loss'])[-1]:.6f}")
    print(f"{args.model_name_pred} final loss: {np.ravel(pred['loss'])[-1]:.6f}")

    rng = np.random.default_rng(args.seed)

    seq_len = min(99, cur["hidden"].shape[0], pred["hidden"].shape[0], cur["x_mini"].shape[1] - 1)
    rate_in = np.tile(cur["x_mini"][0, :seq_len][:, None], (1, args.repeats))

    cur_scale = max(np.max(np.abs(cur["hidden"])), 1e-8)
    pred_scale = max(np.max(np.abs(pred["hidden"])), 1e-8)
    rate_current = np.tile((cur["hidden"][:seq_len, :].T / cur_scale), (1, args.repeats))
    rate_pred = np.tile((pred["hidden"][:seq_len, :].T / pred_scale), (1, args.repeats))

    keep_cur = np.mean(rate_current, axis=1) > 0.1
    keep_pred = np.mean(rate_pred, axis=1) > 0.1

    lags, xrr_rate = xcorr_pairwise(rate_in, rate_current[keep_cur, :], args.maxtau, args.dt)
    _, xrr_rate_pred = xcorr_pairwise(rate_in, rate_pred[keep_pred, :], args.maxtau, args.dt)

    spk_in = (rng.poisson(rate_in) > 0).astype(np.int8)
    spk_current = (rng.poisson(rate_current) > 0).astype(np.int8)
    spk_pred = (rng.poisson(rate_pred) > 0).astype(np.int8)

    plot_rate_ccg(out_dir, lags, xrr_rate, xrr_rate_pred, args.maxtau, args.dt, args.save_pdf)

    lags, infor = xinfo(spk_in, spk_current, args.maxtau, args.dt, args.nsubsets, args.ncode, rng)
    _, infor_pred = xinfo(spk_in, spk_pred, args.maxtau, args.dt, args.nsubsets, args.ncode, rng)
    plot_spike_mi(out_dir, lags, infor, infor_pred, args.dt, args.maxtau)

    print(f"Saved Figure3 outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
