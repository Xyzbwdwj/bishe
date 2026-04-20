import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RNN_Class import ElmanRNN_pred, ElmanSNN_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MNIST public-dataset compare runs (RNN vs SNN).")
    parser.add_argument(
        "--ckpt-dir",
        default="mnist_compare/public",
        type=str,
        help="Directory containing *seed*_e*.pth.tar checkpoints.",
    )
    parser.add_argument(
        "--input-meta",
        default="Elman_SGD/predloss/MNIST_68PC_SeqN100_Ns5.pth.tar",
        type=str,
        help="Figure6 input-meta file with PCA stats.",
    )
    parser.add_argument(
        "--out-dir",
        default="mnist_compare/public/report",
        type=str,
        help="Output report directory.",
    )
    parser.add_argument(
        "--stop-t",
        default=17,
        type=int,
        help="Teacher-forcing stop time for free-run metric.",
    )
    return parser.parse_args()


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def inverse_pca(x, pca_components, pca_mean, center, scale):
    # x: (..., n_components)
    x0 = x * scale + center
    return np.matmul(x0, pca_components) + pca_mean


def infer_model_and_run(ckpt, stop_t):
    state = ckpt["state_dict"]
    x = ckpt["X_mini"].float()
    y = ckpt["Target_mini"].float()
    n = int(x.shape[-1])
    hidden = int(state["linear3.weight"].shape[1])
    use_snn = bool(int(ckpt.get("snn", 0)))

    if use_snn:
        model = ElmanSNN_pred(
            n,
            hidden,
            n,
            lif_alpha=float(ckpt.get("lif_alpha", 0.9)),
            lif_beta=float(ckpt.get("lif_beta", 0.9)),
            lif_threshold=float(ckpt.get("lif_threshold", 1.0)),
            lif_reset=float(ckpt.get("lif_reset", 1.0)),
            sg_beta=float(ckpt.get("sg_beta", 10.0)),
            lif_refractory=int(ckpt.get("lif_refractory", 0)),
            input_spike_mode=str(ckpt.get("input_spike_mode", "analog")),
            input_spike_scale=float(ckpt.get("input_spike_scale", 5.0)),
            learnable_threshold=bool(int(ckpt.get("lif_learn_threshold", 0))),
            readout_mode=str(ckpt.get("snn_readout", "softmax_seq")),
        )
        model_type = "snn"
    else:
        model = ElmanRNN_pred(n, hidden, n)
        model_type = "rnn"

    model.act = nn.Tanh()
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        h0 = torch.zeros(1, x.shape[0], hidden)
        out_tf, _ = model(x, h0)
        tf_mse = float(torch.mean((out_tf - y) ** 2).item())

        stop_t = min(stop_t, x.shape[1] - 1)
        x_null = torch.zeros_like(x)
        x_null[:, :stop_t, :] = x[:, :stop_t, :]
        out_free = torch.zeros_like(y)
        out_t, htp1 = model(x[:, :stop_t, :], torch.zeros_like(h0))
        out_free[:, :stop_t, :] = out_t.detach()
        for t in range(x.shape[1] - stop_t):
            xtp1 = out_free[:, stop_t + t - 1 : stop_t + t + 1, :]
            otp1, htp1 = model(xtp1, htp1)
            out_free[:, stop_t + t : stop_t + t + 1, :] = otp1[:, 1:, :].detach()

        fr_mse = float(torch.mean((out_free[:, stop_t:, :] - y[:, stop_t:, :]) ** 2).item())

    return model_type, tf_mse, fr_mse, out_tf.numpy(), out_free.numpy(), y.numpy()


def seed_from_name(name):
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else -1


def mean_std(rows, key, model_type):
    vals = [r[key] for r in rows if r["model"] == model_type]
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0), len(arr)


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = torch_load_compat(args.input_meta, map_location="cpu")
    pca_components = np.asarray(prep["pca_components"], dtype=np.float32)
    pca_mean = np.asarray(prep["pca_mean"], dtype=np.float32)
    center = float(prep["center"])
    scale = float(prep["scale"])

    rows = []
    for ckpt_path in sorted(ckpt_dir.glob("*.pth.tar")):
        ckpt = torch_load_compat(ckpt_path, map_location="cpu")
        loss = np.asarray(ckpt.get("loss", []), dtype=np.float64)
        if loss.size == 0:
            continue
        model_type, tf_mse, fr_mse, out_tf, out_free, y = infer_model_and_run(ckpt, args.stop_t)

        y_pix = inverse_pca(y.reshape(-1, y.shape[-1]), pca_components, pca_mean, center, scale)
        out_free_pix = inverse_pca(
            out_free.reshape(-1, out_free.shape[-1]), pca_components, pca_mean, center, scale
        )
        fr_pixel_mse = float(np.mean((out_free_pix - y_pix) ** 2))

        rows.append(
            {
                "run": ckpt_path.stem.replace(".pth", ""),
                "model": model_type,
                "seed": seed_from_name(ckpt_path.stem),
                "epochs": int(loss.size),
                "loss_start": float(loss[0]),
                "loss_end": float(loss[-1]),
                "loss_min": float(loss.min()),
                "tf_mse": tf_mse,
                "free_run_mse": fr_mse,
                "free_run_pixel_mse": fr_pixel_mse,
            }
        )

    rows.sort(key=lambda r: (r["model"], r["seed"], r["run"]))

    # per-run csv
    per_run_csv = out_dir / "per_run_metrics.csv"
    fields = [
        "run",
        "model",
        "seed",
        "epochs",
        "loss_start",
        "loss_end",
        "loss_min",
        "tf_mse",
        "free_run_mse",
        "free_run_pixel_mse",
    ]
    with per_run_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # summary csv
    summary_rows = []
    for model_type in ("rnn", "snn"):
        for key in ("loss_end", "tf_mse", "free_run_mse", "free_run_pixel_mse"):
            mean, std, n = mean_std(rows, key, model_type)
            summary_rows.append(
                {"model": model_type, "metric": key, "mean": mean, "std": std, "n": n}
            )

    summary_csv = out_dir / "summary_by_model.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "metric", "mean", "std", "n"])
        w.writeheader()
        w.writerows(summary_rows)

    # simple bar plots
    metrics_for_plot = ["loss_end", "free_run_mse", "free_run_pixel_mse"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, metrics_for_plot):
        rnn_mean, rnn_std, _ = mean_std(rows, metric, "rnn")
        snn_mean, snn_std, _ = mean_std(rows, metric, "snn")
        ax.bar([0, 1], [rnn_mean, snn_mean], yerr=[rnn_std, snn_std], capsize=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["RNN", "SNN"])
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_barplots.png", dpi=160)
    plt.close(fig)

    # markdown summary
    md_path = out_dir / "summary.md"
    lines = []
    lines.append("# Public MNIST Compare (RNN vs SNN)")
    lines.append("")
    lines.append("## Per-Run Metrics")
    lines.append("")
    lines.append(
        "| run | model | seed | epochs | loss_start | loss_end | loss_min | tf_mse | free_run_mse | free_run_pixel_mse |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['model']} | {r['seed']} | {r['epochs']} | "
            f"{r['loss_start']:.6g} | {r['loss_end']:.6g} | {r['loss_min']:.6g} | "
            f"{r['tf_mse']:.6g} | {r['free_run_mse']:.6g} | {r['free_run_pixel_mse']:.6g} |"
        )
    lines.append("")
    lines.append("## Model-Wise Mean ± Std")
    lines.append("")
    lines.append("| model | metric | mean | std | n |")
    lines.append("|---|---|---:|---:|---:|")
    for r in summary_rows:
        lines.append(
            f"| {r['model']} | {r['metric']} | {r['mean']:.6g} | {r['std']:.6g} | {r['n']} |"
        )
    lines.append("")
    lines.append("Artifacts:")
    lines.append("- per_run_metrics.csv")
    lines.append("- summary_by_model.csv")
    lines.append("- compare_barplots.png")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"WROTE {per_run_csv}")
    print(f"WROTE {summary_csv}")
    print(f"WROTE {md_path}")
    print(f"WROTE {out_dir / 'compare_barplots.png'}")
    print(f"RUNS {len(rows)}")


if __name__ == "__main__":
    main()
