import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".cache", "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.RNN_Class import ElmanRNN_pred, ElmanSNN_pred


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use a trained MNIST sequence model to predict and save the next frame."
    )
    parser.add_argument("--model", required=True, type=str, help="Trained Main.py checkpoint.")
    parser.add_argument(
        "--input-meta",
        default="mnist/data/processed/MNIST_68PC_SeqN100_Ns5.pth.tar",
        type=str,
        help="Figure6_InputPrep.py output containing PCA metadata.",
    )
    parser.add_argument("--sample", default=0, type=int, help="Sequence index in X_mini.")
    parser.add_argument(
        "--context-frames",
        default=17,
        type=int,
        help="Number of true MNIST frames to feed before predicting the next frame.",
    )
    parser.add_argument("--device", default="cpu", type=str, help="cpu or cuda:0.")
    parser.add_argument(
        "--out-dir",
        default="mnist/mnist_next_frame",
        type=str,
        help="Directory for saved PNG outputs.",
    )
    return parser.parse_args()


def torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def inverse_pca(x, pca_components, pca_mean, center, scale):
    return np.matmul(x * scale + center, pca_components) + pca_mean


def build_model(checkpoint, n, hidden_n):
    use_snn = bool(int(checkpoint.get("snn", 0))) or str(checkpoint.get("model_name", "")).startswith("ElmanSNN")
    if use_snn:
        model = ElmanSNN_pred(
            n,
            hidden_n,
            n,
            lif_alpha=float(checkpoint.get("lif_alpha", 0.9)),
            lif_beta=float(checkpoint.get("lif_beta", 0.9)),
            lif_threshold=float(checkpoint.get("lif_threshold", 1.0)),
            lif_reset=float(checkpoint.get("lif_reset", 1.0)),
            sg_beta=float(checkpoint.get("sg_beta", 10.0)),
            lif_refractory=int(checkpoint.get("lif_refractory", 0)),
            input_spike_mode=str(checkpoint.get("input_spike_mode", "analog")),
            input_spike_scale=float(checkpoint.get("input_spike_scale", 5.0)),
            learnable_threshold=bool(int(checkpoint.get("lif_learn_threshold", 0))),
            readout_mode=str(checkpoint.get("snn_readout", "softmax_seq")),
        )
    else:
        model = ElmanRNN_pred(n, hidden_n, n)
    model.act = nn.Tanh()
    model.load_state_dict(checkpoint["state_dict"])
    return model


def save_frame(path, image, title):
    plt.figure(figsize=(3, 3))
    plt.imshow(image.reshape(28, 28), cmap="binary_r")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint = torch_load_compat(args.model, map_location=device)
    prep = torch_load_compat(args.input_meta, map_location="cpu")
    x_mini = checkpoint["X_mini"].float().to(device)

    ns, seqn, n = x_mini.shape
    if args.sample < 0 or args.sample >= ns:
        raise ValueError(f"--sample must be in [0, {ns - 1}], got {args.sample}.")
    if args.context_frames < 1 or args.context_frames >= seqn:
        raise ValueError(f"--context-frames must be in [1, {seqn - 1}], got {args.context_frames}.")

    state = checkpoint["state_dict"]
    hidden_n = int(state["linear3.weight"].shape[1])
    model = build_model(checkpoint, n, hidden_n).to(device).eval()

    sample_x = x_mini[args.sample : args.sample + 1]
    model_input = torch.zeros((1, args.context_frames + 1, n), device=device)
    model_input[:, : args.context_frames, :] = sample_x[:, : args.context_frames, :]
    h0 = torch.zeros(1, 1, hidden_n, device=device)

    with torch.no_grad():
        output, _ = model(model_input, h0)
    pred_next = output[:, args.context_frames, :].detach().cpu().numpy()
    target_next = sample_x[:, args.context_frames, :].detach().cpu().numpy()
    last_input = sample_x[:, args.context_frames - 1, :].detach().cpu().numpy()

    pca_components = np.asarray(prep["pca_components"], dtype=np.float32)
    pca_mean = np.asarray(prep["pca_mean"], dtype=np.float32)
    center = float(prep["center"])
    scale = float(prep["scale"])

    pred_img = inverse_pca(pred_next, pca_components, pca_mean, center, scale)[0]
    target_img = inverse_pca(target_next, pca_components, pca_mean, center, scale)[0]
    last_img = inverse_pca(last_input, pca_components, pca_mean, center, scale)[0]

    stem = f"sample{args.sample}_context{args.context_frames}"
    pred_path = os.path.join(args.out_dir, stem + "_pred_next.png")
    compare_path = os.path.join(args.out_dir, stem + "_compare.png")
    save_frame(pred_path, pred_img, "Predicted next")

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    panels = [
        (last_img, "Last input"),
        (pred_img, "Predicted next"),
        (target_img, "True next"),
    ]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image.reshape(28, 28), cmap="binary_r")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(compare_path, dpi=200)
    plt.close(fig)

    mse = float(np.mean((pred_img - target_img) ** 2))
    print(f"Saved predicted next frame: {pred_path}")
    print(f"Saved comparison image: {compare_path}")
    print(f"Pixel MSE vs true next frame: {mse:.6g}")


if __name__ == "__main__":
    main()
