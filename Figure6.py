import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import FastICA

from RNN_Class import ElmanRNN_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce Fig.6 from trained model")
    parser.add_argument(
        "--model",
        default="Elman_SGD/predloss/MNIST_68PC_SeqN100_Ns5_partial.pth.tar",
        type=str,
        help="Trained predictive checkpoint from Main.py",
    )
    parser.add_argument(
        "--input-meta",
        default="Elman_SGD/predloss/MNIST_68PC_SeqN100_Ns5.pth.tar",
        type=str,
        help="Output of Figure6_InputPrep.py containing PCA metadata",
    )
    parser.add_argument("--hidden-n", default=200, type=int, help="Hidden size")
    parser.add_argument("--stop-t", default=17, type=int, help="Teacher-forcing stop time")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or cuda:0")
    parser.add_argument(
        "--out-prefix",
        default="",
        type=str,
        help="Output prefix (default: model path without .pth.tar)",
    )
    return parser.parse_args()


def inverse_pca(x, pca_components, pca_mean):
    return np.matmul(x, pca_components) + pca_mean


def main():
    args = parse_args()
    device = torch.device(args.device)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Missing model file: {args.model}")
    if not os.path.exists(args.input_meta):
        raise FileNotFoundError(f"Missing input meta file: {args.input_meta}")

    net = torch.load(args.model, map_location=device)
    prep = torch.load(args.input_meta, map_location="cpu")
    required_keys = ["pca_components", "pca_mean", "center", "scale"]
    for k in required_keys:
        if k not in prep:
            raise KeyError(
                f"{k} not found in {args.input_meta}. Re-run Figure6_InputPrep.py first."
            )

    x_mini = net["X_mini"].cpu()
    ns, seqn, n = x_mini.shape
    hidden_n = args.hidden_n
    stop_t = args.stop_t
    loop_n = int(prep.get("loop_n", round(seqn / 10)))

    model = ElmanRNN_pred(n, hidden_n, n)
    model.act = nn.Tanh()
    model.load_state_dict(net["state_dict"])
    model = model.to(device).eval()

    h_t = torch.zeros(1, ns, hidden_n, device=device)
    x_null = torch.zeros((ns, seqn, n), device=device)
    x_null[:, :stop_t, :] = x_mini[:, :stop_t, :].to(device)
    with torch.no_grad():
        output, _ = model(x_null, h_t)
        o_future = torch.zeros_like(output)
        output_t, htp1 = model(x_mini[:, :stop_t, :].to(device), torch.zeros(1, ns, hidden_n, device=device))
        o_future[:, :stop_t, :] = output_t.detach()
        for t in range(seqn - stop_t):
            xtp1 = o_future[:, stop_t + t - 1 : stop_t + t + 1, :]
            otp1, htp1 = model(xtp1, htp1)
            o_future[:, stop_t + t : stop_t + t + 1, :] = otp1[:, 1:, :].detach()

    center = float(prep["center"])
    scale = float(prep["scale"])
    pca_components = np.asarray(prep["pca_components"])
    pca_mean = np.asarray(prep["pca_mean"])

    o = output.detach().cpu().numpy().reshape(ns * seqn, n)
    o_re = inverse_pca(o * scale + center, pca_components, pca_mean)

    out_prefix = args.out_prefix if args.out_prefix else args.model[:-8]

    # Panel B/C: reconstruction and prediction
    plt.figure(figsize=(20, ns))
    count = 1
    for i in range(ns):
        for j in range(20):
            plt.subplot(ns, 20, count)
            plt.imshow(o_re[i * seqn + j, :].reshape((28, 28)))
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)
            count += 1
    plt.tight_layout()
    plt.savefig(out_prefix + "_O_re.pdf")
    plt.close()

    # recurrent unit activity
    ht_seq = torch.zeros((ns, seqn, hidden_n), device=device)
    htp1_seq = torch.zeros((ns, seqn, hidden_n), device=device)
    ht = torch.zeros((1, ns, hidden_n), device=device)
    with torch.no_grad():
        for t in range(seqn):
            ht = model.tanh(model.input_linear(x_mini[:, t, :].to(device)) + model.hidden_linear(ht))
            htp1 = model.tanh(model.hidden_linear(ht))
            ht_seq[:, t, :] = ht.detach()
            htp1_seq[:, t, :] = htp1.detach()

    htp1_full = htp1_seq.detach().cpu().numpy().reshape(ns * seqn, hidden_n)

    # Panel D: ICA
    ica = FastICA(n_components=10, max_iter=1000)
    x_ic = ica.fit_transform(htp1_full)

    ic1_list = np.array([0, 1, 2, 3])
    ic2_list = np.array([0, 1, 2, 3])
    plt.figure(figsize=(7, 7))
    for idx in range(len(ic1_list)):
        ic1 = ic1_list[idx]
        ic2 = ic2_list[idx]
        plt.subplot(2, 2, idx + 1)
        for digit in range(10):
            idx_digit = np.arange(loop_n) * 10 + digit
            plt.plot(x_ic[idx_digit, ic1], x_ic[idx_digit, ic2], ".")
        plt.xlabel("IC{}".format(ic1))
        plt.ylabel("IC{}".format(ic2))
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.rc("font", size=12)

    plt.tight_layout()
    plt.savefig(out_prefix + "_htp1_cluster_select.pdf")
    plt.close()


if __name__ == "__main__":
    main()
