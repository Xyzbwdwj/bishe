import argparse
import gzip
import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".cache", "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce Fig.7 / Fig.S7")
    parser.add_argument(
        "--model",
        default="mnist_local",
        type=str,
        help="Output file from Main_local.py",
    )
    parser.add_argument(
        "--mnist-gz",
        default="mnist/data/train-images-idx3-ubyte.gz",
        type=str,
        help="Raw MNIST gzip file used for PCA reconstruction panel",
    )
    parser.add_argument(
        "--n-components", default=68, type=int, help="PCA components"
    )
    parser.add_argument(
        "--n-train", default=100, type=int, help="Number of MNIST images for PCA fit"
    )
    parser.add_argument("--lr", default=0.05, type=float, help="Learning rate for replay")
    parser.add_argument(
        "--out-prefix", default="figure7", type=str, help="Prefix for output figures"
    )
    return parser.parse_args()


def load_last_pickle(path):
    last_obj = None
    with open(path, "rb") as f:
        while True:
            try:
                last_obj = pickle.load(f)
            except EOFError:
                break
    if last_obj is None:
        raise RuntimeError(f"No pickle object found in {path}")
    return last_obj


def save_digits_grid(x, out_path):
    n = x.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    if n == 1:
        axes = [axes]
    c_min = np.min(x)
    c_max = np.max(x)
    for ax, i in zip(axes, range(n)):
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        im = ax.imshow(x[i].reshape(28, 28), cmap="binary_r", vmin=c_min, vmax=c_max)
    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Missing model output: {args.model}")

    dlc = load_last_pickle(args.model)

    # Panel C: reconstruction figures (optional if raw MNIST is available)
    if os.path.exists(args.mnist_gz):
        with gzip.open(args.mnist_gz, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        x = data.reshape(data.shape[0], 28 * 28).astype(np.float64)[: args.n_train]
        pca = PCA(n_components=args.n_components)
        x_pca = pca.fit_transform(x)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        _ = scaler.fit_transform(x_pca)

        x_rec_pca_lc = dlc["output_rep"][-1].T
        x_rec_lc = pca.inverse_transform(x_rec_pca_lc)
        save_digits_grid(x_rec_lc, args.out_prefix + "_panelC_digits.png")
        print(f"Saved: {args.out_prefix}_panelC_digits.png")
    else:
        print(f"Skip panel C: missing {args.mnist_gz}")

    # Fig. S7
    lr = args.lr
    net = dlc["net"]
    grad_list = dlc["grad_list"]
    u = net.U.copy()
    v = net.V.copy()
    r2_list = []
    for i in range(1, len(grad_list[0])):
        slope, intercept, r_value, p_value, std_err = stats.linregress(u.flatten(), v.T.flatten())
        r2_list.append(r_value**2)
        dldu = grad_list[2][-i]
        dldv = grad_list[1][-i]
        u += lr * dldu
        v += lr * dldv

    r2_list = r2_list[::-1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(u.flatten(), v.T.flatten())
    x_line = np.linspace(np.min(u.flatten()), np.max(u.flatten()), 100)
    y_line = slope * x_line + intercept

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    plt.rcParams.update({"font.size": 14})

    im1 = axs[0, 0].imshow(net.U, cmap="bwr")
    axs[0, 0].set_title("U", fontsize=16)
    axs[0, 0].set_xlabel("Columns", fontsize=14)
    axs[0, 0].set_ylabel("Rows", fontsize=14)
    plt.colorbar(im1, ax=axs[0, 0])

    im2 = axs[0, 1].imshow(net.V.T, cmap="bwr")
    axs[0, 1].set_title("V^T", fontsize=16)
    axs[0, 1].set_xlabel("Columns", fontsize=14)
    axs[0, 1].set_ylabel("Rows", fontsize=14)
    plt.colorbar(im2, ax=axs[0, 1])

    axs[1, 0].plot(net.U.flatten(), net.V.T.flatten(), "o", label="Data")
    axs[1, 0].plot(x_line, y_line, color="black", label="Fitted Line", linewidth=2)
    axs[1, 0].set_xlabel("U elements", fontsize=14)
    axs[1, 0].set_ylabel("V^T elements", fontsize=14)
    axs[1, 0].set_title("Linear Regression Plot", fontsize=16)
    axs[1, 0].text(
        0.05,
        0.95,
        f"R-squared: {r_value**2:.4f}\n p-value: {p_value:.2e}",
        transform=axs[1, 0].transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    axs[1, 1].plot(range(len(r2_list)), r2_list)
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("R-squared Value")

    plt.tight_layout()
    fig.savefig(args.out_prefix + "_supp.pdf")
    plt.close(fig)
    print(f"Saved: {args.out_prefix}_supp.pdf")


if __name__ == "__main__":
    main()
