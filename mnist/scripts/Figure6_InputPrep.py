import argparse
import gzip
import os

import numpy as np
import torch
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MNIST PCA sequence input for Fig.6")
    parser.add_argument(
        "--images",
        default="mnist/data/processed/Rotated/MNIST_X_train.npy",
        type=str,
        help="Path to flattened MNIST image array (num_samples x 784)",
    )
    parser.add_argument(
        "--labels",
        default="mnist/data/processed/Rotated/MNIST_labels.npy",
        type=str,
        help="Path to labels array (num_samples,)",
    )
    parser.add_argument(
        "--mnist-images-gz",
        default="mnist/data/train-images-idx3-ubyte.gz",
        type=str,
        help="Fallback raw MNIST images gzip if --images is missing",
    )
    parser.add_argument(
        "--mnist-labels-gz",
        default="mnist/data/train-labels-idx1-ubyte.gz",
        type=str,
        help="Fallback raw MNIST labels gzip if --labels is missing",
    )
    parser.add_argument("--n-components", default=68, type=int, help="PCA components")
    parser.add_argument("--seqn", default=100, type=int, help="Sequence length")
    parser.add_argument("--ns", default=5, type=int, help="Number of sequences")
    parser.add_argument(
        "--out",
        default="mnist/data/processed/MNIST_68PC_SeqN100_Ns5.pth.tar",
        type=str,
        help="Output .pth.tar path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.images) and os.path.exists(args.labels):
        images = np.load(args.images)
        labels = np.load(args.labels)
    elif os.path.exists(args.mnist_images_gz) and os.path.exists(args.mnist_labels_gz):
        with gzip.open(args.mnist_images_gz, "rb") as f:
            images_u8 = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(args.mnist_labels_gz, "rb") as f:
            labels_u8 = np.frombuffer(f.read(), np.uint8, offset=8)
        images = images_u8.reshape(images_u8.shape[0], -1).astype(np.float32) / 255.0
        labels = labels_u8.astype(np.int64)
        print(f"Loaded raw MNIST from {args.mnist_images_gz} / {args.mnist_labels_gz}")
    else:
        raise FileNotFoundError(
            "Missing MNIST inputs. Provide either "
            f"{args.images} + {args.labels} or "
            f"{args.mnist_images_gz} + {args.mnist_labels_gz}."
        )

    pca = PCA(n_components=args.n_components)
    pca.fit(images)
    print("Explained variance ratio sum:", np.sum(pca.explained_variance_ratio_))

    im_de = pca.transform(images)
    center = (np.max(im_de) + np.min(im_de)) / 2
    scale = (np.max(im_de) - np.min(im_de)) / 2
    im_de_scale = (im_de - center) / scale

    im_num = []
    for i in range(10):
        im_num.append(im_de_scale[labels == i, :])

    n = im_de.shape[1]
    seqn = args.seqn
    ns = args.ns
    loop_n = round(seqn / 10)

    tmp = np.zeros((ns, seqn, n), dtype=np.float32)
    for i in range(ns):
        for j in range(10):
            idx = np.arange(0, 10 * loop_n, 10) + j
            im_pre = im_num[j][np.arange(loop_n) + i * loop_n]
            tmp[i, idx, :] = im_pre

    x_mini = torch.tensor(tmp)
    save_dict = {
        "X_mini": x_mini,
        "Target_mini": x_mini,
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "center": float(center),
        "scale": float(scale),
        "loop_n": int(loop_n),
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(save_dict, args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
