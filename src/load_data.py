import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt

from pathlib import Path


def reshape_images(X):
    X = X.reshape(X.shape[0], 3, 32, 32)
    X = np.moveaxis(X, 1, -1)
    return X


def load_data(data_folder: Path, reshape=True) -> Tuple[np.ndarray]:
    Xtr = np.array(
        pd.read_csv(data_folder + "/Xtr.csv", header=None, sep=",", usecols=range(3072))
    )
    Xte = np.array(
        pd.read_csv(data_folder + "/Xte.csv", header=None, sep=",", usecols=range(3072))
    )
    Ytr = np.array(
        pd.read_csv(data_folder + "/Ytr.csv", sep=",", usecols=[1])
    ).squeeze()
    if reshape:
        Xtr = reshape_images(Xtr)
        Xte = reshape_images(Xte)

    return Xtr, Ytr, Xte


if __name__ == "__main__":

    root_folder = Path(__file__).parent.parent
    data_folder = root_folder / "data"
    Xtr, Ytr, Xte = load_data(data_folder, reshape=False)

    train_images = (Xtr - Xtr.min()) / (Xtr.max() - Xtr.min())
    test_images = (Xte - Xte.min()) / (Xte.max() - Xte.min())

    figure_folder = root_folder / "figures"
    # Visualize images
    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(train_images[0].reshape(32, 32, 3))
    plt.title(f"Label {Ytr[0]}")
    plt.axis("off")

    fig.add_subplot(1, 3, 2)
    plt.imshow(train_images[1].reshape(32, 32, 3))
    plt.title(f"Label {Ytr[1]}")
    plt.axis("off")

    fig.add_subplot(1, 3, 3)
    plt.imshow(train_images[2].reshape(32, 32, 3))
    plt.title(f"Label {Ytr[2]}")
    plt.axis("off")

    fig.savefig(figure_folder / "ex_train_images.png")

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(test_images[0].reshape(32, 32, 3))
    plt.axis("off")

    fig.add_subplot(1, 3, 2)
    plt.imshow(test_images[1].reshape(32, 32, 3))
    plt.axis("off")

    fig.add_subplot(1, 3, 3)
    plt.imshow(test_images[2].reshape(32, 32, 3))

    fig.savefig(figure_folder / "ex_test_images.png")
