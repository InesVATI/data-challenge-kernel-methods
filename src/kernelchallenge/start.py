from kernelchallenge.load_data import load_data
import numpy as np
import pandas as pd
from kernelchallenge.classifiers.KFD import KFD, cosine_similarity
from kernelchallenge.classifiers.svm import MultiClassKernelSVM
from pathlib import Path


def compute_accuracy(Y_true, Y_pred):
    return np.mean(Y_true == Y_pred)


if __name__ == "__main__":
    data_folder = Path(__file__).parent.parent / "__data"
    Xtr, Ytr, Xte = load_data(data_folder)
    # print('Nb train samples:', len(Xtr))
    # print('Nb test samples:', len(Xte))
    # define learning algorithm here
    # classifier.train(Ytr, Xtr)
    # predict on the test data
    # Yte = classifier.predict(Xte)

    # --- KFD classifier ---
    n_train = 4000
    classifier = KFD(cosine_similarity)
    classifier.train(Ytr[:n_train], Xtr[:n_train])

    pred = classifier.predict(Xtr[:n_train])
    accuracy = compute_accuracy(Ytr[:n_train], pred)
    print("Accuracy on trained data:", accuracy * 100, "%")
    pred = classifier.predict(Xtr[n_train:])
    accuracy = compute_accuracy(Ytr[n_train:], pred)
    print("Accuracy on unseen images:", accuracy * 100, "%")

    # --- Save predictions ---

    Yte = classifier.predict(Xte)
    Yte = {'Prediction' : Yte}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv(data_folder / 'Yte.csv', index_label='Id')
