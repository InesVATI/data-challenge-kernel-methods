from src.load_data import load_data
import jax.numpy as jnp
import pandas as pd
from src.kernels import RBF
from src.CKN import ModelCKN, ConvKN
from src.svm import MultiClassKernelSVM
import os
import pickle


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), 'data')
    Xtr, Ytr, Xte = load_data(data_folder, reshape=True)
    print(f"Xtr {Xtr.shape}; Ytr {Ytr.shape}; Xte {Xte.shape}")

    models_folder = os.path.join(os.getcwd(), 'models')

    myCKN = ModelCKN(patch_sizes=[3, 2, 2], 
                 out_channels=[64, 128, 256],
                 subsampling_factors=[2, 4, 4],
                 n_patch_per_img_for_kmean=20)
    with open(f'{models_folder}/myckn_f.pkl', 'rb') as f:
        myCKN = pickle.load(f)

    # Compute test features 
    out_test = myCKN(Xte)
    out_test = out_test.reshape(out_test.shape[0], -1)

    # data_mean = jnp.load(os.path.join(models_folder, 'scaler_features_means.npy'))
    # data_std = jnp.load(os.path.join(models_folder, 'scaler_features_std.npy'))

    X = (out_test - out_test.mean(axis=0, keepdims=True))/out_test.std(axis=0, keepdims=True)
    print(f'X mean {X.mean(axis=0)} X var {X.var(axis=0)}')

    kernel_func = RBF(sigma=jnp.sqrt(X.shape[1]))
    my_svm = MultiClassKernelSVM(num_classes=10, kernel_func=kernel_func, c=1)
    with open(f'{models_folder}/classifier_full.pkl', 'rb') as f:
        my_svm = pickle.load(f)

    Yte = my_svm.predict(X)
    Yte = {"Prediction": Yte}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1

    dataframe.to_csv(f"{data_folder}/Yte.csv", index_label="Id")
    