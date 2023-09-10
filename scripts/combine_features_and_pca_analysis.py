from typing import Tuple

import os
from glob import glob
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils import remove_unnamed, get_logger

logger = get_logger("pca_analysis")


def export_object_pickle(obj, obj_name: str):
    with open(f"data/{obj_name}.pickle", "wb") as handle:
        pickle.dump(obj, handle)


def load_features(features_path: str) -> pd.DataFrame:
    logger.info("Loading features")
    feature_files = glob(features_path)
    features = []
    for feature_file in feature_files:
        city_features = pd.read_csv(feature_file)
        city_features["city"] = os.path.basename(feature_file).replace(
            "_features.csv", ""
        )
        features.append(city_features)
    return pd.concat(features, join="inner", ignore_index=True)


def scale_features(features: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def process_pca_1(features: np.ndarray):
    pca = PCA()
    pca.fit(features)
    pca.transform(features)

    export_object_pickle(pca, "pca1")
    return pca


def process_pca_2(features: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca2 = PCA(n_components=0.95)
    pca2.fit(features)
    features_pca = pca2.transform(features)

    export_object_pickle(pca2, "pca2")
    return (features_pca, pca2)


def generate_pca_plot(pca1: PCA, pca2: PCA):
    # Calculate cumulative explained variance across all PCs

    cum_exp_var = []
    var_exp = 0
    for i in pca1.explained_variance_ratio_:
        var_exp += i
        cum_exp_var.append(var_exp)

    # Plot cumulative explained variance for all PCs
    _, ax = plt.subplots(figsize=(20, 10))
    ax.axhline(0.95, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(pca2.n_components_, color="red", linewidth=1.5, linestyle="--")
    ax.bar(
        range(0, pca1.n_components_),
        cum_exp_var,
        color="slategrey",
        alpha=0.5,
    )
    ax.set_xlabel("# Principal Components", size="x-large")
    ax.set_ylabel("% Cumulative Variance Explained", size="x-large")
    plt.savefig("data/plots/pca_analysis.png")


def get_pca_to_feature_mapping(features_df: pd.DataFrame, pca2: PCA) -> pd.DataFrame:
    # number of components
    n_pcs = pca_2.components_.shape[0]

    # get the index of the most important feature on EACH component
    # LIST COMPREHENSION HERE
    most_important = [np.abs(pca_2.components_[i]).argmax() for i in range(n_pcs)]

    initial_feature_names = features_df.columns
    # get the names
    most_important_names = [
        initial_feature_names[most_important[i]] for i in range(n_pcs)
    ]

    # LIST COMPREHENSION HERE AGAIN
    dic = {"PC{}".format(i): most_important_names[i] for i in range(n_pcs)}

    # build the dataframe
    df = pd.DataFrame(dic.items())
    df.to_csv("data/pca_components_to_features_mapping.csv", index=False)
    return df


if __name__ == "__main__":
    logger.info("Start combine features and pca analysis")

    scratch_path = "/lustre/scratch/scratch/zcftkst/features/*features.csv"
    features_df = load_features(scratch_path)

    # local_path = "data/features/*features.csv"
    # features_df = load_features(local_path)

    remove_unnamed(features_df)

    logger.info("Saving combined raw features for later")
    features_df.to_csv("data/raw_features.csv", index=False)

    # Popping buildingsID and city for a bit
    features_buildings_id = features_df.pop("buildingsID")
    features_city = features_df.pop("city")

    logger.info("Imputing missing values in features")
    features_df = features_df.fillna(features_df.mean())
    features = features_df.values

    logger.info("Scaling features")
    features = scale_features(features)

    # Run PCA1 and PCA2
    logger.info("PCA1")
    pca_1 = process_pca_1(features)

    logger.info("PCA2")
    features_pca, pca_2 = process_pca_2(features)

    logger.info("Exporting PCA plot")
    generate_pca_plot(pca_1, pca_2)

    logger.info("Export PCA components to features mapping")
    pca_to_feature_mapping = get_pca_to_feature_mapping(features_df, pca_2)

    logger.info("Building final processed features")
    final_features = pd.concat(
        [features_buildings_id, features_city, pd.DataFrame(features_pca)], axis=1
    )
    final_features.columns = ["buildingID", "city", *pca_to_feature_mapping[1].values]

    logger.info("Exporting final features")
    final_features.to_csv("data/final_features.csv", index=False)
