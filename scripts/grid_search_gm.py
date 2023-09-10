from time import time

import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from utils import remove_unnamed, get_logger

logger = get_logger("grid_search", True)


def load_features():
    features = pd.read_csv("data/final_features.csv")
    remove_unnamed(features)
    return features


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    start = time()
    bic_score = -estimator.bic(X)
    logger.info(
        f"Hyperparameters: {estimator.get_params()}, "
        f"BIC Score: {bic_score}, "
        f"Time taken: {time() - start} seconds"
    )
    return bic_score


if __name__ == "__main__":
    logger.info("Start Grid Search Script")

    logger.info("Loading Features")
    features_df = load_features()
    features_buildings_id = features_df.pop("buildingID")
    features_city = features_df.pop("city")

    features = features_df.values

    param_grid = {
        "n_components": range(10, 40),
        "covariance_type": ["full"],
    }

    grid_search = GridSearchCV(
        GaussianMixture(max_iter=5),
        param_grid=param_grid,
        scoring=gmm_bic_score,
        verbose=3,
        n_jobs=-1,
    )

    logger.info("Start grid search")
    grid_search.fit(features)

    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )
    df.sort_values(by="BIC score", inplace=True)

    logger.info("Exporting grid search results!")
    df.to_csv("data/grid_search_results.csv")
