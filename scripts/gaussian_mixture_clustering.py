import os
from time import time
from glob import glob

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.mixture import GaussianMixture

from utils import remove_unnamed, get_logger, load_geo_data

logger = get_logger("gaussian_mixture_clustering", True)


def load_features() -> pd.DataFrame:
    features = pd.read_csv("data/final_features.csv")
    remove_unnamed(features)
    return features


def load_tessellation(tessellation_path: str) -> gpd.GeoDataFrame:
    euro_places = pd.read_csv("data/euro_case_studies.csv")["place"].unique()

    tessellation_files = glob(tessellation_path)
    tessellation = []
    for tessellation_file in tessellation_files:
        place_name = os.path.basename(tessellation_file).replace(
            "_tessellation.csv", ""
        )

        if place_name not in euro_places:
            continue

        city_tessellation = load_geo_data(tessellation_file)
        tessellation.append(city_tessellation)
    return pd.concat(tessellation, join="inner", ignore_index=True)


if __name__ == "__main__":
    logger.info("Start Gaussian Mixture Model script")

    start = time()

    logger.info("Loading features")
    features_df = load_features()
    features_buildings_id = features_df.pop("buildingID")
    features_city = features_df.pop("city")

    features = features_df.values

    logger.info("Loading and combining tessellation")
    scratch_path = "/lustre/scratch/scratch/zcftkst/tessellations/*_tessellation.csv"
    tessellation = load_tessellation(scratch_path)

    # local_path = "data/tessellations/*_tessellation.csv"
    # tessellation = load_tessellation(local_path)
    remove_unnamed(tessellation)

    logger.info("Removing tessellations not present in features")
    tessellation = tessellation[tessellation.buildingsID.isin(features_buildings_id)]

    # logger.info(
    #     "Saving final tessellation file in data/tessellations/tessellation_combined.csv"
    # )
    # tessellation.to_csv("data/tessellations/tessellation_combined.csv", index=False)

    logger.info(
        "\n"
        "Preparing the Gaussian Mixture Model:\n"
        "n_components=47 (GridSearch lowest BIC Score)\n"
        "covariance_type='full', max_iter=5, random_state=42"
    )
    gmm = GaussianMixture(
        n_components=47, covariance_type="full", max_iter=5, random_state=42
    )

    logger.info("Start fitting features")
    gmm.fit(features)

    logger.info("Predict cluster id and append to tessellation")
    features_data = pd.concat([features_buildings_id, features_city], axis=1)
    features_data = features_data.rename(columns={"buildingID": "buildingsID"})
    features_data.set_index("buildingsID", inplace=True)
    tessellation.set_index("buildingsID", inplace=True)

    tessellation = features_data.join(
        tessellation, on="buildingsID", how="inner"
    ).reset_index()

    tessellation = gpd.GeoDataFrame(tessellation, geometry="geometry")

    tessellation["cluster"] = gmm.predict(features)

    logger.info("Saving tessellations with cluster and metrics just in case")
    tessellation.to_csv(
        "data/tessellations/tessellation_with_cluster_and_metrics.csv", index=False
    )

    logger.info("Dissolving tessellation by cluster within cities into agglomeration")
    all_agglo_urban_types = []
    for city in tessellation.city.unique():
        city_urban_types = tessellation.loc[tessellation["city"] == city]
        agglo_urban_types = city_urban_types.dissolve(by="cluster")
        all_agglo_urban_types.append(agglo_urban_types)

    all_agglo_urban_types = pd.concat(all_agglo_urban_types, axis=0).reset_index()
    agglomeration = all_agglo_urban_types.explode(ignore_index=True)

    logger.info("Saving agglomeration in data/agglomeration.csv")
    agglomeration.to_csv("data/agglomeration.csv", index=False)
    logger.info(f"Completed main GMM script in {time() - start}s")

    logger.info(
        "Reforming agglomeration before sjoining to tessellation to get buildingsID"
    )
    polygons_with_buildingIDs = agglomeration.reset_index().rename(
        columns={"index": "polygonID"}
    )
    polygons_with_buildingIDs = polygons_with_buildingIDs[
        ["city", "cluster", "polygonID", "geometry"]
    ]

    logger.info(
        "Spatial joining the tessellation on the agglomeration polygons to get buildingsID"
    )
    polygons_with_buildingIDs = polygons_with_buildingIDs.sjoin(
        tessellation, how="left", predicate="contains"
    )

    logger.info("Cleaning up resulting dataframe")
    polygons_with_buildingIDs.drop(
        columns=["index_right"], errors="ignore", inplace=True
    )
    polygons_with_buildingIDs.drop_duplicates("buildingsID", inplace=True)
    polygons_with_buildingIDs.dropna(subset=["buildingsID"], inplace=True)

    logger.info("Aggregating all metrics by polygon")
    # Initialize an empty dictionary to store aggregation functions
    agg_dict = {}

    # Loop through each column to determine the aggregation type
    def name_agg(series):
        unique_values = series.unique()
        if len(unique_values) == 1:
            return unique_values[0]
        else:
            return list(unique_values)

    def ID_agg(series):
        return list(series.astype(int).unique())

    for column in polygons_with_buildingIDs.columns:
        if column not in ["polygonID", "geometry"]:
            if column in ["buildingsID", "cluster"]:
                agg_dict[column] = [ID_agg]
            elif np.issubdtype(polygons_with_buildingIDs[column].dtype, np.number):
                # If the column is numerical, use mean and median
                agg_dict[column] = ["mean", "median"]
            else:
                # If the column is non-numerical, use first and mode
                agg_dict[column] = [name_agg]

    polygons_with_buildingIDs = polygons_with_buildingIDs.groupby("polygonID").agg(
        agg_dict
    )
    polygons_with_buildingIDs.columns = [
        "_".join(col).rstrip("_") for col in polygons_with_buildingIDs.columns.values
    ]
    polygons_with_buildingIDs.reset_index(inplace=True)

    logger.info("Saving polygon metrics in data/polygon_metrics.csv")
    polygons_with_buildingIDs.to_csv("data/polygon_metrics.csv", index=False)

    logger.info(f"Completed full GMM script in {time() - start}s")
