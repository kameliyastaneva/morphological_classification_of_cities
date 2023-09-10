import pandas as pd
import geopandas as gpd
import numpy as np
import momepy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


from utils import (
    get_logger,
    load_geo_data,
    remove_unnamed,
    get_utm_zone_from_city_lat_lon,
)

logger = get_logger("agglomerative_clustering")


def calculate_features(
    agglomeration_df: gpd.GeoDataFrame, city_supplement_df: pd.DataFrame
) -> pd.DataFrame:
    def cluster_feature_stats(feature_name, values, weights):
        return pd.Series(
            {
                f"{feature_name}_weighted_mean": weighted_mean(values, weights),
                f"{feature_name}_weighted_med": weighted_median(values, weights),
                f"{feature_name}_weighted_std": weigted_std(values, weights),
            }
        )

    def weighted_mean(values, weights):
        return np.average(values, weights=weights)

    def weighted_median(values, weights):
        # Median, so we sort the values ascending
        df = pd.DataFrame({"values": values, "weights": weights})
        df.sort_values("values", inplace=True)

        weights = df["weights"]
        cumsum = weights.cumsum()
        cutoff = weights.sum() / 2.0
        return df["values"][cumsum >= cutoff].iloc[0]

    def weigted_std(values, weights):
        weigted_mean = weighted_mean(values, weights)
        variance = np.average((values - weigted_mean) ** 2, weights=weights)
        return np.sqrt(variance)

    def compute_cluster_feature(city_gdf, feature_class):
        feature_name = feature_class.__name__.lower()

        def process_cluster_feature_stats(cluster_gdf):
            try:
                feature_values = feature_class(cluster_gdf, verbose=False).series
            except TypeError:
                feature_values = feature_class(cluster_gdf).series
            except AttributeError:
                if feature_class is momepy.CentroidCorners:
                    feature_values = feature_class(cluster_gdf, verbose=False).mean
            return cluster_feature_stats(
                feature_name, feature_values, cluster_gdf.area_share_of_cluster
            )

        return city_gdf.groupby("cluster").apply(process_cluster_feature_stats)

    def compute_cluster_metrics(city_gdf, city_center_point):
        cluster_metrics = pd.DataFrame()

        total_area_per_cluster = city_gdf.groupby("cluster")["geometry"].agg(
            lambda geoms: geoms.area.sum()
        )

        city_gdf["area_share_of_cluster"] = city_gdf.apply(
            lambda row: row.geometry.area / total_area_per_cluster[row["cluster"]],
            axis=1,
        )

        circular_compactness = compute_cluster_feature(
            city_gdf, momepy.CircularCompactness
        )
        orientation = compute_cluster_feature(city_gdf, momepy.Orientation)
        rectangularity = compute_cluster_feature(city_gdf, momepy.Rectangularity)
        elongation = compute_cluster_feature(city_gdf, momepy.Elongation)
        longest_axis_length = compute_cluster_feature(
            city_gdf, momepy.LongestAxisLength
        )
        centroid_corners = compute_cluster_feature(city_gdf, momepy.CentroidCorners)
        convexity = compute_cluster_feature(city_gdf, momepy.Convexity)
        equivalent_rectangular_index = compute_cluster_feature(
            city_gdf, momepy.EquivalentRectangularIndex
        )
        fractal_dimension = compute_cluster_feature(city_gdf, momepy.FractalDimension)
        squareness = compute_cluster_feature(city_gdf, momepy.Squareness)

        number_of_vertices = city_gdf.groupby("cluster").apply(
            lambda cluster_gdf: cluster_feature_stats(
                "no_vertices",
                [len(geom.exterior.coords) for geom in cluster_gdf.geometry],
                cluster_gdf.area_share_of_cluster,
            )
        )

        complexity = city_gdf.groupby("cluster").apply(
            lambda cluster_gdf: cluster_feature_stats(
                "complexity",
                [geom.length / geom.area for geom in cluster_gdf.geometry],
                cluster_gdf.area_share_of_cluster,
            )
        )

        cluster_metrics = pd.concat(
            [
                circular_compactness,
                orientation,
                rectangularity,
                elongation,
                longest_axis_length,
                centroid_corners,
                convexity,
                equivalent_rectangular_index,
                fractal_dimension,
                squareness,
                number_of_vertices,
                complexity,
            ],
            axis=1,
        )

        # Total city area (sum of cluster area)
        total_city_area = city_gdf.area.sum()

        # Total city perimeters (sum of cluster perimeters)
        total_city_perimeter = city_gdf.geometry.length.sum()

        cluster_metrics["area_share_of_perimeter"] = city_gdf.groupby("cluster")[
            "geometry"
        ].agg(lambda geoms: geoms.area.sum() / total_city_area)

        cluster_metrics["perimeter_share_of_cluster"] = city_gdf.groupby("cluster")[
            "geometry"
        ].agg(lambda geoms: geoms.length.sum() / total_city_perimeter)

        cluster_metrics["weighted_distance_from_centerpoint"] = city_gdf.groupby(
            "cluster"
        ).apply(
            lambda cluster_gdf: np.mean(
                cluster_gdf.apply(
                    lambda row: city_center_point.distance(
                        row["geometry"].representative_point()
                    ).iloc[0]
                    * row.area_share_of_cluster,
                    axis=1,
                )
            )
        )

        return cluster_metrics

    cluster_data_per_city = []
    for city in tqdm(
        agglomeration_df["city"].unique(), total=len(agglomeration_df["city"].unique())
    ):
        city_gdf = agglomeration_df.loc[agglomeration_df.city == city].copy()

        city_suppliment_data = city_supplement_df.loc[
            city_supplement_df.place == city
        ].copy()
        lat, lon = city_suppliment_data[["latitude", "longitude"]].values[0]
        city_utm_zone = get_utm_zone_from_city_lat_lon(lat, lon)

        city_gdf = city_gdf.to_crs(city_utm_zone)
        city_suppliment_data = city_suppliment_data.to_crs(city_utm_zone)

        city_center_point = city_suppliment_data.geometry

        cluster_metrics = compute_cluster_metrics(city_gdf, city_center_point)

        # Add city column to the front
        cluster_metrics.insert(0, "city", city)

        cluster_metrics.reset_index(inplace=True)
        cluster_data_per_city.append(cluster_metrics)

    cluster_features = pd.concat(cluster_data_per_city, axis=0, ignore_index=True)

    logger.info("Finished calculating primary cluster features for all 44 cities")

    logger.info("Min Max Scalling features that are not between 0 and 1")
    for metric in [
        "orientation",
        "no_vertices",
        "distance_from_centerpoint",
        "longestaxislength",
        "centroidcorners",
        "squareness",
        "equivalentrectangularindex",
        "fractaldimension",
        "complexity",
    ]:
        for col in cluster_features.columns:
            if metric in col:
                cluster_features[col] = (
                    cluster_features[col] - cluster_features[col].min()
                ) / (cluster_features[col].max() - cluster_features[col].min())

    logger.info("Exporting cluster_features")
    cluster_features.to_csv("data/agglomerative_clustering_features.csv")
    return cluster_features


if __name__ == "__main__":
    logger.info("Start script")

    logger.info("Loading agglomeration data")
    agglomeration = load_geo_data("data/agglomeration.csv")

    logger.info("Loading data from euro_")
    euro_case_studies = pd.read_csv("data/euro_case_studies.csv")
    remove_unnamed(euro_case_studies)

    euro_case_studies["place"] = (
        euro_case_studies["urban_agglomeration"].str.replace(" ", "_").str.lower()
        + "-"
        + euro_case_studies["country_or_area"].str.replace(" ", "_").str.lower()
    )

    logger.info("Start computing features")
    cluster_features = calculate_features(agglomeration, euro_case_studies)

    logger.info("Pivoting table to get cluster features by city")
    cluster_features = cluster_features.pivot(
        index="city",
        columns="cluster",
        values=[
            col for col in cluster_features.columns if col not in ["city", "cluster"]
        ],
    )
    cluster_features.columns = [
        f"{col[0]}_{col[1]}" for col in cluster_features.columns
    ]

    logger.info("Zero imputing NaN values")
    cluster_features = cluster_features.fillna(0)

    logger.info("Loading Agglomerative Cluster Model")
    agg_clustering = AgglomerativeClustering()

    X = cluster_features.reset_index(drop=True)

    logger.info("Model fitting")
    model = agg_clustering.fit(X)

    logger.info("Saving cluster_features with Agglomerative Clustering label")
    cluster_features["label"] = agg_clustering.labels_
    cluster_features.to_csv("agglomerative_clustering_features_with_label.csv")

    logger.info("Generating dendogram from clustering")
    Z = linkage(X, method="ward")
    plt.figure(figsize=(15, 12))
    dendrogram(
        Z,
        orientation="right",
        labels=cluster_features.index.tolist(),
        show_leaf_counts=False,
    )

    plt.savefig("data/dendogram_agglomerative_clustering.png")
    logger.info("Agglomerative clustering finished!")
