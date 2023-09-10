import pandas as pd
import geopandas as gpd
import momepy
import numpy as np
from shapely import wkt

WORLD_CRS = "EPSG:4326"

# Elongation, Number of Vertices and Complexity


def load_geo_data(path: str, crs=WORLD_CRS) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry=df["geometry"], crs=crs)


agglomeration_2 = load_geo_data("clustering_data.csv")
gdf = load_geo_data("city_supplement_data.csv")


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
        lambda row: row.geometry.area / total_area_per_cluster[row["cluster"]], axis=1
    )

    circular_compactness = compute_cluster_feature(city_gdf, momepy.CircularCompactness)
    orientation = compute_cluster_feature(city_gdf, momepy.Orientation)
    rectangularity = compute_cluster_feature(city_gdf, momepy.Rectangularity)
    elongation = compute_cluster_feature(city_gdf, momepy.Elongation)
    longest_axis_length = compute_cluster_feature(city_gdf, momepy.LongestAxisLength)
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
for city in agglomeration_2["city"].unique()[:1]:
    city_gdf = agglomeration_2.loc[agglomeration_2.city == city].copy()

    city_suppliment_data = gdf.loc[gdf.urban_agglomeration == city].copy()
    city_utm_zone = city_suppliment_data.iloc[0]["utm_zone"]

    city_gdf = city_gdf.to_crs(city_utm_zone)
    city_suppliment_data = city_suppliment_data.to_crs(city_utm_zone)

    city_center_point = city_suppliment_data.geometry

    cluster_metrics = compute_cluster_metrics(city_gdf, city_center_point)

    # Add city column to the front
    cluster_metrics.insert(0, "city", city)

    cluster_metrics.reset_index(inplace=True)
    cluster_data_per_city.append(cluster_metrics)


cluster_features = pd.concat(cluster_data_per_city, axis=0, ignore_index=True)

for metric in [
    "orientation",
    "no_vertices",
    "distance_from_centerpoint",
    "longestaxislength",
    "centroidcorners",
    "squareness",
]:
    for col in cluster_features.columns:
        if metric in col:
            cluster_features[col] = (
                cluster_features[col] - cluster_features[col].min()
            ) / (cluster_features[col].max() - cluster_features[col].min())

from pprint import pprint

breakpoint()
pass
