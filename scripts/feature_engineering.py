import collections


import pandas as pd
import numpy as np
import geopandas as gpd
import libpysal
import momepy

from constants import WORLD_CRS
from utils import (
    get_utm_zone_from_city_lat_lon,
    load_geo_data,
    get_logger,
)

logger = get_logger("feature engineering main", log_to_file=True)


FEATURES_COLUMNS = [
    "tessellation_orientation",
    "tessellation_longest_axis_length",
    "tessellation_area",
    "tessellation_circular_compactness",
    "tessellation_equivalent_rectangular_index",
    "tessellation_neighbors",
    "tessellation_covered_area",
    "tessellation_area_ratio",
    "tessellation_street_alignment",
    "buildings_area",
    "buildings_perimeter",
    "buildings_circular_compactness",
    "buildings_corners",
    "buildings_squareness",
    "buildings_equivalent_rectangular_index",
    "buildings_elongation",
    "buildings_centroid_corners_mean",
    "buildings_centroid_corners_std",
    "buildings_orientation",
    "buildings_cell_alignment",
    "buildings_shared_walls_ratio",
    "buildings_courtyard_area",
    "buildings_alignment",
    "buildings_neighbor_distance",
    "buildings_courtyards",
    "buildings_perimeter_wall",
    "buildings_mean_interbuildings_distance",
    "buildings_building_adjacency",
    "buildings_street_alignment",
    "streets_length",
    "streets_width",
    "streets_openness",
    "streets_width_deviation",
    "streets_linearity",
    "streets_reached_sum",
    "streets_count",
    "streets_reached_count_q1",
    "streets_reached_sum_q1",
    "degree",
    # "meshedness",
    "local_closeness",
    "graph_cds_length",
    "graph_nodes_clustering",
    "graph_mean_node_distance",
    # "closeness",
]


class CustomMomepyCount:
    def __init__(self, left, right, left_id, right_id, weighted=False):
        self.left = left
        self.right = right
        self.left_id = left[left_id]
        self.right_id = right[right_id]
        self.weighted = weighted

        count = collections.Counter(right[right_id])
        df = pd.DataFrame.from_dict(count, orient="index", columns=["mm_count"])
        joined = left[[left_id, left.geometry.name]].join(df["mm_count"], on=left_id)
        joined.loc[joined["mm_count"].isna(), "mm_count"] = 0

        if weighted:
            if left.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]:
                joined["mm_count"] = joined["mm_count"] / left.geometry.area
            elif left.geometry.iloc[0].geom_type in ["LineString", "MultiLineString"]:
                joined["mm_count"] = joined["mm_count"] / left.geometry.length
            else:
                raise TypeError("Geometry type does not support weighting.")

        self.series = joined["mm_count"]


def load_data():
    buildings = load_geo_data("sample_data/buildings.csv")
    streets = load_geo_data("sample_data/streets.csv")
    tessellation = load_geo_data("sample_data/tessellation.csv")
    return buildings, streets, tessellation

    # places = buildings["place"].unique().tolist()


def join_buildings_and_streets(
    buildings: gpd.GeoDataFrame, streets: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # Taking the lon, lat of the city from the first building
    lon, lat = buildings.iloc[0][["city_lon", "city_lat"]].to_list()
    city_utm_zone = get_utm_zone_from_city_lat_lon(lat=lat, lon=lon)

    buildings = buildings.to_crs(city_utm_zone)
    streets = streets.to_crs(city_utm_zone)

    buildings_streets = gpd.sjoin_nearest(
        buildings, streets, max_distance=10000, how="left"
    )

    buildings_streets = buildings_streets.to_crs(crs=WORLD_CRS)

    buildings_streets = buildings_streets.drop_duplicates("buildingsID").drop(
        columns="index_right"
    )

    buildings_streets["city"] = buildings_streets["city_left"]
    buildings_streets = buildings_streets.drop(["city_left", "city_right"], axis=1)

    buildings_streets["place"] = buildings_streets["place_left"]
    buildings_streets = buildings_streets.drop(["place_left", "place_right"], axis=1)

    return buildings_streets


def link_buildings_to_tessellation(
    tessellation: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    return tessellation.merge(
        buildings[["buildingsID", "streetsID", "city"]],
        on="buildingsID",
        how="left",
    )


def process_city_features(
    unique_place_name_identification: str,
    buildings: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    tessellation: gpd.GeoDataFrame,
):
    # Remove tessellations outside of the circle

    logger.info(
        f"Removing tessellations outside of the circle: {unique_place_name_identification}"
    )
    tessellation = tessellation[
        tessellation["buildingsID"].isin(buildings["buildingsID"])
    ]

    buildings = buildings[buildings["buildingsID"].isin(tessellation["buildingsID"])]

    logger.info(f"Removed!: {unique_place_name_identification}")

    logger.info(f"Preprocessing: {unique_place_name_identification}")

    streets.reset_index(drop=True, inplace=True)
    streets["streetsID"] = range(len(streets))

    buildings = join_buildings_and_streets(buildings, streets)
    tessellation = link_buildings_to_tessellation(
        buildings=buildings, tessellation=tessellation
    )

    # Taking the lon, lat of the city from the first building
    lon, lat = buildings.iloc[0][["city_lon", "city_lat"]].to_list()
    city_utm_zone = get_utm_zone_from_city_lat_lon(lat=lat, lon=lon)

    # Reproject gdfs to City UTM Zone
    buildings = buildings.to_crs(crs=city_utm_zone)
    streets = streets.to_crs(crs=city_utm_zone)
    tessellation = tessellation.to_crs(crs=city_utm_zone)

    logger.info(
        f"Calculating primary metrics for city: {unique_place_name_identification}"
    )
    calculate_primary_metrics_for_city(buildings=buildings, tessellation=tessellation)

    logger.info(f"Calculating tessellation metrics: {unique_place_name_identification}")
    calculate_tessellation_metrics(buildings=buildings, tessellation=tessellation)

    logger.info(
        f"Calculating spatial weights matrix queen 1 tessellation: {unique_place_name_identification}"
    )
    # Spatial Weights Matrix Calculations
    queen_1 = calculate_spatial_weights_matrix_queen_1_tessellation(
        buildings=buildings, tessellation=tessellation
    )

    logger.info(
        f"Calculating spatial weights matrix queen 1 buildings: {unique_place_name_identification}"
    )
    buildings_queen_1 = calculate_spatial_weights_matrix_queen_1_buildings(
        buildings=buildings
    )

    logger.info(
        f"Calculating spatial weights matrix queen 3 tessellation: {unique_place_name_identification}"
    )
    queen_3 = calculate_spatial_weights_matrix_queen_3_tessellation(
        buildings=buildings,
        tessellation=tessellation,
        streets=streets,
        queen_1=queen_1,
        buildings_queen_1=buildings_queen_1,
    )

    logger.info(
        f"Calculating spatial weights matrix queen 1 streets: {unique_place_name_identification}"
    )
    nodes = calculate_spatial_weights_matrix_queen_1_streets(
        buildings=buildings, streets=streets, tessellation=tessellation
    )

    logger.info(f"Linking all data together: {unique_place_name_identification}")
    # Link all data together
    city_merged = tessellation.merge(
        buildings.drop(columns=["streetsID", "geometry"]), on="buildingsID"
    )
    city_merged = city_merged.merge(
        streets.drop(columns="geometry"), on="streetsID", how="left"
    )
    city_merged = city_merged.merge(
        nodes.drop(columns="geometry"), on="nodeID", how="left"
    )

    city_percentiles = []

    logger.info(
        f"Calculating percentiles from all columns: {unique_place_name_identification}"
    )
    for column in city_merged[FEATURES_COLUMNS]:
        perc = momepy.Percentiles(
            city_merged, column, queen_3, "buildingsID", verbose=False
        ).frame
        perc.columns = [f"{column}_" + str(x) for x in perc.columns]
        city_percentiles.append(perc)

    all_percentiles_joined = pd.concat(
        [city_merged["buildingsID"], *city_percentiles], axis=1
    )

    logger.info(f"Saving percentiles as features!: {unique_place_name_identification}")
    all_percentiles_joined.to_csv(
        f"/lustre/scratch/scratch/zcftkst/features/{unique_place_name_identification}_features.csv",
        index=False,
    )


def calculate_primary_metrics_for_city(buildings, tessellation):
    # Buildings
    buildings["buildings_area"] = buildings.area
    # Perimeter Buildings
    buildings["buildings_perimeter"] = momepy.Perimeter(buildings).series
    # Circular Compactness Buildings
    buildings["buildings_circular_compactness"] = momepy.CircularCompactness(
        buildings
    ).series
    # Corners Buildings
    buildings["buildings_corners"] = momepy.Corners(buildings, verbose=False).series
    # Squareness Buildings
    buildings["buildings_squareness"] = momepy.Squareness(
        buildings, verbose=False
    ).series
    # Equivalent Rectangular Index Buildings
    buildings[
        "buildings_equivalent_rectangular_index"
    ] = momepy.EquivalentRectangularIndex(buildings).series
    # Elongation Buildings
    buildings["buildings_elongation"] = momepy.Elongation(buildings).series
    # Centroid Corners - mean Buildings
    centroid_corners = momepy.CentroidCorners(buildings, verbose=False)
    buildings["buildings_centroid_corners_mean"] = centroid_corners.mean
    # Centroid Corners - std Buildigns
    buildings["buildings_centroid_corners_std"] = centroid_corners.std
    # Orientation Buildings
    buildings["buildings_orientation"] = momepy.Orientation(
        buildings, verbose=False
    ).series
    # Orientation Tessellation
    tessellation["tessellation_orientation"] = momepy.Orientation(
        tessellation, verbose=False
    ).series
    # Cell Alignment Buildings
    buildings["buildings_cell_alignment"] = momepy.CellAlignment(
        buildings,
        tessellation,
        "buildings_orientation",
        "tessellation_orientation",
        "buildingsID",
        "buildingsID",
    ).series


def calculate_tessellation_metrics(buildings, tessellation):
    # Longest Axis Length
    tessellation["tessellation_longest_axis_length"] = momepy.LongestAxisLength(
        tessellation
    ).series
    # Area
    tessellation["tessellation_area"] = tessellation.area
    # Circular Compactness
    tessellation["tessellation_circular_compactness"] = momepy.CircularCompactness(
        tessellation
    ).series
    # Equivalent Rectangular Index
    tessellation[
        "tessellation_equivalent_rectangular_index"
    ] = momepy.EquivalentRectangularIndex(tessellation).series

    # Shared Walls Ratio Buildings
    buildings["buildings_shared_walls_ratio"] = momepy.SharedWallsRatio(
        buildings
    ).series
    # Courtyard area Buildings
    buildings["buildings_courtyard_area"] = momepy.CourtyardArea(buildings).series


def calculate_spatial_weights_matrix_queen_1_tessellation(buildings, tessellation):
    # SWM queen_1 Tessellation
    queen_1 = libpysal.weights.contiguity.Queen.from_dataframe(
        tessellation, ids="buildingsID", silence_warnings=True
    )

    # Alignment Buildings
    buildings["buildings_alignment"] = momepy.Alignment(
        buildings, queen_1, "buildingsID", "buildings_orientation", verbose=False
    ).series
    # Neighbour Distance Buildings
    buildings["buildings_neighbor_distance"] = momepy.NeighborDistance(
        buildings, queen_1, "buildingsID", verbose=False
    ).series
    # Neighbours Tessellation
    tessellation["tessellation_neighbors"] = momepy.Neighbors(
        tessellation, queen_1, "buildingsID", weighted=True, verbose=False
    ).series
    # Covered Area Tessellation
    tessellation["tessellation_covered_area"] = momepy.CoveredArea(
        tessellation, queen_1, "buildingsID", verbose=False
    ).series

    return queen_1


def calculate_spatial_weights_matrix_queen_1_buildings(buildings):
    # SWM queen_1 Buildings
    buildings_queen_1 = libpysal.weights.contiguity.Queen.from_dataframe(
        buildings, silence_warnings=True
    )
    # Courtyards
    buildings["buildings_courtyards"] = momepy.Courtyards(
        buildings, spatial_weights=buildings_queen_1, verbose=False
    ).series
    # Perimeter Wall
    buildings["buildings_perimeter_wall"] = momepy.PerimeterWall(
        buildings, buildings_queen_1, verbose=False
    ).series

    return buildings_queen_1


def calculate_spatial_weights_matrix_queen_3_tessellation(
    buildings, tessellation, streets, queen_1, buildings_queen_1
):
    queen_3 = momepy.sw_high(k=3, weights=queen_1)

    logger.info("Q3 Tessellation MeanInterbuildingDistance")
    # Mean Interbuilding Distance Buildings
    buildings[
        "buildings_mean_interbuildings_distance"
    ] = momepy.MeanInterbuildingDistance(
        gdf=buildings, spatial_weights=queen_1, unique_id="buildingsID", verbose=True
    ).series

    logger.info("Q3 Tessellation BuildingAdjacency")
    # Building Adjacency Buildings
    buildings["buildings_building_adjacency"] = momepy.BuildingAdjacency(
        buildings, queen_3, "buildingsID", buildings_queen_1, verbose=True
    ).series

    logger.info("Q3 Tessellation AreaRatio")
    # Area Ratio
    tessellation["tessellation_area_ratio"] = momepy.AreaRatio(
        tessellation, buildings, "tessellation_area", "buildings_area", "buildingsID"
    ).series

    # Length Streets
    streets["streets_length"] = streets.length

    logger.info("Q3 Tessellation StreetAlignment Tessellation")
    # Street Alignment Tessellation
    tessellation["tessellation_street_alignment"] = momepy.StreetAlignment(
        tessellation, streets, "tessellation_orientation", "streetsID"
    ).series

    logger.info("Q3 Tessellation StreetAlignment Buildings")
    # Street Alignment Buildings
    buildings["buildings_street_alignment"] = momepy.StreetAlignment(
        buildings, streets, "buildings_orientation", "streetsID"
    ).series

    logger.info("Q3 Tessellation StreetProfile")
    # Street Profile:
    profile = momepy.StreetProfile(streets, buildings)
    # w
    streets["streets_width"] = profile.w
    # o
    streets["streets_openness"] = profile.o
    # wd
    streets["streets_width_deviation"] = profile.wd
    # Linearity
    streets["streets_linearity"] = momepy.Linearity(streets).series

    # Reached
    logger.info("Q3 Tessellation StreetsReached")
    streets["streets_reached_sum"] = momepy.Reached(
        streets,
        tessellation,
        "streetsID",
        "streetsID",
        mode="sum",
        values="tessellation_area",
        verbose=True,
    ).series

    # Count
    logger.info("Q3 Tessellation Count")
    streets["streets_count"] = CustomMomepyCount(
        streets, buildings, "streetsID", "streetsID", weighted=True
    ).series

    return queen_3


def calculate_spatial_weights_matrix_queen_1_streets(buildings, streets, tessellation):
    streets_q1 = libpysal.weights.contiguity.Queen.from_dataframe(
        streets, silence_warnings=True, idVariable="streetsID"
    )
    # Reached

    logger.info("Q1 Streets Reached count")
    streets["streets_reached_count_q1"] = momepy.Reached(
        streets,
        tessellation,
        "streetsID",
        "streetsID",
        spatial_weights=streets_q1,
        mode="count",
        verbose=True,
    ).series

    logger.info("Q1 Streets Reached sum")
    # Reached
    streets["streets_reached_sum_q1"] = momepy.Reached(
        streets,
        tessellation,
        "streetsID",
        "streetsID",
        spatial_weights=streets_q1,
        mode="sum",
        verbose=True,
    ).series

    logger.info("Q1 Streets Graph")
    # GRAPH and Connectivity
    graph = momepy.gdf_to_nx(streets)
    graph = momepy.node_degree(graph)

    logger.info("Q1 Streets Subgraph")
    # Subgraph
    graph = momepy.subgraph(
        graph,
        radius=5,
        meshedness=True,
        cds_length=False,
        mode="sum",
        degree="degree",
        length="mm_len",
        mean_node_degree=False,
        cyclomatic=False,
        edge_node_ratio=False,
        gamma=False,
        local_closeness=True,
        closeness_weight="mm_len",
        verbose=True,
    )

    logger.info("Q1 Streets Graph length")
    graph = momepy.cds_length(graph, radius=3, name="graph_cds_length", verbose=True)

    logger.info("Q1 Streets Graph Clustering")
    graph = momepy.clustering(graph, name="graph_nodes_clustering")

    logger.info("Q1 Streets Graph Mean Node Distance")
    graph = momepy.mean_node_dist(graph, name="graph_mean_node_distance", verbose=True)

    logger.info("Q1 Streets Graph Closeness Centrality")

    # graph = momepy.closeness_centrality(
    #     graph, radius=400, distance="graph_mm_len", verbose=True
    # )

    # logger.info("Q1 Streets Graph Meshedness")
    # graph = momepy.meshedness(graph, radius=400, distance="graph_mm_len", verbose=True)

    nodes, edges, _ = momepy.nx_to_gdf(graph, spatial_weights=True)

    logger.info("Q1 Streets Graph Get node id")
    buildings["nodeID"] = momepy.get_node_id(
        buildings, nodes, edges, "nodeID", "streetsID", verbose=True
    )
    return nodes
