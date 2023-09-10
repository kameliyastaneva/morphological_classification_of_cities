import os
from typing import List
from glob import glob
import multiprocessing as mp
import warnings

import pandas as pd
import geopandas as gpd
import osmnx

osmnx.settings.memory=2000000000
osmnx.settings.log_console=True
osmnx.settings.log_file=True

from utils import get_logger, convert_df_to_gdf

warnings.simplefilter(action="ignore", category=UserWarning)

logger = get_logger("process_buildings_and_street_network")
COMMON_LOCK = mp.Lock()


def load_data(path: str = None) -> gpd.GeoDataFrame:
    return pd.read_csv(path or "data/distance_from_centrepoint.csv")


def drop_street_features(street_df: gpd.GeoDataFrame):
    street_features_to_drop = [
        "u",
        "v",
        "key",
        "osmid",
        "oneway",
        "name",
        "length",
        "highway",
        "oneway",
        "maxspeed",
        "reversed",
        "lanes",
        "ref",
        "junction",
        "access",
        "bridge",
        "width",
        "tunnel",
    ]

    street_df.drop(
        street_df.columns.intersection(street_features_to_drop),
        axis=1,
        inplace=True,
    )


def process_buildings_and_streets_for_city(city_row: pd.Series):
    city_name = city_row["urban_agglomeration"]
    area_name = city_row["country_or_area"]

    place = (
        f"{city_name.replace(' ', '_').lower()}-{area_name.replace(' ', '_').lower()}"
    )

    COMMON_LOCK.acquire()
    logger.info(f"{place} start")
    COMMON_LOCK.release()

    point = (city_row["latitude"], city_row["longitude"])

    COMMON_LOCK.acquire()
    try:
        buildings = osmnx.features_from_point(
            point,
            tags={"building": True},
            dist=city_row["distance_from_centrepoint"] * 0.95,
        )
    except Exception as e:
        COMMON_LOCK.release()
        raise Exception(
            f"Something went wrong during buildings downdload for {place}"
        ) from e
    COMMON_LOCK.release()

    buildings = buildings[buildings.geom_type == "Polygon"].reset_index(drop=True)
    buildings = buildings[["geometry"]]
    buildings["city"] = city_row["urban_agglomeration"]
    buildings["city_lat"] = city_row["latitude"]
    buildings["city_lon"] = city_row["longitude"]

    buildings.to_csv(f"data/cities/{place}_buildings.csv", index=False)

    COMMON_LOCK.acquire()
    logger.info(f"{place} buildings - done")
    COMMON_LOCK.release()

    # Get city boundary
    polygon = buildings.unary_union.envelope

    COMMON_LOCK.acquire()
    try:
        osm_graph = osmnx.graph_from_polygon(polygon, network_type="drive")
        osm_graph = osmnx.projection.project_graph(
            osm_graph, to_crs=city_row["utm_zone"]
        )

        streets = osmnx.graph_to_gdfs(
            osm_graph,
            nodes=False,
            edges=True,
            node_geometry=False,
            fill_edge_geometry=True,
        )
    except Exception as e:
        COMMON_LOCK.release()
        raise Exception(
            f"Something went wrong during streets download for {place}"
        ) from e
    COMMON_LOCK.release()

    streets["city"] = city_row["urban_agglomeration"]
    streets = streets.to_crs("EPSG:4326")
    drop_street_features(streets)
    streets.to_csv(f"data/cities/{place}_streets.csv", index=False)

    COMMON_LOCK.acquire()
    logger.info(f"{place} streets - done")
    COMMON_LOCK.release()


def run_process_buildings_and_streets_for_city(row):
    try:
        process_buildings_and_streets_for_city(row)
    except Exception as e:
        logger.exception(e)
        pass


def combine_datasets(output_path: str = None):
    logger.info("Start combining all buildings and streets datasets into one.")

    for dataset_type in ("buildings", "streets"):
        combined_dataset = []
        to_remove = []
        for csv_file in sorted(glob(f"data/cities/*{dataset_type}.csv")):
            # To remove later
            to_remove.append(csv_file)

            combined_dataset.append(load_data(csv_file))

        combined_dataset = pd.concat(combined_dataset)
        combined_dataset = convert_df_to_gdf(combined_dataset)

        combined_dataset_output_path = (
            output_path or "data"
        ) + f"/all_{dataset_type}.csv"

        combined_dataset.to_csv(combined_dataset_output_path, index=False)

        logger.info(f"{dataset_type} data written to {combined_dataset_output_path}")

        if (
            os.path.exists(combined_dataset_output_path)
            and os.path.getsize(combined_dataset_output_path) > 0
        ):
            for f in to_remove:
                os.remove(f)
        logger.info(f"Removed all temporary {dataset_type} datasets in data/cities/*")


def find_remaining_cities(
    gdf: gpd.GeoDataFrame, output_dir: str
) -> List[gpd.GeoSeries]:
    downloaded_streets = glob(f"{output_dir}/*_streets.csv")
    streets = set()
    for city_street in downloaded_streets:
        city_street = os.path.basename(city_street)
        city_street = city_street.replace("_streets.csv", "")
        streets.add(city_street)

    downloaded_buildings = glob(f"{output_dir}/*_buildings.csv")
    buildings = set()
    for city_building in downloaded_buildings:
        city_building = os.path.basename(city_building)
        city_building = city_building.replace("_buildings.csv", "")
        buildings.add(city_building)

    processed = buildings.intersection(streets)
    rows = []
    for _, row in gdf.iterrows():
        city_name = row["urban_agglomeration"]
        area_name = row["country_or_area"]

        place = f"{city_name.replace(' ', '_').lower()}-{area_name.replace(' ', '_').lower()}"
        if not place in processed:
            rows.append(row)

    return rows


if __name__ == "__main__":
    gdf = convert_df_to_gdf(load_data())

    logger.info("Start processing cities")

    no_rows = None
    multiprocessing = False
    workers = 15

    rows = find_remaining_cities(gdf, "data/cities")

    logger.info(f"Number of rows: {len(rows)}")

    from time import time

    start = time()

    if multiprocessing:
        pool = mp.Pool(workers)
        pool.map(
            run_process_buildings_and_streets_for_city,
            iterable=rows,
        )
    else:
        for row in rows:
            run_process_buildings_and_streets_for_city(row)

    logger.critical(f"Time needed - {time() - start}")

    #combine_datasets(output_path="data")