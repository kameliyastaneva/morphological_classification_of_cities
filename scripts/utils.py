import os
import logging
from glob import glob
from typing import List

import pandas as pd
import geopandas as gpd
from shapely import wkt

from constants import WORLD_CRS


def convert_df_to_gdf(df: pd.DataFrame, csr: str = "EPSG:4326") -> gpd.GeoDataFrame:
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry=df["geometry"], crs=csr)


def get_logger(logger_name, log_to_file=False):
    logger = logging.getLogger(
        logger_name,
    )

    # Set the level of the logger. This can be DEBUG, INFO, WARNING, ERROR, or CRITICAL
    logger.setLevel(logging.DEBUG)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter(
            "\033[92m%(asctime)s\033[0m - \033[94m%(message)s\033[0m - \033[92m%(levelname)s\033[0m - (%(funcName)s)"
        )
    )
    logger.addHandler(ch)

    if log_to_file:
        # Create a file handler for outputting log messages to a file
        fh = logging.FileHandler(f"logs/{logger_name}.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(message)s - %(levelname)s - (%(funcName)s)"
            )
        )
        logger.addHandler(fh)
    return logger


def get_utm_zone(point) -> int:
    """
    Helper function to get the UTM Zone EPSG of the input point.
    Parameters:
        point: shapely.geometry.Point
    Returns:
        int: EPSG of the UTM zone of the input point
    """
    prefix = 32600
    if point.y < 0:
        prefix = 32700

    zone = int(((point.x + 180) / 6) + 1)
    return prefix + zone


def get_utm_zone_from_city_lat_lon(lat: float, lon: float) -> str:
    point = gpd.points_from_xy([lon], [lat], crs=WORLD_CRS)[0]
    return get_utm_zone(point)


def load_geo_data(path: str, crs=WORLD_CRS) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry=df["geometry"], crs=crs)


def format_city_country_name(city_name: str, country_area: str):
    return f"{city_name}__{country_area}"


def find_remaining_unprocessed_places(
    gdf: gpd.GeoDataFrame, output_dir: str, report_file_suffix: str
) -> List[gpd.GeoSeries]:
    processed_files = glob(f"{output_dir}/{report_file_suffix}")

    processed_places = set()
    for f in processed_files:
        city = os.path.basename(f).replace(report_file_suffix, "")
        processed_places.add(city)

    rows = []
    for _, row in gdf.iterrows():
        place = format_city_country_name(
            row["urban_agglomeration"], row["country_or_area"]
        )
        if place not in processed_places:
            rows.append(row)
    return rows


def remove_unnamed(df: pd.DataFrame):
    for col in df.columns:
        if "Unnamed: " in col:
            df.drop(col, axis=1, inplace=True)
