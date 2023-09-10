import os
from typing import Generator, Tuple, List
import multiprocessing as mp
from glob import glob

# Ignore User Warnings
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)


import momepy
import geopandas as gpd

from utils import (
    get_utm_zone,
    load_geo_data,
    get_logger,
)

logger = get_logger("get_tessellation")
COMMON_LOCK = mp.Lock()


def find_remaining_unprocessed_places(
    gdf: gpd.GeoDataFrame, output_dir: str, report_file_suffix: str
) -> List[gpd.GeoSeries]:
    processed_files = glob(f"{output_dir}/*{report_file_suffix}")

    processed_places = set()
    for f in processed_files:
        city = os.path.basename(f).replace(report_file_suffix, "")
        processed_places.add(city)

    return gdf.loc[~gdf["place"].isin(processed_places)]


def split_buildings_by_place(
    buildings: gpd.GeoDataFrame,
) -> Generator[Tuple[str, gpd.GeoDataFrame], None, None]:
    # Group by the 'urban_agglomeration' and 'country_or_area' column
    # called "place"
    grouped = buildings.groupby("place")

    # Iterate through the groups and yield each group as a GeoDataFrame
    for place, group in grouped:
        city_buildings = gpd.GeoDataFrame(group, crs="EPSG:4326")
        city_buildings.reset_index(drop=True, inplace=True)
        yield (place, city_buildings)


def process_city_tessellation(city_name: str, city_buildings: gpd.GeoDataFrame):
    logger.info(f"{city_name} start")

    point = gpd.points_from_xy(
        city_buildings.city_lon, city_buildings.city_lat, crs="EPSG:4326"
    )[0]

    # Get UTM for this point
    city_utm_zone = get_utm_zone(point)
    city_buildings = city_buildings.to_crs(crs=city_utm_zone)

    # Create Tessellation
    limit = momepy.buffered_limit(city_buildings, 100)

    city_tessellation = momepy.Tessellation(
        city_buildings, "buildingsID", limit, verbose=False, segment=5
    )
    city_tessellation = city_tessellation.tessellation
    city_tessellation = city_tessellation.to_crs(crs="EPSG:4326")

    city_tessellation.to_csv(
        f"data/tessellations/{city_name}_tessellation.csv",
        index=False,
    )

    logger.info(f"{city_name} tessellation processed!")


def run_process_tessellation(row: gpd.GeoSeries):
    try:
        process_city_tessellation(city_name=row[0], city_buildings=row[1])
    except Exception as e:
        logger.exception(e)
        pass


if __name__ == "__main__":
    gdf = load_geo_data("data/all_buildings.csv")

    logger.info("Start processing tessellations")

    MULTIPROCESSING = True

    logger.info("Start big cities clip and tessellation")

    city_circles = load_geo_data("data/city_circles.csv")

    city_circles["place"] = [
        f"{place[0].replace(' ', '_').lower()}-{place[1].replace(' ', '_').lower()}"
        for place in city_circles[["urban_agglomeration", "country_or_area"]].values
    ]

    to_process = [
        "cartagena-colombia",
        "london-united_kingdom",
        "kinki_m.m.a._(osaka)-japan",
        "tokyo-japan",
        "los_angeles-long_beach-santa_ana-united_states_of_america",
        "manila-philippines",
        "dhaka-bangladesh",
        "paris-france",
        "jakarta-indonesia",
        "são_paulo-brazil",
        "new_york-newark-united_states_of_america",
    ]

    rows_per_place = []
    for place in to_process:
        place_dataframe = gdf[gdf["place"] == place]
        if not place_dataframe.empty:
            logger.info(f"Added {place} to rows to process")
            rows_per_place.append((place, place_dataframe))

    logger.info(f"Places left {len(rows_per_place)}")

    from time import time

    start = time()

    if MULTIPROCESSING:
        pool = mp.Pool(mp.cpu_count())
        pool.map(
            run_process_tessellation,
            iterable=rows_per_place,
        )
    else:
        for row in rows_per_place:
            run_process_tessellation(row)

    logger.critical(f"Time needed - {time() - start}")
