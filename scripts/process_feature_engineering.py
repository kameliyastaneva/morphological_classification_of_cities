import os
from glob import glob
import multiprocessing as mp

# Ignore User Warnings
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)

import geopandas as gpd

from feature_engineering import process_city_features
from utils import (
    load_geo_data,
    get_logger,
    remove_unnamed,
)

logger = get_logger("process_feature_engineering", log_to_file=True)

COMMON_LOCK = mp.Lock()


def run_process_features(
    place: str,
    buildings: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
):
    COMMON_LOCK.acquire()
    logger.info(f"Currently at {place}")
    logger.info(f"Loading tessellation data for {place}")
    COMMON_LOCK.release()

    base_scratch_path = "/lustre/scratch/scratch/zcftkst"
    try:
        tessellation_path = (
            f"{base_scratch_path}/tessellations/{place}_tessellation.csv"
        )
        if not os.path.exists(tessellation_path):
            raise FileExistsError(f"Tessellation file does not exist for {place}")

        tessellation = load_geo_data(tessellation_path)
        remove_unnamed(tessellation)
        process_city_features(place, buildings, streets, tessellation)
    except Exception as e:
        logger.exception(e)
        raise


if __name__ == "__main__":
    base_scratch_path = "/lustre/scratch/scratch/zcftkst"

    case_studies = load_geo_data("data/euro_case_studies.csv")["place"].values

    processed_places = [
        os.path.basename(f).replace("_features.csv", "")
        for f in glob(f"{base_scratch_path}/features/*_features.csv")
    ]

    unprocessed_places = set(case_studies).difference(set(processed_places))

    from time import time

    start = time()

    logger.info("Start processing tessellations")

    logger.info("Loading building data")
    all_buildings = load_geo_data("data/euro_buildings.csv")
    remove_unnamed(all_buildings)

    logger.info("Loading street data")
    all_streets = load_geo_data("data/euro_streets.csv")
    remove_unnamed(all_streets)

    places = []
    for place in unprocessed_places:
        buildings = all_buildings[all_buildings.place == place].copy()
        streets = all_streets[all_streets.place == place].copy()
        places.append((place, buildings, streets))

    pool = mp.Pool(mp.cpu_count())
    pool.starmap(run_process_features, iterable=places)

    logger.critical(f"Time needed - {time() - start}")
