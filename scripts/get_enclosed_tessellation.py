import os
from typing import Tuple, List
from glob import glob
import multiprocessing as mp

# Ignore User Warnings
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)

import momepy
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely import wkt

from utils import get_utm_zone, load_geo_data, get_logger, remove_unnamed
import pickle


logger = get_logger("get_enclosed_tessellation")

CPU_CORES = True


class CustomTessellation(momepy.Tessellation):
    # def worker(
    #     self,
    #     i,
    #     enclosures,
    #     buildings,
    #     inp,
    #     res,
    #     threshold,
    #     unique_id,
    #     kwargs,
    # ):
    #     return self._tess(
    #         i,
    #         enclosures,
    #         buildings,
    #         inp,
    #         res,
    #         threshold=threshold,
    #         unique_id=unique_id,
    #         **kwargs,
    #     )

    def _morphological_tessellation(
        self, gdf, unique_id, limit, shrink, segment, verbose, check=True
    ):
        try:
            return super()._morphological_tessellation(
                gdf, unique_id, limit, shrink, segment, verbose, check
            )
        except Exception as e:
            with open("gdf.pickle", "wb") as file:
                pickle.dump(gdf, file)

            with open("limit.pickle", "wb") as file:
                pickle.dump(limit, file)

            with open("extra_details.txt", "w") as file:
                file.write(f"limit: {limit}\n")
                file.write(f"shrink: {shrink}\n")
                file.write(f"segment: {segment}\n")
                file.write(f"check: {segment}\n")

    def _dense_point_array(self, geoms, distance, index):
        """
        geoms : array of shapely lines
        """
        # interpolate lines to represent them as points for Voronoi
        points = []
        ids = []

        if shapely.get_type_id(geoms[0]) not in [1, 2, 5]:
            lines = shapely.boundary(geoms)
        else:
            lines = geoms
        lengths = shapely.length(lines)
        for ix, line, length in zip(index, lines, lengths):
            if length > distance:  # some polygons might have collapsed
                pts = shapely.line_interpolate_point(
                    line,
                    np.linspace(0.1, length - 0.1, num=int((length - 0.1) // distance)),
                )  # .1 offset to keep a gap between two segments
                points.append(shapely.get_coordinates(pts))
                ids += [ix] * len(pts)

        if not points:
            with open("gdf.pickle", "wb") as file:
                pickle.dump(geoms, file)

            with open("extra_details.txt", "w") as file:
                file.write(f"segment: {distance}\n")

        points = np.vstack(points)

        return points, ids

    # def _enclosed_tessellation(
    #     self,
    #     buildings,
    #     enclosures,
    #     unique_id,
    #     threshold=0.05,
    #     *args,
    #     **kwargs,
    # ):
    #     enclosures = enclosures.reset_index(drop=True)
    #     enclosures["position"] = range(len(enclosures))

    #     # determine which polygons should be split
    #     inp, res = buildings.sindex.query_bulk(
    #         enclosures.geometry, predicate="intersects"
    #     )
    #     unique, counts = np.unique(inp, return_counts=True)
    #     splits = unique[counts > 1]
    #     single = unique[counts == 1]

    #     if kwargs.get("use_dask"):
    #         if CPU_CORES is True:
    #             no_workers = mp.cpu_count() - 1
    #         else:
    #             no_workers = CPU_CORES

    #         pool = mp.Pool(no_workers)
    #         new = pool.starmap(
    #             self.worker,
    #             iterable=[
    #                 (
    #                     i,
    #                     enclosures,
    #                     buildings,
    #                     inp,
    #                     res,
    #                     threshold,
    #                     unique_id,
    #                     kwargs,
    #                 )
    #                 for i in splits
    #             ],
    #         )
    #     else:
    #         new = [
    #             self._tess(
    #                 i,
    #                 enclosures,
    #                 buildings,
    #                 inp,
    #                 res,
    #                 threshold=threshold,
    #                 unique_id=unique_id,
    #                 **kwargs,
    #             )
    #             for i in splits
    #         ]

    #     # finalise the result
    #     clean_blocks = enclosures.drop(splits)
    #     clean_blocks.loc[single, unique_id] = clean_blocks.loc[
    #         single, "position"
    #     ].apply(lambda ix: buildings.iloc[res[inp == ix][0]][unique_id])
    #     return pd.concat(new + [clean_blocks.drop(columns="position")]).reset_index(
    #         drop=True
    #     )


def find_remaining_unprocessed_places(
    gdf: gpd.GeoDataFrame, output_dir: str, report_file_suffix: str
) -> List[gpd.GeoSeries]:
    processed_files = glob(f"{output_dir}/*{report_file_suffix}")

    processed_places = set()
    for f in processed_files:
        city = os.path.basename(f).replace(report_file_suffix, "")
        processed_places.add(city)

    return gdf.loc[~gdf["place"].isin(processed_places)]


def get_building_and_street_data_for_place(
    place: str,
    buildings: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    city_buildings = (
        buildings[buildings["place"] == place].copy().reset_index(drop=True)
    )
    city_streets = streets[streets["place"] == place].copy().reset_index(drop=True)
    return city_buildings, city_streets


def process_tessellation(
    place: str, buildings: gpd.GeoDataFrame, streets: gpd.GeoDataFrame
):
    point = gpd.points_from_xy(buildings.city_lon, buildings.city_lat, crs="EPSG:4326")[
        0
    ]

    # Get UTM for this point
    city_utm_zone = get_utm_zone(point)
    buildings = buildings.to_crs(crs=city_utm_zone)

    logger.info("Creating enclosures")
    # Create enclosures
    streets = streets.to_crs(crs=city_utm_zone)
    convex_hull = streets.unary_union.convex_hull
    enclosures = momepy.enclosures(streets, limit=gpd.GeoSeries([convex_hull]))

    # Create Tessellation
    logger.info("Starting tessellation processing")
    tessellation = momepy.Tessellation(
        buildings,
        "buildingsID",
        enclosures=enclosures,
        use_dask=True,
        segment=5,
    )
    tessellation = tessellation.tessellation
    tessellation = tessellation.to_crs(crs="EPSG:4326")

    logger.info(f"Writing tessellation for {place}")
    tessellation.to_csv(
        f"/lustre/scratch/scratch/zcftkst/enclosed_tessellation/{place}_tessellation.csv",
        index=False,
    )

    logger.info(f"{place} tessellation processed!")


def run_process_tessellation(
    place: str, buildings: gpd.GeoDataFrame, streets: gpd.GeoDataFrame
):
    try:
        process_tessellation(place, buildings, streets)
    except Exception as e:
        logger.exception(e)
        pass


if __name__ == "__main__":
    unprocessed_places = [
        "kinki_m.m.a._(osaka)-japan",
        "tokyo-japan",
        "los_angeles-long_beach-santa_ana-united_states_of_america",
        "london-united_kingdom",
        "manila-philippines",
        "dhaka-bangladesh",
        "paris-france",
        "jakarta-indonesia",
        "são_paulo-brazil",
        "new_york-newark-united_states_of_america",
    ]

    from time import time

    start = time()

    logger.info("Loading all buildings data")
    all_buildings = load_geo_data("data/all_buildings.csv")

    logger.info("Loading all streets data")
    all_streets = load_geo_data("data/all_streets.csv")

    logger.info("Start processing tessellations")

    for place in unprocessed_places:
        logger.info(f"Currently at {place}")

        logger.info(f"Loading {place} building data")
        buildings = all_buildings.loc[all_buildings["place"] == place]
        remove_unnamed(buildings)

        logger.info(f"Loading {place} street data")
        streets = all_streets.loc[all_streets["place"] == place]
        remove_unnamed(streets)

        run_process_tessellation(place, buildings, streets)
        del buildings
        del streets

    logger.critical(f"Time needed - {time() - start}")
