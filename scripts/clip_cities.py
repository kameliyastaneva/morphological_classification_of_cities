from typing import Generator, Tuple

import geopandas as gpd
import pandas as pd

from utils import load_geo_data, get_logger

logger = get_logger("clip_cities")


def split_geodataframe_by_place(
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


def clip_geodataframe_by_city_circle(
    gdf: gpd.GeoDataFrame, city_circles: gpd.GeoDataFrame, gdf_name: str
):
    new_gdfs = []
    for place, new_gdf in split_geodataframe_by_place(gdf):
        logger.info(f"Clipping the {gdf_name} data for {place}")
        city_circle = city_circles.loc[city_circles["place"] == place]
        new_gdf = new_gdf[new_gdf.geometry.within(city_circle.unary_union)]
        new_gdfs.append(new_gdf)

    new_gdfs = pd.concat(new_gdfs, ignore_index=True)

    logger.info(f"Finished clipping all data in {gdf_name}")
    new_gdfs.to_csv(f"data/{gdf_name}.csv")


if __name__ == "__main__":
    logger.info("Loading building and street data")
    all_buildings = load_geo_data("data/all_buildings.csv")
    all_streets = load_geo_data("data/all_streets.csv")

    logger.info("Loading city circles data")
    city_circles = load_geo_data("data/city_circles.csv")
    city_circles["place"] = [
        f"{place[0].replace(' ', '_').lower()}-{place[1].replace(' ', '_').lower()}"
        for place in city_circles[["urban_agglomeration", "country_or_area"]].values
    ]

    clip_geodataframe_by_city_circle(all_buildings, city_circles, "all_buildings")
    clip_geodataframe_by_city_circle(all_streets, city_circles, "all_streets")
