import pandas as pd
import geopandas as gpd

from constants import WORLD_CRS
from utils import (
    get_utm_zone_from_city_lat_lon,
    load_geo_data,
    get_logger,
    remove_unnamed,
)

logger = get_logger("export_europe")


euro_cities = load_geo_data("data/case_studies.csv")["place"].values


logger.info("Loading all buildings")
all_buildings = load_geo_data("data/all_buildings.csv")
remove_unnamed(all_buildings)

logger.info("Loading all streets")
all_streets = load_geo_data("data/all_streets.csv")
remove_unnamed(all_streets)


logger.info("Exporting euro buildings")
euro_buildings = all_buildings.loc[all_buildings.place.isin(euro_cities)]
euro_buildings.to_csv("data/euro_buildings.csv", index=False)
del all_buildings

logger.info("Exporting euro streets")
euro_streets = all_streets.loc[all_streets.place.isin(euro_cities)]
euro_streets.to_csv("data/euro_streets.csv", index=False)
del all_streets
del euro_streets

# logger.info("Computing buildings area sum per city")

# buildings_area = []
# for euro_city in euro_cities:
#     logger.info(f"Start {euro_city}")
#     city_buildings = euro_buildings.loc[euro_buildings.place == euro_city]
#     lat, lon = city_buildings[["city_lat", "city_lon"]].to_list()
#     city_utm_zone = get_utm_zone_from_city_lat_lon(lat=lat, lon=lon)
    
#     city_buildings = city_buildings.to_crs(crs=city_utm_zone)

#     city_buildings["buildings_area"] = city_buildings.geometry.area
#     city_buildings.groupby("place")["buildings_area"].sum()

