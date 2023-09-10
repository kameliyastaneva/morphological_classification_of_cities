import dask.dataframe as dd

from utils import get_logger, load_geo_data, remove_unnamed

logger = get_logger("split_by_city")

# all_buildings = dd.read_csv("data/all_buildings.csv")
all_streets = load_geo_data("data/all_streets.csv")

unprocessed_places = [
    # "kinki_m.m.a._(osaka)-japan",
    # "tokyo-japan",
    # "los_angeles-long_beach-santa_ana-united_states_of_america",
    # "london-united_kingdom",
    # "manila-philippines",
    # "dhaka-bangladesh",
    # "paris-france",
    # "jakarta-indonesia",
    # "são_paulo-brazil",
    # "new_york-newark-united_states_of_america",
    "cartagena-colombia",
    "krasnoyarsk-russian_federation",
    "rabat-morocco",
    "luoyang-china",
]


for place in unprocessed_places:
    # logger.info(f"Processing buildings for {place}")
    # city_buildings = all_buildings[all_buildings["place"] == place].compute()
    # remove_unnamed(city_buildings)

    # logger.info(f"Saving {place} buildings to CSV")
    # city_buildings.to_csv(f"data/cities/{place}_buildings.csv", index=False)

    logger.info(f"Processing streets for {place}")
    city_streets = all_streets[all_streets["place"] == place]
    remove_unnamed(city_streets)

    logger.info(f"Saving {place} streets to CSV")
    city_streets.to_csv(f"data/cities/{place}_streets.csv", index=False)
