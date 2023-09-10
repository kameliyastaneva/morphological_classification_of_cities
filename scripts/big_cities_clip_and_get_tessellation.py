import geopandas as gpd
import momepy

from utils import get_logger, remove_unnamed, load_geo_data, get_utm_zone

logger = get_logger("big_cities_clip_and_tessellation")


def clip_city_buildings(
    buildings: gpd.GeoDataFrame, place: str, city_circle: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    logger.info(f"Start clipping {place} buildings!")

    buildings = buildings[buildings.geometry.within(city_circle.unary_union)]

    logger.info(f"Clipped {place} buildings!")
    logger.info(f"Writing {place} buildings to in data/cities!")

    buildings.to_csv(f"data/cities/{place}_buildings.csv", index=False)
    logger.info(f"Finished processing {place} buildings")

    return buildings


def clip_city_streets(
    streets: gpd.GeoDataFrame, place: str, city_circle: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    logger.info(f"Start clipping {place} streets!")

    streets = streets[streets.geometry.within(city_circle.unary_union)]

    logger.info(f"Clipped {place} streets!")
    logger.info(f"Writing {place} streets to in data/cities!")

    streets.to_csv(f"data/cities/{place}_streets.csv", index=False)
    logger.info(f"Finished processing {place} streets")

    return streets


def process_city_tessellation(city_name: str, city_buildings: gpd.GeoDataFrame):
    logger.info(f"{city_name} start processing tessellation")

    point = gpd.points_from_xy(
        city_buildings.city_lon, city_buildings.city_lat, crs="EPSG:4326"
    )[0]

    # Get UTM for this point
    city_utm_zone = get_utm_zone(point)
    city_buildings = city_buildings.to_crs(crs=city_utm_zone)

    # Create Tessellation
    limit = momepy.buffered_limit(city_buildings, 100)

    city_tessellation = momepy.Tessellation(
        city_buildings, "buildingsID", limit, verbose=False, segment=1
    )
    city_tessellation = city_tessellation.tessellation
    city_tessellation = city_tessellation.to_crs(crs="EPSG:4326")

    city_tessellation.to_csv(
        f"data/tessellations/{city_name}_tessellation.csv",
        index=False,
    )

    logger.info(f"{city_name} tessellation processed!")


if __name__ == "__main__":
    logger.info("Start big_cities_clip_and_tessellation")

    city_circles = load_geo_data("data/city_circles.csv")
    remove_unnamed(city_circles)

    city_circles["place"] = [
        f"{place[0].replace(' ', '_').lower()}-{place[1].replace(' ', '_').lower()}"
        for place in city_circles[["urban_agglomeration", "country_or_area"]].values
    ]

    for place in [
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
    ]:
        try:
            buildings = load_geo_data(f"data/cities/{place}_buildings.csv")
            streets = load_geo_data(f"data/cities/{place}_streets.csv")

            buildings = clip_city_buildings(
                buildings,
                place=place,
                city_circle=city_circles.loc[city_circles["place"] == place],
            )

            streets = clip_city_streets(
                streets,
                place=place,
                city_circle=city_circles.loc[city_circles["place"] == place],
            )

            process_city_tessellation(place, buildings)
        except Exception as e:
            logger.exception(e)
