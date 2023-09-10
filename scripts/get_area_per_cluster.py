import pandas as pd

from tqdm import tqdm

from utils import (
    load_geo_data,
    remove_unnamed,
    get_utm_zone_from_city_lat_lon,
)


agglomeration = load_geo_data("data/agglomeration.csv")
euro_case_studies = pd.read_csv("data/euro_case_studies.csv")
remove_unnamed(euro_case_studies)

euro_case_studies["place"] = (
    euro_case_studies["urban_agglomeration"].str.replace(" ", "_").str.lower()
    + "-"
    + euro_case_studies["country_or_area"].str.replace(" ", "_").str.lower()
)

cluster_data_per_city = []
for city in tqdm(
    agglomeration["city"].unique(), total=len(agglomeration["city"].unique())
):
    city_gdf = agglomeration.loc[agglomeration.city == city].copy()

    city_suppliment_data = euro_case_studies.loc[euro_case_studies.place == city].copy()
    lat, lon = city_suppliment_data[["latitude", "longitude"]].values[0]
    city_utm_zone = get_utm_zone_from_city_lat_lon(lat, lon)

    city_gdf = city_gdf.to_crs(city_utm_zone)

    cluster_data_per_city.append(
        city_gdf.groupby("cluster")["geometry"]
        .agg(lambda geoms: sum(geoms.area))
        .to_frame()
        .reset_index()
        .rename(columns={"index": "cluster"})
    )

cluster_data = pd.concat(cluster_data_per_city, axis=1)
cluster_data.fillna(0, inplace=True)
area_per_cluster = cluster_data.groupby("cluster").sum()
