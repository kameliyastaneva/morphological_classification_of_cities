import math

import pandas as pd
import geopandas as gpd
from shapely import wkt

from utils import get_utm_zone


df = pd.read_csv("data/case_studies.csv")

# Fix geometry
df["geometry"] = df["geometry"].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry=df["geometry"], crs="EPSG:4326")


def generate_dist_from_centre(row):
    row["distance_from_centrepoint"] = round(7 * (math.sqrt(row["2020"])))
    return row


gdf = gdf.apply(generate_dist_from_centre, axis=1)


def get_utm(row):
    row["utm_zone"] = get_utm_zone(row["geometry"])
    return row


gdf = gdf.apply(get_utm, axis=1)
gdf.to_csv("data/distance_from_centrepoint.csv", index=False)
