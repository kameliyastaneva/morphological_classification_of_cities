import os
from glob import glob

import pandas as pd
import geopandas as gpd

from utils import load_geo_data


def load_data(path: str = None) -> gpd.GeoDataFrame:
    return pd.read_csv(path or "data/distance_from_centrepoint.csv")


def combine_datasets(output_path: str = None):
    for dataset_type in ("buildings", "streets"):
        combined_dataset = []

        cities = sorted(glob(f"data/cities/*{dataset_type}.csv"))
        for csv_file in cities:
            gdf = load_geo_data(csv_file)

            place_name = os.path.basename(csv_file)
            place_name = place_name.replace(f"_{dataset_type}.csv", "")

            gdf["place"] = place_name
            combined_dataset.append(gdf)

        combined_dataset = pd.concat(combined_dataset)
        combined_dataset.reset_index(drop=True, inplace=True)
        combined_dataset[f"{dataset_type}ID"] = combined_dataset.reset_index().pop(
            "index"
        )

        combined_dataset_output_path = (
            output_path or "data"
        ) + f"/all_{dataset_type}.csv"

        combined_dataset.to_csv(combined_dataset_output_path, index=False)


combine_datasets("data")
