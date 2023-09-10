from glob import glob

import pandas as pd

from utils import load_geo_data


def combine_tessellations():
    tessellations = sorted(glob("data/tessellations/*.csv"))

    all_tessellations = []
    for csv_file in tessellations:
        gdf = load_geo_data(csv_file)
        all_tessellations.append(gdf)

    all_tessellations = pd.concat(all_tessellations)
    all_tessellations.reset_index(drop=True, inplace=True)

    all_tessellations.to_csv("data/all_tessellations.csv", index=False)


combine_tessellations()
