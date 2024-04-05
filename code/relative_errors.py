
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from scipy import stats
from census_utils import *
import sys

races = ["W", "B", "AI_AN", "AS", "H_PI", "OTH", "TWO_OR_MORE"]
race_names = ["W", "B", "AI/AN", "AS", "H/PI", "OTH", "2+"]

def make_ids(df):
    null_idxs = df[["STATEA", "COUNTYA", "TRACTA", "BLOCKA"]].isnull().any(axis=1) | (df[["STATEA", "COUNTYA", "TRACTA", "BLOCKA"]] == "None").any(axis=1)
    nonnull_idxs = ~null_idxs

    df.loc[null_idxs, "COUNTYID"] = None
    df.loc[null_idxs, "TRACTID"] = None
    df.loc[null_idxs, "GEOID"] = None

    df.loc[nonnull_idxs, "STATEA"] = df.loc[nonnull_idxs, "STATEA"].astype(float).astype(int).astype(str).str.zfill(2)
    df.loc[nonnull_idxs, "COUNTYA"] = df.loc[nonnull_idxs, "COUNTYA"].astype(float).astype(int).astype(str).str.zfill(3)
    df.loc[nonnull_idxs, "TRACTA"] = df.loc[nonnull_idxs, "TRACTA"].astype(float).astype(int).astype(str).str.zfill(6)
    df.loc[nonnull_idxs, "BLOCKA"] = df.loc[nonnull_idxs, "BLOCKA"].astype(float).astype(int).astype(str).str.zfill(4)
    df.loc[nonnull_idxs, "COUNTYID"] = df.loc[nonnull_idxs, "STATEA"] + df.loc[nonnull_idxs, "COUNTYA"]
    df.loc[nonnull_idxs, "TRACTID"] = df.loc[nonnull_idxs, "COUNTYID"] + df.loc[nonnull_idxs, "TRACTA"]
    df.loc[nonnull_idxs, "GEOID"] = df.loc[nonnull_idxs, "TRACTID"] + df.loc[nonnull_idxs, "BLOCKA"]

def clean_blocks(df):
    df_clean = df.copy()
    
    df_clean = df_clean.rename(columns={
        "TABBLKST": "STATEA",
        "TABBLKCOU": "COUNTYA",
        "TABTRACTCE": "TRACTA",
        "TABBLK": "BLOCKA"
    })

    df_clean["TOTAL"] = 1

    df_clean["W"] = (df_clean["CENRACE"] == 1)
    df_clean["B"] = (df_clean["CENRACE"] == 2)
    df_clean["AI_AN"] = (df_clean["CENRACE"] == 3)
    df_clean["AS"] = (df_clean["CENRACE"] == 4)
    df_clean["H_PI"] = (df_clean["CENRACE"] == 5)
    df_clean["OTH"] = (df_clean["CENRACE"] == 6)
    df_clean["TWO_OR_MORE"] = 1 - df_clean["CENRACE"].isin([1, 2, 3, 4, 5, 6])
    
    df_clean["NUM_HISP"] = (df_clean["CENHISP"] == 2)

    if "VOTING_AGE" in df_clean.columns:
        df_clean["18_PLUS"] = (df_clean["VOTING_AGE"] == 2)
    elif "QAGE" in df_clean.columns:
        df_clean["18_PLUS"] = (df_clean["QAGE"] >= 18)

    make_ids(df_clean)
    
    df_clean = df_clean.groupby("COUNTYID").sum().reset_index()

    return df_clean

if __name__ == '__main__':

    db1_name = str(sys.argv[1])
    db2_name = str(sys.argv[2])
    name = str(sys.argv[3])

    topdown = (name == "topdown")
    toydown = (name == "toydown")
    swapped2 = (name == "swapped2")
    swapped10 = (name == "swapped10")

    db1 = pd.read_csv(db1_name, dtype={"GEOID": str, "COUNTYID": str})
    if not ("GEOID" in db1.columns):
        make_ids(db1)
    db1_blocks = db1.groupby("GEOID").sum().reset_index()
    db1_counties = db1.groupby("COUNTYID").sum().reset_index()

    db2 = pd.read_csv(db2_name, dtype={"GEOID": str, "COUNTYID": str})
    if not ("GEOID" in db2.columns):
        make_ids(db2)
    db2_blocks = db2.groupby("GEOID").sum().reset_index()
    db2_counties = db2.groupby("COUNTYID").sum().reset_index()

    print("done loading")

    all_blocks = db1_blocks.join(db2_blocks.set_index("GEOID"), on="GEOID", how="outer", rsuffix="_2").reset_index().fillna(0)
    all_counties = db1_counties.join(db2_counties.set_index("COUNTYID"), on="COUNTYID", how="outer", rsuffix="_2").reset_index().fillna(0)


    block_abs_errors = (all_blocks[races] - all_blocks[["{}_2".format(r) for r in races]]).abs()
    print((all_blocks[races] - all_blocks[["{}_2".format(r) for r in races]]).sum().sum())

    for race in races:
        zero_mask_1 = all_blocks[race] == 0
        zero_mask_2 = all_blocks["{}_2".format(race)] == 0

        all_blocks.loc[zero_mask_1 & ~zero_mask_2, "{}_RATIO_ERROR".format(race)] = 2
        all_blocks.loc[~zero_mask_1 & zero_mask_2, "{}_RATIO_ERROR".format(race)] = 0
        all_blocks.loc[zero_mask_1 & zero_mask_2, "{}_RATIO_ERROR".format(race)] = 1

        all_blocks.loc[~zero_mask_1 & ~zero_mask_2, "{}_RATIO_ERROR".format(race)] = 2 / (1 + all_blocks.loc[~zero_mask_1 & ~zero_mask_2, race] / all_blocks.loc[~zero_mask_1 & ~zero_mask_2, "{}_2".format(race)])

    all_blocks["RATIO_ERROR"] = all_blocks[["{}_RATIO_ERROR".format(r) for r in races]].mean(axis=1)


    for race in races:
        zero_mask_1 = all_counties[race] == 0
        zero_mask_2 = all_counties["{}_2".format(race)] == 0

        all_counties.loc[zero_mask_1 & ~zero_mask_2, "{}_RATIO_ERROR".format(race)] = 2
        all_counties.loc[~zero_mask_1 & zero_mask_2, "{}_RATIO_ERROR".format(race)] = 0
        all_counties.loc[zero_mask_1 & zero_mask_2, "{}_RATIO_ERROR".format(race)] = 1

        all_counties.loc[~zero_mask_1 & ~zero_mask_2, "{}_RATIO_ERROR".format(race)] = 2 / (1 + all_counties.loc[~zero_mask_1 & ~zero_mask_2, race] / all_counties.loc[~zero_mask_1 & ~zero_mask_2, "{}_2".format(race)])

    all_counties["RATIO_ERROR"] = all_counties[["{}_RATIO_ERROR".format(r) for r in races]].mean(axis=1)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 40))
    plt.plot((-0.5, 7.5), (1, 1), color="red", linewidth=0.5)
    b = plt.boxplot([all_counties[all_counties["RATIO_ERROR"].notna()]["RATIO_ERROR"]] + [all_counties[all_counties["{}_RATIO_ERROR".format(r)].notna()]["{}_RATIO_ERROR".format(r)] for r in races], 
        notch=False, meanline=False, showmeans=False, showfliers=True, showcaps=True, whis=1.5, patch_artist=True, boxprops=dict(facecolor = "white"),
        positions=range(len(races) + 1), labels=["Average"] + race_names)
    for median in b['medians']:
        median.set_color('#1f77b4')
    if topdown:
        plt.ylabel("2/(1+released/topdown)")
    elif toydown:
        plt.ylabel("2/(1+synthetic/toydown)")
    elif swapped2:
        plt.ylabel("2/(1+synthetic/swapped)")
    elif swapped10:
        plt.ylabel("2/(1+synthetic/swapped)")
    plt.ylim(-0.1, 2.1)
    plt.xticks(rotation=0)

    if topdown:
        plt.title("TopDown vs Released on Counties")
        plt.savefig("ratio_errors_topdown_counties.png", dpi=1000)
    elif toydown:
        plt.title("ToyDown vs Synthetic on Counties")
        plt.savefig("ratio_errors_toydown_counties.png", dpi=1000)
    elif swapped2:
        plt.title("Swapped (2%) vs Synthetic on Counties")
        plt.savefig("ratio_errors_swapped2_counties.png", dpi=1000)
    elif swapped10:
        plt.title("Swapped (10%) vs Synthetic on Counties")
        plt.savefig("ratio_errors_swapped10_counties.png", dpi=1000)
    

    plt.clf()

    plt.plot((-0.5, 7.5), (1, 1), color="red", linewidth=0.5)
    b = plt.boxplot([all_blocks[all_blocks["RATIO_ERROR"].notna()]["RATIO_ERROR"]] + [all_blocks[all_blocks["{}_RATIO_ERROR".format(r)].notna()]["{}_RATIO_ERROR".format(r)] for r in races], 
        notch=False, meanline=False, showmeans=False, showfliers=True, showcaps=True, whis=1.5, patch_artist=True, boxprops=dict(facecolor = "white"),
        positions=range(len(races) + 1), labels=["Average"] + race_names)
    for median in b['medians']:
        median.set_color('#1f77b4')
    if topdown:
        plt.ylabel("2/(1+released/topdown)")
    elif toydown:
        plt.ylabel("2/(1+synthetic/toydown)")
    elif swapped2:
        plt.ylabel("2/(1+synthetic/swapped)")
    elif swapped10:
        plt.ylabel("2/(1+synthetic/swapped)")
    plt.ylim(-0.1, 2.1)
    plt.xticks(rotation=0)

    if topdown:
        plt.title("TopDown vs Released on Blocks")
        plt.savefig("ratio_errors_topdown_blocks.png", dpi=1000)
    elif toydown:
        plt.title("ToyDown vs Synthetic on Blocks")
        plt.savefig("ratio_errors_toydown_blocks.png", dpi=1000)
    elif swapped2:
        plt.title("Swapped (2%) vs Synthetic on Blocks")
        plt.savefig("ratio_errors_swapped2_blocks.png", dpi=1000)
    elif swapped10:
        plt.title("Swapped (10%) vs Synthetic on Blocks")
        plt.savefig("ratio_errors_swapped10_blocks.png", dpi=1000)
