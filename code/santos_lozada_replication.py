import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from scipy import stats
from census_utils import *
import sys

races = ["W", "B", "AI_AN", "AS", "H_PI", "OTH", "TWO_OR_MORE"]

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
    db1_names = [str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5])]
    db2_names = [str(sys.argv[6]), str(sys.argv[7]), str(sys.argv[8]), str(sys.argv[9]), str(sys.argv[10])]

    db1s = [pd.read_csv(db, dtype={"GEOID": str, "COUNTYID": str}) for db in db1_names]
    for db in db1s:
        if not ("COUNTYID" in db.columns):
            make_ids(db)
    db1 = pd.concat(db1s, ignore_index=True)
    db1 = db1.groupby("COUNTYID").sum().reset_index()


    db2s = [pd.read_csv(db, dtype={"GEOID": str, "COUNTYID": str}) for db in db2_names]
    for db in db2s:
        if not ("COUNTYID" in db.columns):
            make_ids(db)
    db2 = pd.concat(db2s, ignore_index=True)
    db2 = db2.groupby("COUNTYID").sum().reset_index()

    rucc_codes = pd.read_csv("../data/US/ruralurbancodes2013.csv", dtype={"FIPS": str})
    rucc_codes = rucc_codes.rename(columns={"FIPS": "COUNTYID"})

    print("done loading")

    counties1 = db1
    counties2 = db2

    print(counties1.head())
    print(counties2.head())

    all_counties = counties1.join(counties2.set_index("COUNTYID"), on="COUNTYID", how="outer", rsuffix="_2").reset_index().fillna(0)
    all_counties = all_counties.join(rucc_codes.set_index("COUNTYID"), on="COUNTYID").reset_index()

    print("done merging, merged length:", len(all_counties.index))

    for q in races + ["NUM_HISP", "TOTAL"]:
        all_counties["{}_COUNT_RATIO".format(q)] = all_counties[q] / all_counties["{}_2".format(q)]
        all_counties["{}_ERROR".format(q)] = all_counties[q] - all_counties["{}_2".format(q)]
        

    # Create figure and axes
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

    def split_by_rucc(db, q):
        dbs = [db[db["RUCC_2013"] == rucc] for rucc in range(1, 10)]
        return [db[q][(db[q].notnull()) & (db[q] < np.inf)] for db in dbs]

    possible_values = all_counties[["TOTAL_COUNT_RATIO", "W_COUNT_RATIO", "B_COUNT_RATIO", "NUM_HISP_COUNT_RATIO", "AI_AN_COUNT_RATIO"]]

    y_max = possible_values[possible_values.notna().all(axis=1) & (possible_values < np.inf).all(axis=1)].max().max()
    y_min = possible_values[possible_values.notna().all(axis=1) & (possible_values < np.inf).all(axis=1)].min().min()

    print(y_min, y_max)

    print("mean abs errors:")
    print()

    # Plot each category
    b = axs[0].boxplot(split_by_rucc(all_counties, "TOTAL_COUNT_RATIO"), 
        notch=False, meanline=False, showmeans=False, showfliers=True, showcaps=True, whis=1.5, patch_artist=True, boxprops=dict(facecolor = "white"),
        positions=range(1, 10))
    for median in b['medians']:
        median.set_color('#1f77b4')
    axs[0].set_title("Population")
    axs[0].set_ylim(y_min, y_max)

    axs[0].set_ylabel("TopDown/Synthetic")

    axs[0].set_xlabel("RUCC Code (1: Most urban, 9: Most rural)")

    b = axs[1].boxplot(split_by_rucc(all_counties, "W_COUNT_RATIO"), 
        notch=False, meanline=False, showmeans=False, showfliers=True, showcaps=True, whis=1.5, patch_artist=True, boxprops=dict(facecolor = "white"),
        positions=range(1, 10))
    for median in b['medians']:
        median.set_color('#1f77b4')
    axs[1].set_title("White")
    axs[1].set_ylim(y_min, y_max)
    axs[1].set_xlabel("RUCC Code (1: Most urban, 9: Most rural)")

    b = axs[2].boxplot(split_by_rucc(all_counties, "B_COUNT_RATIO"), 
        notch=False, meanline=False, showmeans=False, showfliers=True, showcaps=True, whis=1.5, patch_artist=True, boxprops=dict(facecolor = "white"),
        positions=range(1, 10))
    for median in b['medians']:
        median.set_color('#1f77b4')
    axs[2].set_title("Black")
    axs[2].set_ylim(y_min, y_max)
    axs[2].set_xlabel("RUCC Code (1: Most urban, 9: Most rural)")

    b = axs[3].boxplot(split_by_rucc(all_counties, "NUM_HISP_COUNT_RATIO"), 
        notch=False, meanline=False, showmeans=False, showfliers=True, showcaps=True, whis=1.5, patch_artist=True, boxprops=dict(facecolor = "white"),
        positions=range(1, 10))
    for median in b['medians']:
        median.set_color('#1f77b4')
    axs[3].set_title("Hispanic")
    axs[3].set_ylim(y_min, y_max)
    axs[3].set_xlabel("RUCC Code (1: Most urban, 9: Most rural)")

    b = axs[4].boxplot(split_by_rucc(all_counties, "AI_AN_COUNT_RATIO"), 
        notch=False, meanline=False, showmeans=False, showfliers=True, showcaps=True, whis=1.5, patch_artist=True, boxprops=dict(facecolor = "white"),
        positions=range(1, 10))
    for median in b['medians']:
        median.set_color('#1f77b4')
    axs[4].set_title("AI/AN")
    axs[4].set_ylim(y_min, y_max)
    axs[4].set_xlabel("RUCC Code (1: Most urban, 9: Most rural)")

    plt.savefig("sl_topdown_error.png")

