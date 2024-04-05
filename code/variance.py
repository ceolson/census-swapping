import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from scipy import stats
from census_utils import *
import sys

single_races = ["W", "B", "AI_AN", "AS", "H_PI", "OTH"]

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

    df_clean = df_clean.groupby("GEOID").sum().reset_index()
    
    return df_clean


if __name__ == '__main__':
    file1 = str(sys.argv[1])
    file2 = str(sys.argv[2])

    db1 = pd.read_csv(file1)
    db2 = pd.read_csv(file2)

    if "TABBLKST" in db1.columns:
        db1 = clean_blocks(db1)
    elif "GEOID" in db1.columns:
        db1 = db1.groupby("GEOID").sum().reset_index()
    else:
        make_ids(db1)
        db1 = db1.groupby("GEOID").sum().reset_index()

    if "TABBLKST" in db2.columns:
        db2 = clean_blocks(db2)
    elif "GEOID" in db2.columns:
        db2 = db2.groupby("GEOID").sum().reset_index()
    else:
        make_ids(db2)
        db2 = db2.groupby("GEOID").sum().reset_index()

    print("done loading")

    print("lengths:", len(db1.index), len(db2.index))

    all_blocks = db1.join(db2.set_index("GEOID"), on="GEOID", rsuffix="_2", how="outer").reset_index().fillna(0)
    all_blocks = all_blocks[all_blocks.isna().any(axis=1) == False]

    print("done merging, merged length:", len(all_blocks.index))

    errors = np.array(all_blocks[single_races]) - np.array(all_blocks[["{}_2".format(r) for r in single_races]])
    mse = np.mean(np.square(errors))
    print(0.5 * mse)



