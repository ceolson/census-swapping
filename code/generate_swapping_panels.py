import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from scipy import stats
from census_utils import *
import sys

races = ['W', 'B', 'AI_AN', 'AS', 'H_PI', 'OTH', 'TWO_OR_MORE']
races_w_hisp = races + ["NUM_HISP"]

nh_races = ["NH_" + r for r in races]
nh_races_and_hisp = nh_races = ["NUM_HISP"]

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


def make_identifier_synth(df, ID_COLS=['TRACTA', 'COUNTYA', 'BLOCKA'], id_lens=[6, 3, 4], name='id'):
    str_cols = [col + '_str' for col in ID_COLS]
    for col, l, col_s in zip(ID_COLS, id_lens, str_cols):
        assert max(num_digits(s) for s in df[col].unique()) <= l
        df[col_s] = df[col].astype(str).str.zfill(l)
    df[name] = df[str_cols].astype(str).agg('-'.join, axis=1)
    for col_s in str_cols:
        del df[col_s]


if __name__ == '__main__':
    synthetic_file = str(sys.argv[1])
    target_leaves_file = str(sys.argv[2])
    target_arrives_file = str(sys.argv[3])
    partner_leaves_file = str(sys.argv[4])
    partner_arrives_file = str(sys.argv[5])
    state_fips = str(sys.argv[6])
    swap_rate_str = str(sys.argv[7])

    print("Load data and create tables")
    synthetic_hhs = pd.read_csv(synthetic_file)
    make_ids(synthetic_hhs)
    synthetic_tracts = synthetic_hhs.groupby("TRACTID").sum().reset_index()
    synthetic_tracts["ENTROPY"] = synthetic_tracts.apply(lambda row: stats.entropy(row[races].astype(int)), axis=1)

    hh_dbs = []
    hh_dbs.append(pd.read_csv(target_leaves_file))
    hh_dbs.append(pd.read_csv(target_arrives_file))
    hh_dbs.append(pd.read_csv(partner_leaves_file))
    hh_dbs.append(pd.read_csv(partner_arrives_file))

    tract_dbs = []
    merged_tracts = []
    for db in hh_dbs:
        make_ids(db)
        tract = db.groupby("TRACTID").sum().reset_index()
        tract["ENTROPY"] = tract.apply(lambda row: stats.entropy(row[races].astype(int)), axis=1)
        tract_dbs.append(tract)
        all_tracts = synthetic_tracts[["TRACTID", "TOTAL", "18_PLUS", "ENTROPY", "targeted", "partnered"] + races_w_hisp].copy()
        all_tracts = all_tracts.join(tract[["TRACTID", "ENTROPY"] + races_w_hisp].set_index("TRACTID"), on="TRACTID", rsuffix="_SW")
        merged_tracts.append(all_tracts)


    fig, axs = plt.subplots(2, 2, figsize=(20, 15))

    color = merged_tracts[0]["targeted"] / merged_tracts[0]["TOTAL"]
    im = axs[0, 0].scatter(merged_tracts[0]["ENTROPY"], merged_tracts[0]["ENTROPY_SW"],
               marker=".", c=color, cmap="plasma", vmin=0, vmax=float(swap_rate_str) / 2)
    axs[0, 0].plot((0, merged_tracts[0]["ENTROPY"].max()), (0, merged_tracts[0]["ENTROPY"].max()), color="deepskyblue")
    axs[0, 0].set_title("Targets Leave")
    axs[0, 0].set_xlabel("Entropy before")
    axs[0, 0].set_ylabel("Entropy after")
    fig.colorbar(im, ax=axs[0, 0])


    color = merged_tracts[1]["targeted"] / merged_tracts[1]["TOTAL"]
    im = axs[0, 1].scatter(merged_tracts[1]["ENTROPY"], merged_tracts[1]["ENTROPY_SW"],
               marker=".", c=color, cmap="plasma", vmin=0, vmax=float(swap_rate_str) / 2)
    axs[0, 1].plot((0, merged_tracts[1]["ENTROPY"].max()), (0, merged_tracts[1]["ENTROPY"].max()), color="deepskyblue")
    axs[0, 1].set_title("Targets Arrive")
    axs[0, 1].set_xlabel("Entropy before")
    axs[0, 1].set_ylabel("Entropy after")
    fig.colorbar(im, ax=axs[0, 1])


    color = merged_tracts[2]["partnered"] / merged_tracts[2]["TOTAL"]
    im = axs[1, 0].scatter(merged_tracts[2]["ENTROPY"], merged_tracts[2]["ENTROPY_SW"],
               marker=".", c=color, cmap="plasma", vmin=0, vmax=float(swap_rate_str) / 2)
    axs[1, 0].plot((0, merged_tracts[2]["ENTROPY"].max()), (0, merged_tracts[2]["ENTROPY"].max()), color="deepskyblue")
    axs[1, 0].set_title("Partners Leave")
    axs[1, 0].set_xlabel("Entropy before")
    axs[1, 0].set_ylabel("Entropy after")
    fig.colorbar(im, ax=axs[1, 0])


    color = merged_tracts[3]["partnered"] / merged_tracts[3]["TOTAL"]
    im = axs[1, 1].scatter(merged_tracts[3]["ENTROPY"], merged_tracts[3]["ENTROPY_SW"],
               marker=".", c=color, cmap="plasma", vmin=0, vmax=float(swap_rate_str) / 2)
    axs[1, 1].plot((0, merged_tracts[3]["ENTROPY"].max()), (0, merged_tracts[3]["ENTROPY"].max()), color="deepskyblue")
    axs[1, 1].set_title("Partners Arrive")
    axs[1, 1].set_xlabel("Entropy before")
    axs[1, 1].set_ylabel("Entropy after")
    fig.colorbar(im, ax=axs[1, 1])

    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "panels.png")




