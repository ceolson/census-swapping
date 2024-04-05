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
    swapped_file = str(sys.argv[2])
    state_fips = str(sys.argv[3])
    swap_rate_str = str(sys.argv[4])

    print("Load data and create tables")
    synthetic_households = pd.read_csv(synthetic_file)
    make_ids(synthetic_households)

    swapped_households = pd.read_csv(swapped_file)
    make_ids(swapped_households)

    synthetic_blocks = synthetic_households.groupby("GEOID").sum().reset_index()
    swapped_blocks = swapped_households.groupby("GEOID").sum().reset_index()
    all_blocks = synthetic_blocks[["GEOID", "TOTAL", "18_PLUS"] + races_w_hisp].copy()
    all_blocks = all_blocks.join(swapped_blocks[["GEOID", "swapped"] + races_w_hisp].set_index("GEOID"), on="GEOID", rsuffix="_SW")

    synthetic_tracts = synthetic_households.groupby("TRACTID").sum().reset_index()
    swapped_tracts = swapped_households.groupby("TRACTID").sum().reset_index()
    all_tracts = synthetic_tracts[["TRACTID", "TOTAL", "18_PLUS"] + races_w_hisp].copy()
    all_tracts = all_tracts.join(swapped_tracts[["TRACTID", "swapped"] + races_w_hisp].set_index("TRACTID"), on="TRACTID", rsuffix="_SW")

    synthetic_counties = synthetic_households.groupby("COUNTYID").sum().reset_index()
    swapped_counties = swapped_households.groupby("COUNTYID").sum().reset_index()
    all_counties = synthetic_counties[["COUNTYID", "TOTAL", "18_PLUS"] + races_w_hisp].copy()
    all_counties = all_counties.join(swapped_counties[["COUNTYID", "swapped"] + races_w_hisp].set_index("COUNTYID"), on="COUNTYID", rsuffix="_SW")



    print("Maps")
    all_county_shapes = gpd.read_file("/Users/ceolson/Downloads/nhgis0036_shape/nhgis0036_shapefile_tl2010_us_county_2010/US_county_2010.shp")
    county_shapefiles_al = all_county_shapes[all_county_shapes["STATEFP10"] == state_fips].rename(columns={"GEOID10": "COUNTYID"})

    all_counties = all_counties.join(county_shapefiles_al[["COUNTYID", "geometry"]].set_index("COUNTYID"), on="COUNTYID")
    all_counties = gpd.GeoDataFrame(all_counties, geometry=all_counties["geometry"])

    all_counties["w_change_swapping"] = all_counties["W_SW"] - all_counties["W"]
    all_counties["w_pct_change_swapping"] = all_counties["w_change_swapping"] / all_counties["TOTAL"]
    all_counties["PCT_W"] = all_counties["W"] / all_counties["TOTAL"]
    all_counties["PCT_NW"] = 1 - all_counties["PCT_W"]

    all_counties["hisp_change_swapping"] = all_counties["NUM_HISP_SW"] - all_counties["NUM_HISP"]
    all_counties["hisp_pct_change_swapping"] = all_counties["hisp_change_swapping"] / all_counties["TOTAL"]
    all_counties["PCT_HISP"] = all_counties["NUM_HISP"] / all_counties["TOTAL"]
    all_counties["PCT_NON_HISP"] = 1 - all_counties["PCT_HISP"]

    all_counties["b_change_swapping"] = all_counties["B_SW"] - all_counties["B"]
    all_counties["b_pct_change_swapping"] = all_counties["b_change_swapping"] / all_counties["TOTAL"]
    all_counties["PCT_B"] = all_counties["B"] / all_counties["TOTAL"]
    all_counties["PCT_NB"] = 1 - all_counties["PCT_B"]

    all_counties["as_change_swapping"] = all_counties["AS_SW"] - all_counties["AS"]
    all_counties["as_pct_change_swapping"] = all_counties["as_change_swapping"] / all_counties["TOTAL"]
    all_counties["PCT_AS"] = all_counties["AS"] / all_counties["TOTAL"]
    all_counties["PCT_NAS"] = 1 - all_counties["PCT_AS"]

    # W maps
    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax=axs, column="w_change_swapping", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("Change in White population after swapping")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "w_pop_change_map.png")

    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax = axs, column="W", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("White population")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "w_pop_map.png")

    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax = axs, column="PCT_W", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("White percentage")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "w_pct_map.png")

    # HISP maps
    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax = axs, column="hisp_change_swapping", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("Change in Hispanic population after swapping")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "hisp_pop_change_map.png")

    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax = axs, column="NUM_HISP", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("Hispanic population")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "hisp_pop_map.png")

    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax = axs, column="PCT_HISP", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("Hispanic percentage")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "hisp_pct_map.png")

    # B maps
    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax=axs, column="b_change_swapping", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("Change in Black population after swapping")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "b_pop_change_map.png")

    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax = axs, column="B", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("Black population")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "b_pop_map.png")

    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax = axs, column="PCT_B", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("Black percentage")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "b_pct_map.png")

    # AS maps
    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax=axs, column="as_change_swapping", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("Change in Asian population after swapping")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "as_pop_change_map.png")

    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax = axs, column="AS", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("Asian population")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "as_pop_map.png")

    fig, axs = plt.subplots(figsize=(10, 10))
    all_counties.plot(ax = axs, column="PCT_AS", cmap="plasma", legend=True)
    axs.set(yticklabels=[])
    axs.set(xticklabels=[])
    axs.set_title("Asian percentage")
    axs.tick_params(left=False, bottom=False)
    fig.savefig(state_fips + "_" + swap_rate_str + "/" + "as_pct_map.png")


    print("Which races are swapped for which?")
    swapped_households["MULTIPLE_RACES"] = ((swapped_households[races] > 0).sum(axis=1) > 1)
    swapped_households["MAJORITY_RACE"] = swapped_households[races].idxmax(axis=1)
    swapped_households.loc[swapped_households["MULTIPLE_RACES"], "MAJORITY_RACE"] = "MULTIPLE"


    swapping_races_for_races = swapped_households[["household.id", "MAJORITY_RACE", "swapped", 
                                        "swap.partner.id", "targeted", "partnered"]].rename(
        columns={"swap.partner.id": "pre.swap.id",
                 "MAJORITY_RACE": "MAJORITY_RACE_PRE_SWAP"}
    )

    swapping_races_for_races = swapping_races_for_races.join(
        swapped_households[["household.id", "MAJORITY_RACE"]].rename(
            columns={"household.id": "pre.swap.id",
                     "MAJORITY_RACE": "MAJORITY_RACE_POST_SWAP"}).set_index("pre.swap.id"), on="pre.swap.id")

    races_for_races = pd.DataFrame(columns=races)

    for r1 in races + ["MULTIPLE"]:
        for r2 in races + ["MULTIPLE"]:
            races_for_races.loc[r1, r2] = len(swapping_races_for_races[
                (swapping_races_for_races["targeted"] == 1) & (swapping_races_for_races['MAJORITY_RACE_PRE_SWAP'] == r1) & (swapping_races_for_races['MAJORITY_RACE_POST_SWAP'] == r2)
            ].index)

    races_for_races.to_csv(state_fips + "_" + swap_rate_str + "/" + "races_for_races.csv")
    print("All households", swapped_households.groupby("MAJORITY_RACE").count()["STATEA"])
    print("Targeted households", swapped_households[swapped_households["targeted"] == 1].groupby("MAJORITY_RACE").count()["STATEA"])
    print("Partnered households", swapped_households[swapped_households["partnered"] == 1].groupby("MAJORITY_RACE").count()["STATEA"])

    print("All people", swapped_households[races].sum(axis=0))
    print("Targeted people", swapped_households[swapped_households["targeted"] == 1][races].sum(axis=0))

    print("Household size distribution")
    hh_dist = pd.DataFrame((swapped_households.groupby("TOTAL").count()["STATE"])).rename(columns={"STATE": "TOTAL_COUNT"})
    hh_dist = hh_dist.merge(swapped_households[swapped_households["swapped"] == 1].groupby("TOTAL").count()["STATE"], on="TOTAL", how="outer").rename(
    columns={"STATE":"SWAPPED_COUNT"})
    hh_dist.to_csv(state_fips + "_" + swap_rate_str + "/" + "hh_dist.csv")


    print("Household distribution by race")
    household_race_dist = pd.DataFrame(columns=range(1, np.max(hh_dist.index) + 1, 1))

    for r in races + ["MULTIPLE"]:
        for i in range(1, np.max(hh_dist.index) + 1, 1):
            household_race_dist.loc[r, i] = len(swapped_households[
                (swapped_households["MAJORITY_RACE"] == r) & (swapped_households["TOTAL"] == i)
            ].index)

    household_race_dist.to_csv(state_fips + "_" + swap_rate_str + "/" + "hh_race_dist.csv")


    print("Entropy changes")
    all_tracts["ENTROPY"] = all_tracts.apply(lambda row: stats.entropy(row[races].astype(int)), axis=1)
    all_tracts["ENTROPY_SW"] = all_tracts.apply(lambda row: stats.entropy(row[["{}_SW".format(r) for r in races]].astype(int)), axis=1)

    plt.figure(figsize=(10, 7.5))

    plt.scatter(all_tracts["ENTROPY"], all_tracts["ENTROPY_SW"],
                marker=".", c=all_tracts["swapped"] / all_tracts["TOTAL"], 
                cmap="plasma", vmin=0, vmax=float(swap_rate_str))
    plt.plot((0, all_tracts["ENTROPY"].max()), (0, all_tracts["ENTROPY"].max()), color="deepskyblue")
    plt.title("Effect of Swapping on Entropy")
    plt.colorbar(label="Percent of tract swapped")
    plt.xlabel("Entropy before swapping")
    plt.ylabel("Entropy after swapping")
    plt.savefig(state_fips + "_" + swap_rate_str + "/" + "entropy.png")









