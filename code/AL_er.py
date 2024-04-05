import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from census_utils import *
import sys

races = ['W', 'B', 'AI_AN', 'AS', 'H_PI', 'OTH', 'TWO_OR_MORE']
races_for_printing = ["W", "B", "AI/AN", "AS", "H/PI", "Other", "2+"]
races_w_hisp = races + ["NUM_HISP"]
races_vap_names = ["{}_VAP".format(r) for r in races]

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
    toydown_file = str(sys.argv[3])
    state_fips = str(sys.argv[4])
    state = str(sys.argv[5])
    swap_rate_str = str(sys.argv[6])
    weight_by_pop = int(sys.argv[7])


    print("Load data and create tables")
    synthetic_households = pd.read_csv(synthetic_file)
    make_ids(synthetic_households)

    swapped_households = pd.read_csv(swapped_file)
    make_ids(swapped_households)

    toydown_blocks = pd.read_csv(toydown_file)
    make_ids(toydown_blocks)

    for df in [synthetic_households, swapped_households]:
        for race in races:
            df["{}_VAP".format(race)] = (df[race] / df["TOTAL"] * df["18_PLUS"]).astype(int)
            
        df["MAJORITY_RACE"] = df[races].idxmax(axis=1) + "_VAP"
        for r in races_vap_names:
            maj_r = np.flatnonzero(df["MAJORITY_RACE"] == r)
            df.loc[maj_r, r] += df.loc[maj_r, "18_PLUS"] - df.loc[maj_r, races_vap_names].sum(axis=1) 

    for race in races:
        toydown_blocks["{}_VAP".format(race)] = toydown_blocks[race]

    toydown_blocks["18_PLUS"] = toydown_blocks[["{}_VAP".format(r) for r in races]].sum(axis=1)


    synthetic_blocks = synthetic_households.groupby("GEOID").sum().reset_index()
    swapped_blocks = swapped_households.groupby("GEOID").sum().reset_index()
    all_blocks = synthetic_blocks[["GEOID", "TOTAL", "18_PLUS"] + races_vap_names].copy()
    all_blocks = all_blocks.join(swapped_blocks[["GEOID", "swapped"] + races_vap_names].set_index("GEOID"), on="GEOID", rsuffix="_SW", how="outer").fillna(0)
    all_blocks = all_blocks.join(toydown_blocks[["GEOID", "18_PLUS"] + races_vap_names].set_index("GEOID"), on="GEOID", rsuffix="_TD", how="outer").fillna(0)

    blocks_to_vtd = pd.read_csv("../data/{}/block_to_vtd.csv".format(state), dtype={"GEOID": str, "VTDID": str})

    all_blocks = all_blocks.join(blocks_to_vtd.set_index("GEOID"), on="GEOID", rsuffix="_")

    all_vtds = all_blocks.groupby("VTDID").sum().reset_index()

    election_results = pd.read_csv("../data/{}/election_results.csv".format(state), dtype={"VTDID": str})

    election = "LTG"
    candidates = ['G18LTGRAIN', 'G18LTGDBOY', 'G18LTGOWRI']
    

    election_results["TOTAL_VOTES_{}".format(election)] = election_results[candidates].sum(axis=1)
    candidates = candidates + ["NO_VOTE_{}".format(election)]

    candidates_for_printing = ["Ainsworth (R)", "Boyd (D)", "Write-In", "No Vote"]
    
    all_vtds = all_vtds.join(election_results.set_index("VTDID"), on="VTDID", rsuffix="_")

    print("Mean VTD VAP:", all_vtds["18_PLUS"].mean())

    all_vtds["NO_VOTE_{}".format(election)] = all_vtds["18_PLUS"] - all_vtds["TOTAL_VOTES_{}".format(election)]

    for r in races_vap_names:
        all_vtds["{}_PCT".format(r)] = all_vtds[r] / all_vtds["18_PLUS"]
        all_vtds["{}_SW_PCT".format(r)] = all_vtds["{}_SW".format(r)] / all_vtds["18_PLUS"]
        all_vtds["{}_TD_PCT".format(r)] = all_vtds["{}_TD".format(r)] / all_vtds["18_PLUS_TD"]
        
    for c in candidates:
        all_vtds["{}_PCT".format(c)] = all_vtds[c] / all_vtds["18_PLUS"]

    
    vtd_conditions_er = [
            all_vtds[["{}_PCT".format(r) for r in races_vap_names]].notna().all(axis=1)
        ] + [
            all_vtds[["{}_SW_PCT".format(r) for r in races_vap_names]].notna().all(axis=1)
        ] + [
            all_vtds[["{}_TD_PCT".format(r) for r in races_vap_names]].notna().all(axis=1)
        ] + [
            (all_vtds["NO_VOTE_{}".format(election)] >= 0)
        ] + [
            (all_vtds[races_vap_names].sum(axis=1) == all_vtds["18_PLUS"])
        ] + [
            (all_vtds[["{}_SW".format(r) for r in races_vap_names]].sum(axis=1) == all_vtds["18_PLUS"])
        ] + [
            (all_vtds[candidates].sum(axis=1) == all_vtds["18_PLUS"])
        ]

    print(np.sum(all_vtds["W_VAP"]))
    print(np.sum(all_vtds["W_VAP_SW"]))
    print(np.sum(all_vtds["W_VAP_TD"]))

    vtd_filter_er = True
    for c in vtd_conditions_er:
        print(np.mean(c))
        vtd_filter_er &= c

    print("Kept", np.sum(vtd_filter_er), "out of", len(vtd_filter_er))

    all_vtds = all_vtds[vtd_filter_er]


    fig, axs = plt.subplots(len(races), len(candidates), figsize=(25, 30))

    num_precincts = len(all_vtds.index)

    predictions_unswapped = pd.DataFrame([], columns=candidates, index=races)
    predictions_swapped = pd.DataFrame([], columns=candidates, index=races)
    predictions_toydown = pd.DataFrame([], columns=candidates, index=races)

    for r in range(len(races_vap_names)):
        race = races_vap_names[r]
        for c in range(len(candidates)):
            cand = candidates[c]
        
            xs = np.reshape(np.linspace(0, 1, 100), (100,1))

            cand_pcts = np.reshape(np.array(all_vtds["{}_PCT".format(cand)]), (num_precincts, 1)) 


            unswapped_race_pcts = np.reshape(np.array(all_vtds["{}_PCT".format(race)]), (num_precincts, 1))

            if weight_by_pop:
                unswapped_reg = LinearRegression().fit(unswapped_race_pcts, cand_pcts, all_vtds["18_PLUS"])
            else:
                unswapped_reg = LinearRegression().fit(unswapped_race_pcts, cand_pcts, None)

            axs[r, c].plot(unswapped_race_pcts, cand_pcts, '.', color="b", rasterized=True, alpha=0.2)
            axs[r, c].plot(xs, unswapped_reg.predict(xs), '-', color="b", label="Original")
            
            swapped_race_pcts = np.reshape(np.array(all_vtds["{}_SW_PCT".format(race)]), (num_precincts, 1))

            if weight_by_pop:
                swapped_reg = LinearRegression().fit(swapped_race_pcts, cand_pcts, all_vtds["18_PLUS"])
            else:
                swapped_reg = LinearRegression().fit(swapped_race_pcts, cand_pcts, None)

            axs[r, c].plot(swapped_race_pcts, cand_pcts, '.', color="magenta", rasterized=True, alpha=0.2)
            axs[r, c].plot(xs, swapped_reg.predict(xs), '-', color="magenta", label="Swapped")

            toydown_race_pcts = np.reshape(np.array(all_vtds["{}_TD_PCT".format(race)]), (num_precincts, 1))

            if weight_by_pop:
                toydown_reg = LinearRegression().fit(toydown_race_pcts, cand_pcts, all_vtds["18_PLUS"])
            else:
                toydown_reg = LinearRegression().fit(toydown_race_pcts, cand_pcts, None)

            axs[r, c].plot(toydown_race_pcts, cand_pcts, '.', color="orange", rasterized=True, alpha=0.2)
            axs[r, c].plot(xs, toydown_reg.predict(xs), '-', color="orange", label="ToyDown")

    #         axs[r, c].set_xlim(np.min(unswapped_race_pcts) - 0.01, np.max(unswapped_race_pcts) + 0.01)
    #         axs[r, c].set_ylim(np.min(cand_pcts) - 0.01, np.max(cand_pcts) + 0.01)
            axs[r, c].set_title("Support for {} by {} voters".format(candidates_for_printing[c], races_for_printing[r]))

            axs[r, c].set_xlabel("Percentage " + races_for_printing[r])
            axs[r, c].set_ylabel("Percentage " + candidates_for_printing[c])
            axs[r, c].legend()

            predictions_unswapped.loc[races[r], cand] = unswapped_reg.predict(np.ones(1).reshape(1, -1))[0][0]
            predictions_swapped.loc[races[r], cand] = swapped_reg.predict(np.ones(1).reshape(1, -1))[0][0]
            predictions_toydown.loc[races[r], cand] = toydown_reg.predict(np.ones(1).reshape(1, -1))[0][0]
        
    fig.tight_layout()
    plt.savefig("ER_{}_{}_{}.png".format(state_fips, swap_rate_str, weight_by_pop), dpi=500)

    print(predictions_unswapped)
    print(predictions_swapped)
    print(predictions_toydown)




