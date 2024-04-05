from census_utils import *
import pandas as pd
import geopandas as gpd
from random import random
from math import sqrt
from sklearn.neighbors import KDTree
import numpy as np
import json
import sys

groups = ['W', 'B', 'AI_AN', 'AS', 'H_PI', 'OTH', 'TWO_OR_MORE', 'NUM_HISP', '18_PLUS']
races = ['W', 'B', 'AI_AN', 'AS', 'H_PI', 'OTH', 'TWO_OR_MORE']
races_w_hisp = races + ['NUM_HISP']

def load_data(task_name):
    return pd.read_csv(get_synthetic_out_file(task_name))

def make_identifier_synth(df, ID_COLS=['COUNTYA', 'TRACTA', 'BLOCKA'], id_lens=[3, 6, 4], name='id'):
    str_cols = [col + '_str' for col in ID_COLS]
    for col, l, col_s in zip(ID_COLS, id_lens, str_cols):
        assert max(num_digits(s) for s in df[col].unique()) <= l
        df[col_s] = df[col].astype(str).str.zfill(l)
    df[name] = df[str_cols].astype(str).agg('-'.join, axis=1)
    for col_s in str_cols:
        del df[col_s]

def load_shape_data(area):
    block_map = gpd.read_file(get_shape_file(area))
    return block_map.to_crs('EPSG:3395')

def make_identifier_synth_geo(df):
    ID_COLS = ['COUNTYFP10', 'TRACTCE10', 'BLOCKCE10']
    id_lens = [3, 6, 4]
    str_cols = [col + '_str' for col in ID_COLS]
    for col, l, col_s in zip(ID_COLS, id_lens, str_cols):
        assert max(num_digits(s) for s in df[col].unique()) <= l
        df[col_s] = df[col].astype(str).str.zfill(l)
    df['id'] = df[str_cols].astype(str).agg('-'.join, axis=1)
    for col_s in str_cols:
        del df[col_s]

def build_trees_and_inds(df):
    trees = {}
    indices = {}
    for t, a in all_num_age_pairs:
        matches = df[(df['TOTAL'] == t) & (df['18_PLUS'] == a)]
        pts = np.array([matches['INTPTLAT10'], matches['INTPTLON10']]).T
        indices[(t, a)] = {i: index for i, (index, row) in enumerate(matches.iterrows())}
        trees[(t, a)] = KDTree(pts)
    return trees, indices


def block_distance(row):
    lat1 = row['INTPTLAT10']
    lat2 = row['other_lat']
    lon1 = row['INTPTLON10']
    lon2 = row['other_lon']
    return sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def find_k_closest(row, df, k):
    t = row['TOTAL']
    a = row['18_PLUS']
    tree = trees[(t, a)]
    inds = indices[(t, a)]
    lat = row['INTPTLAT10']
    lon = row['INTPTLON10']
    if np.isnan(lat) or np.isnan(lon):
        return None
    num_to_query = k+1
    while num_to_query <= len(inds):
        dists, candidates = tree.query(np.reshape(np.array((lat, lon)), [1, 2]), num_to_query)
        cand_inds = [inds[c] for c in candidates[0]]
        cand_rows = df.loc[cand_inds].copy()
        cand_rows['distance'] = dists[0]
        if params['out_of_tract']:
            cand_rows = cand_rows[cand_rows['tractid'] != row['tractid']]
        else:
            cand_rows = cand_rows[cand_rows['distance'] != 0]
        cand_rows = cand_rows[cand_rows['swapped'] == 0]
        if len(cand_rows.index) < k:
            num_to_query *= 2
            continue
        return cand_rows.head(k)

    if params['out_of_tract']:
        cand_rows = df[(df['TOTAL'] == t) & (df['18_PLUS'] == a) & (df['swapped'] == 0) & (df['tractid'] != row['tractid'])].head(k).copy()
    else:
        cand_rows = df[(df['TOTAL'] == t) & (df['18_PLUS'] == a) & (df['swapped'] == 0) & (df['grpid'] != row['grpid'])].head(k).copy()

    if len(cand_rows) == 0:
        # Unique in the whole state
        return None

    cand_rows['other_lat'] = lat
    cand_rows['other_lon'] = lon
    cand_rows['distance'] = cand_rows.apply(block_distance, axis=1)
    return cand_rows

def flag_risk(df, flagging=['W', 'B', 'AI_AN', 'AS', 'H_PI', 'OTH', 'TWO_OR_MORE', 'NUM_HISP', 'COUNTYA', 'TRACTA']):
    dist_u = params['risk_dist']
    dist_n = {k: v*num_rows for k, v in dist_u.items()}

    if 'frequency' in df.columns:
        print('Removed previous risk ranking')
        del df['frequency']

    counts = df[flagging].groupby(flagging).size().reset_index()
    merged = df.merge(counts,
             how='left',
             on=flagging,
             validate='many_to_one',
    ).rename({0: 'frequency'}, axis=1)
    merged.sort_values(by=['frequency', 'BLOCK_TOTAL'], axis=0, inplace=True)
    vec_n = [[i] * int(dist_n[i]) for i in (4, 3, 2, 1)]
    l = []
    for v in vec_n:
        l += v
    if len(l) < num_rows:
        l += [1] * (num_rows - len(l))
    merged['U'] = l
    merged['prob'] = merged['U'].replace(params['swap_probs'])
    return merged

def get_swap_partners(df, swap_rate):
    # Only modification to df: changes the `swapped` column
    hh_1s = []
    hh_2s = []
    dists = []
    df['swapped'] = 0
    df['targeted'] = 0
    df['partnered'] = 0
    num_matches = params['num_matches']
    print('Total number of swaps', int(swap_rate*num_rows)//2)
    print('Beginning swapping...')
    ordering = pd.DataFrame(index=df.index)
    ordering['ordering'] = df['U'] + np.random.uniform(size=num_rows)
    counter = 0
    for i, _ in ordering.sort_values('ordering', ascending=False).iterrows():
        counter += 1
        row = df.loc[i]
        j = df['swapped'].sum()
        if j % 5000 == 0:
            print(j, '/', int(swap_rate*num_rows))
        if j >= num_rows*swap_rate:
            print('Stopped at:', counter, 'Number of rows:', num_rows)
            break
        if df.loc[i, 'swapped'] == 1:
            continue
        do_swap = random() < row['prob']
        if not do_swap:
            continue
        matches = find_k_closest(row, df, num_matches)
        if matches is None:
            continue
        m = matches.sample()
        partner_index = m.index[0]
        m = m.reset_index().iloc[0]
        hh_1s.extend([row['household.id']])
        hh_2s.extend([m['household.id']])
        dists.extend([m['distance']])
        assert i != partner_index
        df.loc[[i, partner_index], 'swapped'] = 1
        df.loc[i, 'targeted'] = 1
        df.loc[partner_index, 'partnered'] = 1
    partners = pd.DataFrame({'hh_1': hh_1s, 'hh_2': hh_2s, 'distance': dists})
    return partners

def finish_swap(df, pairs):
    # DOES NOT modify df
    other_pairs = pairs.copy().rename(columns={'hh_1': 'temp'})
    other_pairs = other_pairs.rename(columns={'hh_2': 'hh_1'})
    other_pairs = other_pairs.rename(columns={'temp': 'hh_2'})

    pairs_both = pairs.append(other_pairs)
    
    swapped_df = df.merge(
        pairs_both,
        left_on = 'household.id',
        right_on = 'hh_1',
        how = 'left',
        validate = 'one_to_one',
    )
    swapped_df.drop(columns=['hh_1', 'INTPTLAT10', 'INTPTLON10', 'COUNTY', 'NAME', 'COUSUBA', 'BLKGRPA', 'ACCURACY', 'AGE_ACCURACY'], inplace=True)
    swapped_df.head()
    
    swap_subset = swapped_df['hh_2'].notna()
    
    expanded = swapped_df.loc[swap_subset, 'hh_2'].str.split('-', expand=True)
    swapped_df.loc[swap_subset, 'COUNTYA'] = pd.to_numeric(expanded[0])
    swapped_df.loc[swap_subset, 'TRACTA'] = pd.to_numeric(expanded[1])
    swapped_df.loc[swap_subset, 'BLOCKA'] = pd.to_numeric(expanded[2])
    swapped_df.loc[swap_subset, 'household.id'] = swapped_df.loc[swap_subset, 'hh_2']
    swapped_df.rename({'id': 'blockid'}, inplace=True, axis=1)
    return swapped_df

def finish_swap_partial(df, pairs, mode):
    target_leaves, target_arrives, partner_leaves, partner_arrives = mode
    
    # DOES NOT modify df
    partner_pairs = pairs.copy().rename(columns={'hh_1': 'partner_hh_2'})
    partner_pairs = partner_pairs.rename(columns={'hh_2': 'partner_hh_1'})
    
    swapped_df = df.merge(
        pairs,
        left_on = 'household.id',
        right_on = 'hh_1',
        how = 'left',
        validate = 'one_to_one',
    )
    
    swapped_df = swapped_df.merge(
        partner_pairs,
        left_on = 'household.id',
        right_on = 'partner_hh_1',
        how = 'left',
        validate = 'one_to_one',
    )
    
    swapped_df.drop(columns=['hh_1', 'partner_hh_1', 'INTPTLAT10', 'INTPTLON10', 'COUNTY', 'NAME', 'COUSUBA', 'BLKGRPA', 'ACCURACY', 'AGE_ACCURACY'], inplace=True)
    swapped_df.head()
    
    target_swap_subset = swapped_df[swapped_df['hh_2'].notna()].index
    partner_swap_subset = swapped_df[swapped_df['partner_hh_2'].notna()].index
    
    print('Number of targets:', len(target_swap_subset))
    print('Number of partners:', len(partner_swap_subset))
    
    if target_leaves:
        swapped_df.loc[target_swap_subset, 'COUNTYA'] = None
        swapped_df.loc[target_swap_subset, 'TRACTA'] = None
        swapped_df.loc[target_swap_subset, 'BLOCKA'] = None
        swapped_df.loc[target_swap_subset, 'household.id'] = None
    
    if target_arrives:
        expanded = swapped_df.loc[target_swap_subset, 'hh_2'].str.split('-', expand=True)
        if target_leaves:
            swapped_df.loc[target_swap_subset, 'COUNTYA'] = pd.to_numeric(expanded[0]).astype('Int64')
            swapped_df.loc[target_swap_subset, 'TRACTA'] = pd.to_numeric(expanded[1]).astype('Int64')
            swapped_df.loc[target_swap_subset, 'BLOCKA'] = pd.to_numeric(expanded[2]).astype('Int64')
            swapped_df.loc[target_swap_subset, 'household.id'] = swapped_df.loc[target_swap_subset, 'hh_2']
        else:
            new_rows = swapped_df.loc[target_swap_subset].copy()
            new_rows['COUNTYA'] = pd.to_numeric(expanded[0]).astype('Int64')
            new_rows['TRACTA'] = pd.to_numeric(expanded[1]).astype('Int64')
            new_rows['BLOCKA'] = pd.to_numeric(expanded[2]).astype('Int64')
            new_rows['household.id'] = swapped_df.loc[target_swap_subset, 'hh_2']
            swapped_df = swapped_df.append(new_rows)
        
    if partner_leaves:
        swapped_df.loc[partner_swap_subset, 'COUNTYA'] = None
        swapped_df.loc[partner_swap_subset, 'TRACTA'] = None
        swapped_df.loc[partner_swap_subset, 'BLOCKA'] = None
        swapped_df.loc[partner_swap_subset, 'household.id'] = None
    
    if partner_arrives:
        expanded = swapped_df.loc[partner_swap_subset, 'partner_hh_2'].str.split('-', expand=True)
        if partner_leaves:
            swapped_df.loc[partner_swap_subset, 'COUNTYA'] = pd.to_numeric(expanded[0]).astype('Int64')
            swapped_df.loc[partner_swap_subset, 'TRACTA'] = pd.to_numeric(expanded[1]).astype('Int64')
            swapped_df.loc[partner_swap_subset, 'BLOCKA'] = pd.to_numeric(expanded[2]).astype('Int64')
            swapped_df.loc[partner_swap_subset, 'household.id'] = swapped_df.loc[partner_swap_subset, 'hh_2']
        else:
            new_rows = swapped_df.loc[partner_swap_subset].copy()
            new_rows['COUNTYA'] = pd.to_numeric(expanded[0]).astype('Int64')
            new_rows['TRACTA'] = pd.to_numeric(expanded[1]).astype('Int64')
            new_rows['BLOCKA'] = pd.to_numeric(expanded[2]).astype('Int64')
            new_rows['household.id'] = swapped_df.loc[target_swap_subset, 'hh_2']
            swapped_df = swapped_df.append(new_rows)
        
    swapped_df.rename({'id': 'blockid'}, inplace=True, axis=1)
    return swapped_df

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        task_name = sys.argv[1] + '_'
    else:
        task_name = ''
    swap_rate = float(sys.argv[2])
    num_runs = int(sys.argv[3])
    print('Loading data...')
    df = load_data(task_name)
   
    num_rows = len(df)
    print(num_rows, 'rows')
    params = None
    with open('swapping_params.json') as f:
        params = json.load(f)
    params['swap_probs'] = {int(k): v for k, v in params['swap_probs'].items()}

    # Set risk distribution so that all of the 4s and half of the 3s are targeted, in expectation
    # Note that swap rate has to be sufficiently small (<1/6) for this to be a valid distribution 
    if swap_rate < 1.6 / 6:
        params['risk_dist'] = {
                4: swap_rate / (params['swap_probs'][4] + params['swap_probs'][3]),
                3: 2 * swap_rate / (params['swap_probs'][4] + params['swap_probs'][3]),
                2: 3 * swap_rate / (params['swap_probs'][4] + params['swap_probs'][3]),
                1: 1 - 6 * swap_rate / (params['swap_probs'][4] + params['swap_probs'][3])
            }
    else:
        params['risk_dist'] = {
                4: swap_rate / (params['swap_probs'][4] + params['swap_probs'][3]),
                3: 2 * swap_rate / (params['swap_probs'][4] + params['swap_probs'][3]),
                2: 1 - 3 * swap_rate / (params['swap_probs'][4] + params['swap_probs'][3]),
                1: 0
            }
    
    merged = df
    del merged['identifier']
    print('Adding identifier...')
    make_identifier_synth(merged)
    make_identifier_synth(merged, ID_COLS=['COUNTYA', 'TRACTA', 'BLKGRPA'], id_lens=[3, 6, 4], name='grpid')
    make_identifier_synth(merged, ID_COLS=['COUNTYA', 'TRACTA'], id_lens=[3, 6], name='tractid')

    merged['hh_str'] = merged['HH_NUM'].astype(str).str.zfill(4)
    merged['household.id'] = merged[['id', 'hh_str']].astype(str).agg('-'.join, axis=1)
    merged['swap.partner.id'] = merged['household.id']
    del merged['hh_str']

    print('Loading shape data...')
    block_geo = load_shape_data('BLOCK')

    print('Adding geo identifier...')
    make_identifier_synth_geo(block_geo)

    merged = merged.merge(
        block_geo[['INTPTLAT10', 'INTPTLON10', 'id']],
        on='id',
        how='left',
        validate='many_to_one',
    )

    merged['INTPTLAT10'] = pd.to_numeric(merged['INTPTLAT10'])
    merged['INTPTLON10'] = pd.to_numeric(merged['INTPTLON10'])

    all_num_age_pairs = set(zip(merged['TOTAL'], merged['18_PLUS']))

    print('Building trees...')
    trees, indices = build_trees_and_inds(merged)

    merged = flag_risk(merged)

    for n in range(num_runs):
        merged_temp = merged.copy()

        partners = get_swap_partners(merged_temp, swap_rate)

        just_pairs = partners[['hh_1', 'hh_2']]
        print(len(just_pairs), 'pairs')
        print(merged_temp['swapped'].sum(), 'total swapped')

        swapped_df = finish_swap(merged_temp, just_pairs)
                
        if WRITE:
            with open(get_swapped_file(task_name, str(swap_rate), str(n)), 'w') as f:
                swapped_df.to_csv(f, index=False)