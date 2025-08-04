# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:34:49 2025

@author: madel
"""

# %% Add the location of the module to the system paths so that it can be found

import sys
sys.path.append('C:\\Users\\madel\\OneDrive\\Documenten\\BiBC\\ADS_stage\\pupil-dilation-analysis\\pupilanalysis')

# %% Load data

from pupilanalysis.config import data_eyetracking_path
from pupilanalysis.data import read_data

dm = read_data(data_eyetracking_path)

# %%

from pupilanalysis.custom_funcs import perform_trial_exclusion
import pandas as pd
from datamatrix import operations as ops 

# %% Perform trial exclusion with threshold to 0.6 so that all trials that are 
# excluded, are set to 0, all others are set to 1
trial_excl = perform_trial_exclusion(dm, threshold=0.6, t_end=dm.ptrace.depth)
dm.trial_incl = trial_excl[1].tolist() * 1

# Split the dataset into different segments
# Set 1: noise_1 to noise_9
noise_set = {f"noise_{i}" for i in range(1, 10)}
noise_trials = dm.trialid == noise_set

# Set 2: familiarization_1 to familiarization_24
fam1_24_set = {f"familiarization_{i}" for i in range(1, 25)}
fam1 = dm.trialid == fam1_24_set

# Set 3: familiarization_25 to familiarization_48
fam25_48_set = {f"familiarization_{i}" for i in range(25, 49)}
fam2 = dm.trialid == fam25_48_set

# Set 4: familiarization_49 to familiarization_72
fam49_72_set = {f"familiarization_{i}" for i in range(49, 73)}
fam3 = dm.trialid == fam49_72_set

# Loop over all segments and participants to calculate the number of missing
# trials
        
# Define row and column labels
rows = ['total', 'noise', 'fam1', 'fam2', 'fam3']
cols = ['total', 'inf2', 'inf3', 'inf4', 'inf5']

# Define your subsets (must align with 'rows')
subsets = [dm, noise_trials, fam1, fam2, fam3]

# Initialize results dictionary
results = {col: [] for col in cols}

# Loop over subsets
for subset in subsets:
    # --- Total across all participants in subset ---
    total_trials = len(subset)
    print(total_trials)
    included = sum(subset.trial_incl)
    excluded = total_trials - included
    included_pct = round(included / total_trials * 100, 1)
    results['total'].append(f"{included} ({included_pct}%)")

    # --- For each participant group ---
    for inf_subset in ops.split(subset.participant, "inf2", "inf3", "inf4", "inf5"):
        inf_label = inf_subset.participant[0]
        if len(inf_subset.participant) > 0:
            n_trials = len(inf_subset)
            print(n_trials)
            included = sum(inf_subset.trial_incl)
            excluded = n_trials - included
            included_pct = round(included / n_trials * 100, 1)
            results[inf_label].append(f"{included} ({included_pct}%)")
        else:
            results[inf_label].append("0 (0.0%)")  # or NaN
            
selected_trials_df = pd.DataFrame(results, index=rows)
selected_trials_df