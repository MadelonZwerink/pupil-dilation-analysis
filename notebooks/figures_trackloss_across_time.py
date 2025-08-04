# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 17:54:45 2025

@author: madel
"""

# %% Add the location of the module to the system paths so that it can be found

import sys
sys.path.append('C:\\Users\\madel\\OneDrive\\Documenten\\BiBC\\ADS_stage\\pupil-dilation-analysis\\pupilanalysis')

# %% Load data

from pupilanalysis.config import data_eyetracking_path
from pupilanalysis.data import read_data

dm = read_data(data_eyetracking_path)

# %% Visual inspection of trackloss across time by participants

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datamatrix import operations as ops 

cols = ['inf2', 'inf3', 'inf4', 'inf5']

# Initialize results dictionary
results = {col: [] for col in cols}

for inf_subset in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
    inf_label = inf_subset.participant[0]
    missing_timepoints = [0]*992
    for row in inf_subset:
        missing_timepoints += np.isnan(row.ptrace) * 1
    results[inf_label] = missing_timepoints
    
missing_ptrace = pd.DataFrame(results)

# Create 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

for i, participant in enumerate(cols):
    ax = axes[i // 2, i % 2]  # Determine the axis to plot on (2x2 grid)
    # Only select data until row 987, because the last timepoints are mising in 
    # most trials, because they are not EXACTLY 4 seconds, but usually a few ms
    # shorter
    missing_data = getattr(missing_ptrace[0:987], participant)
    missing_perc = round((sum(missing_data) / (81*len(missing_data))) * 100, 2)
    print(f"{participant} - percentage of missing data: {missing_perc}%")
    ax.bar(x=range(0, 4*len(missing_data), 4), height=missing_data, width=3.5)    
    ax.set_title(participant)
plt.tight_layout(rect=[0.04, 0.04, 1, 1])
fig.supxlabel('Time (ms)')
fig.supylabel('Number of missing (count)')
plt.show()
    
# inf2 - percentage of missing data: 53.58%
# inf3 - percentage of missing data: 41.2%
# inf4 - percentage of missing data: 58.82%
# inf5 - percentage of missing data: 66.79%

# %% Trackloss for the different segments

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

# Define your subsets (must align with 'rows')
subsets = [dm, noise_trials, fam1, fam2, fam3]

for subset in subsets:
    # Initialize results dictionary
    results = {col: [] for col in cols}

    for inf_subset in ops.split(subset.participant, "inf2", "inf3", "inf4", "inf5"):
        inf_label = inf_subset.participant[0]
        missing_timepoints = [0]*992
        for row in inf_subset:
            missing_timepoints += np.isnan(row.ptrace) * 1
        results[inf_label] = missing_timepoints
        
    missing_ptrace = pd.DataFrame(results)

    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    for i, participant in enumerate(cols):
        ax = axes[i // 2, i % 2]  # Determine the axis to plot on (2x2 grid)
        # Only select data until row 987, because the last timepoints are mising in 
        # most trials, because they are not EXACTLY 4 seconds, but usually a few ms
        # shorter
        missing_data = getattr(missing_ptrace[0:987], participant)
        missing_perc = round((sum(missing_data) / (len(subset[subset.participant == participant])*len(missing_data))) * 100, 2)
        print(f"{participant} - percentage of missing data: {missing_perc}%")
        ax.bar(x=range(0, 4*len(missing_data), 4), height=missing_data, width=3.5)    
        ax.set_title(f"{participant} ({missing_perc}% missing)")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.ylim(0, len(subset[subset.participant == participant]))
    fig.supxlabel('Time (ms)')
    fig.supylabel('Number of missing (count)')
    fig.suptitle(f"Trackloss for {subset.trialid[0]} to {subset.trialid[-1]}", fontsize=15, fontweight='bold')
    plt.show()

