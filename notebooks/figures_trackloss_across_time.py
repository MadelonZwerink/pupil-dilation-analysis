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
import matplotlib.pyplot as plt

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
    ax.bar(x=range(0, 4*len(missing_data), 4), height=missing_data, width=3.5)    
    ax.set_title(participant)
plt.tight_layout(rect=[0.04, 0.04, 1, 1])
fig.supxlabel('Time (ms)')
fig.supylabel('Number of missing (count)')
plt.show()
    
