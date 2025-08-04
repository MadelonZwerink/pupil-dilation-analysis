# -*- coding: utf-8 -*-
"""
Created on Tue May  6 20:08:27 2025

@author: madel
"""
# %%

from pupilanalysis.config import data_eyetracking_path
from pupilanalysis.data import read_data

dm = read_data(data_eyetracking_path)

# %% Standard function from eyelinkparser

from datamatrix import multidimensional, series as srs
import numpy as np

dm.baseline = multidimensional.reduce(srs.window(dm.ptrace, 
                                               start=0, end=10),
                                    operation=np.mean
                                    )

# Number of nan-baselines
print(f"Number of set baselines: {len(dm.baseline) - np.sum(np.isnan(dm.baseline))}")

# %% Custom function that searches for first 40 ms stretch of non-nan values 
# within the first 200 ms

from pupilanalysis.custom_funcs import baseline_correction

bl_manual = baseline_correction(dm, window_len=900)
dm.baseline_flex = bl_manual[0]
dm.blf_start_index = bl_manual[1]

from matplotlib import pyplot as plt
plt.scatter(range(len(dm)), bl_manual[2], c=range(len(dm)))
plt.show()

plt.scatter(range(len(dm)), bl_manual[3], c=range(len(dm)))
plt.show()

plt.scatter(bl_manual[3], bl_manual[2], c=range(len(dm)))
plt.show()

import pandas as pd
pd.value_counts(bl_manual[1], bins=[0,1,50,100,1000])

bl_start_index = np.array(bl_manual[1])
indices_late_start = np.where(bl_start_index > 50)
bl_percent_empty_total = np.array(bl_manual[3])

plt.scatter(indices_late_start, bl_percent_empty_total[indices_late_start], c=bl_start_index[indices_late_start])

bl_start_index = np.array(bl_manual[1])
indices_late_start = np.where(bl_start_index > 100)
bl_percent_empty_total = np.array(bl_manual[3])

plt.scatter(indices_late_start, bl_percent_empty_total[indices_late_start], c=bl_start_index[indices_late_start])

# %% Visualize baselines for both methods

from pupilanalysis.visualise import plot_baselines

plot_baselines(dm)
plot_baselines(dm, col='baseline_flex')

# %% Visualize estimated baseline per row, indicate difference between two 
# methods by vertical lines

from pupilanalysis.visualise import plot_compare_baselines

plot_compare_baselines(dm, 'baseline', "baseline_flex")

# %% 
# =============================================================================
# Now I would like to try these on the blink-corrected data, ptrace4
# =============================================================================
# %% Standard function from eyelinkparser

from datamatrix import multidimensional, series as srs
import numpy as np

dm.baseline = multidimensional.reduce(srs.window(dm.ptrace4, 
                                               start=0, end=10),
                                    operation=np.mean
                                    )

# Number of nan-baselines
print(f"Number of set baselines: {len(dm.baseline) - np.sum(np.isnan(dm.baseline))}")

# %% Custom function that searches for first 40 ms stretch of non-nan values 
# within the first 200 ms

from pupilanalysis.custom_funcs import baseline_correction

bl_manual = baseline_correction(dm, col='ptrace4')
dm.baseline_flex = bl_manual[0]
dm.blf_start_index = bl_manual[1]

# %% Visualize baselines for both methods

from pupilanalysis.visualise import plot_baselines

plot_baselines(dm)
plot_baselines(dm, col='baseline_flex')

# %% Visualize estimated baseline per row, indicate difference between two 
# methods by vertical lines

from pupilanalysis.visualise import plot_compare_baselines

plot_compare_baselines(dm, 'baseline', "baseline_flex")

# %% 
# =============================================================================
# I would like to see the effect of different lengths of baseline periods
# =============================================================================

# Conclusion: Reducing the window length to 2 leads to 143/146 baselines in
# comparison to 137/140 baselines for a window length of 9. It seems like most
# baselines that are added when choosing a shorter baseline window are outliers
# Therefore, reducing the window length is not a feasible method to increase 
# the number of estimated baselines.

from pupilanalysis.visualise import plot_baselines, plot_compare_baselines
from pupilanalysis.custom_funcs import baseline_correction
  
for baseline_len in range(2,11):  
    print(f"Baseline window length: {baseline_len}")
    dm.baseline = multidimensional.reduce(srs.window(dm.ptrace, 
                                                   start=0, end=baseline_len),
                                        operation=np.mean
                                        )
    
    # Number of nan-baselines
    print(f"Number of set baselines: {len(dm.baseline) - np.sum(np.isnan(dm.baseline))}")
    
    bl_manual = baseline_correction(dm, col='ptrace', baseline_len=baseline_len)
    dm.baseline_flex = bl_manual[0]
    dm.blf_start_index = bl_manual[1]
    
    plot_baselines(dm, title=f'Baselines (window: {baseline_len})')
    plot_baselines(dm, col='baseline_flex', title=f'Baselines (flexible window: {baseline_len})')
    
    plot_compare_baselines(dm, 'baseline', "baseline_flex")
    
# %% Converting baselines to z-scores

# =============================================================================
# Method 1
# =============================================================================
import numpy as np

z_scores = []

# Get a list of unique participants
participants = set(dm.participant)

# Loop over each row in dm
for row in dm:
    participant = row.participant
    # Get all baseline values for this participant
    values = [r.baseline for r in dm if r.participant == participant]
    values = np.array(values, dtype=np.float64)

    # Calculate mean and std, ignoring NaNs
    mean = np.nanmean(values)
    std = np.nanstd(values)

    # Compute z-score
    if np.isnan(row.baseline) or std == 0:
        z = np.nan  # Avoid division by zero or nan inputs
    else:
        z = (row.baseline - mean) / std

    z_scores.append(z)

# =============================================================================
# Method 2
# =============================================================================

from scipy.stats import zscore
from datamatrix import operations as ops

z_scores_2 = []

for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
    z = zscore(inf.baseline, nan_policy='omit')
    print(np.nanmean(inf.baseline))
    z_scores_2.append(z.flatten())

z_scores_2 = np.concatenate(z_scores_2)

# =============================================================================
# Check if methods are identical: yes they are!
# =============================================================================

z_scores - z_scores_2
np.nansum(z_scores - z_scores_2)
    
# Assign to a new column
dm.baselines_z_1 = z_scores
dm.baselines_z_2 = z_scores_2
plot_baselines(dm, col='baselines_z_1', title='Z-scores 1')
plot_baselines(dm, col='baselines_z_2', title='Z-scores 2')

# %% Does baseline change with increasing trial number?

import matplotlib.pyplot as plt

# Assuming `dm` is your DataFrame
# Group by trialnr and compute mean and SEM (standard error of the mean)
grouped = dm.groupby('trialnr')['baseline'].agg(['mean', 'sem']).reset_index()

# Plot mean baseline per trial with error bars
plt.errorbar(grouped['trialnr'], grouped['mean'], yerr=grouped['sem'], fmt='o', capsize=3)

plt.xlabel('Trial Number')
plt.ylabel('Average Baseline')
plt.title('Baseline per Trial with Error Bars')
plt.grid(True)
plt.show()

from datamatrix import operations as ops
import seaborn as sns

for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
    plt.scatter(inf.trialnr, inf.baseline)
    sns.regplot(x='trialnr', y='baseline', data=inf)
    plt.title(f'{inf.participant[1]}')
    plt.show()
    # What happens if we only include the first two segments?
    inf_short = inf[0:57]
    plt.scatter(inf_short.trialnr, inf_short.baseline)
    sns.regplot(x='trialnr', y='baseline', data=inf_short)
    plt.title(f'{inf_short.participant[1]}')
    plt.show()