# -*- coding: utf-8 -*-
"""
Created on Fri May  9 00:23:01 2025

@author: madel
"""

# %% Add the location of the module to the system paths so that it can be found

import sys
sys.path.append('C:\\Users\\madel\\OneDrive\\Documenten\\BiBC\\ADS_stage\\pupil-dilation-analysis\\pupilanalysis')

# %% Load data

from pupilanalysis.config import data_eyetracking_path
from pupilanalysis.data import read_data

dm = read_data(data_eyetracking_path)

# %% Plot raw data

from pupilanalysis.visualise import plot_pupiltrace

plot_pupiltrace(dm, by='split', signal='ptrace', show_individual_trials=True, ymin=0, ymax=[2000, 1750, 1500, 2500], title="Unprocessed pupil traces")
plot_pupiltrace(dm, by='all', signal='ptrace', show_individual_trials=True, ymin=0, ymax=2500, title="Unprocessed pupil traces", min_n_valid=500)

# %% Plot raw data grid

from pupilanalysis.visualise import plot_grid_trials

plot_grid_trials(dm, ptrace=True, bl_corrected=False)

# %% Baseline correction

from pupilanalysis.custom_funcs import baseline_correction

bl_manual = baseline_correction(dm)
dm.baseline_flex = bl_manual[0]
dm.blf_start_index = bl_manual[1]

dm.bl_ptrace = dm.ptrace - bl_manual[0]

plot_pupiltrace(dm, by='split', signal='bl_ptrace', show_individual_trials=True, ymin=-150, ymax=[400, 300, 200, 500], title="Preprocessed pupil traces")
plot_pupiltrace(dm, by='all', signal='bl_ptrace', show_individual_trials=True, ymin=-150, ymax=500, title="Preprocessed pupil traces", min_n_valid=500)
plot_pupiltrace(dm, by='all', signal='bl_ptrace', show_individual_trials=True, ymin=-150, ymax=500, title="Preprocessed pupil traces")
plot_pupiltrace(dm, by='condition_grouped', signal='bl_ptrace', show_individual_trials=True, ymin=-200, ymax=400, title="Preprocessed pupil traces for inf2 and inf3")

# Only plot participants 2 and 3
plot_pupiltrace(dm[0:242], by='all', signal='bl_ptrace', show_individual_trials=True, ymin=-200, ymax=400, title="Preprocessed pupil traces (inf5 excluded)")
plot_pupiltrace(dm[0:242], by='condition_grouped', signal='bl_ptrace', show_individual_trials=True, ymin=-200, ymax=400, title="Preprocessed pupil traces (inf5 excluded)")

#======================================================================
# %% Blink reconstruction

from datamatrix import series as srs

# Vt as default, margin divided by 2, rest divided by 4
dm.ptrace = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=10, 
                                 vt_end=5,
                                 maxdur=250, 
                                 margin=5,
                                 gap_margin=10,
                                 gap_vt=10,
                                 smooth_winlen=5,
                                 std_thr=3, 
                                 mode='advanced')

# %% Plot raw data

from pupilanalysis.visualise import plot_pupiltrace

plot_pupiltrace(dm, by='split', signal='ptrace', show_individual_trials=True, ymin=250, ymax=[1500, 1300, 1200, 2500], title="Preprocessed pupil traces")
plot_pupiltrace(dm, by='all', signal='ptrace', show_individual_trials=True, ymin=0, ymax=2500, title="Preprocessed pupil traces", min_n_valid=500)

# %% Plot raw data grid

from pupilanalysis.visualise import plot_grid_trials

plot_grid_trials(dm, ptrace=True, bl_corrected=False)

# %% Baseline correction

from pupilanalysis.custom_funcs import baseline_correction

bl_manual = baseline_correction(dm)
dm.baseline_flex = bl_manual[0]
dm.blf_start_index = bl_manual[1]

dm.bl_ptrace = dm.ptrace - bl_manual[0]

plot_pupiltrace(dm, by='split', signal='bl_ptrace', show_individual_trials=True, ymin=-150, ymax=[400, 300, 200, 500], title="Preprocessed pupil traces")
plot_pupiltrace(dm, by='all', signal='bl_ptrace', show_individual_trials=True, ymin=-150, ymax=500, title="Preprocessed pupil traces", min_n_valid=500)
plot_pupiltrace(dm, by='all', signal='bl_ptrace', show_individual_trials=True, ymin=-150, ymax=500, title="Preprocessed pupil traces")
plot_pupiltrace(dm, by='condition_grouped', signal='bl_ptrace', show_individual_trials=True, ymin=-200, ymax=400, title="Preprocessed pupil traces")

# Only plot participants 2 and 3
plot_pupiltrace(dm[0:242], by='all', signal='bl_ptrace', show_individual_trials=True, ymin=-200, ymax=400, title="Preprocessed pupil traces (inf5 excluded)")
plot_pupiltrace(dm[0:242], by='condition_grouped', signal='bl_ptrace', show_individual_trials=True, ymin=-200, ymax=400, title="Preprocessed pupil traces (inf5 excluded)")

# %% Data loss

import numpy as np

def count_valid_traces(traces, nan_threshold=1.0):
    """
    Count the number of valid traces based on allowed fraction of NaNs.

    Parameters:
        traces: array-like, shape (n_trials, timepoints)
        nan_threshold: float (default=1.0)
            Maximum allowed fraction of NaNs per trace.
            Default (1.0) removes only fully-NaN traces.

    Returns:
        valid_count: int, number of valid traces
        valid_traces: np.ndarray with only valid rows
    """
    traces = np.asarray(traces)

    # Compute fraction of NaNs per trace
    nan_frac = np.mean(np.isnan(traces), axis=1)

    # Keep rows with NaN fraction less than threshold
    mask = nan_frac < nan_threshold
    valid_traces = traces[mask]

    return len(valid_traces), valid_traces, mask

import pandas as pd
from datamatrix import operations as ops

# Define thresholds
thresholds = np.arange(0, 1.0, 0.1)

# Initialize result storage
results = []

# List of participant IDs
inf_list = ["inf2", "inf3", "inf4", "inf5"]

# Loop over thresholds
for threshold in thresholds:
    row = {"threshold": round(threshold, 2)}
    
    # Individual participants
    for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
        valid_n_inf, _, i = count_valid_traces(inf.ptrace, threshold)
        row[inf.participant[0]] = round(valid_n_inf / 81, ndigits=2)
        
    # Total DM
    valid_n_total, _, i = count_valid_traces(dm.ptrace, threshold)
    row["dm"] = round(valid_n_total / 324, ndigits=2)
    
    results.append(row)

# Convert to DataFrame
summary_df = pd.DataFrame(results)
summary_df.set_index("threshold", inplace=True)

# Show result
print(summary_df)

import matplotlib.pyplot as plt

# Plot
plt.figure(figsize=(10, 6))
for column in summary_df.columns:
    plt.plot(summary_df.index, summary_df[column], marker='o', label=column)

# Formatting
plt.title("Fraction of Valid Traces vs. NaN Threshold")
plt.xlabel("NaN Threshold")
plt.ylabel("Valid Trace Fraction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% Only use first 2.5 seconds 

# Define thresholds
thresholds = np.arange(0, 1.1, 0.1)

# Initialize result storage
results = []

# List of participant IDs
inf_list = ["inf2", "inf3", "inf4", "inf5"]

# Loop over thresholds
for threshold in thresholds:
    row = {"threshold": round(threshold, 2)}
    
    # Individual participants
    for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
        valid_n_inf, _, i = count_valid_traces(inf.ptrace[:,0:625], threshold)
        row[inf.participant[0]] = round(valid_n_inf / 81, ndigits=2)
    
    # Total DM
    valid_n_total, _, i = count_valid_traces(dm.ptrace[:,0:625], threshold)
    row["dm"] = round(valid_n_total / 324, ndigits=2)
    
    results.append(row)

# Convert to DataFrame
summary_df = pd.DataFrame(results)
summary_df.set_index("threshold", inplace=True)

# Show result
print(summary_df)

import matplotlib.pyplot as plt

# Plot
plt.figure(figsize=(10, 6))
for column in summary_df.columns:
    plt.plot(summary_df.index, summary_df[column], marker='o', label=column)

# Formatting
plt.title("Fraction of Valid Traces vs. NaN Threshold")
plt.xlabel("NaN Threshold")
plt.ylabel("Valid Trace Fraction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

valid_n, valid_traces, i = count_valid_traces(dm.ptrace, nan_threshold=0.3)
valid_n, valid_traces, i = count_valid_traces(dm.bl_ptrace, nan_threshold=0.4)
dm_filtered = dm[np.where(i)[0].tolist()]

i[243:324] = False
# Only plot participants 2 and 3
plot_pupiltrace(dm[np.where(i)[0].tolist()], by='all', signal='bl_ptrace', show_individual_trials=True, ymin=-200, ymax=400, title="Preprocessed pupil traces")
plot_pupiltrace(dm[np.where(i)[0].tolist()], by='condition_grouped', signal='bl_ptrace', show_individual_trials=True, ymin=-200, ymax=200, title="Preprocessed pupil traces")
plot_pupiltrace(dm[np.where(i)[0].tolist()], by='split', signal='bl_ptrace', show_individual_trials=True, ymin=-200, ymax=400, title="Preprocessed pupil traces")
plot_pupiltrace(dm[np.where(i)[0].tolist()], by='participant', signal='bl_ptrace', show_individual_trials=True, ymin=-200, ymax=400, title="Preprocessed pupil traces")


