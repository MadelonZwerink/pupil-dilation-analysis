# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 22:22:43 2025

@author: madel
"""

# %% 1. Preparation
# =============================================================================
# =============================================================================

# %% Import required modules and functions

import pandas as pd
import numpy as np
from datamatrix import operations as ops
from datamatrix import series as srs
from datamatrix import SeriesColumn, FloatColumn, NAN

# 2. Importing data
import sys
sys.path.append('C:\\Users\\madel\\OneDrive\\Documenten\\BiBC\\ADS_stage\\pupil-dilation-analysis\\pupilanalysis')
from pupilanalysis.config import data_eyetracking_path
from pupilanalysis.data import read_data

# 3. Data preprocessing
from pupilanalysis.config import smooth_winlen, vt_start, vt_end, maxdur, margin, gap_margin, gap_vt, std_thr

from pupilanalysis.config import baseline_len, window_len
from pupilanalysis.custom_funcs import baseline_correction

# 4. Visualizing
from matplotlib import pyplot as plt
from pupilanalysis.visualise import plot_pupiltrace, plot_baselines, plot_fixations, plot_coordinates
from pupilanalysis.custom_funcs import bl_to_zscore, perform_trial_exclusion

# %% 2. Importing data
# =============================================================================
# =============================================================================

dm = read_data(data_eyetracking_path)

# %% 3. Data preprocessing
# =============================================================================
# =============================================================================

# %% 3.1 Blink reconstruction and smoothing

dm.ptraceraw = dm.ptrace

dm.ptrace = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=vt_start, 
                                 vt_end=vt_end,
                                 maxdur=maxdur, 
                                 margin=margin,
                                 gap_margin=gap_margin,
                                 gap_vt=gap_vt,
                                 smooth_winlen=smooth_winlen,
                                 std_thr=std_thr, 
                                 mode='advanced')

# %% 3.2 Baseline correction

bl = baseline_correction(dm, col='ptrace', baseline_len=baseline_len, window_len=window_len)
dm.baseline = bl[0]
dm.bl_start_index = bl[1]
dm.pupil = SeriesColumn(depth = dm.ptrace.depth)
dm.pupil = dm.ptrace - bl[0]

# Transform baseline to z-scores
dm.bl_zscore = FloatColumn
dm.bl_zscore = bl_to_zscore(dm, bl_col='baseline')

# %% 4. Data sampling - examine data quality
# =============================================================================
# =============================================================================

# %% 4.1 Plot average pupil response for each participant

plot_pupiltrace(dm, by='split', show_individual_trials=False, signal='pupil', title='')

plot_pupiltrace(dm.participant != 'inf5', by='participant', show_individual_trials=False, signal='pupil', title='')

# %% 4.2 Plot all traces per participant

plot_pupiltrace(dm, by='split', show_individual_trials=True, signal='pupil', min_n_valid=100, title='')

# %% 4.3 Plot histogram for pupil size during baseline period

plot_baselines(dm, col='baseline', title='')

plot_baselines(dm, col='bl_zscore', title='')

# %% 4.4 Plot fixation positions

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

participants = ["inf2", "inf3", "inf4", "inf5"]
colors = ["green", "blue", "purple", "pink"]
groups = ops.split(dm.participant, *participants)

for ax, group, participant, color in zip(axes, groups, participants, colors):
    ax.plt = plot_fixations(group, ax, plot_type='heatmap', title=participant)
    
fig.supxlabel("X-coordinate", x=0.53)
fig.supylabel("Y-coordinate")
plt.tight_layout(rect=[0, 0, 1, 1])

# %% 4.4 Plot gaze position

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

participants = ["inf2", "inf3", "inf4", "inf5"]
colors = ["green", "blue", "purple", "pink"]
groups = ops.split(dm.participant, *participants)

for ax, group, participant, color in zip(axes, groups, participants, colors):
    ax.plt = plot_coordinates(group, ax, plot_type='heatmap', title=participant)
    
fig.supxlabel("X-coordinate", x = 0.53)
fig.supylabel("Y-coordinate")
plt.tight_layout(rect=[0, 0, 1, 1])

# %% 4.5 Plot gaze position over time, average

dm.dxtrace = dm.xtrace - 600
dm.abs_dxtrace = dm.dxtrace @ (lambda x: abs(x))
dm.dytrace = dm.ytrace - 400
dm.abs_dytrace = dm.dytrace @ (lambda x: abs(x))

plot_pupiltrace(dm, by='condition_grouped', signal='abs_dxtrace', show_individual_trials=False, ymin=0, ymax=600, ylab='Horizontal gaze deviation (pixels)')
plot_pupiltrace(dm, by='condition_grouped', signal='abs_dytrace', show_individual_trials=False, ymin=0, ymax=400, ylab='Vertical gaze deviation (pixels)')

# %% 5. Trial exclusion
# =============================================================================
# =============================================================================

# %% 5.1 Remove trials based on pupil size during baseline (Z-scores)

# Create a boolean column: 0 if baseline is missing or abs(Z-score) > 2,
# if a baseline is succesfully calculated and not an outlier, bl_incl = 1
# NAN works as a general selector in datamatrix for FloatColumns specifically
dm.bl_incl = 1
dm.bl_incl[(dm.bl_zscore > 2.0) | (dm.bl_zscore < -2.0) | (dm.bl_zscore == NAN)] = 0

# %% 5.2 Remove trials based on missing data (trials with >70% missing data are excluded)

#trial_excl = perform_trial_exclusion(dm, threshold=0.7, t_end=dm.pupil.depth, col='ptrace')
#dm.trial_incl = trial_excl[1].tolist() * 1

# %% 5.3 Combine both baseline based exclusion and missing data based exclusion

#dm.include = 0
#dm.include[(dm.bl_incl == 1) & (dm.trial_incl == 1)] = 1

# %% Save the dataset

from datamatrix import convert as cnv

dm_json = cnv.to_json(dm)
# Replace NaN and Infinity with null in the JSON string
dm_json_clean = dm_json.replace("NaN", "null").replace("Infinity", "null")

with open('C:/Users/madel/OneDrive/Documenten/BiBC/ADS_stage/pupil-dilation-analysis/data/processed/dm_raw.json', 'w', encoding='utf-8') as f:
    f.write(dm_json_clean)