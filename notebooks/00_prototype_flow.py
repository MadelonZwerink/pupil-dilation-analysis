# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 11:37:40 2025

@author: madel
"""

# %% Add the location of the module to the system paths so that it can be found

import sys
sys.path.append('C:\\Users\\madel\\OneDrive\\Documenten\\BiBC\\ADS_stage\\pupil-dilation-analysis\\pupilanalysis')

# %% Load data

from pupilanalysis.config import data_eyetracking_path
from pupilanalysis.data import read_data

dm = read_data(data_eyetracking_path)

# %% 01: Blink reconstruction: interpolating and removing missing and invalid data

from pupilanalysis.config import smooth_winlen, vt_start, vt_end, maxdur, margin, gap_margin, gap_vt, std_thr
from datamatrix import series as srs

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

# %% 02: Gaze correction

# %% 03: Downsampling

# %% 04: Baseline correction

from pupilanalysis.config import baseline_len
from pupilanalysis.custom_funcs import baseline_correction

bl = baseline_correction(dm, col='ptrace', baseline_len=baseline_len)
dm.baseline = bl[0]
dm.bl_start_index = bl[1]

from pupilanalysis.visualise import plot_baselines

plot_baselines(dm)

from pupilanalysis.custom_funcs import bl_to_zscore

dm.bl_zscore = bl_to_zscore(dm, bl_col='baseline')

# %% Does baseline change with increasing trial number?
## ADDED LATER, STILL PLACE TO CORRECT PLACE

dm.trialnr = list(range(1, 82)) * 4

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


# %% 05: Trial exclusion

from pupilanalysis.custom_funcs import perform_trial_exclusion

trial_excl = perform_trial_exclusion(dm, threshold=0.6, t_end=dm.ptrace.depth)
dm.trial_incl = trial_excl[1].tolist() * 1

for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
    print(sum(inf.trial_incl))
    print(sum(inf.trial_incl)/81)
    
# %% Overlap between excluded trials and trials that miss a baseline

import numpy as np

# Create a boolean column: True if baseline is missing, False otherwise
dm['bl_incl'] = ~np.isnan(dm.baseline) * 1

# Group by the two states
result = dm.groupby(['bl_incl', 'trial_incl']).size().reset_index(name='count')

print(result)

# %% Visualisations

from pupilanalysis.visualise import plot_pupiltrace

plot_pupiltrace(dm, by="condition", 
                show_individual_trials=True, 
                signal='ptrace',
                ymax=2000, 
                min_n_valid=10)
