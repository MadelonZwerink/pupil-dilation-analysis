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

dm.bl_zscore = bl_to_zscore(dm, col='baseline')

# %% 05: Trial exclusion

# %% Visualisations

from pupilanalysis.visualise import plot_pupiltrace

plot_pupiltrace(dm, by="condition", 
                show_individual_trials=True, 
                signal='ptrace',
                ymax=2000, 
                min_n_valid=10)
