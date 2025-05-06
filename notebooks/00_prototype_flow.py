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

# %% Blink reconstruction: interpolating and removing missing and invalid data

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

# %% Baseline correction

from datamatrix import multidimensional
import numpy as np
from pupilanalysis.config import start_bl, end_bl

dm.pupil = srs.baseline(
    series=dm.ptrace,
    baseline=dm.ptrace,
    bl_start=start_bl,
    bl_end=end_bl
)

dm.baseline = multidimensional.reduce(srs.window(dm.ptrace, 
                                               start=start_bl, end=end_bl),
                                    operation=np.nanmedian
                                    )
dm.baseline = dm.ptrace[:, 0] - dm.pupil[:, 0]

from pupilanalysis.visualise import plot_baselines

plot_baselines(dm)
dm.baseline

# %% Visualisations

from pupilanalysis.visualise import plot_pupiltrace

plot_pupiltrace(dm, by="condition", 
                show_individual_trials=True, 
                signal='ptrace',
                ymax=2000, 
                min_n_valid=10)
