# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:49:38 2025

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

from pupilanalysis.visualise import plot_grid_trials

plot_grid_trials(dm, manual_events=False, xtrace=True, ytrace=True, bl_corrected=False, fixations=True, fix_xy=True, auto_ymax=False, ymax=1500)

# %%

from datamatrix import operations as ops
from pupilanalysis.visualise import plot_grid_trials, plot_fixations

for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
    plot_fixations(inf)

# %%
from pupilanalysis.visualise import plot_pupiltrace

dm.dxtrace = dm.xtrace - 600
dm.dytrace = dm.ytrace - 400

for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
    plot_pupiltrace(inf, by='all', signal='dxtrace', show_individual_trials=True)
    plot_pupiltrace(inf, by='all', signal='dytrace', show_individual_trials=True)
