# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:13:16 2025

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

from datamatrix import series as srs

dm.ptrace = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=10, 
                                 vt_end=5,
                                 maxdur=200, 
                                 margin=3,
                                 gap_margin=5,
                                 gap_vt=20,
                                 smooth_winlen=3,
                                 std_thr=4, 
                                 mode='advanced')

# %% Baseline correction

from datamatrix import multidimensional
import numpy as np

dm.pupil = srs.baseline(
    series=dm.ptrace,
    baseline=dm.ptrace,
    bl_start=0,
    bl_end=10
)

dm.baseline = multidimensional.reduce(srs.window(dm.ptrace, 
                                               start=0, end=10),
                                    operation=np.nanmedian
                                    )
dm.baseline = dm.ptrace[:, 0] - dm.pupil[:, 0]

from pupilanalysis.visualise import plot_baselines

plot_baselines(dm)
dm.baseline

# %% Visualisations

from pupilanalysis.visualise import plot_pupiltrace
from datamatrix import operations as ops

for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
    ymax = np.nanmax(inf.pupil.std*2)
    ymax2 = np.nanmax(inf.ptrace.mean*2)
    
    plot_pupiltrace(inf, by="all", 
                    show_individual_trials=True, 
                    signal='pupil',
                    ymax=ymax, 
                    ymin=-ymax,
                    min_n_valid=1000)
    
    plot_pupiltrace(inf, by="all", 
                    show_individual_trials=True, 
                    signal='ptrace',
                    ymax=ymax2, 
                    ymin=0,
                    min_n_valid=1000)
    
for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
    ymax = np.nanmax(inf.pupil.std*2)
    ymax2 = np.nanmax(inf.ptrace.mean*2)
    
    plot_pupiltrace(inf, by="all", 
                    show_individual_trials=True, 
                    signal='pupil',
                    ymax=ymax, 
                    ymin=-ymax,
                    min_n_valid=10)
    
    plot_pupiltrace(inf, by="all", 
                    show_individual_trials=True, 
                    signal='ptrace',
                    ymax=ymax2, 
                    ymin=0,
                    min_n_valid=10)
    
plot_pupiltrace(dm, by="all", 
                show_individual_trials=True, 
                signal='pupil',
                ymax=400, 
                ymin=-400,
                min_n_valid=1000)

plot_pupiltrace(dm, by="all", 
                show_individual_trials=True, 
                signal='ptrace',
                ymax=2500, 
                ymin=0,
                min_n_valid=1000)

plot_pupiltrace(dm, by="participant", 
                show_individual_trials=True, 
                signal='pupil',
                ymax=400, 
                ymin=-400)

plot_pupiltrace(dm, by="participant", 
                show_individual_trials=True, 
                signal='ptrace',
                ymax=2500, 
                ymin=0)

plot_pupiltrace(dm, by="condition", 
                show_individual_trials=True, 
                signal='pupil',
                ymax=400, 
                ymin=-400)

plot_pupiltrace(dm, by="condition", 
                show_individual_trials=True, 
                signal='ptrace',
                ymax=2500, 
                ymin=0)