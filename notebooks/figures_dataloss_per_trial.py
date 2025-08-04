# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 19:38:59 2025

@author: madel
"""

# %% Add the location of the module to the system paths so that it can be found

import sys
sys.path.append('C:\\Users\\madel\\OneDrive\\Documenten\\BiBC\\ADS_stage\\pupil-dilation-analysis\\pupilanalysis')

# %% Load data

from pupilanalysis.config import data_eyetracking_path
from pupilanalysis.data import read_data

dm = read_data(data_eyetracking_path)

# %% Make a new column prop_missing that contains the proportion of data missing per trial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datamatrix import operations as ops 

nr_missing = np.isnan(dm.ptrace)*1
dm.prop_missing = np.sum(nr_missing, axis=1) / dm.ptrace.depth

# %% Visualize the proportion of missing data per trial in a histogram

plt.hist(dm.prop_missing[dm.stim_grouped == 'noise'], bins=10)
plt.hist(dm.prop_missing[dm.stim_grouped == 'familiarization'], bins=10)
