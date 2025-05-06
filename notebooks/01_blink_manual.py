# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:16:10 2025

@author: madel
"""
# %% Parse the data

from pupilanalysis.config import data_eyetracking_path
from pupilanalysis.data import read_data

dm = read_data(data_eyetracking_path)

# %% Read the data on events that was analysed manually for comparison

from pupilanalysis.manual_events import read_manual_events, process_video_events, correct_times_to_relative, create_event_df

eyetracking_manual = read_manual_events()
dm = correct_times_to_relative(dm)
dm = process_video_events(dm, eyetracking_manual)
manual_events_df = create_event_df(dm, eyetracking_manual)

# %% Visualize data before blink reconstruction

from pupilanalysis.visualise import plot_grid_trials

plot_grid_trials(dm, manual_events=True, ptrace=True, bl_corrected=False)

# %%

from datamatrix import series as srs
import numpy as np

dm.ptrace0 = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=10, 
                                 vt_end=5,
                                 maxdur=500, 
                                 margin=10,
                                 gap_margin=20,
                                 gap_vt=10,
                                 smooth_winlen=21,
                                 std_thr=3, 
                                 mode='advanced')

dm.ptrace1 = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=10, 
                                 vt_end=5,
                                 maxdur=125, 
                                 margin=3,
                                 gap_margin=5,
                                 gap_vt=10,
                                 smooth_winlen=7,
                                 std_thr=3, 
                                 mode='advanced')

dm.ptrace2 = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=40, 
                                 vt_end=20,
                                 maxdur=125, 
                                 margin=3,
                                 gap_margin=5,
                                 gap_vt=40,
                                 smooth_winlen=7,
                                 std_thr=3, 
                                 mode='advanced')

dm.ptrace3 = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=10, 
                                 vt_end=5,
                                 maxdur=200, 
                                 margin=3,
                                 gap_margin=5,
                                 gap_vt=20,
                                 smooth_winlen=3,
                                 std_thr=4, 
                                 mode='advanced')

dm.ptrace4 = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=20, 
                                 vt_end=10,
                                 maxdur=200, 
                                 margin=4,
                                 gap_margin=5,
                                 gap_vt=40,
                                 smooth_winlen=5,
                                 std_thr=4, 
                                 mode='advanced')

# %% Only plot the blinks to quickly see the effect of blink parameters
from matplotlib import pyplot as plt, lines

dm.row_i = range(len(dm))

blink_rows = manual_events_df[manual_events_df.event_type == "blink"].row_i.unique()
blink_rows = [246, 247, 264, 143, 163, 166, 170, 177, 201, 207, 215, 216, 217, 230, 239]
param_sets = ['ptrace', 'ptrace0', 'ptrace1', 'ptrace2', 'ptrace3', 'ptrace4']

fig, axs = plt.subplots(len(param_sets), len(blink_rows), figsize=(40, 20), sharex=True, sharey=True)
ymax = np.nanmax(dm.ptrace.mean*2.5)
ymin = 0

for j, row_nr in enumerate(blink_rows): 
    video_bl = (dm.video_blink[dm.row_i == row_nr, :] - dm.cor_t_onset[dm.row_i == row_nr]) / 4
    lookaway_start = (dm.video_lookaway_start[dm.row_i == row_nr, :] - dm.cor_t_onset[dm.row_i == row_nr]) / 4
    lookaway_end = (dm.video_lookaway_end[dm.row_i == row_nr, :] - dm.cor_t_onset[dm.row_i == row_nr]) / 4
    
    for i, params in enumerate(param_sets):
        axs[i,j].set_ylim(ymin, ymax)
        axs[i,j].set_xlim(0, 992)
        axs[i,j].set_xticks([0, 250, 500, 750], labels=[0, 1, 2, 3])
        
        axs[i,j].plot(np.arange(992), 
                      np.array(getattr(dm, params)[row_nr, :]), 
                      label = f"{params}",
                      alpha = 0.8)
        # Add manual blinks
        axs[i,j].vlines(video_bl, ymin, ymax, colors='red', linestyles='dashed', alpha=0.7, label = "manual blink")
        
        # Mask out NaNs and loop over valid lookaway events
        for start, end in zip(lookaway_start, lookaway_end):
            if not np.isnan(start) and not np.isnan(end):
                axs[i,j].axvspan(start, end, alpha=0.3, color='gray')
                
        axs[i,j].set_title(f"Row: {row_nr}\tParam: {params}")
    





