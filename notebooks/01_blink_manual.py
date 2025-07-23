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

# %% Use different parameters for blink reconstruction

from datamatrix import series as srs
import numpy as np

# Default parameter settings, suitable for 1000 Hz data
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

# Simply divide/multiply by 4 to make parameters suitable for 250 Hz data
dm.ptrace1 = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=40, # multiply
                                 vt_end=20, # multiply
                                 maxdur=125, # divide
                                 margin=3, # divide
                                 gap_margin=5, # divide
                                 gap_vt=40, # multiply
                                 smooth_winlen=5,  # divide
                                 std_thr=3, # keep the same
                                 mode='advanced')

# Divide/multiply by 2
dm.ptrace2 = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=20, 
                                 vt_end=10,
                                 maxdur=250, 
                                 margin=5,
                                 gap_margin=10,
                                 gap_vt=20,
                                 smooth_winlen=10,
                                 std_thr=3, 
                                 mode='advanced')

# Combination of ptrace1 and ptrace2: vt multiplied by 2, windows divided by 4
dm.ptrace3 = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=20, 
                                 vt_end=10,
                                 maxdur=250, 
                                 margin=3,
                                 gap_margin=5,
                                 gap_vt=20,
                                 smooth_winlen=5,
                                 std_thr=3, 
                                 mode='advanced')

# Vt as default, margin divided by 2, rest divided by 4
dm.ptrace4 = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=10, 
                                 vt_end=5,
                                 maxdur=250, 
                                 margin=5,
                                 gap_margin=10,
                                 gap_vt=10,
                                 smooth_winlen=5,
                                 std_thr=3, 
                                 mode='advanced')

# Best parameters from previous tuning tryout
dm.ptrace5 = srs.blinkreconstruct(dm.ptrace,
                                 vt_start=10, 
                                 vt_end=5,
                                 maxdur=200, 
                                 margin=3,
                                 gap_margin=5,
                                 gap_vt=20,
                                 smooth_winlen=3,
                                 std_thr=4, 
                                 mode='advanced')



# %% Only plot the blinks to quickly see the effect of blink parameters

from matplotlib import pyplot as plt

dm.row_i = range(len(dm))

blink_rows = manual_events_df[manual_events_df.event_type == "blink"].row_i.unique()
blink_rows = [143, 163, 166, 170, 177, 201, 207, 215, 216, 217, 230, 239]
param_sets = ['ptrace', 'ptrace0', 'ptrace1', 'ptrace2', 'ptrace3', 'ptrace4', 'ptrace5']

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
    
# %% Plot pupil traces after blink reconstruction

from pupilanalysis.visualise import plot_pupiltrace

plot_pupiltrace(dm, by='split', signal='ptrace', show_individual_trials=True, min_n_valid=100, ymax=2000, ymin=0)
plot_pupiltrace(dm, by='split', signal='ptrace0', show_individual_trials=True, min_n_valid=100, ymax=2000, ymin=0)
plot_pupiltrace(dm, by='split', signal='ptrace1', show_individual_trials=True, min_n_valid=100, ymax=2000, ymin=0)
plot_pupiltrace(dm, by='split', signal='ptrace2', show_individual_trials=True, min_n_valid=100, ymax=2000, ymin=0)
plot_pupiltrace(dm, by='split', signal='ptrace3', show_individual_trials=True, min_n_valid=100, ymax=2000, ymin=0)
plot_pupiltrace(dm, by='split', signal='ptrace4', show_individual_trials=True, min_n_valid=100, ymax=2000, ymin=0)
plot_pupiltrace(dm, by='split', signal='ptrace5', show_individual_trials=True, min_n_valid=100, ymax=2000, ymin=0)

# %% Data loss for different methods

from pupilanalysis.config import data_dir

# =============================================================================
# Data loss
# =============================================================================

# Initialize an empty list to collect rows
data = []

# Iterate through each "inf" group
for inf_label, inf in enumerate(ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5")):
    for i in range(len(inf)):
        row = {
            "inf": f"inf{inf_label+2}",  # Adjust label to match "inf2", "inf3", etc.
            "i": i,
            "ptrace": np.sum(~np.isnan(inf.ptrace[i]))/992,
            "ptrace0": np.sum(~np.isnan(inf.ptrace0[i]))/992,
            "ptrace1": np.sum(~np.isnan(inf.ptrace1[i]))/992,
            "ptrace2": np.sum(~np.isnan(inf.ptrace2[i]))/992,
            "ptrace3": np.sum(~np.isnan(inf.ptrace3[i]))/992,
            "ptrace4": np.sum(~np.isnan(inf.ptrace4[i]))/992,
            "ptrace5": np.sum(~np.isnan(inf.ptrace5[i]))/992
        }
        data.append(row)  # Append each row to the list

# Convert collected data into a DataFrame
df = pd.DataFrame(data)
df.to_csv(f"{data_dir}/processed/data_loss_blink_reconstruction.csv", index=False)

# Compute averages per 'inf'
average_per_inf = df.groupby("inf").mean(numeric_only=True).reset_index()
average_per_inf.to_csv(f"{data_dir}/processed/data_loss_blink_reconstruction_per_inf.csv", index=False)

# Compute averages per 'i' (trial)
average_per_trial = df.groupby("i").mean(numeric_only=True).reset_index()
average_per_trial.to_csv(f"{data_dir}/processed/data_loss_blink_reconstruction_per_trial.csv", index=False)

# Display results
print("Average per 'inf':")
print(average_per_inf)

print("\nAverage per 'i' (trial):")
print(average_per_trial)




