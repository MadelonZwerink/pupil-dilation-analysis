# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:06:41 2025

@author: madel
"""

# %% Code to change the time trace (to make the trace continuous)
corrected_ttrace = []
for row in dm:
    corrected = np.array(row['ttrace'] + row['cor_t_onset'])
    corrected_ttrace.append(corrected) 
    
dm.cor_ttrace = SeriesColumn(depth = 992)
dm.cor_ttrace = corrected_ttrace

# %% Optional print statements to check output blink

from datamatrix import operations as ops

for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
    #print(inf.participant[1])
    #print(min(inf.t_onset)) 
    #print(inf.t_onset[1:] - inf.t_offset[:-1])
    #print(inf.t_offset - inf.t_onset)
    print(inf.t_onset[1:] - inf.t_offset[:-1] + inf.t_offset[0:-1] - inf.t_onset[0:-1])
    #print(inf.t_onset - min(inf.t_onset))

# --- Optional: Print to verify ---
print("Sample output:")
for i, row in enumerate(dm[:10]):
    print(f"Row {i}:")
    print(f"  Blinks: {row.video_blink}")
    print(f"  Lookaway start: {row.video_lookaway_start}")
    print(f"  Lookaway end: {row.video_lookaway_end}")

# %% Custom algorithm that searches first 200ms for first 40ms of non-nan values
# Written as function baseline_correction in pupilanalysis.custom_funcs

import pandas as pd

mean_values = []
start_indices = []

for row in dm:
    signal = row.ptrace[:50]  # first 50 values only
    found = False

    for i in range(41):  # up to index 40 (inclusive) so i:i+10 is within bounds
        window = signal[i:i+10]
        if np.all(~np.isnan(window)):
            mean_values.append(np.mean(window))
            start_indices.append(i)
            found = True
            break

    if not found:
        mean_values.append(np.nan)
        start_indices.append(np.nan)

dm.baseline_flex = mean_values
dm.blf_start_index = start_indices

# Number of nan-baselines        
np.sum(np.isnan(mean_values))

# Start indices of baseline period
pd.value_counts(np.array(start_indices))