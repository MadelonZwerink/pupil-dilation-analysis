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

# %%
