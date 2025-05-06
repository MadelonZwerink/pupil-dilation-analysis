# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:50:24 2025

@author: madel
"""

from datamatrix import DataMatrix, MultiDimensionalColumn, SeriesColumn, NAN, series as srs, operations as ops
import pandas as pd
import numpy as np

# %% read_manual_events

def read_manual_events(relative_path = 'processed/eyetracking_events_manual.csv'):
    from pupilanalysis.config import data_dir
    import pandas as pd

    eyetracking_manual = pd.read_csv(data_dir / relative_path, sep=";")
    
    return(eyetracking_manual)

# %% correct_times_to_relative

def correct_times_to_relative(dm):
    corrected_time_onset = []
    corrected_time_offset = []
    for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
        corrected_time_onset.extend(inf.t_onset - min(inf.t_onset))
        corrected_time_offset.extend(inf.t_offset - min(inf.t_onset))

    dm.cor_t_onset = corrected_time_onset
    dm.cor_t_offset = corrected_time_offset
    return dm
    
# %% match_trials
# --- Helper function to match trials based on time and participant ---
def match_trials(event_row, dm):
    matches = []
    for row in dm:
        if row.participant == str(event_row['participant']):
            if row.cor_t_onset <= event_row['time_start'] <= row.cor_t_offset:
                matches.append(row)
    return matches

# %% process manually added events based on video data

def process_video_events(dm, eyetracking_manual):
    
    # --- Initialize MultiDimensionalColumns with variable-length dimension ---
    dm.video_blink = SeriesColumn(depth=3)
    dm.video_lookaway_start = SeriesColumn(depth=3)
    dm.video_lookaway_end = SeriesColumn(depth=3)

    # --- Iterate through the eyetracking events ---
    for _, event in eyetracking_manual.iterrows():
        participant = str(event['participant'])
        start_time = event['time_start']
        end_time = event['time_end']

        if event['event'] == 'blink':
            matched_rows = match_trials(event, dm)
            for row in matched_rows:
                row.video_blink[np.min(np.where(np.isnan(row.video_blink)))] = start_time

        elif event['event'] == 'look away' and pd.notna(end_time):
            for row in dm:
                if str(row.participant) != participant:
                    continue
                if row.cor_t_offset < start_time or row.cor_t_onset > end_time:
                    continue

                # print(f"Match found for trial with onset {row.cor_t_onset}, offset {row.cor_t_offset}")
                
                if row.cor_t_onset <= start_time <= row.cor_t_offset:
                    row.video_lookaway_start[np.min(np.where(np.isnan(row.video_lookaway_start)))] = start_time
                    if end_time <= row.cor_t_offset:
                        row.video_lookaway_end[np.min(np.where(np.isnan(row.video_lookaway_end)))] = end_time
                    else:
                        row.video_lookaway_end[np.min(np.where(np.isnan(row.video_lookaway_end)))] = row.cor_t_offset

                elif start_time <= row.cor_t_onset < end_time:
                    row.video_lookaway_start[np.min(np.where(np.isnan(row.video_lookaway_start)))] = row.cor_t_onset
                    if end_time <= row.cor_t_offset:
                        row.video_lookaway_end[np.min(np.where(np.isnan(row.video_lookaway_end)))] = end_time
                    else:
                        row.video_lookaway_end[np.min(np.where(np.isnan(row.video_lookaway_end)))] = row.cor_t_offset
                        
    return dm

# %% create event dataframe
# Contains the same information as eyetracking manual, and the added rows from
# process_video_events, but in a separate dataframe, with trial per event

def create_event_df(dm, eyetracking_manual):
    event_log = []

    for _, event in eyetracking_manual.iterrows():
        part = event['participant']
        start = event['time_start']
        end = event['time_end'] if not pd.isna(event['time_end']) else None
        etype = event['event']

        for i, row in enumerate(dm):
            if row.participant != part:
                continue
            trial_start = row.cor_t_onset
            trial_end = row.cor_t_offset

            if trial_start <= start <= trial_end:
                # Blink or start of look away
                if etype == 'blink':
                    event_log.append({
                        'participant': part,
                        'row_i': i,
                        'trial_id': dm.trialid[i],
                        'event_type': 'blink',
                        'event_start': start,
                        'event_end': None
                    })
                elif etype == 'look away':
                    if end is None:
                        continue  # Skip invalid
                    # Add first trial
                    event_log.append({
                        'participant': part,
                        'row_i': i,
                        'trial_id': dm.trialid[i],
                        'event_type': 'look away',
                        'event_start': start,
                        'event_end': min(end, trial_end)
                    })

                    # Add continuation rows if spans multiple trials
                    next_time = trial_end
                    while end > next_time:
                        i += 1
                        if i >= len(dm):
                            break
                        next_row = dm[i]
                        if next_row.participant != part:
                            break
                        new_start = next_row.cor_t_onset
                        new_end = next_row.cor_t_offset
                        if new_start >= end:
                            break
                        event_log.append({
                            'participant': part,
                            'row_i': i,
                            'trial_id': dm.trialid[i],
                            'event_type': 'look away',
                            'event_start': max(new_start, start),
                            'event_end': min(end, new_end)
                        })
                        next_time = new_end
                break  # Only assign once per event

    # Convert to DataFrame or DataMatrix
    event_df = pd.DataFrame(event_log)
    return event_df