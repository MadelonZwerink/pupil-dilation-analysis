# custom_funcs.py

import numpy as np
import pandas as pd
from scipy.stats import zscore
from datamatrix import operations as ops

# %% count_valid_traces

def count_valid_traces(traces, nan_threshold=1.0):
    """
    Count the number of valid traces based on allowed fraction of NaNs.

    Parameters:
        traces: array-like, shape (n_trials, timepoints)
        nan_threshold: float (default=1.0)
            Maximum allowed fraction of NaNs per trace.
            Default (1.0) removes only fully-NaN traces.

    Returns:
        valid_count: int, number of valid traces
        valid_traces: np.ndarray with only valid rows
    """
    traces = np.asarray(traces, dtype=float)

    # Compute fraction of NaNs per trace
    nan_frac = np.mean(np.isnan(traces), axis=1)

    # Keep rows with NaN fraction less than threshold
    mask = nan_frac < nan_threshold
    valid_traces = traces[mask]

    return len(valid_traces), valid_traces, mask

# %% perform_trial_exclusion

def perform_trial_exclusion(dm, threshold, t_end):
    #If i == True: trial is included, contains less missing data than threshold
    row = {"threshold": round(threshold, ndigits=2)}
        
    for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
        valid_n_inf, _, i = count_valid_traces(inf.ptrace[:,0:round(t_end)], threshold)
        row[inf.participant[0]] = round(valid_n_inf / len(inf), ndigits=2)
        
    # Total DM
    valid_n_total, _, i = count_valid_traces(dm.ptrace[:,0:round(t_end)], threshold)
    row["dm"] = round(valid_n_total / len(dm), ndigits=2)
    
    return(row, i)

# %%

def baseline_correction(dm, col='ptrace', baseline_len=10, window_len=50):
    mean_values = []
    start_indices = []
    percent_empty_window = []
    percent_empty_total = []
    empty_window = 0

    for row in dm:
        signal = getattr(row, col)
        percent_empty_total.append(np.mean(np.isnan(signal)))
        signal = signal[:window_len]  # first 50 values only
        if np.all(~np.isnan(signal)): 
            empty_window += 1
        percent_empty_window.append(np.mean(np.isnan(signal)))
        found = False

        for i in range(window_len-baseline_len+1):  # up to index 40 (inclusive) so i:i+10 is within bounds
            window = signal[i:i+baseline_len]
            if np.all(~np.isnan(window)):
                mean_values.append(np.mean(window))
                start_indices.append(i)
                found = True
                break

        if not found:
            mean_values.append(np.nan)
            start_indices.append(np.nan)
    
    print(f"Number of set baselines: {len(mean_values) - np.sum(np.isnan(mean_values))}")
    print(f"Start index of baselines: {pd.value_counts(np.array(start_indices))}")
    print(f"Number of non-empty windows: {len(dm) - empty_window}")
    
    return(mean_values, start_indices, percent_empty_window, percent_empty_total)

# %%

def bl_to_zscore(dm, bl_col='baseline'):
    z_scores = []

    for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
        bl = getattr(inf, bl_col)
        z = zscore(bl, nan_policy='omit')
        z_scores.append(z.flatten())

    z_scores = np.concatenate(z_scores)
    
    return(z_scores)