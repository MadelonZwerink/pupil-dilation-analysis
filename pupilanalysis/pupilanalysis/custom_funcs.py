# custom_funcs.py

import numpy as np

# %% count_valid_traces

def count_valid_traces(traces):
    """
    Count the number of valid traces (not all NaN).

    Parameters:
        traces: array-like, shape (n_trials, timepoints)

    Returns:
        valid_count: int, number of non-empty traces
        valid_traces: np.ndarray with only valid rows
    """
    traces = np.asarray(traces)
    mask = ~np.all(np.isnan(traces), axis=1)
    valid_traces = traces[mask]
    return len(valid_traces), valid_traces

