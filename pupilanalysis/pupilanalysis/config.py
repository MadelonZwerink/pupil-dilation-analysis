# config.py
# Contains all variables and paths that are required for the analysis

# %% Loading data

from pathlib import Path  

data_dir = Path('/Users/madel/OneDrive/Documenten/BiBC/ADS_stage/pupil-dilation-analysis/data')
data_eyetracking_path = data_dir / 'raw/eyetracking/asc' 

# Ideally add dictionary for participant id here (is now hardcoded in read_data)
# Can then also be used dynamically in plot_pupiltrace
# Ideally add dictionary for conditions here (is now hardcoded in read_data)
# Can then also be used dynamically in plot_pupiltrace

# %% Blink reconstruction

vt_start = 10
vt_end = 5
maxdur = 250
margin = 5
gap_margin = 10
gap_vt = 10
smooth_winlen = 5
std_thr = 3

# %% Baseline correction

baseline_len = 10

