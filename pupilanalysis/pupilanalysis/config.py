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
maxdur = 125
margin = 3
gap_margin = 5
gap_vt = 10
smooth_winlen = 7
std_thr = 3

# %% Baseline correction

start_bl = 0        # Start of baseline correction window
end_bl = 10          # End of baseline correction window

