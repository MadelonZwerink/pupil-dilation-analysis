# -*- coding: utf-8 -*-
"""
Created on Tue May  6 20:08:29 2025

@author: madel
"""

# %% Add the location of the module to the system paths so that it can be found

import sys
sys.path.append('C:\\Users\\madel\\OneDrive\\Documenten\\BiBC\\ADS_stage\\pupil-dilation-analysis\\pupilanalysis')

# %% Parse the data

from pupilanalysis.config import data_eyetracking_path
from pupilanalysis.data import read_data

dm = read_data(data_eyetracking_path)

# %% 

from pupilanalysis.config import data_dir
import numpy as np
import pandas as pd
from datamatrix import operations as ops 

# =============================================================================
# Data loss
# =============================================================================

# Initialize an empty list to collect rows
data = []

# Iterate through each "inf" group
for inf_label, inf in enumerate(ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5")):
    for i in range(len(inf)):
        row = {
            "inf": f"{inf.participant[i]}",  # Adjust label to match "inf2", "inf3", etc.
            "i": i,
            "ptrace": np.sum(~np.isnan(inf.ptrace[i]))/992
        }
        data.append(row)  # Append each row to the list

# Convert collected data into a DataFrame
df = pd.DataFrame(data)

# Compute averages per 'inf'
average_per_inf = df.groupby("inf").mean(numeric_only=True).reset_index()

# Compute averages per 'i' (trial)
average_per_trial = df.groupby("i").mean(numeric_only=True).reset_index()

# Display results
print("Average per 'inf':")
print(average_per_inf)

print("\nAverage per 'i' (trial):")
print(average_per_trial)

# Save results
df.to_csv(f"{data_dir}/processed/data_loss_blink_reconstruction.csv", index=False)
average_per_inf.to_csv(f"{data_dir}/processed/data_loss_blink_reconstruction_per_inf.csv", index=False)
average_per_trial.to_csv(f"{data_dir}/processed/data_loss_blink_reconstruction_per_trial.csv", index=False)

# %% Plot with missing fraction of trials, average for all infants

import matplotlib.pyplot as plt

# Plot
plt.figure(figsize=(10, 6))
plt.plot(average_per_trial.i, average_per_trial.ptrace, marker='o')

# Formatting
plt.title("Data quality decreases with time")
plt.xlabel("Trial number")
plt.ylabel("Fraction of data that is valid")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Plot with missing fraction of trials, average for all infants, with sd

import matplotlib.pyplot as plt
import seaborn as sns

plot_missing_data = sns.relplot(data=df, x="i", y="ptrace", 
                                kind="line", errorbar="sd",
                                height=5, aspect=2)
plot_missing_data.set_axis_labels("Trial number", "Recorded fraction")
plot_missing_data.ax.margins(0)
plot_missing_data.ax.set_ylim(0, 1)

# %%

import numpy as np
import pandas as pd
from datamatrix import operations as ops
from pupilanalysis.custom_funcs import count_valid_traces

def perform_trial_exclusion(dm, threshold, t_end=dm.ptrace.depth):
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

trial_exl = perform_trial_exclusion(dm, 0.6)

# %% Plot fraction of valid traces per threshold

results = []

for threshold in np.arange(0, 1.1, 0.1):
    result = perform_trial_exclusion(dm, threshold)
    results.append(result[0])

df = pd.DataFrame(results)
df.set_index("threshold", inplace=True)

# Plot
plt.figure(figsize=(10, 6))
for column in df.columns:
    plt.plot(df.index, df[column], marker='o', label=column)

# Formatting
plt.title("Fraction of Valid Traces vs. NaN Threshold")
plt.xlabel("NaN Threshold")
plt.ylabel("Valid Trace Fraction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

# Set 1: noise_1 to noise_9
noise_set = {f"noise_{i}" for i in range(1, 10)}
noise_trials = dm.trialid == noise_set

# Set 2: familiarization_1 to familiarization_24
fam1_24_set = {f"familiarization_{i}" for i in range(1, 25)}
fam1 = dm.trialid == fam1_24_set

# Set 3: familiarization_25 to familiarization_48
fam25_48_set = {f"familiarization_{i}" for i in range(25, 49)}
fam2 = dm.trialid == fam25_48_set

# Set 4: familiarization_49 to familiarization_72
fam49_72_set = {f"familiarization_{i}" for i in range(49, 73)}
fam3 = dm.trialid == fam49_72_set

long_results = []

for subset in (noise_trials, fam1, fam2, fam3):
    results = []
    
    for threshold in np.arange(0, 1.1, 0.1):
        result = perform_trial_exclusion(subset, threshold, t_end=992)
        results.append(result[0])
    
    df = pd.DataFrame(results)
    df.set_index("threshold", inplace=True)
    
    long_results.append(results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], marker='o', label=column)
    
    plt.plot(df.index, df.dm, linewidth=5, color="purple")
    
    # Formatting
    plt.title(f"Valid traces for {subset.trialid[0]} until {subset.trialid[-1]} (full 4s)")
    plt.xlabel("NaN Threshold")
    plt.ylabel("Valid Trace Fraction")
    plt.legend()
    plt.grid(True)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

import pandas as pd

# Flatten while adding outer index
flattened = [
    {**d, 'outer_idx': i}
    for i, sublist in enumerate(long_results)
    for d in sublist
]

# %%

long_results = []

for subset in (noise_trials, fam1, fam2, fam3):
    results = []
    
    for threshold in np.arange(0, 1.1, 0.1):
        result = perform_trial_exclusion(subset, threshold, t_end=dm.ptrace.depth*0.625)
        results.append(result[0])
    
    df = pd.DataFrame(results)
    df.set_index("threshold", inplace=True)
    
    long_results.append(results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], marker='o', label=column)
    
    plt.plot(df.index, df.dm, linewidth=5, color="purple")
    
    # Formatting
    plt.title(f"Valid traces for {subset.trialid[0]} until {subset.trialid[-1]} (first 2.5s)")
    plt.xlabel("NaN Threshold")
    plt.ylabel("Valid Trace Fraction")
    plt.ylim(0,1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# %%

perform_trial_exclusion(noise_trials, 0.6)
perform_trial_exclusion(fam1, 0.6)
perform_trial_exclusion(fam2, 0.6)
perform_trial_exclusion(fam3, 0.6)
