# -*- coding: utf-8 -*-
"""
Created on Tue May  6 20:08:29 2025

@author: madel
"""

# %% 

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
