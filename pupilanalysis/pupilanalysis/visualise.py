# visualise.py
# Contains functions to visualise the data.

import numpy as np
from matplotlib import pyplot as plt, lines
from datamatrix import operations as ops
from pupilanalysis.custom_funcs import count_valid_traces
import math

 # %% plot_baselines
 
def plot_baselines(dm, by='participant', col='baseline', title='Baselines per participant'):
    if by == 'participant':
        groups = ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5")
        labels = ["Inf2", "Inf3", "Inf4", "Inf5"]
        colors = ["green", "blue", "purple", "pink"]

        # Create 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        for i, (group, label, color) in enumerate(zip(groups, labels, colors)):
            ax = axes[i // 2, i % 2]  # Determine the axis to plot on (2x2 grid)
            baseline = getattr(group, col)
            ax.hist(baseline, color=color, bins=20)
            ax.set_title(label)
            ax.set_xlabel('Baseline Value')
            ax.set_ylabel('Frequency')

        # Adjust layout to avoid overlap
        fig.suptitle(f"{title}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

 
 # %% plot_series

def plot_series(x, s, color, label, show_individual_trials=False, min_n_valid=5):
    valid_n, valid_data = count_valid_traces(s)

    # Count how many non-NaN values at each time point
    valid_counts = np.sum(~np.isnan(valid_data), axis=0)

    # Compute mean and SE across valid data
    mean = np.nanmean(valid_data, axis=0)
    se = np.nanstd(valid_data, axis=0) / np.sqrt(valid_counts)

    # Mask time points with too few valid samples
    mask = valid_counts < min_n_valid
    mean[mask] = np.nan
    se[mask] = np.nan

    # Plot individual traces
    if show_individual_trials:
        for trace in valid_data:
            plt.plot(x, trace, alpha=0.5, linewidth=0.6, zorder=1)

    # Plot mean and confidence interval
    plt.fill_between(x, mean - se, mean + se, color=color, alpha=0.25, zorder=2)
    plt.plot(x, mean, color=color, label=f"{label} (N={valid_n})", linewidth=2, zorder=3)
    
# %% plot_series_panel

def plot_series_panel(ax, x, s, color, label, show_individual_trials=False, min_n_valid=5):
    valid_n, valid_data = count_valid_traces(s)
    valid_counts = np.sum(~np.isnan(valid_data), axis=0)
    mean = np.nanmean(valid_data, axis=0)
    se = np.nanstd(valid_data, axis=0) / np.sqrt(valid_counts)

    mask = valid_counts < min_n_valid
    mean[mask] = np.nan
    se[mask] = np.nan

    if show_individual_trials:
        for trace in valid_data:
            ax.plot(x, trace, alpha=0.5, linewidth=0.6, zorder=1)

    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.25, zorder=2)
    ax.plot(x, mean, color=color, label=f"{label} (N={valid_n})", linewidth=0.8, zorder=3)

# %% plot_pupiltrace

def plot_pupiltrace(dm, 
                    by='condition', 
                    show_individual_trials=False, 
                    ymin=None, 
                    ymax=None, 
                    xmin=0,
                    xmax=4,
                    signal='pupil', 
                    min_n_valid=5,
                    title='Pupil trace'):
    """
    Plot pupil traces over time, grouped by 'condition', 'participant', or all together.

    Parameters:
        dm: data matrix containing pupil data.
        by: str, one of 'condition', 'participant', 'split', or 'all'.
        show_individual_trials: bool, whether to show individual traces.
        ymin: float, min y-axis value.
        ymax: float, max y-axis value.
        signal: str, either 'pupil' or 'ptrace', depending on which trace to use.
    """
    x = np.linspace(0, 4, len(dm.ttrace[0]))

    plt.figure()
    
    if ymin < 0:
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1) 

    if by == 'condition':
        groups = ops.split(dm.stim_type, 
                           "emphasized", 
                           "functional", 
                           "emphasized_noise",
                           "functional_noise")
        labels = ["Emphasized", "Functional", "Emphasized noise", "Functional noise"]
        colors = ["blue", "red", "green", "red"]

        for group, label, color in zip(groups, labels, colors):
            traces = getattr(group, signal)
            plot_series(x, traces, color=color, label=label, 
                        show_individual_trials=show_individual_trials,
                        min_n_valid=min_n_valid)

        plt.legend(frameon=False, title='Stimulus type')
        plt.title(f"{title}")
        
    if by == 'condition_grouped':
        groups = ops.split(dm.stim_grouped, 
                           "noise", 
                           "familiarization")
        labels = ["noise", "familiarization"]
        colors = ["blue", "green"]

        for group, label, color in zip(groups, labels, colors):
            traces = getattr(group, signal)
            plot_series(x, traces, color=color, label=label, 
                        show_individual_trials=show_individual_trials,
                        min_n_valid=min_n_valid)

        plt.legend(frameon=False, title='Stimulus type')
        plt.title(f"{title}")

    elif by == 'participant':
        groups = ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5")
        labels = ["Inf2", "Inf3", "Inf4", "Inf5"]
        colors = ["green", "blue", "purple", "red"]

        for group, label, color in zip(groups, labels, colors):
            traces = getattr(group, signal)
            plot_series(x, traces, color=color, label=label, 
                        show_individual_trials=show_individual_trials,
                        min_n_valid=min_n_valid)

        plt.legend(frameon=False, title='Participant')
        plt.title(f"{title}")

    elif by == 'all':
        traces = getattr(dm, signal)
        plot_series(x, traces, color='black', label='All traces', 
                    show_individual_trials=show_individual_trials,
                    min_n_valid=min_n_valid)
        plt.legend(frameon=False)
        plt.title(f"{title}")
    
    elif by == "split":
        participants = ["inf2", "inf3", "inf4", "inf5"]
        colors = ["green", "blue", "purple", "red"]
        groups = ops.split(dm.participant, *participants)

        n = len(participants)
        fig, axes = plt.subplots(2, 2, figsize=(16, 6), sharex=True, sharey=False)
        axes = axes.flatten()  # Convert to 1D array for easier indexing

        if n == 1:
            axes = [axes]  # Ensure iterable if only one subplot

        for ax, group, participant, color in zip(axes, groups, participants, colors):
            traces = getattr(group, signal)
            plot_series_panel(ax, x, traces, color='black', label=participant,
                              show_individual_trials=show_individual_trials,
                              min_n_valid=min_n_valid)
            ax.set_title(f"Participant: {participant}")
            ax.set_ylabel(f"{signal}")

        axes[-1].set_xlabel("Time relative to onset stimulus (s)")
        if ymin is not None or ymax is not None:
            for i, ax in enumerate(axes):
                ax.set_ylim(ymin, ymax[i])
        ax.set_xlim(xmin, xmax)

        fig.suptitle(f"{title}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    else:
        raise ValueError("Invalid `by` argument. Must be 'condition', 'participant', or 'all'.")

    plt.ylabel(f"{signal}")
    plt.xlabel('Time relative to onset stimulus (s)')

    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)
        
    plt.xlim(xmin, xmax)

    plt.show()


# %% plot_trials

def plot_grid_trials(dm, 
                     nr_trials=81, 
                     manual_events=False, 
                     ymin=-5, 
                     auto_ymax=True, 
                     ymax=None, 
                     blinklist=True, 
                     fixations=False, 
                     fix_xy=False,
                     ptrace=False, 
                     xtrace=False,
                     ytrace=False,
                     bl_corrected=True):
    
    gridsize = math.ceil(math.sqrt(nr_trials))
    
    for inf in ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5"):
        fig, axs = plt.subplots(gridsize, gridsize, figsize=(20, 20), sharex=True, sharey=True)
        axs = axs.flatten()  # Convert to 1D array for easier indexing
        
        if auto_ymax:
            ymax = np.nanmax(inf.ptrace.mean*2)
        
        for i in range(len(inf)):
            if i >= nr_trials:  # Avoid indexing out of bounds
                break 
                
            axs[i].set_ylim(ymin, ymax)
            axs[i].set_xlim(0, 992)
            axs[i].set_xticks([0, 250, 500, 750], labels=[0, 1, 2, 3])
            
            if ptrace:
                trace = np.array(inf.ptrace[i, :])
                axs[i].plot(np.arange(len(trace)), trace, label="ptrace", alpha=0.8)
                
                # Highlight NaN regions
                nan_mask = np.isnan(trace)
                if np.any(nan_mask):
                    # Find start and end of NaN runs
                    in_nan = False
                    for idx, val in enumerate(nan_mask):
                        if val and not in_nan:
                            start = idx
                            in_nan = True
                        elif not val and in_nan:
                            end = idx
                            axs[i].axvspan(start, end, color='lightgray', alpha=0.4, label="NaN region" if start == idx else None)
                            in_nan = False
                    # If ends in NaN
                    if in_nan:
                        axs[i].axvspan(start, len(trace), color='lightgray', alpha=0.4)
 
            if xtrace:
                axs[i].plot(np.arange(len(inf.ptrace[i])), 
                            np.array(inf.xtrace[i, :]), 
                            label = "xtrace",
                            alpha = 0.8)    
            if ytrace:
                axs[i].plot(np.arange(len(inf.ptrace[i])), 
                            np.array(inf.ytrace[i, :]), 
                            label = "ytrace",
                            alpha = 0.8) 
                
            if bl_corrected:
               axs[i].plot(np.arange(len(inf.pupil[i])), 
                           np.array(inf.pupil[i, :]), 
                           label = "ptrace (baseline-corrected)",
                           alpha = 0.8)  
            
            if blinklist:
                # Calculate blink start time relative to onset trial
                blst = (inf.blinkstlist[i, :] - inf.t_onset[i]) / 4
                blst[blst < 0] = 0 # Blinks that started before trial onset will be indicated at t=0
                axs[i].plot(blst, np.repeat(0, len(inf.blinkstlist[i])), 'd', label = "blinkstlist")
                
                # Calculate blink end time relative to onset trial
                blend = (inf.blinketlist[i, :] - inf.t_onset[i]) / 4
                blend[blend > 992] = 992 # Blinks that ended after trial ended will be indicated at the end of the trial
                axs[i].plot(blend, np.repeat(0, len(inf.blinketlist[i])), 'd', label = "blinketlist")
                
            if fixations:
                fixst = (inf.fixstlist[i, :]) / 4
                fixst[fixst < 0] = 0
                fixet = (inf.fixetlist[i, :]) / 4
                fixet[fixet > 992] = 992
                
                if fix_xy:
                    fixx = inf.fixxlist[i, :]
                    fixy = inf.fixylist[i, :]
                    for start, end, x, y in zip(fixst, fixet, fixx, fixy):
                        if not np.isnan(start) and not np.isnan(end):
                            values = np.arange(start, end)
                            axs[i].plot(values,
                                        np.repeat(x, len(values)),
                                        color="orange")
                            axs[i].plot(values,
                                        np.repeat(y, len(values)),
                                        color="purple")
                            axs[i].axvspan(start, end, alpha=0.2, color='pink')
                            
                for start, end in zip(fixst, fixet):
                    if not np.isnan(start) and not np.isnan(end):   
                        axs[i].axvspan(start, end, alpha=0.2, color='pink')
                        

            
            if manual_events:
                # Add manual blinks
                video_bl = (inf.video_blink[i, :] - inf.cor_t_onset[i]) / 4
                axs[i].vlines(video_bl, ymin, ymax, colors='red', linestyles='dashed', alpha=0.7, label = "manual blink")
                
                # Add manual look aways
                lookaway_start = (inf.video_lookaway_start[i, :] - inf.cor_t_onset[i]) / 4
                lookaway_end = (inf.video_lookaway_end[i, :] - inf.cor_t_onset[i]) / 4
                
                # Mask out NaNs and loop over valid lookaway events
                for start, end in zip(lookaway_start, lookaway_end):
                    if not np.isnan(start) and not np.isnan(end):
                        axs[i].axvspan(start, end, alpha=0.3, color='gray')
            
            axs[i].set_title(f"{inf.trialid[i]}")
        
        # Set overall title
        fig.suptitle(f'{inf.participant[0]}', x=0.1, y=0.99, fontsize=30, fontweight='bold')
    
        # Create a single legend (pick the first subplot's lines)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', fontsize=12, ncol=6, bbox_to_anchor=(0.75, 1))
    
        plt.tight_layout(pad=0, rect=[0, 0, 1, 0.98], h_pad=1.1)  # Adjust layout to fit suptitle and legend
        
        plt.show()
        
        #, save_fig=False, fig_name=None
        #if save_fig:
        #    if fig_name == None:
        #        break
        #    plt.savefig(f'figures/{fig_name}_{inf.participant[0]}.jpg')

 # %% plot_fixations
 
def plot_fixations(dm, plot_type='heatmap'):
    x = np.array(dm.fixxlist)
    y = np.array(dm.fixylist)
    x = x.flatten()
    y = y.flatten()
    if plot_type == 'heatmap':
        plt.hexbin(x, y, gridsize=25, extent=(0, 1099, 0, 799))
    elif plot_type == 'scatterplot':
        plt.scatter(x, y)
        plt.axis((0, 1099, 0, 799))
    plt.show()
 
    # %% plot_coordinates
    
def plot_coordinates(dm, plot_type='heatmap'):
    x = np.array(dm.xtrace)
    y = np.array(dm.ytrace)
    x = x.flatten()
    y = y.flatten()
    if plot_type == 'heatmap':
        plt.hexbin(x, y, gridsize=25, extent=(0, 1099, 0, 799))
    elif plot_type == 'scatterplot':
        plt.scatter(x, y)
        plt.axis((0, 1099, 0, 799))
    plt.show()
    
 # %% plot_nr_blinks
 
# %% plot_compare_baselines

def plot_compare_baselines(dm, baselines1, baselines2):
    baseline1 = np.array(getattr(dm, baselines1))
    baseline2 = np.array(getattr(dm, baselines2))
    x = np.arange(len(dm))

    nan_display_value = 0  # You can change this to another placeholder if desired

    # Replace NaNs with placeholder y-value for plotting
    baseline1_plot = np.where(np.isnan(baseline1), nan_display_value, baseline1)
    baseline2_plot = np.where(np.isnan(baseline2), nan_display_value, baseline2)

    plt.figure(figsize=(12, 6))

    # Define block width and colors
    block_width = 81
    colors = ['#f0f0f0', '#d0e0f0', '#f0d0d0', '#e0f0d0']  # Light gray/blue/pink/green

    # Draw the background blocks
    for i in range(4):
        start = i * block_width
        end = start + block_width
        plt.axvspan(start, end, facecolor=colors[i % len(colors)], alpha=0.3, zorder=0)

    # Draw a line between each baseline and baseline_flex point (even if one or both were NaN)
    for i in x:
        plt.plot([i, i], [baseline1_plot[i], baseline2_plot[i]], color='red', alpha=0.5, zorder=1)

    # Plot the baseline and baseline_flex points
    plt.scatter(x, baseline1_plot, label=baselines1, color='blue', zorder=2)
    plt.scatter(x, baseline2_plot, label=baselines2, color='orange', marker='x', zorder=2)

    plt.axhline(nan_display_value, color='black', linestyle='--', alpha=0.3, label='NaN marker')

    plt.xlim(0, len(dm))
    plt.xlabel('Row index')
    plt.ylabel('Baseline value')
    plt.title('Comparison of baseline values (NaNs Displayed at y=0)')
    plt.legend()
    plt.xticks(ticks=np.arange(0, len(dm), 9))  # Set your regular x-ticks if needed
    plt.grid(axis='x', which='major', linewidth=0.2)
    plt.tight_layout()
    plt.show()

