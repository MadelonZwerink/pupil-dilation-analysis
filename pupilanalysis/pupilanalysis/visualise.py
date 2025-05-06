# visualise.py
# Contains functions to visualise the data.

import numpy as np
from matplotlib import pyplot as plt, lines
from datamatrix import operations as ops
from pupilanalysis.custom_funcs import count_valid_traces
import math

 # %% plot_baselines
 
def plot_baselines(dm, by='participant'):
    if by == 'participant':
        groups = ops.split(dm.participant, "inf2", "inf3", "inf4", "inf5")
        labels = ["Inf2", "Inf3", "Inf4", "Inf5"]
        colors = ["green", "blue", "purple", "pink"]

        # Create 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        for i, (group, label, color) in enumerate(zip(groups, labels, colors)):
            ax = axes[i // 2, i % 2]  # Determine the axis to plot on (2x2 grid)
            ax.hist(group.baseline, color=color, bins=20)
            ax.set_title(label)
            ax.set_xlabel('Baseline Value')
            ax.set_ylabel('Frequency')

        # Adjust layout to avoid overlap
        plt.tight_layout()
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
    

# %% plot_pupiltrace

def plot_pupiltrace(dm, by='condition', show_individual_trials=False, 
                    ymin=None, ymax=None, signal='pupil', min_n_valid=5):
    """
    Plot pupil traces over time, grouped by 'condition', 'participant', or all together.

    Parameters:
        dm: data matrix containing pupil data.
        by: str, one of 'condition', 'participant', or 'all'.
        show_individual_trials: bool, whether to show individual traces.
        ymin: float, min y-axis value.
        ymax: float, max y-axis value.
        signal: str, either 'pupil' or 'ptrace', depending on which trace to use.
    """
    x = np.linspace(0, 4, len(dm.ttrace[0]))

    plt.figure()
    plt.axvline(0, linestyle=':', color='black')
    plt.axhline(1, linestyle=':', color='black')

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

    elif by == 'all':
        traces = getattr(dm, signal)
        plot_series(x, traces, color='black', label='All traces', 
                    show_individual_trials=show_individual_trials,
                    min_n_valid=min_n_valid)
        plt.legend(frameon=False)

    else:
        raise ValueError("Invalid `by` argument. Must be 'condition', 'participant', or 'all'.")

    plt.ylabel('Pupil size (norm)')
    plt.xlabel('Time relative to onset retention interval (s)')

    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)

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
                axs[i].plot(np.arange(len(inf.ptrace[i])), 
                            np.array(inf.ptrace[i, :]), 
                            label = "ptrace",
                            alpha = 0.8)    
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
 
def plot_fixations(dm):
    for i, row in zip(range(420,500), dm):
        print('Trial %d' % i)
        for x, y in zip(
            row.fixxlist,
            row.fixylist
        ):
            print('\t', x, y)
    
    x = np.array(dm.fixxlist)
    y = np.array(dm.fixylist)
    x = x.flatten()
    y = y.flatten()
    plt.hexbin(x, y, gridsize=25, extent=(0, 1099, 0, 799))
    plt.show()
 
 # %% plot_nr_blinks
 


