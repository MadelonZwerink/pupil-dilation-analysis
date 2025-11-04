
library(dplyr)
library(purrr)
library(tidyr)
library(jsonlite)
library(tuneR)
library(stringr)
library(tidyverse)
library(signal)
library(zoo)

#==============================================================
# Helper functions
#==============================================================

# Compute duration of sound files in seconds
# Returns a tibble with file name, stimulus label, and duration
get_duration <- function(path) {
  files <- list.files(path, pattern = "\\.(wav|wac|mp3|flac)$", ignore.case = TRUE)
  map_dfr(files, function(file) {
    sound <- readWave(file.path(path, file))
    tibble(
      stimulus = str_remove(file, "\\.wav$"), # drop extension
      dur      = length(sound) / 48000        # convert samples to seconds (48 kHz)
    )
  })
}

# Helper: returns fixation x,y for each timepoint in t
assign_fix_xy <- function(t, start, end, fixx, fixy) {
  n <- length(t)
  if (length(start) == 0 || length(end) == 0 || length(fixx) == 0) {
    return(tibble(fix_x = rep(NA_real_, n), fix_y = rep(NA_real_, n)))
  }
  
  # Initialize output
  fix_x <- rep(NA_real_, n)
  fix_y <- rep(NA_real_, n)
  
  # Assign coordinates for samples inside each fixation interval
  for (i in seq_along(start)) {
    inside <- t >= start[i] & t <= end[i]
    if (any(inside)) {
      fix_x[inside] <- fixx[i]
      fix_y[inside] <- fixy[i]
    }
  }
  tibble(fix_x, fix_y)
}

# Helper: fast check if any interval contains t
in_any_interval <- function(t, start, end) {
  if (length(start) == 0 || length(end) == 0) return(rep(FALSE, length(t)))
  Reduce("|", Map(function(s, e) t >= s & t <= e, start, end))
}

#==============================================================
# Load and preprocess JSON data
#==============================================================

# Read JSON and extract relevant part (columns with trial data)
dm <- jsonlite::fromJSON("~/BiBC/ADS_stage/pupil-dilation-analysis/data/processed/dm_raw.json", 
                         simplifyVector = TRUE)$columns

nr_trials <- length(dm$trial[[2]])
segment_breaks <- c(0, 9, 33, 57, 81)
segment_labels <- c('noise', 'fam1', 'fam2', 'fam3')

# Flatten nested JSON structure into a tibble
# Keep only columns of length = nr_trials
dm <- map(dm, ~ if (length(.x[[2]]) == nr_trials) .x[[2]] else .x[[2]][[1]]) %>% 
  as_tibble() %>%
  mutate(
    index = seq_len(nr_trials), # add trial index
    segment = cut(trial, breaks=segment_breaks, labels = segment_labels)) %>% # add segment labels 
  select(-data_error)

#==============================================================
# Create nested trace columns and apply low-pass filter to pupil
#==============================================================

trace_cols <- c(
  "xtrace", "ytrace", "pupil", "ttrace", "ptrace", "ptraceraw",
  "dxtrace", "dytrace", "abs_dxtrace", "abs_dytrace")
#,"fixstlist", "fixetlist", "fixxlist", "fixylist")

fs <- 250 # Hz, sampling rate
fc <- 10  # Hz, cutoff frequency
Wn <- fc / (fs / 2)

# Apply Butterworth low-pass filter
bf <- butter(n = 3, W = Wn, type = "low")

# Convert each matrix column into a list-column of tibbles
dm <- dm %>%
  mutate(across(
    all_of(intersect(trace_cols, names(dm))),
    ~ map(asplit(.x, 1), function(row) {
      # add names (time indices) before tibble conversion
      names(row) <- seq_along(row)
      as_tibble_row(row)
    }),
    .names = "{.col}"
  )) %>%
  mutate(
    lppupil = map(
      pupil,
      ~ {
        # convert 1×992 tibble → numeric vector
        x <- as.numeric(unlist(.x))
        
        # handle missing values
        x <- zoo::na.approx(x, na.rm = FALSE)
        x <- zoo::na.locf(x, na.rm = FALSE, fromLast = TRUE)
        x <- zoo::na.locf(x, na.rm = FALSE)
        
        # apply filter
        if (sum(!is.na(x)) > 10) {
          y <- signal::filtfilt(bf, x)
        } else {
          y <- rep(NA_real_, length(x))
        }
        
        # create names so columns aren't empty
        names(y) <- names(.x)
        
        # convert back to 1×992 tibble
        tibble::as_tibble_row(as.list(y))
      }
    )
  )

#==============================================================
# Baseline medians (per trial)
#==============================================================

# Compute median x and y position during baseline window (10 samples after bl_start_index)
dm <- dm %>%
  mutate(
    med_x = map2_dbl(xtrace, bl_start_index, ~ 
                       if (!is.na(.y)) median(unlist(.x[(.y + 1):(.y + 10)])) else NA_real_
    ),
    med_y = map2_dbl(ytrace, bl_start_index, ~ 
                       if (!is.na(.y)) median(unlist(.x[(.y + 1):(.y + 10)])) else NA_real_
    )
  )

#==============================================================
# Deviation of baseline gaze coordinates from participant median gaze
#==============================================================

# Calculate median gaze positions for each participant
# Create column gaze_incl that specifies whether the gaze position during 
# baseline deviated too much from the median gaze position for the participant
dm <- dm %>%
  group_by(participant) %>%
  mutate(
    medx_participant = median(unlist(map(xtrace, as.numeric)), na.rm = TRUE),
    medy_participant = median(unlist(map(ytrace, as.numeric)), na.rm = TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    bl_gaze_dev = sqrt((medx_participant - med_x)^2 + (medy_participant - med_y)^2),
    gaze_incl = ifelse(bl_gaze_dev < 250, 1, 0)
  )

#==============================================================
# Deviation columns (centered gaze measures)
#==============================================================

# Subtract baseline medians from traces and compute distance from center
dm <- dm %>%
  mutate(
    devx      = map2(xtrace, med_x, ~ .x - .y),             # x deviation from baseline
    devy      = map2(ytrace, med_y, ~ .x - .y),             # y deviation from baseline
    devcenter = map2(dxtrace, dytrace, ~ sqrt(.x^2 + .y^2)),# distance to center
    devbl     = map2(devx, devy, ~ sqrt(.x^2 + .y^2))       # distance relative to baseline
  )

#==============================================================
# Factor variables and experimental conditions
#==============================================================

dm <- dm %>%
  mutate(
    stim_grouped = factor(stim_grouped, levels = c("noise", "familiarization"), ordered = TRUE),
    stim_type    = factor(stim_type, levels = c("emphasized_noise", "emphasized", 
                                                "functional_noise", "functional"), ordered = TRUE),
    participant  = factor(participant),
    type         = if_else(stim_grouped == "familiarization", "language", "noise"),
    condition    = if_else(str_detect(stim_type, "emphasized"), "emphasized", "functional") %>% factor()
  )

#==============================================================
# Stimulus durations
#==============================================================

# Read stimulus durations from both emphasized and functional folders
durations_df <- bind_rows(
  get_duration("~/BiBC/ADS_stage/pupil-dilation-analysis/data/raw/stimuli_infant_experiment/emphasized/training_L1"),
  get_duration("~/BiBC/ADS_stage/pupil-dilation-analysis/data/raw/stimuli_infant_experiment/functional/training_L1")
)

# Use headturn_fam to add the trialid column to dm (to determine duration of noise trials)
headturn_fam <- read.csv("~/BiBC/ADS_stage/pupil-dilation-analysis/data/raw/headturn/headturn-familiarization-1.csv", skip = 1, sep = ";")
headturn_fam <- headturn_fam %>%
  mutate(participant = case_when(ppid == 'inf_pilot2' ~ 'inf2',
                                 ppid == 'test_pilot1' ~ 'inf3',
                                 ppid == 'inf_pilot4' ~ 'inf4',
                                 ppid == 'inf_pilot5' ~ 'inf5')) %>%
  dplyr::filter(!is.na(participant))

dm <- dm %>%
  left_join(
    headturn_fam %>% select(participant, trial, id),
    by = c("participant", "trial")
  )

# Join durations to trial data and assign missing values for noise trials
dm <- dm %>%
  left_join(durations_df, by = "stimulus") %>%
  group_by(condition, id) %>%
  mutate(dur = if_else(type == "noise" & is.na(dur), first(na.omit(dur)), dur)) %>%
  ungroup()

#==============================================================
# Convert blink/fix start and end matrices into list-columns
#==============================================================

dm <- dm %>%
  mutate(
    # Each row of blinkstlist/blinketlist becomes a numeric vector
    blinkstlist = split(as.data.frame(blinkstlist), seq_len(nrow(blinkstlist))) %>%
      map(~ unlist(.x) %>% discard(is.na)),
    blinketlist = split(as.data.frame(blinketlist), seq_len(nrow(blinketlist))) %>%
      map(~ unlist(.x) %>% discard(is.na)),
    fixstlist   = split(as.data.frame(fixstlist), seq_len(nrow(fixstlist))) %>%
      map(~ unlist(.x) %>% discard(is.na)),
    fixetlist   = split(as.data.frame(fixetlist), seq_len(nrow(fixetlist))) %>%
      map(~ unlist(.x) %>% discard(is.na)),
    fixxlist   = split(as.data.frame(fixxlist), seq_len(nrow(fixxlist))) %>%
      map(~ unlist(.x) %>% discard(is.na)),
    fixylist   = split(as.data.frame(fixylist), seq_len(nrow(fixylist))) %>%
      map(~ unlist(.x) %>% discard(is.na))
  )

#==============================================================
# Convert to long format (old code)
#==============================================================

dm_long <- dm %>%
  select(-starts_with("fix"), -starts_with("blink")) %>%
  # Unnest nested trace columns into long tibble
  unnest(c(pupil, lppupil, ptrace, ptraceraw, xtrace, ytrace, dxtrace, dytrace, ttrace, 
           devx, devy, devcenter, devbl),
         names_sep = "_") %>%
  
  # Reshape: one row per sample per variable
  pivot_longer(
    cols = matches("^(pupil|lppupil|ptrace|ptraceraw|xtrace|ytrace|dxtrace|dytrace|ttrace|devx|devy|devcenter|devbl)_"),
    names_to = c("var", "time"),
    values_to = "val",
    names_sep = "_"
  ) %>%
  
  # Spread variables back into wide form (columns per variable)
  pivot_wider(names_from = var, values_from = val) %>%
  
  # Add timing and inclusion criteria
  mutate(
    time             = as.numeric(time) * 4,                  # sample index → ms (4 ms resolution)
    gaze_sample_incl = if_else(devbl < 250, 1, 0)             # mark valid gaze samples
  ) 

#==============================================================
# Perform trial exclusion based on missing data
#==============================================================

dm_long <- dm_long %>%
  mutate(pupil_sample_incl = ifelse(is.na(ptrace), 0, 1)) %>%
  group_by(index) %>%
  mutate(trial_samples = sum(pupil_sample_incl[time < 3000], na.rm = TRUE),
         trial_incl = ifelse(trial_samples/n() > 0.4, 1, 0),
         include = trial_incl * bl_incl * gaze_incl) %>%
  ungroup()

dm_long <- dm_long %>%
  mutate(across(
    c(gaze_incl, bl_incl, trial_incl, include, pupil_sample_incl, gaze_sample_incl),
    ~ replace_na(.x, 0)
  ))

# Compute trial_prop and trial_incl per trial in dm_long
trial_summary <- dm_long %>%
  group_by(index) %>%
  summarise(
    trial_samples = mean(trial_samples, na.rm = TRUE),
    trial_incl = first(trial_incl),
    missing_samples = sum(is.na(ptraceraw)),
    .groups = "drop"
  )

# Join back to dm by index
dm <- dm %>%
 # select(-c(trial_samples, trial_incl)) %>%
  left_join(trial_summary, by = "index")

dm$include <- dm$trial_incl * dm$bl_incl * dm$gaze_incl

dm <- dm %>%
  mutate(across(
    c(gaze_incl, bl_incl, trial_incl, include),
    ~ replace_na(.x, 0)
  ))
