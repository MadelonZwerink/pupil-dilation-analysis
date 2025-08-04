setwd("~/BiBC/ADS_stage/pupil-dilation-analysis")
library('tidyverse')
library('jsonlite')
library('RJSONIO')
library(patchwork)
library(broom.mixed)
library(lme4)

create_nested_col <- function(data, column){
  dat <- data[column][[1]]
  n_obs <- length(dat[[1]])
  df <- matrix(nrow = length(dat), ncol = n_obs)
  
  for (i in 1:length(dat)){
    df[i,] <- dat[[i]]
  }
  
  df <- as_tibble(df)
  colnames(df) <- seq(1, ncol(df))
  df$index = seq(1, 100)
  
  return(nest(df, .by=index)$data) 
}

# Read JSON file as character values
dm <- read_json('./data/processed/dm.json')
# Convert to large list, with two elements
dm <- RJSONIO::fromJSON(dm, nullValue=NA, simplify=TRUE, simplifyWithNames = TRUE)
# First element is only index, keep only the second element: column values
dm <- dm$columns
# For each column, keep only the data
for (i in 1:length(dm)){
  new_col <- dm[[i]][[2]]
  
  if (length(new_col) != 100){
    new_col <- new_col[[1]]
  }
  
  dm[[i]] <- new_col
}
# Change dm into a tibble and remove irrelevant columns
dm <- dm %>% 
  as_tibble() %>%
  select(!c(data_error, ptrace, xtrace, ytrace, ptrace_ds, bl_incl, bl_zscore, blinketlist, blinkstlist, fixetlist, fixstlist, fixxlist, fixylist, dxtrace, dytrace, abs_dxtrace, abs_dytrace, include, path, sessionid, t_offset, t_onset, trace_length, trial_incl, ttrace))

# Create a column with indices to easily identify individual trials
dm$index <- seq(1, 100)

# Convert series columns to nested columns
dm$xtrace <- create_nested_col(dm, 'xtrace_ds')
dm$ytrace <- create_nested_col(dm, 'ytrace_ds')
dm$pupil <- create_nested_col(dm, 'pupil')

dm <- dm %>% select(!c(xtrace_ds, ytrace_ds))

# Print column names
colnames(dm)

# Now try converting all data to longform
dm_long <- unnest(dm, cols=c(pupil, xtrace, ytrace), names_sep = '_')
dm_long <- pivot_longer(dm_long, cols = starts_with(c('pupil', 'xtrace', 'ytrace')), names_to = c('var', 'timepoint'), values_to = 'val', names_sep = '_')
dm_long <- pivot_wider(dm_long, names_from = var, values_from = val)
dm_long$timepoint <- as.numeric(dm_long$timepoint)

dm_long$stim_grouped <- factor(dm_long$stim_grouped, ordered=T, levels = c('noise', 'familiarization'))
dm_long$stim_type <- factor(dm_long$stim_type, ordered=T, levels = c('emphasized_noise', 'emphasized', 'functional_noise', 'functional'))
dm_long$participant <- as.factor(dm_long$participant)

# Plot all trials, colored by grouped type of stimulus; noise or linguistic
ggplot(dm_long, aes(x=timepoint, y=pupil, group = index, colour = stim_grouped)) +
  geom_line() +
  geom_smooth(aes(group = stim_grouped, colour = stim_grouped))

# Plot all trials, colored by type of stimulus; emphasized or functional, noise or linguistic
ggplot(dm_long, aes(x=timepoint, y=pupil, group = index, colour = stim_type)) +
  geom_line() +
  geom_smooth(aes(group = stim_type, colour = stim_type))

# ==============================================================================

df_max_pupil <- dm_long %>%
  group_by(index) %>%
  slice_max(pupil, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(index, max_pupil = pupil, timepoint, stim_grouped, stim_type, participant, baseline)

ggplot(df_max_pupil, aes(x = index, y = max_pupil, group = index, colour = as.factor(index))) +
  geom_boxplot() +
  geom_point() +
  theme(legend.position = 'none')

# difference in maximum pupil size for all different groups
kruskal.test(max_pupil ~ stim_type, data = df_max_pupil)
pairwise.wilcox.test(df_max_pupil$max_pupil, df_max_pupil$stim_type,
                     p.adjust.method = "BH")
ggplot(df_max_pupil, aes(x = stim_type, y = max_pupil, group = stim_type, colour = stim_type)) +
  geom_boxplot() +
  geom_point() 

kruskal.test(timepoint ~ stim_type, data = df_max_pupil)
ggplot(df_max_pupil, aes(x = stim_type, y = timepoint, group = stim_type, colour = stim_type)) +
  geom_boxplot() +
  geom_point() 

wilcox.test(max_pupil ~ stim_grouped, 
            data = df_max_pupil,
            exact = FALSE)
ggplot(df_max_pupil, aes(x = stim_grouped, y = max_pupil, group = stim_grouped, colour = stim_grouped)) +
  geom_boxplot() +
  geom_point() 

wilcox.test(timepoint ~ stim_grouped, 
            data = df_max_pupil,
            exact = FALSE)
ggplot(df_max_pupil, aes(x = stim_grouped, y = timepoint, group = stim_grouped, colour = stim_grouped)) +
  geom_boxplot() +
  geom_point() 
