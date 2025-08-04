# ==============================================================================
# Analysis try
# ==============================================================================

# ------------------------------------------------------------------------------

models_by_time <- dm_long %>%
  filter(timepoint <= 494) %>%
  group_by(timepoint) %>%
  nest() %>%
  mutate(
    model = map(data, ~ lmer(pupil ~ stim_grouped + (1 + stim_grouped | participant), data = .x)),
    results = map(model, tidy)
  )

# View results for one timepoint
models_by_time$results[[1]]

# Or unnest all results
all_results <- models_by_time %>% unnest(results)

# ------------------------------------------------------------------------------

models_by_time <- dm_long %>%
  filter(timepoint <= 494) %>%
  group_by(timepoint) %>%
  nest() %>%
  mutate(
    model = map(data, ~ lmer(pupil ~ stim_type + 
                               (1 + stim_grouped | participant), data = .x)),
    results = map(model, ~ tidy(.x, effects = "fixed"))  # Only fixed effects
  )

stim_tvalues <- models_by_time %>%
  unnest(results) %>%
  filter(term != "(Intercept)") %>%
  select(timepoint, term, estimate, std.error, statistic)  # statistic = t-value

# Plot 1: Timecourse of pupil data
p1 <- ggplot(dm_long, aes(x = timepoint, y = pupil, group = index, colour = stim_type)) +
  geom_line(alpha = 0.2) +
  geom_smooth(aes(group = stim_type, colour = stim_type), se = FALSE, size = 1.2) +
  labs(y = "Pupil size") +
  theme_minimal()

# Plot 2: t-values over time
p2 <- ggplot(stim_tvalues, aes(x = timepoint, y = statistic, color = term)) +
  geom_point() +
  geom_hline(yintercept = c(-2, 2), linetype = "dashed", color = "red") +
  geom_hline(yintercept = 0, linetype = "solid", color = "blue") +
  labs(y = "t-value") +
  theme_minimal()

# Combine vertically
(p1 / p2) + plot_layout(heights = c(0.7, 0.3))

# ------------------------------------------------------------------------------

models_by_time <- dm_long %>%
  filter(timepoint <= 494 & 
           trialnr <= 49 &
           (stim_type == 'functional' | stim_type == 'functional_noise')) %>%
  group_by(timepoint) %>%
  nest() %>%
  mutate(
    model = map(data, ~ lmer(pupil ~ stim_type + (1 + stim_type | participant), data = .x)),
    results = map(model, ~ tidy(.x, effects = "fixed"))  # Only fixed effects
  )

stim_tvalues <- models_by_time %>%
  unnest(results) %>%
  filter(term != "(Intercept)") %>%
  select(timepoint, term, estimate, std.error, statistic)  # statistic = t-value

# Plot 1: Timecourse of pupil data
p1 <- dm_long %>%
  filter(timepoint <= 494 & (stim_type == 'functional' | stim_type == 'functional_noise')) %>%
  ggplot(aes(x = 8*timepoint, y = pupil, group = index, colour = stim_type)) +
  geom_line(alpha = 0.2) +
  geom_smooth(aes(group = stim_type, colour = stim_type), se = FALSE, size = 1.2) +
  labs(y = "Pupil size", x = '') +
  theme_minimal()

# Plot 2: t-values over time
p2 <- ggplot(stim_tvalues, aes(x = 8*timepoint, y = statistic, color = term)) +
  geom_point() +
  geom_hline(yintercept = c(-2, 2), linetype = "dashed", color = "red") +
  geom_hline(yintercept = 0, linetype = "solid", color = "blue") +
  labs(y = "t-value", x = 'Time since onset stimulus (ms)') +
  theme_minimal()

# Combine vertically
(p1 / p2) + plot_layout(heights = c(0.7, 0.3))

# ------------------------------------------------------------------------------

models_by_time <- dm_long %>%
  filter(timepoint <= 494 & 
           trialnr <= 49 &
           (stim_type == 'emphasized' | stim_type == 'emphasized_noise')) %>%
  group_by(timepoint) %>%
  nest() %>%
  mutate(
    model = map(data, ~ lm(pupil ~ stim_type, data = .x)),
    results = map(model, ~ broom::tidy(.x))
  )

stim_tvalues <- models_by_time %>%
  unnest(results) %>%
  filter(term != "(Intercept)") %>%
  select(timepoint, term, estimate, std.error, statistic)  # statistic = t-value

# Plot 1: Timecourse of pupil data
p1 <- dm_long %>%
  filter(timepoint <= 494 & (stim_type == 'emphasized' | stim_type == 'emphasized_noise')) %>%
  ggplot(aes(x = 8*timepoint, y = pupil, group = index, colour = stim_type)) +
  geom_line(alpha = 0.2) +
  geom_smooth(aes(group = stim_type, colour = stim_type), se = FALSE, size = 1.2) +
  labs(y = "Pupil size", x = '') +
  theme_minimal()

# Plot 2: t-values over time
p2 <- ggplot(stim_tvalues, aes(x = 8*timepoint, y = statistic, color = term)) +
  geom_point() +
  geom_hline(yintercept = c(-2, 2), linetype = "dashed", color = "red") +
  geom_hline(yintercept = 0, linetype = "solid", color = "blue") +
  labs(y = "t-value", x = 'Time since onset stimulus (ms)') +
  theme_minimal()

# Combine vertically
(p1 / p2) + plot_layout(heights = c(0.7, 0.3))

# ==============================================================================
# Analysis try 2: random effect for trial number
# ==============================================================================

models_by_time <- dm_long %>%
  filter(timepoint <= 494) %>%
  group_by(timepoint) %>%
  nest() %>%
  mutate(
    model = map(data, ~ lmer(pupil ~ stim_grouped + 
                               (1 + stim_grouped || participant) +
                               (1 | trialnr), data = .x)),
    results = map(model, tidy)
  )

# View results for one timepoint
models_by_time$results[[1]]

# Or unnest all results
all_results <- models_by_time %>% unnest(results)

models_by_time <- dm_long %>%
  filter(timepoint <= 494) %>%
  group_by(timepoint) %>%
  nest() %>%
  mutate(
    model = map(data, ~ lmer(pupil ~ stim_grouped + (1 + stim_grouped | participant), data = .x)),
    results = map(model, ~ tidy(.x, effects = "fixed"))  # Only fixed effects
  )

stim_tvalues <- models_by_time %>%
  unnest(results) %>%
  filter(term != "(Intercept)") %>%
  select(timepoint, term, estimate, std.error, statistic)  # statistic = t-value

# Plot 1: Timecourse of pupil data
p1 <- ggplot(dm_long, aes(x = timepoint, y = pupil, group = index, colour = stim_grouped)) +
  geom_line(alpha = 0.2) +
  geom_smooth(aes(group = stim_grouped, colour = stim_grouped), se = FALSE, size = 1.2) +
  labs(y = "Pupil size") +
  theme_minimal()

# Plot 2: t-values over time
p2 <- ggplot(stim_tvalues, aes(x = timepoint, y = statistic)) +
  geom_point() +
  geom_hline(yintercept = c(-2, 2), linetype = "dashed", color = "red") +
  geom_hline(yintercept = 0, linetype = "solid", color = "blue") +
  labs(y = "t-value") +
  theme_minimal()

# Combine vertically
(p1 / p2) + plot_layout(heights = c(0.7, 0.3))
