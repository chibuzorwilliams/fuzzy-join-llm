library(tidyverse)
library(cowplot)

options(dplyr.width = Inf)

font <- 'Roboto Condensed'

# Shared label cleaning ------------------------------------------------------
clean_labels <- function(d) {
  d %>%
    mutate(
      method = case_match(
        method,
        'llm'                           ~ 'GPT-4o-mini\nStandard',
        'llm_gpt4o_mini_token_fallback' ~ 'GPT-4o-mini\nToken Fallback',
        'llm_haiku'                     ~ 'Claude Haiku\nStandard',
        'llm_haiku_token_fallback'      ~ 'Claude Haiku\nToken Fallback',
        'llm_claude'                    ~ 'Claude Sonnet\nStandard',
        'llm_gpt4o'                     ~ 'GPT-4o\nStandard',
        'tfidf'                         ~ 'TFIDF',
        'soft_tfidf'                    ~ 'Soft TFIDF',
        'openai_embeddings'             ~ 'OpenAI Emb.',
        'sentence_transformer'          ~ 'Transformer',
        .default = str_replace_all(method, "_", " ") %>% str_to_title()
      ),
      dataset = str_replace_all(dataset, "-", " ") %>% str_to_title(),
      transformation = str_replace_all(transformation, "_", " ") %>%
        str_to_title()
    )
}

# Methods to exclude from main plots (expensive models shown separately)
expensive_methods <- c('llm_claude', 'llm_gpt4o')

# ---------------------------------------------------------------------------
# Plot 1: F1 by transformation, with 5-fold CV error bars (mean +/- 1 SD).
# Excludes expensive models (Sonnet, GPT-4o) — those are in a separate chart.
# ---------------------------------------------------------------------------
cv <- read_csv(file.path('results', 'cv_results.csv')) %>%
  filter(!method %in% expensive_methods) %>%
  clean_labels() %>%
  group_by(method, dataset) %>%
  mutate(max_f1 = max(f1_mean)) %>%
  ungroup() %>%
  mutate(
    method = fct_reorder(method, -max_f1),
    transformation = factor(
      transformation,
      levels = c('Scrambled', 'Ciphered Words', 'Ciphered Letters', 'Original')
    )
  )

cv_baseline <- cv %>%
  filter(transformation == 'Original') %>%
  select(method, dataset, f1_baseline = f1_mean, sd_baseline = f1_std)

cv_ablations <- cv %>%
  filter(transformation != 'Original') %>%
  mutate(transformation = fct_drop(transformation))

cv_ablations %>%
  ggplot(aes(x = f1_mean, y = transformation, color = transformation)) +
  geom_rect(
    data = cv_baseline, inherit.aes = FALSE,
    aes(xmin = f1_baseline - sd_baseline, xmax = f1_baseline + sd_baseline,
        ymin = -Inf, ymax = Inf),
    fill = 'grey70', alpha = 0.35
  ) +
  geom_vline(
    data = cv_baseline,
    aes(xintercept = f1_baseline)
  ) +
  geom_errorbarh(
    aes(xmin = f1_mean - f1_std, xmax = f1_mean + f1_std),
    height = 0.3, alpha = 0.9, linewidth = 0.5
  ) +
  geom_point(size = 1.6) +
  facet_grid(method ~ dataset) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(
    x = 'F1 Score',
    y = 'Transformation'
  ) +
  theme_minimal_vgrid(font_family = font, font_size = 16) +
  theme(
    strip.background = element_rect("grey80"),
    panel.grid.minor = element_blank(),
    plot.title.position = "plot",
    legend.position = 'none',
    panel.background = element_rect(fill = 'white', color = NA),
    plot.background = element_rect(fill = 'white', color = NA)
  ) +
  panel_border()

ggsave(
  filename = file.path('results', 'plots', 'f1_comparison.png'),
  width = 8,
  height = 10
)

# ---------------------------------------------------------------------------
# Plot 2: Precision-recall scatter (point estimates from summary.csv).
# Excludes expensive models.
# ---------------------------------------------------------------------------
df <- read_csv(file.path('results', 'summary.csv')) %>%
  filter(!method %in% expensive_methods) %>%
  clean_labels() %>%
  group_by(method, dataset) %>%
  mutate(max_f1 = max(f1)) %>%
  ungroup() %>%
  mutate(
    method = fct_reorder(method, -max_f1),
    transformation = fct_reorder(transformation, -f1)
  )

df %>%
  ggplot() +
  geom_point(
    aes(
      x = recall,
      y = precision,
      color = transformation
    )
  ) +
  facet_grid(method ~ dataset) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    x = 'Recall',
    y = 'Precision',
    color = 'Transformation'
  ) +
  theme_minimal_grid(font_family = font, font_size = 16) +
  theme(
    strip.background = element_rect("grey80"),
    panel.grid.minor = element_blank(),
    plot.title.position = "plot",
    legend.position = 'right',
    panel.background = element_rect(fill = 'white', color = NA),
    plot.background = element_rect(fill = 'white', color = NA)
  ) +
  guides(color = guide_legend(ncol = 1)) +
  panel_border()

ggsave(
  filename = file.path('results', 'plots', 'precision_recall_scatter.png'),
  width = 8,
  height = 13
)

# ---------------------------------------------------------------------------
# Plot 2b: Robustness check — GPT-4o vs GPT-4o-mini (Amazon-Google only).
# Shows all transformations as points + error bars, dodged by model tier.
# ---------------------------------------------------------------------------
robustness_lookup <- tribble(
  ~method,      ~tier,
  'llm',        'GPT-4o-mini',
  'llm_gpt4o',  'GPT-4o',
)

cv_robustness <- read_csv(file.path('results', 'cv_results.csv')) %>%
  inner_join(robustness_lookup, by = 'method') %>%
  filter(dataset == 'amazon-google') %>%
  mutate(
    dataset = str_replace_all(dataset, "-", " ") %>% str_to_title(),
    transformation = str_replace_all(transformation, "_", " ") %>%
      str_to_title(),
    transformation = factor(
      transformation,
      levels = c('Scrambled', 'Ciphered Words', 'Ciphered Letters', 'Original')
    ),
    tier = factor(tier, levels = c('GPT-4o', 'GPT-4o-mini'))
  )

cv_robustness %>%
  ggplot(aes(x = f1_mean, y = transformation, color = tier)) +
  geom_errorbarh(
    aes(xmin = f1_mean - f1_std, xmax = f1_mean + f1_std),
    height = 0.3, alpha = 0.9, linewidth = 0.5,
    position = position_dodge(width = 0.5)
  ) +
  geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
  facet_wrap(~ dataset) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c(
    'GPT-4o'         = '#1a1a1a',
    'GPT-4o-mini'    = '#636363'
  )) +
  labs(
    x = 'F1 Score',
    y = 'Transformation',
    color = 'Model'
  ) +
  theme_minimal_vgrid(font_family = font, font_size = 14) +
  theme(
    strip.background = element_rect("grey80"),
    panel.grid.minor = element_blank(),
    plot.title.position = "plot",
    legend.position = 'bottom',
    panel.background = element_rect(fill = 'white', color = NA),
    plot.background = element_rect(fill = 'white', color = NA)
  ) +
  panel_border()

ggsave(
  filename = file.path('results', 'plots', 'llm_robustness.png'),
  width = 8,
  height = 6
)

# ---------------------------------------------------------------------------
# Plots 3 & 4: Per-model LLM prompt comparison (standard vs token-fallback).
# One figure per model family, faceted by dataset. Ablations on y-axis,
# dodged points + error bars colored by prompt variant. Original shown as
# reference line.
# ---------------------------------------------------------------------------

cv_raw <- read_csv(file.path('results', 'cv_results.csv'))

# Map raw method names to model family + prompt variant
llm_lookup <- tribble(
  ~method,                            ~model,          ~prompt,
  'llm',                              'GPT-4o-mini',   'Standard',
  'llm_gpt4o_mini_token_fallback',    'GPT-4o-mini',   'Token Fallback',
  'llm_haiku',                        'Claude Haiku',   'Standard',
  'llm_haiku_token_fallback',         'Claude Haiku',   'Token Fallback',
)

cv_llm <- cv_raw %>%
  inner_join(llm_lookup, by = 'method') %>%
  mutate(
    dataset = str_replace_all(dataset, "-", " ") %>% str_to_title(),
    transformation = str_replace_all(transformation, "_", " ") %>%
      str_to_title(),
    transformation = factor(
      transformation,
      levels = c('Scrambled', 'Ciphered Words', 'Ciphered Letters', 'Original')
    ),
    prompt = factor(prompt, levels = c('Standard', 'Token Fallback'))
  )

llm_baseline <- cv_llm %>%
  filter(transformation == 'Original', prompt == 'Standard') %>%
  select(model, dataset, f1_baseline = f1_mean, sd_baseline = f1_std)

llm_ablations <- cv_llm %>%
  filter(transformation != 'Original') %>%
  mutate(transformation = fct_drop(transformation))

llm_ablations %>%
  ggplot(aes(x = f1_mean, y = transformation, color = prompt)) +
  geom_rect(
    data = llm_baseline, inherit.aes = FALSE,
    aes(
      xmin = f1_baseline - sd_baseline,
      xmax = f1_baseline + sd_baseline,
      ymin = -Inf, ymax = Inf
    ),
    fill = 'grey70', alpha = 0.35
  ) +
  geom_vline(
    data = llm_baseline,
    aes(xintercept = f1_baseline)
  ) +
  geom_errorbarh(
    aes(xmin = f1_mean - f1_std, xmax = f1_mean + f1_std),
    height = 0.3, alpha = 0.9, linewidth = 0.5,
    position = position_dodge(width = 0.5)
  ) +
  geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
  facet_grid(model ~ dataset) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c('Standard' = '#2166ac', 'Token Fallback' = '#b2182b')) +
  labs(
    x = 'F1 Score',
    y = 'Transformation',
    color = 'Prompt'
  ) +
  theme_minimal_vgrid(font_family = font, font_size = 14) +
  theme(
    strip.background = element_rect("grey80"),
    panel.grid.minor = element_blank(),
    plot.title.position = "plot",
    legend.position = 'bottom',
    panel.background = element_rect(fill = 'white', color = NA),
    plot.background = element_rect(fill = 'white', color = NA)
  ) +
  panel_border()

ggsave(
  filename = file.path('results', 'plots', 'llm_prompt_comparison.png'),
  width = 8,
  height = 6
)
