library(tidyverse)
library(cowplot)
library(ggh4x)

options(dplyr.width = Inf)

font <- 'Roboto Condensed'

# Shared label cleaning ------------------------------------------------------
clean_labels <- function(d) {
  d %>%
    mutate(
      method = case_match(
        method,
        'llm' ~ 'GPT-4o-mini',
        'llm_gpt4o_mini_token_fallback' ~ 'GPT-4o-mini\nToken Fallback',
        'llm_haiku' ~ 'Claude Haiku',
        'llm_haiku_token_fallback' ~ 'Claude Haiku\nToken Fallback',
        'llm_claude' ~ 'Claude Sonnet\nStandard',
        'llm_gpt4o' ~ 'GPT-4o\nStandard',
        'tfidf' ~ 'TFIDF',
        'soft_tfidf' ~ 'Soft TFIDF',
        'openai_embeddings' ~ 'OpenAI Emb.',
        'sentence_transformer' ~ 'Transformer',
        .default = str_replace_all(method, "_", " ") %>% str_to_title()
      ),
      dataset = str_replace_all(dataset, "-", " ") %>% str_to_title(),
      transformation = str_replace_all(transformation, "_", " ") %>%
        str_to_title()
    )
}

# Methods to exclude from main plots (expensive models + token-fallback variants
# shown separately in robustness/prompt-comparison charts)
expensive_methods <- c(
  'llm_claude',
  'llm_gpt4o',
  'llm_gpt4o_mini_token_fallback',
  'llm_haiku_token_fallback'
)

# LLM/embedding methods (raw names) — used to assign method_group in main plots
llm_methods_raw <- c('llm', 'llm_haiku', 'openai_embeddings', 'sentence_transformer')

# ---------------------------------------------------------------------------
# Plot 1: F1 by transformation, with 5-fold CV error bars (mean +/- 1 SD).
# Excludes expensive models (Sonnet, GPT-4o) — those are in a separate chart.
# ---------------------------------------------------------------------------
cv <- read_csv(file.path('results', 'cv_results.csv')) %>%
  filter(!method %in% expensive_methods) %>%
  mutate(method_group = if_else(
    method %in% llm_methods_raw, 'LLM / Embeddings', 'String Distance'
  )) %>%
  clean_labels() %>%
  group_by(method, dataset) %>%
  mutate(max_f1 = max(f1_mean)) %>%
  ungroup()

# Order methods by max F1 within each group (LLM first, then string distance)
cv_method_order <- cv %>%
  distinct(method, method_group) %>%
  left_join(
    cv %>% group_by(method) %>% summarise(top_f1 = max(max_f1), .groups = 'drop'),
    by = 'method'
  ) %>%
  arrange(method_group, desc(top_f1)) %>%
  pull(method)

cv <- cv %>%
  mutate(
    method_group = factor(method_group, levels = c('LLM / Embeddings', 'String Distance')),
    method = factor(method, levels = cv_method_order),
    transformation = factor(
      transformation,
      levels = c('Scrambled', 'Ciphered Words', 'Ciphered Letters', 'Original')
    )
  )

cv_baseline <- cv %>%
  filter(transformation == 'Original') %>%
  select(method, method_group, dataset, f1_baseline = f1_mean, sd_baseline = f1_std)

cv_ablations <- cv %>%
  filter(transformation != 'Original') %>%
  mutate(transformation = fct_drop(transformation))

group_header <- function(label) {
  ggdraw() +
    draw_label(label, fontface = 'bold', color = 'white', fontfamily = font, size = 14) +
    theme(plot.background = element_rect(fill = '#555555', color = NA))
}

f1_panel <- function(ablations, baseline, show_x = FALSE) {
  p <- ablations %>%
    ggplot(aes(x = f1_mean, y = transformation, color = transformation)) +
    geom_rect(
      data = baseline, inherit.aes = FALSE,
      aes(xmin = f1_baseline - sd_baseline, xmax = f1_baseline + sd_baseline,
          ymin = -Inf, ymax = Inf),
      fill = 'grey70', alpha = 0.35
    ) +
    geom_vline(data = baseline, aes(xintercept = f1_baseline)) +
    geom_errorbarh(
      aes(xmin = f1_mean - f1_std, xmax = f1_mean + f1_std),
      height = 0.3, alpha = 0.9, linewidth = 0.5
    ) +
    geom_point(size = 1.6) +
    facet_grid(method ~ dataset) +
    scale_x_continuous(limits = c(0, 1)) +
    labs(x = if (show_x) 'F1 Score' else NULL, y = 'Transformation') +
    theme_minimal_vgrid(font_family = font, font_size = 16) +
    theme(
      strip.background = element_rect('grey80'),
      panel.grid.minor = element_blank(),
      legend.position = 'none',
      panel.background = element_rect(fill = 'white', color = NA),
      plot.background = element_rect(fill = 'white', color = NA)
    ) +
    panel_border()
  if (!show_x) p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  p
}

n_llm <- n_distinct(cv_ablations$method[cv_ablations$method_group == 'LLM / Embeddings'])
n_str <- n_distinct(cv_ablations$method[cv_ablations$method_group == 'String Distance'])

plot_grid(
  group_header('LLM / Embeddings'),
  f1_panel(
    cv_ablations %>% filter(method_group == 'LLM / Embeddings'),
    cv_baseline  %>% filter(method_group == 'LLM / Embeddings'),
    show_x = FALSE
  ),
  group_header('String Distance'),
  f1_panel(
    cv_ablations %>% filter(method_group == 'String Distance'),
    cv_baseline  %>% filter(method_group == 'String Distance'),
    show_x = TRUE
  ),
  ncol = 1,
  rel_heights = c(0.2, n_llm, 0.2, n_str),
  align = 'v', axis = 'lr'
)

ggsave(
  filename = file.path('results', 'plots', 'f1_comparison.png'),
  width = 8,
  height = 11.5
)

# ---------------------------------------------------------------------------
# Plot 2: Precision-recall scatter (point estimates from summary.csv).
# Excludes expensive models.
# ---------------------------------------------------------------------------
df <- read_csv(file.path('results', 'summary.csv')) %>%
  filter(!method %in% expensive_methods) %>%
  mutate(method_group = if_else(
    method %in% llm_methods_raw, 'LLM / Embeddings', 'String Distance'
  )) %>%
  clean_labels() %>%
  group_by(method, dataset) %>%
  mutate(max_f1 = max(f1)) %>%
  ungroup()

df_method_order <- df %>%
  distinct(method, method_group) %>%
  left_join(
    df %>% group_by(method) %>% summarise(top_f1 = max(max_f1), .groups = 'drop'),
    by = 'method'
  ) %>%
  arrange(method_group, desc(top_f1)) %>%
  pull(method)

df <- df %>%
  mutate(
    method_group = factor(method_group, levels = c('LLM / Embeddings', 'String Distance')),
    method = factor(method, levels = df_method_order),
    transformation = fct_reorder(transformation, -f1)
  )

pr_panel <- function(data, show_legend = FALSE) {
  p <- data %>%
    ggplot() +
    geom_point(aes(x = recall, y = precision, color = transformation)) +
    facet_grid(method ~ dataset) +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(x = 'Recall', y = 'Precision', color = 'Transformation') +
    theme_minimal_grid(font_family = font, font_size = 16) +
    theme(
      strip.background = element_rect('grey80'),
      panel.grid.minor = element_blank(),
      legend.position = if (show_legend) 'right' else 'none',
      panel.background = element_rect(fill = 'white', color = NA),
      plot.background = element_rect(fill = 'white', color = NA)
    ) +
    guides(color = guide_legend(ncol = 1)) +
    panel_border()
  p
}

n_llm_pr <- n_distinct(df$method[df$method_group == 'LLM / Embeddings'])
n_str_pr <- n_distinct(df$method[df$method_group == 'String Distance'])

plot_grid(
  group_header('LLM / Embeddings'),
  pr_panel(df %>% filter(method_group == 'LLM / Embeddings'), show_legend = FALSE),
  group_header('String Distance'),
  pr_panel(df %>% filter(method_group == 'String Distance'), show_legend = TRUE),
  ncol = 1,
  rel_heights = c(0.2, n_llm_pr, 0.2, n_str_pr),
  align = 'v', axis = 'lr'
)

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
  ~method     , ~tier         ,
  'llm'       , 'GPT-4o-mini' ,
  'llm_gpt4o' , 'GPT-4o'      ,
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
    tier = factor(tier, levels = c('GPT-4o-mini', 'GPT-4o'))
  )

robustness_baseline <- cv_robustness %>%
  filter(transformation == 'Original') %>%
  select(tier, dataset, f1_baseline = f1_mean, sd_baseline = f1_std)

robustness_ablations <- cv_robustness %>%
  filter(transformation != 'Original') %>%
  mutate(transformation = fct_drop(transformation))

robustness_ablations %>%
  ggplot(aes(x = f1_mean, y = transformation)) +
  geom_rect(
    data = robustness_baseline,
    inherit.aes = FALSE,
    aes(
      xmin = f1_baseline - sd_baseline,
      xmax = f1_baseline + sd_baseline,
      ymin = -Inf,
      ymax = Inf
    ),
    fill = 'grey70',
    alpha = 0.35
  ) +
  geom_vline(
    data = robustness_baseline,
    aes(xintercept = f1_baseline)
  ) +
  geom_errorbarh(
    aes(xmin = f1_mean - f1_std, xmax = f1_mean + f1_std),
    height = 0.3,
    alpha = 0.9,
    linewidth = 0.5
  ) +
  geom_point(size = 2.5) +
  facet_grid(tier ~ dataset) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(
    x = 'F1 Score',
    y = 'Transformation'
  ) +
  theme_minimal_vgrid(font_family = font, font_size = 14) +
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
  ~method                         , ~model         , ~prompt          ,
  'llm'                           , 'GPT-4o-mini'  , 'Standard'       ,
  'llm_gpt4o_mini_token_fallback' , 'GPT-4o-mini'  , 'Token Fallback' ,
  'llm_haiku'                     , 'Claude Haiku' , 'Standard'       ,
  'llm_haiku_token_fallback'      , 'Claude Haiku' , 'Token Fallback' ,
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
    data = llm_baseline,
    inherit.aes = FALSE,
    aes(
      xmin = f1_baseline - sd_baseline,
      xmax = f1_baseline + sd_baseline,
      ymin = -Inf,
      ymax = Inf
    ),
    fill = 'grey70',
    alpha = 0.35
  ) +
  geom_vline(
    data = llm_baseline,
    aes(xintercept = f1_baseline)
  ) +
  geom_errorbarh(
    aes(xmin = f1_mean - f1_std, xmax = f1_mean + f1_std),
    height = 0.3,
    alpha = 0.9,
    linewidth = 0.5,
    position = position_dodge(width = 0.5)
  ) +
  geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
  facet_grid(model ~ dataset) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_color_manual(
    values = c('Standard' = '#2166ac', 'Token Fallback' = '#b2182b')
  ) +
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
