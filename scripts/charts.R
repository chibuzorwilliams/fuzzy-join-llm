library(tidyverse)
library(cowplot)

options(dplyr.width = Inf)

font <- 'Roboto Condensed'

# Shared label cleaning ------------------------------------------------------
clean_labels <- function(d) {
  d %>%
    mutate(
      method = str_replace_all(method, "_", " ") %>%
        str_to_title() %>%
        str_replace_all("Tfidf", "TFIDF") %>%
        str_replace_all("Llm", "LLM") %>%
        str_replace_all("Openai Embeddings", "OpenAI Emb.") %>%
        str_replace_all("Sentence Transformer", "Transformer"),
      dataset = str_replace_all(dataset, "-", " ") %>% str_to_title(),
      transformation = str_replace_all(transformation, "_", " ") %>%
        str_to_title() %>%
        str_replace_all("Tfidf", "TFIDF") %>%
        str_replace_all("Llm", "LLM")
    )
}

# ---------------------------------------------------------------------------
# Plot 1: F1 by transformation, with 5-fold CV error bars (mean +/- 1 SD).
# The three ablations are points + horizontal error bars. Original text is the
# reference: a solid vertical line at its mean F1 with a faint shaded band for
# its +/- 1 SD, so the reader can see whether an ablation's drop is within noise.
# ---------------------------------------------------------------------------
cv <- read_csv(file.path('results', 'cv_results.csv')) %>%
  clean_labels() %>%
  group_by(method, dataset) %>%
  mutate(max_f1 = max(f1_mean)) %>%
  ungroup() %>%
  mutate(
    method = fct_reorder(method, -max_f1),
    # bottom-to-top order of the ablation rows
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
# ---------------------------------------------------------------------------
df <- read_csv(file.path('results', 'summary.csv')) %>%
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
