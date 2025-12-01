
library(tidyverse)
library(cowplot)

options(dplyr.width = Inf)

font <- 'Roboto Condensed'

# Load data
df <- read_csv(file.path('results', 'summary.csv')) %>%
  group_by(method, dataset) %>%
  mutate(max_f1 = max(f1)) %>%
  ungroup() %>%
  mutate(
    # Convert to proper title case
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
      str_replace_all("Llm", "LLM") %>%
      str_replace_all("Openai Embeddings", "OpenAI Emb.") %>%
      str_replace_all("Sentence Transformer", "Transformer"),
    # Reorder method by max F1
    method = fct_reorder(method, -max_f1)
  )

df_original <- df %>%
  filter(transformation == 'Original') %>%
  select(method, dataset, f1_original = f1)

df %>%
  filter(transformation != 'Original') %>%
  left_join(df_original, by = c('method', 'dataset')) %>%
  ggplot() +
  geom_segment(
    aes(
      x = f1_original,
      xend = f1,
      y = reorder(transformation, f1),
      yend = reorder(transformation, f1),
      color = transformation
    ),
    alpha = 0.5
  ) +
  geom_vline(
    data = df %>%
      filter(transformation == 'Original'),
    aes(xintercept = f1)
  ) +
  geom_point(
    aes(
      x = f1,
      y = reorder(transformation, f1),
      color = transformation
    ),
    size = 1
  ) +
  facet_grid(method ~ dataset) +
  scale_x_continuous(limits = c(0,1)) +
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

df %>% 
  mutate(
    transformation = fct_reorder(transformation, -f1)
  ) %>% 
  ggplot() +
  geom_point(
    aes(
      x = recall, 
      y = precision,
      color = transformation
    )
  ) +
  facet_grid(method ~ dataset) +
  scale_x_continuous(limits = c(0,1)) +
  scale_y_continuous(limits = c(0,1)) +
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
    legend.position = 'bottom',
    panel.background = element_rect(fill = 'white', color = NA),
    plot.background = element_rect(fill = 'white', color = NA)
  ) +
  guides(color = guide_legend(nrow = 2)) +
  panel_border()

ggsave(
  filename = file.path('results', 'plots', 'precision_recall_scatter.png'),
  width = 7,
  height = 13
)
