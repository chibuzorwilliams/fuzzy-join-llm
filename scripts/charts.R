library(tidyverse)
library(cowplot)

options(dplyr.width = Inf)

font <- 'Roboto Condensed'

# Load data
df <- read_csv(file.path('results', 'summary.csv')) %>% 
  group_by(method) %>% 
  mutate(max_f1 = max(f1)) %>% 
  ungroup() %>% 
  mutate(method = fct_reorder(method, -max_f1))

df_original <- df %>%
  filter(transformation == 'original') %>%
  select(method, f1_original = f1)

df %>%
  filter(transformation != 'original') %>%
  left_join(df_original, by = 'method') %>%
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
      filter(transformation == 'original'),
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
  facet_wrap(vars(method), ncol = 1) +
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
  ggplot() +
  geom_point(
    aes(
      x = recall, 
      y = precision,
      color = transformation
    )
  ) +
  facet_wrap(vars(method)) +
  theme_minimal_grid(font_family = font, font_size = 16) +
  theme(
    strip.background = element_rect("grey80"),
    panel.grid.minor = element_blank(),
    plot.title.position = "plot",
    panel.background = element_rect(fill = 'white', color = NA),
    plot.background = element_rect(fill = 'white', color = NA)
  ) +
  panel_border()

ggsave(
  filename = file.path('results', 'plots', 'precision_recall_scatter.png'),
  width = 8,
  height = 6
)
