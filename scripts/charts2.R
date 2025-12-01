library(tidyverse)
library(cowplot)

options(dplyr.width = Inf)

font <- "Roboto Condensed"

# -------------------------------------------------------------------
# Load data and prepare transformations (fixed order)
# -------------------------------------------------------------------
df <- read_csv(file.path("results", "summary.csv")) %>%
  mutate(
    transformation = factor(
      transformation,
      levels = c("scrambled", "ciphered_letters", "ciphered_words", "original")
    )
  )

df_abt <- df %>% filter(dataset == "abt-buy")
df_amz <- df %>% filter(dataset == "amazon-google")

# -------------------------------------------------------------------
# Helper: F1 comparison plot for a single dataset
# -------------------------------------------------------------------
make_f1_plot <- function(df_ds, fname, title_ds) {
  # reorder methods *within this dataset* by their max F1
  df_ds <- df_ds %>%
    group_by(method) %>%
    mutate(max_f1 = max(f1)) %>%
    ungroup() %>%
    mutate(method = fct_reorder(method, -max_f1))

  df_original <- df_ds %>%
    filter(transformation == "original") %>%
    select(method, f1_original = f1)

  p <- df_ds %>%
    filter(transformation != "original") %>%
    left_join(df_original, by = "method") %>%
    ggplot() +
    geom_segment(
      aes(
        x = f1_original,
        xend = f1,
        y = transformation,
        yend = transformation,
        color = transformation
      ),
      alpha = 0.5
    ) +
    geom_vline(
      data = df_ds %>% filter(transformation == "original"),
      aes(xintercept = f1)
    ) +
    geom_point(
      aes(
        x = f1,
        y = transformation,
        color = transformation
      ),
      size = 1.5
    ) +
    facet_wrap(vars(method), ncol = 1) +
    labs(
      x = "F1 score",
      y = NULL,
      title = paste0("Effect of Text Transformations on F1 (", title_ds, ")")
    ) +
    theme_minimal_vgrid(font_family = font, font_size = 16) +
    theme(
      strip.background = element_rect("grey80"),
      panel.grid.minor = element_blank(),
      plot.title.position = "plot",
      legend.position = "none",
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    panel_border()

  ggsave(
    filename = file.path("results", "plots", fname),
    plot = p,
    width = 8,
    height = 10,
    dpi = 300
  )
}

# -------------------------------------------------------------------
# Helper: precision–recall scatter for a single dataset
# -------------------------------------------------------------------
make_pr_plot <- function(df_ds, fname, title_ds) {
  # same per-dataset method ordering for consistency
  df_ds <- df_ds %>%
    group_by(method) %>%
    mutate(max_f1 = max (f1)) %>%
    ungroup() %>%
    mutate(method = fct_reorder(method, -max_f1))

  p <- df_ds %>%
    ggplot() +
    geom_point(
      aes(
        x = recall,
        y = precision,
        color = transformation
      ),
      size = 2
    ) +
    facet_wrap(vars(method)) +
    labs(
      x = "Recall",
      y = "Precision",
      title = paste0(
        "Precision–Recall by Method and Transformation (",
        title_ds,
        ")"
      )
    ) +
    theme_minimal_grid(font_family = font, font_size = 16) +
    theme(
      strip.background = element_rect("grey80"),
      panel.grid.minor = element_blank(),
      plot.title.position = "plot",
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    panel_border()

  ggsave(
    filename = file.path("results", "plots", fname),
    plot = p,
    width = 8,
    height = 6,
    dpi = 300
  )
}

# -------------------------------------------------------------------
# Generate plots for Abt–Buy
# -------------------------------------------------------------------
make_f1_plot(
  df_ds = df_abt,
  fname = "f1_comparison_abt_buy.png",
  title_ds = "Abt–Buy"
)

make_pr_plot(
  df_ds = df_abt,
  fname = "precision_recall_scatter_abt_buy.png",
  title_ds = "Abt–Buy"
)

# -------------------------------------------------------------------
# Generate plots for Amazon–Google
# -------------------------------------------------------------------
make_f1_plot(
  df_ds = df_amz,
  fname = "f1_comparison_amazon_google.png",
  title_ds = "Amazon–Google"
)

make_pr_plot(
  df_ds = df_amz,
  fname = "precision_recall_scatter_amazon_google.png",
  title_ds = "Amazon–Google"
)
