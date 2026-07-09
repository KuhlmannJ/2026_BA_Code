library(dplyr)
library(ggplot2)

gs_slim <- read.csv("evaluations/gs_slim.csv", stringsAsFactors = FALSE)

## Python schreibt scopes_present / years_present als String-Repräsentation
## einer Liste (z.B. "['1', '2lb', '2mb', '3']") -> in echte R-Listen parsen
parse_list_col <- function(x) {
  lapply(x, function(s) {
    vals <- regmatches(s, gregexpr("'([^']*)'", s))[[1]]
    gsub("'", "", vals)
  })
}

# Ein Report = eine Zeile (scopes_present/years_present sind je Report konstant)
reports <- distinct(gs_slim, report_name, scopes_present, years_present)
reports$scopes_present <- parse_list_col(reports$scopes_present)
reports$years_present  <- parse_list_col(reports$years_present)

########################################
## Plot 1: Häufigkeit der Jahre über alle Reports
########################################

years_count <- table(unlist(reports$years_present))
years_df <- data.frame(year = names(years_count), n = as.integer(years_count))

p_years <- ggplot(years_df, aes(x = year, y = n)) +
  geom_col(fill = "#2a78d6", width = 0.6) +
  geom_text(aes(label = n), vjust = -0.4, size = 3.5) +
  labs(
    title = "Häufigkeit belegter Jahre",
    x = "Jahr", y = "Anzahl Reports"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())

ggsave("rScripts/plot_years_present.png", p_years, width = 7, height = 5, dpi = 300)

########################################
## Plot 2: Häufigkeit der Scopes über alle Reports
########################################

scopes_count <- table(factor(unlist(reports$scopes_present), levels = c("1", "2lb", "2mb", "3")))
scopes_df <- data.frame(scope = names(scopes_count), n = as.integer(scopes_count))

p_scopes <- ggplot(scopes_df, aes(x = scope, y = n)) +
  geom_col(fill = "#2a78d6", width = 0.6) +
  geom_text(aes(label = n), vjust = -0.4, size = 3.5) +
  labs(
    title = "Häufigkeit abgedeckter Scopes",
    x = "Scope", y = "Anzahl Reports"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())

ggsave("rScripts/plot_scopes_present.png", p_scopes, width = 6, height = 5, dpi = 300)
