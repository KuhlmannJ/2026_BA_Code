#install.packages("rjson")
library("rjson")

df_dense <- read.csv(file="src/pipelines/pipelineB/PipelineB-Answers/Qwen3-VL-32B-Thinking/***results.csv")
df_moe   <- read.csv(file="src/pipelines/pipelineB/PipelineB-Answers/Qwen3-VL-30B-A3B-Thinking/***results.csv")

# summary(df_dense)
# summary(df_moe)

compare_per_report <- merge(df_dense, df_moe, by = "report", )

compare_per_report <- compare_per_report[c("report", "pages.x", "duration.x", "duration.y")]
# compare_per_report <- compare_per_report[c("report", "duration.x", "t_inf.page.x", "duration.y", "t_inf.page.y")]

names(compare_per_report)[names(compare_per_report) == "pages.x"] <- "pages"
names(compare_per_report)[names(compare_per_report) == "duration.x"] <- "duration_dense"
names(compare_per_report)[names(compare_per_report) == "duration.y"] <- "duration_moe"
# names(compare_per_report)[names(compare_per_report) == "t_inf.page.x"] <- "t_inf.page_dense"
# names(compare_per_report)[names(compare_per_report) == "t_inf.page.y"] <- "t_inf.page_moe"

compare_per_report$moe_faster <- compare_per_report$duration_moe < compare_per_report$duration_dense

compare_per_report[compare_per_report$moe_faster == TRUE & compare_per_report$pages == 7, ]