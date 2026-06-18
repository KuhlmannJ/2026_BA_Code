df_1st <- read.csv(file="src/pipelines/pipelineB/PipelineB-Answers/1st_Qwen3-VL-32B-Thinking/***results.csv")
df_2nd <- read.csv(file="src/pipelines/pipelineB/PipelineB-Answers/2nd_Qwen3-VL-32B-Thinking/***results.csv")

summary(df_1st)
summary(df_2nd)

compare_per_report <- merge(df_1st, df_2nd, by = "report", )

compare_per_report <- compare_per_report[c("report", "pages.x", "duration.x", "duration.y")]

names(compare_per_report)[names(compare_per_report) == "pages.x"] <- "pages"
names(compare_per_report)[names(compare_per_report) == "duration.x"] <- "duration_first"
names(compare_per_report)[names(compare_per_report) == "duration.y"] <- "duration_second"

compare_per_report$second_faster    <- compare_per_report$duration_second < compare_per_report$duration_first
compare_per_report$second_faster_s  <- compare_per_report$duration_second - compare_per_report$duration_first
compare_per_report$abs_differnece   <- abs(compare_per_report$duration_second - compare_per_report$duration_first)

summary(compare_per_report)

write.csv(compare_per_report, file="compare_per_report.csv", row.names=FALSE)

# Top 5 most huge differences
compare_per_report_diff <- compare_per_report[order(compare_per_report$abs_differnece, decreasing = TRUE), ]
head(compare_per_report_diff$report)
