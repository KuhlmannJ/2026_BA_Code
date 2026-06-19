rm(list=ls())

df_think <- read.csv(file="src/pipelines/pipelineB/PipelineB-Answers/1st_Qwen3-VL-32B-Thinking/***results.csv")
df_instr <- read.csv(file="src/pipelines/pipelineB/PipelineB-Answers/1st Qwen3-VL-32B-Instruct/***results.csv")

summary(df_think)
summary(df_instr)

compare_per_report <- merge(df_think, df_instr, by = "report", )

compare_per_report <- compare_per_report[c("report", "pages.x", "duration.x", "duration.y")]

names(compare_per_report)[names(compare_per_report) == "pages.x"] <- "pages"
names(compare_per_report)[names(compare_per_report) == "duration.x"] <- "duration_think"
names(compare_per_report)[names(compare_per_report) == "duration.y"] <- "duration_instr"

compare_per_report$instr_faster     <- compare_per_report$duration_instr < compare_per_report$duration_think
compare_per_report$second_faster_s  <- compare_per_report$duration_think - compare_per_report$duration_instr
compare_per_report$abs_differnece   <- abs(compare_per_report$duration_think - compare_per_report$duration_instr)

summary(compare_per_report)

write.csv(compare_per_report, file="PipelineB_Moe_V1-V2.csv", row.names=FALSE)

# Top 5 most huge differences
compare_per_report_diff <- compare_per_report[order(compare_per_report$abs_differnece, decreasing = TRUE), ]
head(compare_per_report_diff$report)
