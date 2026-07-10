### Listing all downloadable reports
reports_downloaded <- list.files("localdata/esg_reports_all/", pattern = ".pdf$")
###

### Loading Gold_Standard in full
gs <- read.csv(file="gold_standard.csv")

# Finding wrong field name in gs and mapping it
setdiff(reports_downloaded, gs$report_name) #just printing the one missmatch
gs$report_name[gs$report_name == "viacomcbs_2020_report.pdf"] <- "ViacomCBS_ESG Report_2020-2021_vFINAL.pdf"

# Defining Status Column for each report, beginning with "notavail"
gs$status <- ifelse(gs$report_name %in% reports_downloaded, NaN , "notavail")

# not_downloadable reports
reports_unavail <- setdiff(gs$report_name, reports_downloaded)

# Stipping unnecessary columns
toKeep  <- c("report_name", "year", "scope", "page", "value", "unit", "unit_normalized", "status")
gs_slim <- subset(gs, select = toKeep)
gs_slim <- gs_slim[order(gs_slim$report_name), ] #Sort ascending by report_name
