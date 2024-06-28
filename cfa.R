#install.packages("lavaan")
#install.packages("jsonlite")
library(lavaan)
library(jsonlite)

args = commandArgs(trailingOnly=TRUE)


if (length(args) != 2) {
  stop("Exactly two arguments are required: 1) data path, 2) questionnaire type (PVQ or SVS).")
}

datapath = args[1]
questionnaire_type = args[2]


# PVQ model
pvq_model <- '
  selfdir =~ item1 + item11 + item22 + item34
  stim =~ item6 + item15 + item30
  hed =~ item10 + item26 + item37
  ach =~ item4 + item13 + item24 + item32
  pow =~ item2 + item17 + item39
  sec =~ item5 + item14 + item21 + item31 + item35
  conf =~ item7 + item16 + item28 + item36
  trad =~ item9 + item20 + item25 + item38
  ben =~ item12 + item18 + item27 + item33
  uni =~ item3 + item8 + item19 + item23 + item29 + item40
'

# SVS model (example model, replace with actual SVS model definition)
svs_model <- '
  selfdir =~ item5 + item16 + item31 + item41 + item53
  stim =~ item9 + item25 + item37
  hed =~ item4 + item50 + item57
  ach =~ item34 + item39 + item43 + item55
  pow =~ item3 + item12 + item27 + item46
  sec =~ item8 + item13 + item15 + item22 + item56
  conf =~ item11 + item20 + item40 + item47
  trad =~ item18 + item32 + item36 + item44 + item51
  ben =~ item33 + item45 + item49 + item52 + item54
  uni =~ item1 + item17 + item24 + item26 + item29 + item30 + item35 + item38
'

# Select the appropriate model based on the questionnaire type
model <- ifelse(questionnaire_type == "SVS", svs_model, pvq_model)

# Load your data (assuming it is in a data frame named 'data')
data <- read.csv(datapath)

fit <- cfa(model, data = data, std.lv = TRUE)
converged <- inspect(fit, "converged")

fit_measures <- fitMeasures(fit)
cfi <- fit_measures["cfi"]
tli <- fit_measures["tli"]
srmr <- fit_measures["srmr"]
rmsea <- fit_measures["rmsea"]


# Combine the measures into a list
results <- list(cfi=cfi, tli=tli, srmr=srmr, rmsea=rmsea)

# Convert the results to JSON and print
cat(toJSON(results, pretty=TRUE))
