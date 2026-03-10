# install.packages("lavaan")
# install.packages("jsonlite")

library(lavaan)
library(jsonlite)

args = commandArgs(trailingOnly=TRUE)


if (length(args) != 3) {
  stop("Exactly three arguments are required: 1) data path, 2) questionnaire type (PVQ or SVS). 3) high level value")
}

datapath = args[1]
questionnaire_type = args[2]
high_level_value = args[3]

# PVQ models
pvq_model_conservation <- '
  sec =~ item5 + item14 + item21 + item31 + item35
  conf =~ item7 + item16 + item28 + item36
  trad =~ item9 + item20 + item25 + item38
'

pvq_model_openness_to_change <- '
  selfdir =~ item1 + item11 + item22 + item34
  stim =~ item6 + item15 + item30
  hed =~ item10 + item26 + item37
'
pvq_model_self_transcendence <- '
  ben =~ item12 + item18 + item27 + item33
  uni =~ item3 + item8 + item19 + item23 + item29 + item40
'
pvq_model_self_enhancement <- '
  ach =~ item4 + item13 + item24 + item32
  pow =~ item2 + item17 + item39
'

## SVS model (example model, replace with actual SVS model definition)
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
## SVS models split into four parts (example models, replace with actual SVS model definitions)
svs_model_conservation <- '
  sec =~ item8 + item13 + item15 + item22 + item56
  conf =~ item11 + item20 + item40 + item47
  trad =~ item18 + item32 + item36 + item44 + item51
'
svs_model_openness_to_change <- '
  selfdir =~ item5 + item16 + item31 + item41 + item53
  stim =~ item9 + item25 + item37
  hed =~ item4 + item50 + item57
'
svs_model_self_transcendence <- '
  ben =~ item33 + item45 + item49 + item52 + item54
  uni =~ item1 + item17 + item24 + item26 + item29 + item30 + item35 + item38
'
svs_model_self_enhancement <- '
  ach =~ item34 + item39 + item43 + item55
  pow =~ item3 + item12 + item27 + item46
'

# Select the appropriate model based on the questionnaire type
# model <- ifelse(questionnaire_type == "SVS", svs_model, pvq_model)

if (questionnaire_type == "SVS") {
  "SVS"
  model_conservation <- svs_model_conservation
  model_openness_to_change <- svs_model_openness_to_change
  model_self_transcendence <- svs_model_self_transcendence
  model_self_enhancement <- svs_model_self_enhancement
} else {
  "PVQ"
  model_conservation <- pvq_model_conservation
  model_openness_to_change <- pvq_model_openness_to_change
  model_self_transcendence <- pvq_model_self_transcendence
  model_self_enhancement <- pvq_model_self_enhancement
}

if (high_level_value == "conservation") {
 model <- model_conservation
} else if (high_level_value == "openness_to_change") {
 model <- model_openness_to_change
} else if (high_level_value == "self_transcendence") {
 model <- model_self_transcendence
} else if (high_level_value == "self_enhancement") {
 model <- model_self_enhancement
}

# Load your data (assuming it is in a data frame named 'data')
data <- read.csv(datapath)

fit <- cfa(model, data = data, std.lv = TRUE)

cov_lv <- lavInspect(fit, "cov.lv")
off_diag_vals <- cov_lv[upper.tri(cov_lv)]

# print(inspect(fit, "estimates"))
# print(lavInspect(fit, "cov.lv"))
# var_table <- lavInspect(fit, "est")$theta
# print(diag(var_table))  # Check for negative variances

if (any(off_diag_vals >= 1)) {
    warning("Covariance between latent variables is >= 1.")
    # this means that the data does not distinguish between the factors - and the model is not fit
    return(-1)
}


fit_measures <- fitMeasures(fit)
cfi <- fit_measures["cfi"]
tli <- fit_measures["tli"]
srmr <- fit_measures["srmr"]
rmsea <- fit_measures["rmsea"]


# Combine the measures into a list
results <- list(cfi=cfi, tli=tli, srmr=srmr, rmsea=rmsea)

# Convert the results to JSON and print
cat(toJSON(results, pretty=TRUE))
