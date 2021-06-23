
# Code to create the table 1 that appears in the appendix that describe the predictors used as clinical variables in the CNN
library(tableone)

# read the final data that was preprocessed and then save as "clinical_data.csv"
dt <- read.table("clinical_data.csv", header=TRUE, sep=",")


# Vector of categorical variables that need transformation
catVars <- c("race", "ethnicity", "pathologic_stage","molecular_subtype")
# Vector with all variables
myVars <- c("time_to_event", "event", "age", "race", "ethnicity", "pathologic_stage","molecular_subtype"  )
# Table
tab2 <- CreateTableOne(vars = myVars, data = dt, factorVars = catVars)
tab2

tab2 = print(tab2, printToggle = FALSE, noSpaces = TRUE)
library(knitr)
kable(tab2, format = "latex")
