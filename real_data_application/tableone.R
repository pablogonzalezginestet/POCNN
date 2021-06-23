
library(tableone)

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
