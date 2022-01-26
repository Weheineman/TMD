library(MASS)
library(reshape2)

data(crabs)

summary(crabs)

write.csv(crabs, file = "crabs.csv")