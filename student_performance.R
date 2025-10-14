# Install (only the first time)
install.packages(c("tidyverse", "caret", "randomForest", "ROSE", "xgboost", "pROC"))


# Load libraries
library(tidyverse)    # Data handling and visualization
library(caret)        # ML utilities (train/test split, training)
library(randomForest) # Random forest model
library(ROSE)         # Handle imbalance
library(pROC)         # ROC/AUC plots

#setwd("C:/Users/User/Desktop/R")

#getwd()

# Load your CSV manually
data <- read.csv("StudentsPerformance.csv")


# View structure
str(data)
head(data)


# Summary of dataset
summary(data)

# Check missing values
colSums(is.na(data))

# Optional: create Pass/Fail target
# For example, if Placement_Score >= 75 → pass
data$Pass_Fail <- ifelse(data$Placement_Score >= 75, "pass", "fail")
data$Pass_Fail <- factor(data$Pass_Fail, levels = c("fail", "pass"))
