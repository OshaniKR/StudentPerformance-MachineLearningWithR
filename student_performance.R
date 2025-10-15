# Install (only the first time)
install.packages(c("tidyverse", "caret", "randomForest", "ROSE", "xgboost", "pROC"))


# Load libraries
library(tidyverse)    # Data handling and visualization

library(caret)        # ML utilities (train/test split, training)
library(randomForest) # Random forest model
library(ROSE)         # Handle imbalance
library(pROC)         # ROC/AUC plots

setwd("C:/Users/User/Desktop/R")

getwd()

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


# Check class distribution
table(data$Pass_Fail)
prop.table(table(data$Pass_Fail))


# Convert Club_Join_Date to factor (categorical)
data$Club_Join_Date <- as.factor(data$Club_Join_Date)

# Optionally scale numeric columns
numeric_cols <- c("Math_Score", "Reading_Score", "Writing_Score", "Placement_Score")
data[numeric_cols] <- scale(data[numeric_cols])

# Final dataset ready for ML
str(data)
head(data)





set.seed(123)
train_index <- createDataPartition(data$Pass_Fail, p = 0.75, list = FALSE)
train <- data[train_index, ]
test  <- data[-train_index, ]


# Check class balance in train/test
table(train$Pass_Fail)
table(test$Pass_Fail)

ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)


set.seed(123)
model_log <- train(Pass_Fail ~ ., data = train,
                   method = "glm", family = "binomial",
                   metric = "ROC", trControl = ctrl)

set.seed(123)
model_rf <- train(Pass_Fail ~ ., data = train,
                  method = "rf", metric = "ROC", trControl = ctrl,
                  ntree = 200)


# Predictions on training data
train_pred_log <- predict(model_log, newdata = train)

# Training confusion matrix
confusionMatrix(train_pred_log, train$Pass_Fail, positive = "pass")





# Predictions for test 
pred_log <- predict(model_log, newdata = test)
pred_rf  <- predict(model_rf, newdata = test)

# Confusion Matrices for test
confusionMatrix(pred_log, test$Pass_Fail, positive = "pass")
confusionMatrix(pred_rf, test$Pass_Fail, positive = "pass")

# ROC & AUC (Random Forest example)
pred_rf_prob <- predict(model_rf, newdata = test, type = "prob")
roc_rf <- roc(response = test$Pass_Fail, predictor = pred_rf_prob[,"pass"])
plot(roc_rf, main = "Random Forest ROC Curve")
auc(roc_rf)

varImp(model_rf)
plot(varImp(model_rf), main = "Random Forest Variable Importance")

saveRDS(model_rf, "best_student_model.rds")
# Later load with:
# model <- readRDS("best_student_model.rds")

