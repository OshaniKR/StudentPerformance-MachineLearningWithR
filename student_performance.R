# -------------------------------
# 1. Load Required Libraries
# -------------------------------
library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)

# -------------------------------
# 2. Load Dataset
# -------------------------------
# (Make sure you've already set your working directory)
# Example:
#setwd("C:/Desktop/R/StudentsPerformance.csv")

data <- read.csv("StudentsPerformance.csv")

# -------------------------------
# 3. Data Preprocessing
# -------------------------------

# Convert Club_Join_Date to factor (if it represents a categorical year)
data$Club_Join_Date <- as.factor(data$Club_Join_Date)

# Create new feature: average score
data$Average_Score <- rowMeans(data[, c("Math_Score", "Reading_Score", "Writing_Score")], na.rm = TRUE)

# Create binary target variable: Pass or Fail (pass if avg >= 60)
data$Pass_Fail <- ifelse(data$Average_Score >= 60, "Pass", "Fail")
data$Pass_Fail <- as.factor(data$Pass_Fail)

# Optional: Adjust threshold for better balance (uncomment if too imbalanced)
# data$Pass_Fail <- ifelse(data$Average_Score >= 50, "Pass", "Fail")  # Lower to 50 for ~50/50 split
# data$Pass_Fail <- as.factor(data$Pass_Fail)

# Handle missing values on the full dataset (remove rows with any NA for simplicity)
sapply(data, function(x) sum(is.na(x)))
data <- na.omit(data)

summary(data)

# Check overall class distribution
cat("Overall Class Distribution:\n")
overall_table <- table(data$Pass_Fail)
print(overall_table)
minority_prop <- min(overall_table) / sum(overall_table)
if (minority_prop < 0.05) {
  warning("Extreme imbalance (<5% minority class)! Consider lowering threshold or collecting more data.")
}

# Optional: Check correlations among numeric predictors (base R version)
numeric_cols <- data %>% select_if(is.numeric) %>% select(-Average_Score)  # Exclude derived avg
if (ncol(numeric_cols) > 1) {
  cor_matrix <- cor(numeric_cols, use = "complete.obs")
  cat("\nCorrelation Matrix (High values indicate potential multicollinearity):\n")
  print(round(cor_matrix, 2))
}

# -------------------------------
# 4. Split Data into Train/Test (Robust: Tries multiple seeds, then resamples if needed)
# -------------------------------
# Function to check if both classes are present
has_both_classes <- function(df) {
  length(unique(df$Pass_Fail)) == 2
}

# Try multiple seeds for stratified split until train has both classes
possible_seeds <- c(123, 456, 789, 101, 202)
train <- NULL
test <- NULL
successful_seed <- NULL

for (seed in possible_seeds) {
  set.seed(seed)
  train_index <- createDataPartition(data$Pass_Fail, p = 0.6, list = FALSE)  # 60/40 for larger train
  temp_train <- data[train_index, ]
  temp_test <- data[-train_index, ]
  
  if (has_both_classes(temp_train) && has_both_classes(temp_test)) {
    train <- temp_train
    test <- temp_test
    successful_seed <- seed
    cat(paste("\nSuccessful split with seed:", seed, "(60/40 ratio)\n"))
    break
  }
}

# If no good split found, use first seed and resample train to balance
if (is.null(train)) {
  cat("\nNo balanced split found. Using seed 123 and oversampling minority class in train.\n")
  set.seed(123)
  train_index <- createDataPartition(data$Pass_Fail, p = 0.6, list = FALSE)
  train <- data[train_index, ]
  test <- data[-train_index, ]
  
  # Oversample minority class in train using caret::upSample
  train <- upSample(x = train[, !names(train) %in% "Pass_Fail"], 
                    y = train$Pass_Fail, 
                    data = train)
  # Note: upSample duplicates minority samples; order may be shuffled
}

# Verify final split balance
cat("\nFinal Train Class Distribution (after fixes):\n")
print(table(train$Pass_Fail))
cat("\nFinal Test Class Distribution:\n")
print(table(test$Pass_Fail))

has_both_train <- has_both_classes(train)
has_both_test <- has_both_classes(test)

if (!has_both_train) {
  stop("Could not balance train set! Dataset may have too few minority samples. Check data.")
}
if (!has_both_test) {
  warning("Test set still has only one class after fixes. Predictions will be majority-class baseline.")
}

# -------------------------------
# 5. Train Models
# -------------------------------

## (a) Logistic Regression (Switched to glmnet for regularization)
# Formula: Use non-score features to avoid multicollinearity
predictors <- c("Placement_Score", "Club_Join_Date")  # Adjust if columns missing
formula_str <- paste("Pass_Fail ~", paste(predictors, collapse = " + "))
model_formula <- as.formula(formula_str)

model_log <- train(
  model_formula,
  data = train,
  method = "glmnet",  # Handles convergence/separation
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
  tuneLength = 10,
  preProcess = c("center", "scale"),
  metric = "ROC"
)

print(model_log)  # Print model summary

## (b) Random Forest
model_rf <- randomForest(
  model_formula,
  data = train,
  ntree = 200,
  importance = TRUE
)

# Optional: Variable importance plot for RF
varImpPlot(model_rf)

# -------------------------------
# 6. Predictions and Accuracy
# -------------------------------
pred_log <- predict(model_log, newdata = test)
pred_rf <- predict(model_rf, newdata = test)

acc_log <- mean(pred_log == test$Pass_Fail)
acc_rf <- mean(pred_rf == test$Pass_Fail)

cat("Logistic Regression Accuracy:", round(acc_log * 100, 2), "%\n")
cat("Random Forest Accuracy:", round(acc_rf * 100, 2), "%\n")

# Optional: Confusion matrices (skip if test lacks both classes)
if (has_both_test) {
  print(confusionMatrix(pred_log, test$Pass_Fail))
  print(confusionMatrix(pred_rf, test$Pass_Fail))
} else {
  cat("Skipping confusion matrix: Test set lacks both classes.\n")
}

# -------------------------------
# 7. Visualization
# -------------------------------
accuracy_df <- data.frame(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(acc_log * 100, acc_rf * 100)
)

p <- ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = paste0(round(Accuracy, 2), "%")), vjust = -0.5) +
  theme_minimal() +
  ggtitle("Model Accuracy Comparison") +
  ylab("Accuracy (%)") +
  xlab("Machine Learning Model")
print(p)

# -------------------------------
# 8. Save Result Plot
# -------------------------------
ggsave("results.png", plot = p, width = 6, height = 4)

# Optional: Save models for later use
# save(model_log, model_rf, file = "trained_models.RData")

