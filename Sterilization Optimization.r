# Load required libraries
library(tidyverse)
library(randomForest)
library(caret)

# Load dataset
data <- read.csv("sterilization_data.csv")

# Convert relevant columns to factors
data$Sterilization.Type <- as.factor(data$Sterilization.Type)
data$Instrument.Type <- as.factor(data$Instrument.Type)

# Handle missing values (simple imputation)
data <- data %>%
  mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .))) 

# Define features and target
features <- data %>%
  select(-Timestamp, -Post.Sterilization.CFU) # drop Timestamp and target
target <- data$Post.Sterilization.CFU

# Split data into training and test sets
set.seed(123)
train_index <- createDataPartition(target, p = 0.8, list = FALSE)
train_data <- features[train_index, ]
train_target <- target[train_index]
test_data <- features[-train_index, ]
test_target <- target[-train_index]

# Train Random Forest model
rf_model <- randomForest(train_data, y = train_target, ntree = 100)

# Predict on test data
predictions <- predict(rf_model, test_data)

# Evaluate model
rmse <- sqrt(mean((predictions - test_target)^2))
r2 <- cor(predictions, test_target)^2

cat("RMSE:", rmse, "\n")
cat("R²:", r2, "\n")

# Feature importance
importance <- importance(rf_model)
varImpPlot(rf_model)
