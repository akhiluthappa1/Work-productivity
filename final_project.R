setwd("/Users/kevaldave/Library/CloudStorage/GoogleDrive-kmd1@bu.edu/My Drive/Boston University/Spring 2023 Semester/Data Mining/Final Project")

library(caret)
library(rsample)
library(RWeka)
library(rpart)
library(MASS)
library(kernlab)
library(pROC)
library(dplyr)
library(MLmetrics)
library(e1071)
library(tibble)
library(FSelector)
library(CORElearn)
library(corrplot)
library(caretFeatureSelection)
library(caretFeatureSelection)

# (Suppress warnings)
options(warn = -1)

df<-read.csv('2021_rws_edited.csv')

colnames(df)

## Label Encoding multiple columns in R:

cols_to_encode <- c("job_duration", "industry", "current_occupation", "total_employees", "household", "x10", "Connectivity", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10", "q11", "q12", "q13", "q14", "q15", "q16", "q17" )
df[cols_to_encode] <- lapply(df[cols_to_encode], factor)  ## as.factor() could also be used

sapply(df,class)

## Metro.or.regional, x9, q18, and gender -- One hot encode

cols_to_hot_encode <- c("Metro.or.Regional", "x9", "q18", "gender")

df_encoded <- predict(dummyVars(~., data = df[, cols_to_hot_encode]), newdata = df[, cols_to_hot_encode])

sapply(df_encoded, class)
df_encoded

## Concatenating the original df and the encoded df.
new <- cbind(df,df_encoded)
colnames(new)

## Dropping columns which were one-hot encoded above
drops <- c("gender", "Metro.or.Regional", "x9", "q18")
new <- new[ , !(names(new) %in% drops)]

colnames(new)

sapply(new,class)

## Relocating the class attribute to the end
new <- new %>% relocate(Class, .after = last_col())

## Splitting the data:

set.seed(31)

df <- new

df$Class <- factor(df$Class)
df <- df[, ]
head(df,1)
set.seed(31)
colnames(df) <- make.names(colnames(df))
split <- initial_split(df, prop = 0.66, strata = Class)
train <- training(split)
test <- testing(split)

## Removing NULL values from the train set
sum(is.na(train))
train <- na.omit(train)
names(train)
train$Class <- make.names(train$Class)

## Removing NULL values from the test set
sum(is.na(test))
test <- na.omit(test)
names(test)
test$Class <- make.names(test$Class)

cols <- ncol(train)
cols

# -------------------------------------
# FEATURE SELECTION

# Chi-Square:
trainIndex <- createDataPartition(df$Class, p = .7, list = FALSE)
training <- train[trainIndex, ]
testing <- test[-trainIndex, ]

# Subset the training data into features and target variable
X_train <- train[, -ncol(train)]
y_train <- train$Class

# Perform the chi-squared test for feature selection
chi_square_test <- function(x, y) {
  # Create a contingency table
  tab <- table(x, y)
  
  # Compute the chi-square test statistic and p-value
  result <- chisq.test(tab)
  
  # Return the p-value
  return(result$p.value)
}
chiSQ <- apply(X_train, 2, chi_square_test, y_train)
chiSQ

# Sort the p-values in ascending order
sorted_p_values <- sort(chiSQ)

# Select the top 10 features with the smallest p-values
top_10_features <- names(sorted_p_values)[1:10]
top_20_features <- names(sorted_p_values)[1:20]

top_10_features

train_chi <- training[, c(top_10_features, "Class")]
test_chi <- testing[, c(top_10_features, "Class")]

train_chi_20 <- training[, c(top_20_features, "Class")]
test_chi_20 <- testing[, c(top_20_features, "Class")]

# RFE

# Subset the training and test data to only include the top 10 features
train_chi <- training[, c(top_20_features, "Class")]
test_chi <- testing[, c(top_20_features, "Class")]

# Create a control object for the RFE algorithm
ctrl <- rfeControl(method = "cv", number = 5)
ncol(test_chi)
# Perform RFE with the top 10 features and the target variable
rfe_result <- rfe(x = train_chi[, -ncol(train_chi)], y = as.factor(train_chi$Class),
                  sizes = 1:ncol(train_chi)-1, rfeControl = ctrl)

# Select the top 5 features from the RFE results
top_10_features <- rfe_result$optVariables[1:10]
top_10_features
train_rfe <- train_chi[, c(top_10_features, "Class")]
test_rfe <- test_chi[, c(top_10_features, "Class")]

# ReliefF Algorithm:
top_20_features
train_chi <- training[, c(top_20_features, "Class")]
test_chi <- testing[, c(top_20_features, "Class")]

# create a control object for feature selection
ctrl <- rfeControl(method = "cv", number = 10)

# perform feature selection using the Gini index with rpart algorithm
relief_result <- rfe(x = train_chi[, -ncol(train_chi)], y = as.factor(train_chi$Class),
                   sizes = 1:ncol(train_chi)-1, rfeControl = ctrl, metric = "Accuracy")

relief_result

top_10_features <- relief_result$optVariables[1:10]
top_10_features
train_relief <- train_chi[, c(top_10_features, "Class")]
test_relief <- test_chi[, c(top_10_features, "Class")]

# Gini Index

train_chi <- training[, c(top_20_features, "Class")]
test_chi <- testing[, c(top_20_features, "Class")]

# create a control object for feature selection
ctrl <- rfeControl(method = "cv", number = 10, verbose = FALSE)

# perform feature selection using the Gini index with rpart algorithm
gini_result <- rfe(x = train_chi[, -ncol(train_chi)], y = as.factor(train_chi$Class),
                  sizes = 1:ncol(train_chi)-1, rfeControl = ctrl, method = "rpart", metric = "Gini")

gini_result

top_8_features <- rfe_result$optVariables[1:8]
top_8_features
train_gini <- train_chi[, c(top_8_features, "Class")]
test_gini <- test_chi[, c(top_8_features, "Class")]

# Information Gain:

# Compute the information gain of each feature
ig <- information.gain(Class ~ ., train)

# Print the information gain values
print(ig)

# Select the top k features with the highest information gain
k <- 10
ig <- as.data.frame(ig)
head(ig,5)
ig <- ig %>% rownames_to_column("id")
selected_features <- ig[order(ig$attr_importance, decreasing = TRUE), ]

# Print the selected features
selected_features <- head(selected_features$id,10)
print(selected_features)

train_ig <- train[,selected_features]
test_ig <- test[,selected_features]

Class <- train$Class
class(Class)
train_ig <- cbind(train_ig,Class)
Class <- test$Class
test_ig <- cbind(test_ig,Class)

# -----------------------------------------------

library(caret)
library(e1071)
library(rpart)
library(rpart.plot)
library(RWeka)
library(mlbench)
library(party)
library(gbm)

classify_data <- function(train, test) {
  # Train Control: 10-fold CV
  train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                                summaryFunction = defaultSummary)
  
  ### Rpart:
  
  model <- train(Class ~ ., data = train, method = "rpart", trControl = train_control,
                 tuneLength = 10)
  
  rpart_plot <- rpart.plot(model$finalModel, extra = 1)
  test_pred_rpart <- predict(model, newdata = test)
  cm_rpart <- confusionMatrix(test_pred_rpart, as.factor(test$Class))
  
  ### J48:
  
  # Tune Grid:
  J48Grid <-  expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
  
  model <- train(Class ~ ., data = train, method = "J48", trControl = train_control,
                 tuneGrid = J48Grid
  )
  
  j48_plot <- plot(model)
  test_pred_j48 <- predict(model, newdata = test)
  cm_j48 <- confusionMatrix(test_pred_j48, as.factor(test$Class))
  
  ### KNN:
  
  knnModel <- train(Class ~., data = train, method = "knn",
                    trControl=train_control,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
  
  knn_plot <- plot(knnModel)
  test_pred_knn <- predict(knnModel, newdata = test)
  cm_knn <- confusionMatrix(test_pred_knn, as.factor(test$Class))
  
  ### Gradient Boosting:
  
  ctrl <- trainControl(method = "CV",
                       summaryFunction = multiClassSummary,
                       classProbs = TRUE,
                       savePredictions = TRUE)
  
  gbmGrid <- expand.grid(interaction.depth = c(1),
                         n.trees = (3)*100,
                         shrinkage = c(.01),
                         n.minobsinnode = 3)
  
  
  gbmFit <- train(x = train[, -(ncol(train))], 
                  y = train$Class,
                  method = "gbm",
                  tuneGrid = gbmGrid,
                  metric = "ROC",
                  verbose = FALSE,
                  trControl = ctrl)
  
  # gbm_plot <- plot(gbmFit)
  test_pred_gbm <- predict(gbmFit, test)
  cm_gbm <- confusionMatrix(test_pred_gbm, as.factor(test$Class))
  
  # NaiveBayes:
  
  nbModel <- naiveBayes(Class ~ ., data = train)
  test_pred_nb <- predict(nbModel, newdata = test)
  cm_nb <- confusionMatrix(test_pred_nb, as.factor(test$Class))
  
  # Store the results in a list
  results <- list(
    rpart = list(
      model = model$finalModel,
      plot = rpart_plot,
      cm = cm_rpart
    ),
    J48 = list(
      model = model$finalModel,
      cm = cm_j48
    ),
    knn = list(
      model = knnModel,
      plot = knn_plot,
      cm = cm_knn
    ),
    gbm = list(
      model = gbmFit,
      cm = cm_gbm
    ),
    nb = list(
      model = nbModel,
      cm = cm_nb
    )
)}

# Train the Info Gain data models:
results_ig <- classify_data(train_ig, test_ig)

# Train the Chi-Square data models:
levels(train_chi_20$Class) <- c("less_productivity", "more_productivity", "same_productivity")
levels(test_chi_20$Class) <- c("less_productivity", "more_productivity", "same_productivity")
results_chi <- classify_data(train_chi_20, test_chi_20)

# Train the Gini data models:
results_gini <- classify_data(train_gini, test_gini)

# Train the RFE data models:
results_rfe <- classify_data(train_rfe, test_rfe)

# Train the relief data models:
levels(train_relief$Class) <- c("less_productivity", "more_productivity", "same_productivity")
levels(test_relief$Class) <- c("less_productivity", "more_productivity", "same_productivity")
results_relief <- classify_data(train_relief, test_relief)
library(partykit)
results_relief$gbm$cm
summary(results_relief$knn$model, class="T")
results_relief$gbm$model
results_relief$gbm$plot
results_relief
print(results_relief$nb$cm)
results_chi$nb$model
summary(results_relief$knn$model)

plot(results_relief$J48$model)

# Write dataframe to CSV file
write.csv(train, file = "Initial_Train.csv", row.names = FALSE)
write.csv(test, file = "Initial_Test.csv", row.names = FALSE)

write.csv(train_rfe, file = "RFE_Train_BestModel.csv", row.names = FALSE)
write.csv(test_rfe, file = "RFE_Test_BestModel.csv", row.names = FALSE)
