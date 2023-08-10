# set wd
setwd("/home/user/Downloads/info_tud/statistical_learning_UDE/gender_pay_gap_project")
setwd("C:/Users/lbergmann/OneDrive - RWI–Leibniz-Institut für Wirtschaftsforschung e.V/Dokumente/Promotion/StatisticalLearning/gender_pay_gap_project_1")

# packages
library(dplyr)
library(GGally)
library(caret)
library(ISLR)
library(tidyverse)
library(caret)
library(glmnet)



#load data 
ds <- read.csv("data/full_data_w_dummies_interaction.csv")

#split in train, validation and test data
set.seed(123)

trainindex <- createDataPartition(y = ds$realrinc, p = 0.7, list = FALSE) # Split the data into training (70%) and remaining (30%)
train <- ds[trainindex, ]
remaining <- ds[-trainindex, ]

valid_test_index <- createDataPartition(y = remaining$realrinc, p = 0.5, list = FALSE) # Split the remaining data into validation (50%) and test (50%)
validation <- remaining[valid_test_index, ]
test <- remaining[-valid_test_index, ]



#Lasso
lasso <- train(realrinc ~ female + . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                method = "glmnet", trControl = trainControl(method = "cv"), 
                tuneGrid = expand.grid(alpha = 1, lambda = seq(0,500,1)))
caret::RMSE(pred = predict(lasso, validation), obs = validation$realrinc) #19987.04


#Ridge
ridge <- train(realrinc ~ female + . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train,
  method = "glmnet", trControl = trainControl(method = "cv"),
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0, 500, 1)))
caret::RMSE(pred = predict(ridge_fit, College_test), obs = College_test$Apps)

#Elastic Net