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

#Support Vector Machines with Linear Kernel
lin_kern <- train(realrinc ~ age + prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                    installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                    + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                    graduate + highschool + juniorcollege +  divorced + married + nevermarried + separated  + female, data = train, 
      method = 'svmLinear', preProcess = c("center","scale"),
      trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )
rmse <- caret::RMSE(pred = predict(lin_kern, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #25414.6


#L2 Regularized Linear Support Vector Machines with Class Weights
rsv_weight <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'svmLinearWeights2', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )


#L2 Regularized Support Vector Machine (dual) with Linear Kernel
rsv_linkern <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'svmLinear3', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

#Least Squares Support Vector Machine
ls <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'lssvmLinear', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

#Least Squares Support Vector Machine with Polynomial Kernel
ls_pol_kern <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'lssvmPoly', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

#Least Squares Support Vector Machine with Radial Basis Function Kernel
ls_rad_kern <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'lssvmRadial', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

#Linear Support Vector Machines with Class Weights
lweight <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'svmLinearWeights', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

#Support Vector Machines with Boundrange String Kernel
boundrange_string <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'svmBoundrangeString', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

#Support Vector Machines with Class Weights
weight <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'svmRadialWeights', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

#Support Vector Machines with Exponential String Kernel
expon_string <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'svmExpoString', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

#Support Vector Machines with Polynomial Kernel
poly_kern <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'svmPoly', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

#Support Vector Machines with Radial Basis Function Kernel
rad_kern <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'svmRadial', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

#Support Vector Machines with Spectrum String Kernel
spec_kern <- train(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr, data = train, 
                  method = 'svmSpectrumString', preProcess = c("center","scale"),
                  trControl = trainControl(method = "cv"), tuneGrid = data.frame(C = c(0.01,0.1,1,10,100)) )

