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
test <- ds[-trainindex, ]





#OLS regressions
mod_full <- train(realrinc ~ female + . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr,
                  data = train, 
                  method = "lm",  
                  trControl = trainControl(method = "cv"))
summary(mod_full)
pred <-predict(mod_full, newdata = validation)
mse <- ModelMetrics::mse(validation$realrinc, pred)
cat("Mean Squared Error (MSE):", mse, "\n") #403030161
rmse <- caret::RMSE(pred = predict(mod_full, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #20075.61 

mod_wo_interactions <-train(realrinc ~ age + prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                            installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                             + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                            graduate + highschool + juniorcollege +  + divorced + married + nevermarried + separated  + female,
                            data = train, 
                            method = "lm",  
                            trControl = trainControl(method = "cv"))
summary(mod_wo_interactions)
pred <-predict(mod_wo_interactions, newdata = validation)
mse <- ModelMetrics::mse(validation$realrinc, pred)
cat("Mean Squared Error (MSE):", mse, "\n") #605727520 
rmse <- caret::RMSE(pred = predict(mod_wo_interactions, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #24611.53

mod_loginc <- train(log_realrinc ~ female + . - occrecode -wrkstat - gender -educcat -maritalcat - age_sqr,
                    data = train, 
                    method = "lm",  
                    trControl = trainControl(method = "cv"))
summary(mod_loginc)
pred <-predict(mod_loginc, newdata = validation)
mse <- ModelMetrics::mse(validation$realrinc, pred)
cat("Mean Squared Error (MSE):", mse, "\n") #1284835881  
rmse <- caret::RMSE(pred = predict(mod_loginc, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #35844.61 


mod_age2 <- train(realrinc  ~ . - occrecode -wrkstat - gender -educcat -maritalcat, 
                  data = train, 
                  method = "lm",  
                  trControl = trainControl(method = "cv"))
summary(mod_age2)
pred <-predict(mod_age2, newdata = validation)
mse <- ModelMetrics::mse(validation$realrinc, pred)
cat("Mean Squared Error (MSE):", mse, "\n") #403041571   
rmse <- caret::RMSE(pred = predict(mod_age2, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #20075.9 



m2 <- with(train,lm(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat)) ### this does not work for me
summary(pool(m2))
