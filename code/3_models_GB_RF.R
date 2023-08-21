# set wd
setwd("/home/user/Downloads/info_tud/statistical_learning_UDE/gender_pay_gap_project")
# import functions file
source('code_old/utils_functions.R')

# packages
library(caret)
library(dplyr)
library(lightgbm)
library(Matrix)
library(parallel)
library(sgd)
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(data.table)
# remotes::install_github("mlr-org/mlr3extralearners@*release")
library(mlr3extralearners)

# load data
ds <- fread("data/full_data_w_dummies_interaction.csv")
ds$wrkstat <- as.factor(ds$wrkstat)
ds$educcat <- as.factor(ds$educcat)
ds$gender <- as.factor(ds$gender)
ds$occrecode <- as.factor(ds$occrecode)
ds$maritalcat <- as.factor(ds$maritalcat)

# split in train, validation and test data
set.seed(123)

trainindex_filter <- createDataPartition(y = ds$realrinc, p = 0.7, list = FALSE) # Split the data into training (70%) and remaining (30%)
train_filter <- ds[trainindex_filter, ]
remaining_filter <- ds[-trainindex_filter, ]

# valid_test_index_filter <- createDataPartition(y = remaining_filter$realrinc, p = 0.5, list = FALSE) # Split the remaining data into validation (50%) and test (50%)
# validation_filter <- remaining_filter[valid_test_index_filter, ]
# test_filter <- remaining_filter[-valid_test_index_filter, ]

cat("training dim: ", nrow(train_filter), ", test dim: ", nrow(remaining_filter), "\n")

full.col.names <- colnames(ds)
occrecode.labels <- full.col.names[match("armed_forces", full.col.names):match("transportation", full.col.names)]
occrecode.labels <- occrecode.labels[occrecode.labels != "armed_forces"]
educcat.labels <- full.col.names[match("bachelor", full.col.names):match("less_than_high_school", full.col.names)]
educcat.labels <- educcat.labels[educcat.labels != "junior_college"]
prest.qt.labels <- c("prestg16until30", "prestg31until50", "prestggreater51")
all.label.inter <- c(occrecode.labels, educcat.labels, prest.qt.labels)

x_cols_inter <- list()
for (name.col in full.col.names[c(60:ncol(ds))]){
  split_strings <- strsplit(name.col, ".", fixed = TRUE)
  # Access the individual elements
  first_element <- split_strings[[1]][1]
  second_element <- split_strings[[1]][2]
  if ((first_element %in% all.label.inter) & (second_element %in% all.label.inter)){
    x_cols_inter[[length(x_cols_inter) + 1]] <- paste(first_element, second_element, sep = ".")
  }
}
x_cols_inter <- unlist(x_cols_inter)
  
# create data sets with lbg format
# select regressors and outcome
# x_cols <- colnames(ds)[match("age", colnames(ds)):match("age_sqr", colnames(ds))]

x_cols <- c(colnames(ds)[match("age", colnames(ds)):match("childs", colnames(ds))],
            "inter_age_prestg","inter_age_childs", "inter_prestg_childs",
            "age_sqr", "wrkstat","educcat","gender", "occrecode", "maritalcat", x_cols_inter)
ytarget <- "realrinc"

x_cols_dummys <- c( colnames(ds)[match("age", colnames(ds)):match("childs", colnames(ds))],
                    colnames(ds)[match("armed_forces", colnames(ds)):match("transportation", colnames(ds))],
                    colnames(ds)[match("full_time", colnames(ds)):match("unemployed_laid_off" , colnames(ds))],
                    colnames(ds)[match("bachelor", colnames(ds)):match("less_than_high_school" , colnames(ds))],
                    colnames(ds)[match("divorced", colnames(ds)):match("widowed", colnames(ds))],
                    "female", "age_sqr",
                    "inter_age_prestg","inter_age_childs", "inter_prestg_childs", x_cols_inter)
x_cols_dummys <- x_cols_dummys[!x_cols_dummys %in% c("armed_forces", "other_wrkstat", "junior_college", "separated")]

# lgb.format.train <- lgb.Dataset(data = as.matrix( train_filter %>% 
#                                            select(all_of(x_cols)) ),
#                         label = as.vector( as.matrix(train_filter[,ytarget]) ) 
#                         )
# lgb.format.test <- lgb.Dataset(data = as.matrix( test_filter %>% 
#                                            select(all_of(x_cols)) ),
#                        label = as.vector(as.matrix(test_filter[,ytarget] ) ) 
#                        )
# lgb.format.valid <- lgb.Dataset(data = as.matrix( validation_filter %>% 
#                                             select(all_of(x_cols)) ),
#                         label = as.vector(as.matrix(validation_filter[,ytarget] ) ) 
#                         )

# ******************************************************************************
# RF experiments
# establish hyper parameters RF
# Convert the data to a Task object, define task
task <- TaskRegr$new(id = "gap_task",
                     backend = ( train_filter %>% 
                                   select(all_of(c(x_cols,ytarget))) ),
                     target = ytarget)

# define tuning type, batch_size: # of config./combinations per batch
tnr_rdgrid_search = tnr("random_search", batch_size = 10)

# define cross-validation and error metric
rsmp_cv3 = rsmp("cv", folds = 3)
msr_ce = msr("regr.rmse")

# define learner with hyper parameters
# For example, if bagging_freq = 5, bagging will be performed every 5 iterations. 
# This means that the first 5 trees (iterations 1 to 5) will be trained on the entire training data, 
# then the next 5 trees (iterations 6 to 10) will be trained on a random subsample of the training data, and so on.
# For example, if bagging_fraction = 0.8, each tree will be trained on a random 80% subsample of the training data, 
# selected with replacement.

learner = lrn("regr.lightgbm",
              boosting = "rf",
              objective = "regression",
              max_depth = to_tune(seq(5,12,1)),
              num_leaves = to_tune(seq(270,280,1)),
              num_iterations  = to_tune(seq(132,137,1)),
              min_data_in_leaf = to_tune(seq(70,85,1)),
              min_data_in_bin = to_tune(seq(10,15, 1)),
              feature_fraction_bynode = to_tune(seq(0.3,0.4,0.1)),
              bagging_fraction = to_tune(seq(0.2,0.3,0.1)),
              bagging_freq = to_tune(seq(3,5,1)),
              feature_fraction = to_tune(seq(0.7,0.9,0.1)),
              convert_categorical = TRUE,
              force_col_wise = TRUE,
              verbose = 1,
              num_threads = 5
)

# 3rd version
# learner = lrn("regr.lightgbm",
#               boosting = "rf",
#               objective = "regression",
#               max_depth = to_tune(seq(5,10,1)),
#               num_leaves = to_tune(seq(230,240,1)),
#               num_iterations  = to_tune(seq(80,90,1)),
#               min_data_in_leaf = to_tune(seq(100,110,1)),
#               min_data_in_bin = to_tune(seq(7,10, 1)),
#               feature_fraction_bynode = to_tune(seq(0.3,0.4,0.1)),
#               bagging_fraction = to_tune(seq(0.2,0.3,0.1)),
#               bagging_freq = to_tune(seq(5,8,1)),
#               feature_fraction = to_tune(seq(0.7,0.8,0.1)),
#               convert_categorical = TRUE,
#               force_col_wise = TRUE,
#               verbose = 1,
#               num_threads = 5
# )

# 2nd version
# learner = lrn("regr.lightgbm",
#               boosting = "rf",
#               objective = "regression",
#               max_depth = to_tune(seq(15,23,1)),
#               num_leaves = to_tune(seq(70,150,1)),
#               num_iterations  = to_tune(seq(70,87,1)),
#               min_data_in_leaf = to_tune(seq(11,20,1)),
#               min_data_in_bin = to_tune(seq(50,70, 1)),
#               feature_fraction_bynode = to_tune(seq(0.3,0.5,0.1)),
#               bagging_fraction = to_tune(seq(0.7,0.8,0.1)),
#               bagging_freq = to_tune(seq(8,13,1)),
#               convert_categorical = TRUE,
#               force_col_wise = TRUE,
#               verbose = 1,
#               num_threads = 5
# )

# begin training process, with 100 configurations
init_time <- Sys.time()
instance.rf = tune(
  tuner = tnr_rdgrid_search,
  task = task,
  learner = learner,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 150,
  store_models = FALSE
)
cat('finish gb.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.rf <- (instance.rf$result)
result.table.rf <- (as.data.table(instance.rf$archive, measures = msrs(c("regr.mse","regr.mae")) ))
View(result.table.rf)
# plot(result.table.rf$feature_fraction, result.table.rf$regr.rmse)

# result.cv.table.rf <- (instance.rf$archive$benchmark_result$score())
# id_min_cv_rf <- which.min(result.table.rf$regr.mse)
# best_cv_results_rf <- result.cv.table.rf[(result.cv.table.rf$nr == id_min_cv_rf),]
# id_min_cv_model_rf <- which.min(best_cv_results_rf$regr.mse)
# best model of the best CV
# best_model_cv_rf <- instance.rf$archive$learners(
#   uhash = instance.rf$archive$benchmark_result$uhashes[id_min_cv_rf])[[id_min_cv_model_rf]]$model

# fit final model on complete data set with the best combination
lrn_rf_tuned = lrn("regr.lightgbm")
lrn_rf_tuned$param_set$values = instance.rf$result_learner_param_vals
lrn_rf_tuned$train(task)

# feature importance
# Gain: Represents the relative contribution of the feature to the model's accuracy. 
# A higher gain value indicates a more significant impact on the model's predictions
lgb.importance(lrn_rf_tuned$model, percentage = TRUE) %>% arrange(desc("Gain"))

# prediction over test data set
y.predict_rf <- predict(lrn_rf_tuned$model,
                        remaining_filter %>% 
                          select(all_of(x_cols)) %>% 
                          as.matrix() 
)
# y.predict_rf_bestcv <- predict(best_model_cv_rf,
#                                remaining_filter %>% 
#                                  select(all_of(x_cols)) %>% 
#                                  as.matrix() 
# )
rmse.test_rf <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - y.predict_rf)^2)) 
# 2ndhyper.... 33522.43 , 32438.98 , 32575.16
# 3rdhyper.... 1) 33143.26 , 2) 32748.17 , 3) 32785.78
# 3rdhyper+marital(includ.).... 4) 32264.96 , 5) 32066.77
# 4thhyper... 1) 32260.44 , 2) 32431.75
# rmse.test_rf_algcv <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - y.predict_rf_bestcv)^2)) #33121.16 , 


# ******************************************************************************
# GB experiments
# establish hyper parameters GB

learner.gb = lrn("regr.lightgbm",
                 boosting = "gbdt",
                 objective = "regression",
                 max_depth = to_tune(seq(3, 6, 1)),
                 num_leaves = to_tune(seq(6,8,1)),
                 min_data_in_leaf = to_tune(seq(39,45,1)),
                 min_data_in_bin = to_tune(seq(8,11,1)),
                 feature_fraction = to_tune(seq(0.3,0.5,0.1)),
                 feature_fraction_bynode = to_tune(seq(0.1,0.2,0.1)),
                 learning_rate = to_tune(seq(0.03, 0.04, 0.01)),
                 num_iterations  = to_tune(seq(39,45,1)), #40
                 lambda_l1 = to_tune(seq(0.4, 0.5, 0.1)),
                 lambda_l2 = to_tune(seq(0.30, 0.33, 0.01)),
                 convert_categorical = TRUE,
                 force_col_wise = TRUE,
                 verbose = 1,
                 num_threads = 5
)

# 3rd version
# learner.gb = lrn("regr.lightgbm",
#                  boosting = "gbdt",
#                  objective = "regression",
#                  max_depth = to_tune(seq(5, 8, 1)),
#                  num_leaves = to_tune(seq(19,26,1)),
#                  min_data_in_leaf = to_tune(seq(40,60,1)),
#                  min_data_in_bin = to_tune(seq(70,71,1)),
#                  feature_fraction = to_tune(seq(0.2,0.4,0.1)),
#                  feature_fraction_bynode = to_tune(seq(0.1,0.3,0.1)),
#                  learning_rate = to_tune(seq(0.03, 0.05, 0.01)),
#                  num_iterations  = to_tune(seq(29,33,1)), #33
#                  lambda_l1 = to_tune(seq(0.08, 0.1, 0.01)),
#                  lambda_l2 = to_tune(seq(0.04, 0.05, 0.01)),
#                  convert_categorical = TRUE,
#                  force_col_wise = TRUE,
#                  verbose = 1,
#                  num_threads = 5
# )
    
# 2nd version   
# learner.gb = lrn("regr.lightgbm",
#                  boosting = "gbdt",
#                  objective = "regression",
#                  max_depth = to_tune(seq(3, 15, 1)),
#                  num_leaves = to_tune(seq(15,90,2)),
#                  min_data_in_leaf = to_tune(seq(15,25,1)),
#                  min_data_in_bin = to_tune(seq(5,30,1)),
#                  feature_fraction = to_tune(seq(0.6,0.8,0.1)),
#                  feature_fraction_bynode = to_tune(seq(0.4,0.6,0.1)),
#                  bagging_fraction = to_tune(seq(0.2,0.7,0.1)),
#                  learning_rate = to_tune(c(seq(0.001, 0.1, 0.001),seq(0.11, 0.2, 0.01))), #seq(0.01, 0.15, 0.01)
#                  num_iterations  = to_tune(seq(50,120,2)),
#                  lambda_l1 = to_tune(c(seq(0.001, 0.1, 0.001),seq(0.11, 0.2, 0.01))),#seq(0.01, 0.15, 0.005)
#                  lambda_l2 = to_tune(c(seq(0.001, 0.1, 0.001),seq(0.11, 0.2, 0.01))), #seq(0.001, 0.09, 0.002)
#                  convert_categorical = TRUE,
#                  force_col_wise = TRUE,
#                  verbose = 1,
#                  num_threads = 5
# )

# begin training process, with 100 configurations
set.seed(123)
init_time <- Sys.time()
instance.gb = tune(
  tuner = tnr_rdgrid_search,
  task = task,
  learner = learner.gb,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 170,
  store_models = FALSE
)
cat('finish gb.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.gb <- (instance.gb$result)
result.table.gb <- (as.data.table(instance.gb$archive, measures = msrs(c("regr.mse","regr.mae")) ))
View(result.table.gb)

# plot(result.table.gb$lambda_l2, sqrt(result.table.gb$regr.mse))
# instance.gb$archive$#(uhash = "8ad2938f-3c45-404b-a5f9-afa25238dce2")#(uhash = "8ad2938f-3c45-404b-a5f9-afa25238dce2")
# instance.gb$archive$benchmark_result$uhashes

# fit final model on complete data set with the best combination
lrn_gb_tuned = lrn("regr.lightgbm")
lrn_gb_tuned$param_set$values = instance.gb$result_learner_param_vals
lrn_gb_tuned$train(task)

# feature importance
lgb.importance(lrn_gb_tuned$model, percentage = TRUE) %>% arrange(desc("Gain"))

# prediction over test data set
y.predict_gb <- predict(lrn_gb_tuned$model, 
                        remaining_filter %>% 
                          select(all_of(x_cols)) %>% 
                          as.matrix() 
                        )
rmse.test_gb <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - y.predict_gb)^2)) 
#29076.05, 0) 33146.65, 1) 31829.46, 2) 32496.88 , 3) 28990.77, 4) 29726.42
# (marital includ..) 5) 29580.22 
# 4rdhyper... 1) 29796.47 , 2) 29796.47


# ******************************************************************************
# DART experiments
# establish hyper parameters GB

learner.dart = lrn("regr.lightgbm",
                   boosting = "dart",
                   objective = "regression",
                   max_depth = to_tune(seq(4,6,1)),
                   max_bin = to_tune(seq(55,57,1)),
                   num_leaves = to_tune(seq(8,11,1)),
                   min_data_in_leaf = to_tune(seq(67,70,1)), #68
                   min_data_in_bin = to_tune(seq(4,10,1)),
                   feature_fraction = to_tune(seq(0.3,0.4, 0.1)),
                   feature_fraction_bynode = to_tune(seq(0.1,0.2, 0.1)),
                   learning_rate = to_tune(seq(0.08, 0.1, 0.01) ), #0.09
                   num_iterations  = to_tune(seq(40,45,1)),
                   lambda_l1 = to_tune( seq(0.24, 0.27, 0.01) ),
                   lambda_l2 = to_tune( seq(0.01, 0.02, 0.01) ), #0.43,0.24
                   drop_rate = to_tune(c(0.3)), #0.1,0.6
                   max_drop = to_tune(seq(14, 17, 1)),
                   xgboost_dart_mode = TRUE,
                   uniform_drop = TRUE,
                   convert_categorical = TRUE,
                   force_col_wise = TRUE,
                   verbose = 1,
                   num_threads = 5
)

# 3rd version
# learner.dart = lrn("regr.lightgbm",
#                    boosting = "dart",
#                    objective = "regression",
#                    max_depth = to_tune(seq(5,15,1)),
#                    max_bin = to_tune(seq(40,60,2)),
#                    num_leaves = to_tune(seq(15,22,1)),
#                    min_data_in_leaf = to_tune(seq(50,54,1)), #53
#                    min_data_in_bin = to_tune(seq(12,18,1)),
#                    feature_fraction = to_tune(seq(0.4,0.6, 0.1)),
#                    feature_fraction_bynode = to_tune(seq(0.3,0.4, 0.1)),
#                    learning_rate = to_tune(seq(0.1, 0.13, 0.01)), #0.11
#                    num_iterations  = to_tune(seq(38,40,1)), #39
#                    lambda_l1 = to_tune(seq(0.14, 0.3, 0.01) ),
#                    lambda_l2 = to_tune(seq(0.2, 0.3, 0.01) ), #0.43,0.24
#                    drop_rate = to_tune(c(0.1)), #0.1,0.6
#                    max_drop = to_tune(seq(15, 20, 1)),
#                    xgboost_dart_mode = TRUE,
#                    uniform_drop = TRUE,
#                    convert_categorical = TRUE,
#                    force_col_wise = TRUE,
#                    verbose = 1,
#                    num_threads = 5
# )

# 2nd version
# learner.dart = lrn("regr.lightgbm",
#                  boosting = "dart",
#                  objective = "regression",
#                  max_depth = to_tune(seq(5,12,1)),
#                  max_bin = to_tune(seq(15,100,2)),
#                  num_leaves = to_tune(seq(5,25,1)),# to_tune(seq(17,40,1)),
#                  min_data_in_leaf = to_tune(seq(17,40,1)),
#                  min_data_in_bin = to_tune(seq(10,20,1)),
#                  feature_fraction = to_tune(seq(0.7,0.8, 0.1)),
#                  feature_fraction_bynode = to_tune(seq(0.2,0.4, 0.1)),
#                  bagging_fraction = to_tune(seq(0.3,0.8, 0.1)),
#                  learning_rate = to_tune(seq(0.05, 0.2, 0.01)),
#                  num_iterations  = to_tune(seq(45,99,2)),
#                  lambda_l1 = to_tune(seq(0.01, 0.2, 0.007)),
#                  lambda_l2 = to_tune(seq(0.001, 0.05, 0.002)),
#                  drop_rate = to_tune(c(0.1,0.2,0.3)),
#                  max_drop = to_tune(seq(20, 60, 2)),
#                  xgboost_dart_mode = TRUE,
#                  uniform_drop = TRUE,
#                  convert_categorical = TRUE,
#                  force_col_wise = TRUE,
#                  verbose = 1,
#                  num_threads = 5
# )

# begin training process, with 100 configurations
init_time <- Sys.time()
instance.dart = tune(
  tuner = tnr_rdgrid_search,
  task = task,
  learner = learner.dart,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 150,
  store_models = FALSE
)
cat('finish dart.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.dart <- (instance.dart$result)
result.table.dart <- (as.data.table(instance.dart$archive, measures = msrs(c("regr.mse","regr.mae")) ))
View(result.table.dart)

# plot(result.table.dart$max_drop, sqrt(result.table.dart$regr.mse))
# result.cv.table.dart <- (instance.dart$archive$benchmark_result$score())
# id_min_cv <- which.min(result.table.dart$regr.mse)
# best_cv_results <- result.cv.table.dart[result.cv.table.dart$nr == id_min_cv,]
# id_min_cv_model <- which.min(best_cv_results$regr.mse)
# best model of the best CV
# best_model_cv <- instance.dart$archive$learners(
#   uhash = instance.dart$archive$benchmark_result$uhashes[id_min_cv])[[id_min_cv_model]]$model


# fit final model on complete data set with the best combination
lrn_dart_tuned = lrn("regr.lightgbm")
lrn_dart_tuned$param_set$values = instance.dart$result_learner_param_vals
lrn_dart_tuned$train(task)

# feature importance
lgb.importance(lrn_dart_tuned$model, percentage = TRUE) %>% arrange(desc("Gain"))

# prediction over test data set
y.predict_dart <- predict(lrn_dart_tuned$model, 
                        remaining_filter %>% 
                          select(all_of(x_cols)) %>% 
                          as.matrix() 
)

rmse.test_dart <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - y.predict_dart)^2)) #32709.68
# 1) 30121.02 , 2) 32594.04 , 3) 29999.78 , 4) 32354.96, 5) 30713.56 , 6) 32010.74 ,
# (includ. marital) 7) 32341.13
# 4thhyper... 1) 30183.02 , 2) 29722.61


# ******************************************************************************
# SVM experiments
# establish hyper parameters SVM

# Convert the data to a Task object, define task
task_dummys <- TaskRegr$new(id = "gap_task_dummys",
                     backend = ( train_filter %>% 
                                   select(all_of(c(x_cols_dummys,ytarget))) ),
                     target = ytarget)

learner.svm = lrn("regr.svm",
                  cost = to_tune( c(seq(0.001, 0.1, 0.001),seq(0.11,1, 0.1),seq(1,10,1),seq(11,110,10)) ),
                  kernel = 'linear',
                  type =	'eps-regression'
)

# begin training process,
init_time <- Sys.time()
instance.svm = tune(
  tuner = tnr_rdgrid_search,
  task = task_dummys,
  learner = learner.svm,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 5,
  store_models = FALSE
)
cat('finish linear SVM.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.sv <- (instance.svm$result)
result.table.sv <- (as.data.table(instance.svm$archive, measures = msrs(c("regr.mse","regr.mae")) ))
View(result.table.sv)

# fit final model on complete data set with the best combination
lrn_sv_tuned = lrn("regr.svm")
lrn_sv_tuned$param_set$values = instance.svm$result_learner_param_vals
lrn_sv_tuned$train(task_dummys)

# prediction over test data set
y.predict_sv <- predict(lrn_sv_tuned$model, 
                          remaining_filter %>% 
                            select(all_of(x_cols_dummys)) %>% 
                            as.matrix() 
)

rmse.test_sv <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - y.predict_sv)^2))


# ******************************************************************************
# linear models experiments with regularization
# establish hyper parameters

lambdas <- c( c(0.0)  )
alphas <- c( seq(0.0, 0.02, 0.001) )
elastic.net.grid <- expand.grid(list(alpha = alphas, lambda = lambdas) )
set.seed(3)
samp <- sample(1:nrow(elastic.net.grid ), 20)
elastic.net.gridFilter <- elastic.net.grid[samp,]

# Create a cross-validation object
cv_folds <- createFolds(train_filter[[ytarget]], k = 3, list = TRUE, returnTrain = TRUE)

estimate_sgd_cv <- function(id_val, filter_data, sgdmodelFilter.grid, x_cols_dummys, ytarget, cv_folds) {
  lambda_val <- sgdmodelFilter.grid[id_val, 'lambda']
  alphas <- sgdmodelFilter.grid[id_val, 'alpha']
  
  rmse_values <- numeric(length(cv_folds))
  for (fold_id in seq_along(cv_folds)) {
    train_indices <- cv_folds[[fold_id]]
    train_data <- filter_data[train_indices, ]
    valid_data <- filter_data[-train_indices, ]
    
    sgd_reg <- sgd(
      x = as.matrix( train_data %>%  select(all_of(x_cols_dummys)) ),
      y = train_data[[ytarget]], model = "lm",
      model.control = list(lambda1 = alphas, lambda2 = lambda_val),
      sgd.control = list(method = 'ai-sgd' )
    )
    sgd_pred_val <- predict(sgd_reg, newdata = as.matrix(valid_data %>% select( all_of(x_cols_dummys) )) )
    rmse_values[fold_id] <- RMSE(valid_data[[ytarget]], sgd_pred_val)
  }
  
  avg_rmse <- mean(rmse_values)
  return(list(lambda = lambda_val, alphas = alphas, avg_rmse = avg_rmse, mod = sgd_reg ) )
}

# Parallel processing using mclapply
results <- mclapply(1:nrow(elastic.net.gridFilter),
                    FUN = function(i) estimate_sgd_cv(i, ds, elastic.net.gridFilter, x_cols_dummys, ytarget, cv_folds )
                    )

# Convert results to data frame
result_df <- data.frame(
  lambdas = sapply(results, function(x) x$lambda),
  alphas = sapply(results, function(x) x$alphas),
  avg_rmse = sapply(results, function(x) x$avg_rmse)
)
result_df %>% arrange(avg_rmse, desc(TRUE))

id_min_sgd_cv <- which.min(result_df$avg_rmse)
best_sgd_cv_results <- results[[id_min_sgd_cv]]$mod

sgd_cv_pred_test <- predict(best_sgd_cv_results, 
                            newdata = as.matrix(remaining_filter %>% select( all_of(x_cols_dummys) )) )
rmse_sgd_cv_pred_test <- RMSE( remaining_filter[[ytarget]], sgd_cv_pred_test) # 27580.98, 2) incld.maritalcat... 27592.93





