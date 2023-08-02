# set wd
setwd("/home/user/Downloads/info_tud/statistical_learning_UDE/gender_pay_gap_project")
# import functions file
source('code/utils_functions.R')

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

# split in train, validation and test data
set.seed(123)

trainindex_filter <- createDataPartition(y = ds$realrinc, p = 0.7, list = FALSE) # Split the data into training (70%) and remaining (30%)
train_filter <- ds[trainindex_filter, ]
remaining_filter <- ds[-trainindex_filter, ]

# valid_test_index_filter <- createDataPartition(y = remaining_filter$realrinc, p = 0.5, list = FALSE) # Split the remaining data into validation (50%) and test (50%)
# validation_filter <- remaining_filter[valid_test_index_filter, ]
# test_filter <- remaining_filter[-valid_test_index_filter, ]

cat("training dim: ", nrow(train_filter), ", test dim: ", nrow(remaining_filter), "\n")

# create data sets with lbg format
# select regressors and outcome
x_cols <- colnames(ds)[match("age", colnames(ds)):match("age_sqr", colnames(ds))]
ytarget <- "realrinc"

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

# define learner with hyper parameters
# For example, if bagging_freq = 5, bagging will be performed every 5 iterations. 
# This means that the first 5 trees (iterations 1 to 5) will be trained on the entire training data, 
# then the next 5 trees (iterations 6 to 10) will be trained on a random subsample of the training data, and so on.
# For example, if bagging_fraction = 0.8, each tree will be trained on a random 80% subsample of the training data, 
# selected with replacement.
learner = lrn("regr.lightgbm",
              boosting = "rf",
              objective = "regression",
              max_depth = to_tune(seq(7,16,1)),
              num_leaves = to_tune(seq(15,50,5)),
              num_iterations  = to_tune(seq(25,50,2)),
              min_data_in_leaf = to_tune(seq(15,30,3)),
              min_data_in_bin = to_tune(seq(30,62, 2)),
              feature_fraction_bynode = to_tune(c(0.7,0.8)),
              bagging_fraction = to_tune(seq(0.1,0.8,0.2)),
              bagging_freq = to_tune(seq(6,12,1)),
              verbose = to_tune(1),
              num_threads = to_tune(5)
)

# define cross-validation and error metric
rsmp_cv3 = rsmp("cv", folds = 3)
msr_ce = msr("regr.rmse")

# begin training process, with 100 configurations
init_time <- Sys.time()
instance = tune(
  tuner = tnr_rdgrid_search,
  task = task,
  learner = learner,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 50,
  store_models = FALSE
)
cat('finish rf.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.rf <- (instance$result)
result.table.rf <- (as.data.table(instance$archive, measures = msrs(c("regr.mse","regr.mae")) ))
# View(result.table.rf)

# fit final model on complete data set with the best combination
lrn_rf_tuned = lrn("regr.lightgbm")
lrn_rf_tuned$param_set$values = instance$result_learner_param_vals
lrn_rf_tuned$train(task)

# feature importance
# Gain: Represents the relative contribution of the feature to the model's accuracy. 
# A higher gain value indicates a more significant impact on the model's predictions
lgb.importance(lrn_rf_tuned$model, percentage = TRUE) %>% arrange(desc("Gain"))

# prediction over test data set
y.predict_rf <- predict(lrn_rf_tuned$model,
                        data = Matrix( as.matrix(remaining_filter 
                                                %>% select(all_of(x_cols))), sparse=TRUE )
                          )
rmse.test_rf <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - y.predict_rf)^2))


# ******************************************************************************
# GB experiments
# establish hyper parameters GB
learner.gb = lrn("regr.lightgbm",
                 boosting = "gbdt",
                 objective = "regression",
                 max_depth = to_tune(seq(5, 12, 1)),
                 num_leaves = to_tune(seq(20,55,3)),
                 min_data_in_leaf = to_tune(seq(25,65,3)),
                 min_data_in_bin = to_tune(seq(5,60,3)),
                 feature_fraction = to_tune(c(0.7,0.8)),
                 feature_fraction_bynode = to_tune(c(0.3,0.4,0.5)),
                 bagging_fraction = to_tune(seq(0.7,0.8,0.1)),
                 learning_rate = to_tune(seq(0.01, 0.2, 0.01)),
                 num_iterations  = to_tune(seq(50,100,2)),
                 lambda_l1 = to_tune(seq(0.001, 0.2, 0.005)),
                 lambda_l2 = to_tune(seq(0.001, 0.2, 0.005)),
                 verbose = to_tune(1),
                 num_threads = to_tune(5)
)

# begin training process, with 100 configurations
init_time <- Sys.time()
instance.gb = tune(
  tuner = tnr_rdgrid_search,
  task = task,
  learner = learner.gb,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 65,
  store_models = FALSE
)
cat('finish gb.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.gb <- (instance.gb$result)
result.table.gb <- (as.data.table(instance.gb$archive, measures = msrs(c("regr.mse","regr.mae")) ))
# View(result.table.gb)

# fit final model on complete data set with the best combination
lrn_gb_tuned = lrn("regr.lightgbm")
lrn_gb_tuned$param_set$values = instance.gb$result_learner_param_vals
lrn_gb_tuned$train(task)

# feature importance
lgb.importance(lrn_gb_tuned$model, percentage = TRUE) %>% arrange(desc("Gain"))

# prediction over test data set
y.predict_gb <- predict(lrn_gb_tuned$model,
                        data = Matrix( as.matrix(remaining_filter 
                                                 %>% select(all_of(x_cols))), sparse=TRUE )
)
rmse.test_gb <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - y.predict_gb)^2)) #29076.05




