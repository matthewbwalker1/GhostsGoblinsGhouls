# Loading in Libraries --------------------------------------------------
library(tidyverse)
library(tidymodels)
library(vroom)
library(bonsai)
library(lightgbm)
library(agua)
library(naivebayes)

# Reading In Data -------------------------------------------------------
# Import train and test data
trainData <- vroom("train.csv")
testData <- vroom("test.csv")

# # EDA -------------------------------------------------------------------
# 
# trainData %>%
#   ggplot(mapping = aes(x = hair_length,
#                        y = rotting_flesh,
#                        color = type)) +
#   geom_point() +
#   facet_wrap(~color)

# Cleaning Data ---------------------------------------------------------
# Recipe
ggg_recipe <- recipe(type ~ .,
                     data = trainData) %>%
  step_rm(id) %>%
  step_string2factor(color)

# This code can be used to test that the recipe works as intended.
# bake(prep(ggg_recipe),
#      new_data = trainData) %>%
#   summary()

# # Random Forests Model --------------------------------------------------
# 
# ggg_random_forest_model <- rand_forest(mtry = tune(),
#                                        min_n = tune(),
#                                        trees = tune()) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# ggg_preliminary_workflow <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(ggg_random_forest_model)

# # Making a bart-boosted tree model --------------------------------------
# 
# ggg_trees_bart <- parsnip::bart(trees = tune()) %>%
#   set_engine("dbarts") %>%
#   set_mode("classification")
# 
# # preliminary workflow
# ggg_preliminary_workflow <- workflow() %>%
#   add_recipe(ggg_recipe) %>%
#   add_model(ggg_trees_bart)

# Naive Bayes Model -----------------------------------------------------

ggg_naive_bayes_model <- naive_Bayes(Laplace = tune(),
                                      smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")


ggg_preliminary_workflow <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(ggg_naive_bayes_model)


# Cross-validation ------------------------------------------------------
# grid of tuning parameters
tuning_grid <- grid_space_filling(
  Laplace(range = c(0, 10)),
  smoothness(range = c(0.1, 3.1)),
  # mtry(range = c(1, 6)),
  # min_n(),
  # trees(),
  size = 5)

# splitting data into folds
folds <- vfold_cv(trainData,
                  v = 3,
                  repeats = 1)

# cross_validation
cv_results <- ggg_preliminary_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# pulling off best tuning parameter values
best_tuning_parameters <- cv_results %>%
  select_best(metric = "accuracy")

# printing best tuning parameters
best_tuning_parameters

# # using saved tuning parameters
# best_tuning_parameters <- vroom("naiveBayesBestTune.csv")
  
# Making final workflow -------------------------------------------------
# making final workflow
ggg_workflow <- ggg_preliminary_workflow %>%
  finalize_workflow(best_tuning_parameters) %>%
  fit(data = trainData)

# Making Predictions ----------------------------------------------------
ggg_predictions <- predict(ggg_workflow,
                           new_data = testData,
                           type = "class")

# format the predictions as a submission file
ggg_predictions_formatted <- ggg_predictions %>%
  mutate(id = testData$id) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

# Writing submission file -----------------------------------------------

vroom_write(x = ggg_predictions_formatted,
            file = "./preds.csv",
            delim = ",")

# saving best tuning parameters
vroom_write(x = best_tuning_parameters,
            file = "./cv_bart_forests.csv",
            delim = ",")
