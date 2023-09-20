##
## Bike Share Analysis
##

library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)

bike_train <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/KagglebikeShare/train.csv")
bike_test <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/KagglebikeShare/test.csv")

################
### Cleaning ###
################

# Bike_train w/o casual and registered (bike_log takes the count and logs it)
bike_train <- bike_train %>%
  select(-casual,-registered)

bike_log <- bike_train %>%
  mutate(count=log(count))

# Engineering section ### IF YOU WANT TO DO SOMETHING TO BOTH, DO IT IN RECIPE#
my_recipe <- recipe(count~., data=bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_rm(datetime)
  
# Baking the recipe
# prepped_recipe <- prep(my_recipe)
# bike_bake <- bake(prepped_recipe, bike_train)




#########################
### Linear regression ###
#########################
my_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine = What R function to use

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike_train) # Fit the workflow
  
bike_predictions <- predict(bike_workflow,
                            new_data=bike_test) # Use fit to predict

# Make the submission csv
bike_linear <- cbind(bike_test,bike_predictions) %>%
  select(datetime,.pred)
colnames(bike_linear) <- c("datetime", "count")

# Assuming 'df' is your data frame and 'negative_column' is the column with negative values
bike_linear$count[bike_linear$count < 0] <- 0

bike_linear$datetime <- format(bike_linear$datetime, "%Y-%m-%d %H:%M:%S")

# Get the csv file
vroom_write(bike_linear, "bike_linear.csv", ",")


#############################
### Log-Linear Regression ###
#############################
my_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine = What R function to use

log_bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike_log) # Fit the workflow

## Get Predictions for test set AND format for Kaggle
log_lin_preds <- predict(log_bike_workflow, new_data = bike_test) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write predictions to CSV
vroom_write(x=log_lin_preds, file="./bike_log_linear.csv", delim=",")


##########################
### Poisson Regression ###
##########################

pois_mod <- poisson_reg() %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

bike_pois_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pois_mod) %>%
  fit(data = bike_train) # Fit the workflow

bike_predictions_pois <- predict(bike_pois_workflow,
                            new_data=bike_test) # Use fit to predict

# Make the submission csv
bike_pois <- cbind(bike_test,bike_predictions) %>%
  select(datetime,.pred)
colnames(bike_pois) <- c("datetime", "count")


bike_pois$datetime <- format(bike_pois$datetime, "%Y-%m-%d %H:%M:%S")

vroom_write(bike_pois, "bike_pois.csv", ",")

############################
### Penalized Regression ###
############################
# With log transformation

preg_model <- linear_reg(penalty=1, mixture=0.5) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=bike_log)

preg_preds <- predict(preg_wf, new_data=bike_test)%>% 
  mutate(.pred=exp(.pred)) %>% 
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=preg_preds, file="./bike_log_preg.csv", delim=",")


### Poisson model, penalized regression

preg_pois_mod <- poisson_reg(penalty=0, mixture=0.5) %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

preg_pois_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_pois_mod) %>%
  fit(data = bike_log) # Fit the workflow

preg_pois_preds <- predict(preg_pois_wf, new_data=bike_test)%>% 
  mutate(.pred=exp(.pred)) %>% 
  bind_cols(., bike_test) %>% 
  select(datetime, .pred) %>% 
  rename(count=.pred) %>% 
  mutate(count=pmax(0, count)) %>% 
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=preg_pois_preds, file="./bike_log_preg_pois.csv", delim=",")
