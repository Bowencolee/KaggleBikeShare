##
## Bike Share Analysis
##

library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)

bike_train <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/KagglebikeShare/train.csv")
bike_test <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/KagglebikeShare/test.csv")

# Bike_train w/o casual and registered
bike_train <- bike_train %>%
  select(-casual,-registered)

################
### Cleaning ###
################

# my_recipe <- recipe(count~., data = bike_train) %>%
#  ste_mutate(weather = ifelse(weather == 4,3,weather)) %>%
#  step_time(datetime, features=c("hour", "minute")) %>%
#  step_dummy(all_nominal_predictors()) %>%
#  step_zv(all_predictors())

# Engineering section ### IF YOU WANT TO DO SOMETHING TO BOTH, DO IT IN RECIPE#
my_recipe <- recipe(count~., data=bike_train) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime)
  
# Baking the recipe
prepped_recipe <- prep(my_recipe)
bike_bake <- bake(prepped_recipe, bike_train)




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

# Get the csv file
vroom_write(bike_pois, "bike_pois.csv", ",")
