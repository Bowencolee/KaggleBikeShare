##
## Bike Share Analysis
##

library(tidyverse)
library(tidymodels)
library(vroom)

bike_train <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/KagglebikeShare/train.csv")
bike_test <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/KagglebikeShare/test.csv")

# Bike_train w/o casual and registered
bike_train <- bike_train %>%
  select(-casual,-registered)


# Cleaning section
bike_clean <- bike_train %>%
  mutate(weather = ifelse(weather == 4,3,weather)) %>% # get rid of the one level 4 weather var.
  mutate(season = factor(season, # change the season to factor with names
                         levels = 1:4,
                         labels = c("spring", "summer", "fall", "winter"))) %>%
  select(datetime,season,temp,atemp,humidity,windspeed,count)

# mutate(weather = ifelse(weather == 4,3,weather))

# Engineering section
my_recipe <- recipe(count~., data = bike_train) %>%
  ste_mutate(weather = ifelse(weather == 4,3,weather)) %>%
  step_time(datetime, features=c("hour", "minute")) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())
  
# Baking the recipe
prepped_recipe <- prep(my_recipe)
bike_bake <- bake(prepped_recipe, bike_train)

#Linear regression
my_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine = What R function to use

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike_train) # Fit the workflow
  
bike_predictions <- predict(bike_workflow,
                            new_data=bike_test) # Use fit to predict

# Make the submission csv
bike_submit <- cbind(bike_test,bike_predictions) %>%
  select(datetime,.pred)
colnames(bike_submit) <- c("datetime", "count")

# Assuming 'df' is your data frame and 'negative_column' is the column with negative values
bike_submit$count[bike_submit$count < 0] <- 0

bike_submit$datetime <- format(bike_submit$datetime, "%Y-%m-%d %H:%M:%S")

# Get the csv file
vroom_write(bike_submit, "bike_submit.csv", ",")

