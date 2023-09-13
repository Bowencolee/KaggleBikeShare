##
## Bike Share Analysis
##

library(tidyverse)
library(tidymodels)

bike_train <- vroom::vroom("C:/Users/bowen/Desktop/Stat348/KagglebikeShare/train.csv")

# Cleaning section
bike_clean <- bike_train %>%
  filter(weather != 4) %>% # get rid of the one level 4 weather var.
  mutate(season = factor(season, # change the season to factor with names
                         levels = 1:4,
                         labels = c("spring", "summer", "fall", "winter"))) %>%
  select(datetime,season,temp,atemp,humidity,windspeed,count)

# Engineering section
my_recipe <- recipe(count~., data = bike_train) %>%
  step_time(datetime, features=c("hour", "minute")) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_select(datetime_hour,season,temp,atemp,humidity,windspeed,count)
  

prepped_recipe <- prep(my_recipe)
bike_bake <- bake(prepped_recipe, bike_train)
