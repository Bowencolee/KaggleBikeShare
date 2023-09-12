##
## Bike Share EDA Code
##

## Libraries
library(tidyverse)
library(vroom)
library(DataExplorer)
library(patchwork)

## Read in data
bike <- vroom("C:/Users/bowen/Desktop/Stat348/KaggleBikeShare/train.csv")
dplyr::glimpse(bike)
skimr::skim(bike)
DataExplorer::plot_correlation(bike)
plot_bar(bike)
plot_histogram(bike)

# Plot 3
bike_s <- bike %>%
  mutate(season = factor(season, levels = 1:4, labels = c("spring", "summer", "fall", "winter")))
plot3 <- ggplot(bike_s, aes(x = season, y = count)) +
  geom_boxplot()

# Plot 2
plot2 <- ggplot(bike, aes(x = humidity, y = count)) +
  geom_point() +
  geom_smooth(se=F)

# Plot 1
bike_select <- bike %>%
  select(season, weather, count)
plot1 <- plot_correlation(bike_select)

# Plot 4
plot4 <- ggplot(bike, aes(x = atemp, y = count)) + 
  geom_point() +
  geom_smooth(se=F)

# 4 panel plot
(plot1 + plot2) / (plot3 + plot4)

