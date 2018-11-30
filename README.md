# bikerent
The objective of this Case Study is to predict daily bike rental count based on the environmental and seasonal settings. 

## Tools Used:
##### R
##### Python
##### Jupyter Notebook
##### RStudio
##### Git

## R Packages:
##### psych: Used for visualization
##### ggplot2: Used for visualization
##### DMWR2: Used for knnImputation
##### DMWR: Same as DMWR2
##### corrgram: Used to create correlation plot
##### randomForest: Used to model Random Forest
##### rpart: Used to model Decision Trees
##### caTools: Used to partition the data in train and test
##### FNN: used for KNN regression
##### dummies: used to create dummy variables
##### car: used to calculate VIF values


## Python libraries
##### sklearn
##### pandas
##### numpy
##### os
##### fancyimpute
##### seaborn

## Information about the data:
The details of data attributes in the dataset are as follows -
##### instant: Record index
##### dteday: Date
##### season: Season (1:spring, 2:summer, 3:fall, 4:winter)
##### yr: Year (0: 2011, 1:2012)
##### mnth: Month (1 to 12)
##### hr: Hour (0 to 23)
##### holiday: weather day is holiday or not (extracted fromHoliday Schedule)
##### weekday: Day of the week
##### workingday: If day is neither weekend nor holiday is 1, otherwise is 0.
##### weathersit: (extracted fromFreemeteo)
1: Clear, Few clouds, Partly cloudy, Partly cloudy
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered
clouds
##### temp: Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min),t_min=-8, t_max=+39 (only in hourly scale)
#####  atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_maxt_min),t_min=-16, t_max=+50 (only in hourly scale)
##### hum: Normalized humidity. The values are divided to 100 (max)
##### windspeed: Normalized wind speed. The values are divided to 67 (max)
##### casual: count of casual users
##### registered: count of registered users
##### cnt: count of total rental bikes including both casual and registered

## Findings of EDA:
No missing values present in the dataset. Outliers present in Windspeed and Humidity. Correlation present in numerical variables between temp and atemp, so temp was removed. ANOVA test conducted to find association and identify significant predictors. VIF to reduce dimension after converting categorical to dummy variables. Random Forest importance metric used to reduce the variables further, these variables used for RF Model, Decision Tree, and KNN. Linear Backward Elimination used to find the best variables for Linear Model. Feature Scaling performed on all variables except the target. This ends the Exploratory Data Analysis.

## Models used:
##### Linear Regression: Used backward elimination to find the best model
##### Decision Tree Regression: Used ANOVA method to construct the Decision Tree.
##### Random Forest: Forest consisted of 100 trees, on plotting the error rate decreased with increase in no. of estimators.
##### KNN Regression: K was chosen as 7 after analyzing the test error and select k for which test error is the least.

## Model Evaluation:
##### Linear Regression resulted in the best adjusted R^2 value and hence was chosen as the best model.