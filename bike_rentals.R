day <- read.csv("~/Desktop/261_project/bike+sharing+dataset/day.csv")
View(day)
str(day)
anyNA(day)
#converting categorical variables into factors
day$season <- as.factor(day$season)
day$holiday <- as.factor(day$holiday)
day$workingday <- as.factor(day$workingday)
day$weathersit <- as.factor(day$weathersit)
#standardizing continuous variables for better model performance
day$temp <- scale(day$temp)
day$hum <- scale(day$hum)
day$windspeed <- scale(day$windspeed)
str(day)

library(ggplot2)
ggplot(day, aes(x=cnt)) +
geom_histogram(binwidth=100, fill="blue", color="black") +
ggtitle("Distribution of Bike Rentals (cnt)") +
xlab("Bike Rentals") +
ylab("Frequency")

# boxplot of bike rentals by season
ggplot(day, aes(x=season, y=cnt)) +
geom_boxplot(fill="skyblue") +
ggtitle("Bike Rentals by Season") +
xlab("Season") +
ylab("Bike Rentals")

# scatterplot of Temperature vs Bike Rentals
ggplot(day, aes(x=temp, y=cnt)) +
geom_point(alpha=0.5, color="darkgreen") +
geom_smooth(method="lm", color="red") +
ggtitle("Bike Rentals vs Temperature") +
xlab("Standardized Temperature") +
ylab("Bike Rentals")

# Correlation between numerical variables
cor(day[, c("cnt", "temp", "hum", "windspeed")])

install.packages("caTools")
library(caTools)

#split data into train and test sets
set.seed(123)
split <- sample.split(day$cnt, SplitRatio = 0.7)
train <- subset(day, split == TRUE)
test <- subset(day, split == FALSE)

#fitting a multiple linear regression model
model <- lm(cnt ~ temp + hum + windspeed + season + holiday + workingday + weathersit, data=train)
summary(model)

# making predictions on the testing set
predictions <- predict(model, test)
#calculating RMSE and MAE
rmse <- sqrt(mean((predictions - test$cnt)^2))
mae <- mean(abs(predictions - test$cnt))

# printing the performance metrics
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")

#plotting actual vs predicted values 
plot(test$cnt, predictions,xlab = "Actual Bike Rentals",ylab = "Predicted Bike Rentals",main = "Actual vs Predicted Bike Rentals",col = "blue", pch = 16)
abline(0, 1, col = "red", lwd = 2)

final_model <- lm(I(cnt^0.5) ~ poly(temp,2) + hum + windspeed + season + yr + holiday + weekday + weathersit, data=day)

# Load required libraries
library(caret)
library(ggplot2)

# Precompute polynomial terms for `temp`
day$poly_temp_1 <- poly(day$temp, 2)[, 1]  # First-degree term
day$poly_temp_2 <- poly(day$temp, 2)[, 2]  # Second-degree term

# Set up 10-fold cross-validation repeated 1000 times
set.seed(123)  # For reproducibility
train_control <- trainControl(
  method = "repeatedcv",  # Repeated k-fold CV
  number = 10,           # 10 folds
  repeats = 1000         # Repeat 1000 times
)

# Fit the model using repeated cross-validation
model_cv <- train(
  I(cnt^0.5) ~ poly_temp_1 + poly_temp_2 + hum + windspeed + season + yr + holiday + weekday + weathersit,
  data = day,
  method = "lm",  # Linear regression
  trControl = train_control
)

# Print summary of cross-validation results
print(model_cv)

# Extract resample metrics (RMSE, Rsquared, MAE) for visualization
metrics <- model_cv$resample

# Print first few rows to inspect
head(metrics)


# Summarize RMSE, R-squared, and MAE from the repeated cross-validation
summary_stats <- model_cv$resample %>%
  summarize(
    Mean_RMSE = mean(RMSE),
    SD_RMSE = sd(RMSE),
    Mean_R2 = mean(Rsquared),
    SD_R2 = sd(Rsquared),
    Mean_MAE = mean(MAE),
    SD_MAE = sd(MAE)
  )

# Print the summary statistics
print("Summary of 10-Fold Cross-Validation Results After 1000 Iterations:")
print(summary_stats)

# Optional: Print the first few rows of raw resample data for inspection
print("First Few Rows of Resampled Metrics:")
head(model_cv$resample)

# Ensure all categorical variables retain their full levels
day$season <- factor(day$season, levels = unique(day$season))
day$weekday <- factor(day$weekday, levels = unique(day$weekday))
day$weathersit <- factor(day$weathersit, levels = unique(day$weathersit))

# Precompute polynomial terms for `temp`
day$poly_temp_1 <- poly(day$temp, 2)[, 1]  # First-degree term
day$poly_temp_2 <- poly(day$temp, 2)[, 2]  # Second-degree term

# Initialize storage for coefficients across iterations
set.seed(123)  # For reproducibility
coefficients_storage <- list()
predictors <- c("Intercept", "poly_temp_1", "poly_temp_2", "hum", "windspeed", 
                "season", "yr", "holiday", "weekday", "weathersit")  # Replace with factor levels if needed

for (predictor in predictors) {
  coefficients_storage[[predictor]] <- numeric()
}

MSP <- numeric()  # Mean squared prediction error

# Repeated validation: 1000 iterations
for (i in 1:1000) {
  # Split data into 90% estimation, 10% validation
  obs <- 1:nrow(day)
  sample.est <- sort(sample(obs, size = round(0.9 * length(obs))))
  sample.val <- setdiff(obs, sample.est)
  
  # Create estimation and validation datasets
  day.est <- day[sample.est, ]
  day.val <- day[sample.val, ]
  
  # Ensure all factor levels are retained in the estimation set
  day.est$season <- factor(day.est$season, levels = levels(day$season))
  day.est$weekday <- factor(day.est$weekday, levels = levels(day$weekday))
  day.est$weathersit <- factor(day.est$weathersit, levels = levels(day$weathersit))
  
  # Fit the model
  fit <- lm(I(cnt^0.5) ~ poly_temp_1 + poly_temp_2 + hum + windspeed + season + yr + holiday + weekday + weathersit,
            data = day.est)
  
  # Store coefficients for each predictor
  for (predictor in names(coefficients(fit))) {
    coefficients_storage[[predictor]][i] <- coef(fit)[predictor]
  }
  
  # Calculate MSP
  y_hat <- predict(fit, day.val)
  pred_error <- day.val$cnt^0.5 - y_hat
  MSP[i] <- mean(pred_error^2)
}

# Convert coefficients to a data frame for visualization
coefficients_df <- do.call(cbind, coefficients_storage)
coefficients_long <- tidyr::pivot_longer(
  as.data.frame(coefficients_df),
  cols = everything(),
  names_to = "Predictor",
  values_to = "Coefficient"
)

# Plot histograms for each predictor's coefficients
ggplot(coefficients_long, aes(x = Coefficient)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black", alpha = 0.7) +
  facet_wrap(~ Predictor, scales = "free", ncol = 3) +
  labs(
    title = "Coefficient Distributions Across Cross-Validation Iterations",
    x = "Coefficient Value",
    y = "Frequency"
  ) +
  theme_minimal()


# Summarize RMSE, R-squared, and MAE from the repeated cross-validation
summary_stats <- model_cv$resample %>%
  +     summarise(    Mean_RMSE = mean(RMSE),
SD_RMSE = sd(RMSE),
Mean_R2 = mean(Rsquared),
  SD_R2 = sd(Rsquared),
Mean_MAE = mean(MAE),
  SD_MAE = sd(MAE) )

# Print the summary statistics
print("Summary of 10-Fold Cross-Validation Results After 1000 Iterations:")
print(summary_stats)


