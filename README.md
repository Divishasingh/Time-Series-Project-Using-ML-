# Sales Forecasting using ARIMA, SARIMA and Machine Learning Models

## Project Overview

This project focuses on forecasting retail sales using both classical time series models and machine learning models. The main objective is to analyze historical sales patterns, identify trend and seasonality, check stationarity, build forecasting models, and compare their performance using standard evaluation metrics.

The project is designed from a statistical and business analytics perspective. Since sales forecasting is important for demand planning, inventory management, staffing, and revenue estimation, this project demonstrates how statistical modeling and machine learning can be used together for better forecasting decisions.

## Abstract

Retail businesses depend heavily on accurate sales forecasting to manage inventory, plan resources, and improve operational efficiency. In this project, daily sales data was analyzed using time series techniques and machine learning models. The data was first cleaned and transformed into a daily sales time series. Exploratory Data Analysis was performed to understand sales behavior over time.

Time series decomposition revealed a strong weekly seasonal pattern. The Augmented Dickey-Fuller test confirmed that the sales series was stationary. ACF and PACF plots were used to study autocorrelation patterns and support model selection. ARIMA and SARIMA models were developed as statistical forecasting models, while Random Forest and XGBoost were implemented using lag-based features.

The models were evaluated using MAE, RMSE, and MAPE. Random Forest achieved the lowest MAE and RMSE values, making it the best-performing model overall. SARIMA performed better than ARIMA by capturing weekly seasonality, while XGBoost also produced competitive results.

## Business Objective

The objective of this project is to forecast future sales based on historical sales patterns.

Accurate sales forecasting can help businesses in:

* Demand planning
* Inventory management
* Revenue forecasting
* Resource allocation
* Staffing decisions
* Promotional planning

## Dataset

The dataset used in this project is the Rossmann Store Sales dataset from Kaggle.

Main files used:

* `train.csv`
* `store.csv`

Important columns used from the dataset:

* `Date`
* `Sales`
* `Store`
* `DayOfWeek`
* `Promo`
* `Open`
* `StateHoliday`
* `SchoolHoliday`

For this project, daily sales were aggregated by date to create a univariate time series.

Dataset Limitation

The available dataset covered approximately 9.5 months of sales data (October 2014 to July 2015). Therefore, a 90-day forecast was generated instead of a long-term yearly forecast to maintain forecasting reliability and reduce uncertainty in future predictions.

## Tools and Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Statsmodels
* Scikit-learn
* XGBoost
* Google Colab

## Project Workflow

## 1. Data Cleaning and Preprocessing

The dataset was imported and checked for missing values, duplicate records, and incorrect data types. The `Date` column was converted into datetime format to make the data suitable for time series analysis.

Daily sales were aggregated using the `Date` column to create a time series where each observation represents total sales for a particular day.

### Observation

The cleaned time series contained daily sales values from October 2014 to July 2015. This time range was sufficient for short-term forecasting and weekly seasonality analysis.

## 2. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the basic structure and behavior of the sales data.

The following analyses were performed:

* Daily sales visualization
* Monthly sales trend
* Distribution of daily sales
* Boxplot for outlier detection

### Observation

The daily sales plot showed repeated fluctuations, indicating the presence of seasonality. The distribution of sales was not perfectly normal and showed variation due to business cycles, weekly effects, and possible holiday/store closure effects.

The yearly sales chart was removed from the final analysis because the dataset contained less than two complete years of data, making yearly comparison less meaningful.

## 3. Time Series Decomposition

Time series decomposition was performed to break the sales series into three components:

* Trend
* Seasonality
* Residuals

Since the data showed a strong weekly pattern, decomposition was performed using a seasonal period of 7.

### Observation

The decomposition showed a clear weekly seasonal pattern in the sales data. The trend component showed moderate fluctuations, while the residual component captured irregular movements. This confirmed that weekly seasonality plays an important role in sales behavior.

## 4. Stationarity Check using ADF Test

The Augmented Dickey-Fuller test was used to check whether the sales series was stationary.

### ADF Test Result

* ADF Statistic: -5.0369
* p-value: 0.0000187

### Interpretation

Since the p-value was less than 0.05, the null hypothesis of non-stationarity was rejected. This means the sales series was stationary.

Therefore, differencing was not required, and the differencing order (d) was set to 0 while fitting the ARIMA and SARIMA models.

## 5. ACF and PACF Analysis

ACF and PACF plots were used to study autocorrelation and partial autocorrelation patterns in the sales series.

### Observation

The ACF plot showed significant spikes at lags 7, 14, 21, 28, and 35. This confirmed the presence of strong weekly seasonality.

The PACF plot also showed significant spikes around lag 7 and lag 14, suggesting autoregressive behavior and supporting the use of seasonal models.

## 6. ARIMA Model

An ARIMA model was fitted as the baseline statistical time series model.

The model used was:

ARIMA(1,0,1)

### Observation

The ARIMA model captured short-term dependencies in the sales data but did not explicitly model weekly seasonality. As a result, its forecasting performance was weaker compared to SARIMA and machine learning models.

## 7.SARIMA Model

A SARIMA model was fitted to capture both non-seasonal and seasonal patterns.

The model used was:

SARIMA(1,0,1)(1,0,1,7)

The seasonal period was taken as 7 because the data showed strong weekly seasonality.

SARIMA was selected because the decomposition plot, ACF plot, and PACF plot all indicated strong weekly seasonality with a seasonal period of 7 days.

Observation

SARIMA performed better than ARIMA because it captured weekly seasonal patterns in the data. Its lower AIC and BIC values also indicated a better statistical fit compared to ARIMA.

## 8. Feature Engineering for Machine Learning

To apply machine learning models to the time series data, lag-based and rolling features were created.

Features created:

* Lag 1
* Lag 7
* Lag 14
* Rolling Mean 7
* Rolling Standard Deviation 7
* Month
* Day of Week

### Observation

These features helped machine learning models understand recent sales behavior, weekly seasonality, and short-term sales fluctuations.

## 9. Random Forest Regressor

A Random Forest Regressor was trained using the engineered features. Random Forest was used because it can capture nonlinear relationships and interactions between variables.

### Observation

Random Forest performed very well and captured the weekly sales pattern effectively. It achieved the lowest MAE and RMSE values among all models.

## 10. XGBoost Regressor

An XGBoost Regressor was also trained and evaluated. XGBoost is a powerful boosting algorithm often used for structured data and regression problems.

### Observation

XGBoost produced competitive results and achieved the lowest MAPE value. However, its MAE and RMSE were higher than Random Forest.

## 11. Model Evaluation

All models were evaluated using:

* MAE: Mean Absolute Error
* RMSE: Root Mean Squared Error
* MAPE: Mean Absolute Percentage Error

## Final Model Comparison

| Model         |          MAE |         RMSE |   MAPE |
| ------------- | -----------: | -----------: | -----: |
| ARIMA         | 2,200,265.52 | 3,046,588.75 | 3.1991 |
| SARIMA        | 1,197,844.72 | 1,673,164.59 | 0.2216 |
| Random Forest |   754,922.48 | 1,084,303.22 | 0.4817 |
| XGBoost       |   834,060.71 | 1,252,846.11 | 0.2078 |

### Result

Among all models, Random Forest achieved the lowest MAE and RMSE values and emerged as the best overall forecasting model. SARIMA was the strongest statistical forecasting model because it successfully captured weekly seasonal patterns present in the sales data, while XGBoost achieved the lowest MAPE value.

Overall, this project demonstrates the effectiveness of combining classical statistical forecasting techniques with machine learning methods for retail sales prediction. The results show that incorporating seasonality, historical sales behavior, and engineered features can significantly improve forecasting accuracy and support better business decision-making.

## 12. Feature Importance Analysis

Feature importance analysis was performed for the Random Forest model.

### Feature Importance Results

| Feature        | Importance |
| -------------- | ---------: |
| Day of Week    |     0.5439 |
| Lag 7          |     0.1771 |
| Rolling Mean 7 |     0.1067 |
| Lag 1          |     0.0979 |
| Lag 14         |     0.0608 |
| Month          |     0.0137 |

### Interpretation

Day of Week was the most important feature, contributing more than 54% importance. This confirms that sales are strongly affected by weekly patterns.

Lag 7 was the second most important feature, showing that sales from the same day of the previous week strongly influence current sales.

Rolling Mean 7 also contributed meaningfully, indicating that recent sales trends are useful for forecasting.

## 13. Forecasting

The final forecasting step was performed to estimate future sales for the next 90 days.

A 90-day forecast was selected instead of a 12-month forecast because the available time series covered less than one full year. Forecasting too far into the future using limited historical data may reduce reliability.

### Observation

The forecast showed stable sales behavior with recurring weekly fluctuations. This indicates that weekly seasonality is expected to continue in the near future.

## Business Insights

1. Weekly seasonality is the strongest driver of sales.
2. Day of Week has the highest impact on sales forecasting.
3. Sales from the previous week are highly useful for predicting current sales.
4. SARIMA performs better than ARIMA because it captures weekly seasonality.
5. Random Forest provides the best overall forecasting accuracy based on MAE and RMSE.
6. Businesses should consider weekly sales patterns while planning inventory and staffing.
7. Short-term forecasting can support demand planning and operational decision-making.

## Conclusion

This project compared classical time series forecasting models with machine learning models for retail sales forecasting. ARIMA and SARIMA were used as statistical forecasting methods, while Random Forest and XGBoost were applied using lag-based feature engineering.

The ADF test confirmed that the sales series was stationary. ACF and PACF plots revealed strong weekly seasonality, which was further confirmed by time series decomposition. SARIMA improved over ARIMA by incorporating seasonal effects.

Among all models, Random Forest achieved the lowest MAE and RMSE values, making it the best-performing model overall. XGBoost achieved the lowest MAPE, while SARIMA was the strongest statistical model.

Overall, this project demonstrates the effectiveness of combining classical statistical forecasting techniques with machine learning methods for retail sales prediction. The results show that incorporating seasonality, historical sales behavior, and engineered features can significantly improve forecasting accuracy and support better business decision-making.
## Future Scope

This project can be further improved by:

* Performing hyperparameter tuning for Random Forest and XGBoost
* Including external features such as holidays and promotions
* Forecasting store-wise sales separately
* Comparing additional models such as Prophet or LSTM
* Building a dashboard for forecast visualization

## Repository Structure

```text
Sales-Forecasting-Time-Series-ML/
│
├── README.md
├── Sales_Forecasting_Project.ipynb
├── data/
│   ├── train.csv
│   └── store.csv
├── images/
│   ├── daily_sales_plot.png
│   ├── decomposition_plot.png
│   ├── acf_plot.png
│   ├── pacf_plot.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── forecast_plot.png
└── requirements.txt
```

## Author

Divisha Singh
M.Sc. Statistics
University of Delhi
