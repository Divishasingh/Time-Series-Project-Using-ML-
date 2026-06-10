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
  
## Statistical Techniques Used

- Time Series Decomposition
- Augmented Dickey-Fuller (ADF) Test
- Autocorrelation Function (ACF)
- Partial Autocorrelation Function (PACF)
- ARIMA Modeling
- SARIMA Modeling
- Forecast Error Evaluation (MAE, RMSE, MAPE)

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
 <img width="826" height="451" alt="image" src="https://github.com/user-attachments/assets/fda4447a-2efa-4d57-a84d-55f5309ba7d2" />
 <img width="1233" height="470" alt="image" src="https://github.com/user-attachments/assets/fe8452fd-b5ea-4537-9289-40c70b2c99e0" />
 <img width="822" height="469" alt="image" src="https://github.com/user-attachments/assets/eb4d7463-2cd4-4215-816a-593349cc0c60" />
 <img width="644" height="470" alt="image" src="https://github.com/user-attachments/assets/1203b729-0b4b-405d-8c96-c4fd02c49c77" />


### Observation

The daily sales plot showed repeated fluctuations, indicating the presence of seasonality. The distribution of sales was not perfectly normal and showed variation due to business cycles, weekly effects, and possible holiday/store closure effects.

The yearly sales chart was removed from the final analysis because the dataset contained less than two complete years of data, making yearly comparison less meaningful.

## 3. Time Series Decomposition

Time series decomposition was performed to break the sales series into three components:

* Trend
* Seasonality
* Residuals

Since the data showed a strong weekly pattern, decomposition was performed using a seasonal period of 7.
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/15bdaf08-e672-4b70-9464-fb740e547a3a" />


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
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/37f55fac-41fc-4dad-8dcb-43fd83b09ecb" />
<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/a1f46f0e-b03f-4917-a0de-abdbe6b516be" />

### Observation

The ACF plot showed significant spikes at lags 7, 14, 21, 28, and 35. This confirmed the presence of strong weekly seasonality.

The PACF plot also showed significant spikes around lag 7 and lag 14, suggesting autoregressive behavior and supporting the use of seasonal models.

## 6. ARIMA Model

ARIMA (AutoRegressive Integrated Moving Average) is a widely used statistical forecasting model that combines autoregressive and moving average components. It assumes that future observations can be explained using past observations and past forecast errors.

An ARIMA model was fitted as the baseline statistical time series model.

The model used was:

ARIMA(1,0,1)

### Observation

The ARIMA(1,0,1) model was fitted after confirming stationarity through the Augmented Dickey-Fuller test. The model was able to capture short-term autocorrelation present in the sales series and provided a baseline statistical forecasting framework.

The forecasting performance of ARIMA was evaluated using MAE, RMSE, and MAPE. Although the model successfully captured short-term dependencies, it was unable to effectively model the strong weekly seasonal patterns present in the data. Consequently, the forecasting errors were relatively high compared to SARIMA and machine learning models.

The results indicate that while ARIMA is useful for modeling non-seasonal time series behavior, additional seasonal components are required when significant periodic patterns exist in the data.


## 7.SARIMA Model

SARIMA (Seasonal AutoRegressive Integrated Moving Average) extends ARIMA by incorporating seasonal autoregressive and moving average terms. It is particularly useful when recurring seasonal patterns are present in the time series.
A SARIMA model was fitted to capture both non-seasonal and seasonal patterns.

The model used was:

SARIMA(1,0,1)(1,0,1,7)

The seasonal period was taken as 7 because the data showed strong weekly seasonality.

SARIMA was selected because the decomposition plot, ACF plot, and PACF plot all indicated strong weekly seasonality with a seasonal period of 7 days.

### Observation

SARIMA outperformed ARIMA because it incorporated both non-seasonal and seasonal components. The seasonal period of 7 days was selected based on the weekly seasonal pattern observed in the decomposition, ACF, and PACF plots.

The model was able to capture recurring weekly fluctuations more effectively than ARIMA, resulting in substantially lower forecasting errors. The lower AIC and BIC values further indicated a better model fit.

The improvement in performance demonstrates the importance of accounting for seasonality in retail sales forecasting. Among the statistical forecasting approaches considered in this study, SARIMA emerged as the best-performing model.

## 8. Feature Engineering for Machine Learning

Since machine learning models cannot directly understand temporal dependence, time-based features were created to capture historical sales behavior and seasonal information.

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

A Random Forest Regressor was trained using the engineered lag-based and calendar-based features. Random Forest is an ensemble learning technique that combines multiple decision trees and aggregates their predictions to improve accuracy and reduce overfitting.

The model was trained using the following predictors:

* Lag 1
* Lag 7
* Lag 14
* Rolling Mean 7
* Rolling Standard Deviation 7
* Month
* Day of Week

These features enabled the model to learn both short-term dependencies and recurring seasonal behavior present in the sales data.

### Observation

The Random Forest model successfully captured the weekly sales pattern and produced forecasts that closely followed the actual sales values. The Actual vs Predicted plot demonstrated that the model was able to track major fluctuations in sales while maintaining stable forecasting performance.

The model achieved the lowest MAE and RMSE values among all forecasting approaches considered in this project, indicating the highest overall predictive accuracy.

The strong performance of Random Forest highlights the effectiveness of feature engineering in time series forecasting and demonstrates how machine learning models can successfully capture nonlinear relationships within sales data.

<img width="1221" height="443" alt="image" src="https://github.com/user-attachments/assets/a5d3d0f9-85b1-4110-a46e-aa7a9915b82e" />

## 10. XGBoost Regressor

An XGBoost Regressor was developed using the same engineered features. XGBoost is a gradient boosting algorithm that builds decision trees sequentially, where each new tree attempts to correct the prediction errors of the previous trees.

The model was trained using:

* Lag 1
* Lag 7
* Lag 14
* Rolling Mean 7
* Rolling Standard Deviation 7
* Month
* Day of Week

The objective was to evaluate whether a boosting-based machine learning model could improve forecasting performance relative to Random Forest and statistical forecasting approaches.

### Observation

The XGBoost model effectively captured sales fluctuations and weekly seasonal patterns. It achieved competitive forecasting performance and produced lower forecasting errors than both ARIMA and SARIMA in certain evaluation metrics.

Although its MAE and RMSE values were slightly higher than those of Random Forest, it achieved the lowest MAPE value among all models. This indicates strong relative forecasting accuracy and demonstrates the capability of gradient boosting methods in retail sales forecasting.

The results confirm that machine learning models can provide substantial forecasting improvements when appropriate lag and rolling-window features are incorporated.


## 11. Model Evaluation

All models were evaluated using:

MAE (Mean Absolute Error) measures the average magnitude of forecasting errors.

RMSE (Root Mean Squared Error) penalizes larger forecasting errors more heavily and is useful for evaluating prediction accuracy.

MAPE (Mean Absolute Percentage Error) expresses forecasting error as a percentage of actual values, allowing relative comparison across models.

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

<img width="755" height="451" alt="image" src="https://github.com/user-attachments/assets/20199f63-998b-45d4-b0fc-a3bccd051c0b" />


### Interpretation

Day of Week was the most important feature, contributing more than 54% importance. This confirms that sales are strongly affected by weekly patterns.

Lag 7 was the second most important feature, showing that sales from the same day of the previous week strongly influence current sales.

Rolling Mean 7 also contributed meaningfully, indicating that recent sales trends are useful for forecasting.

## 13. Forecasting

Forecasting is the process of predicting future observations based on historical patterns present in the data. The primary objective of forecasting is to estimate future demand and support decision-making under uncertainty.

In this project, forecasting was performed using the best-performing time series model to predict future sales for the next 90 days. The forecast was generated using historical sales patterns, trend information, and weekly seasonal behavior identified during the exploratory and statistical analysis stages.

A 90-day forecast was selected instead of a 12-month forecast because the available time series covered less than one full year. Forecasting too far into the future using limited historical data may reduce reliability.

<img width="1214" height="451" alt="image" src="https://github.com/user-attachments/assets/4100ca31-a662-491c-b7bc-b15f080636c0" />


### Observation

The forecast showed stable sales behavior with recurring weekly fluctuations. This indicates that weekly seasonality is expected to continue in the near future.

## Business Insights

### Demand Planning
Weekly seasonality was identified as the strongest driver of sales. Businesses can use these recurring patterns to improve demand forecasting and reduce forecasting errors.

### Inventory Planning
Lag-based features such as Lag 7 and Rolling Mean 7 were found to be important predictors. This suggests that previous sales behavior can be used to optimize inventory replenishment decisions.

### Resource Allocation
Since sales vary systematically across days of the week, staffing and operational resources can be allocated more efficiently based on expected demand levels.

### Revenue Planning
Accurate forecasting models such as Random Forest can support revenue estimation and budgeting by providing reliable short-term sales forecasts.

### Key Findings
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

The consistency between decomposition results, ADF testing, ACF/PACF analysis, and model evaluation demonstrates the importance of combining statistical inference with machine learning techniques in forecasting applications.

## Future Scope

This project can be further improved by:

* Performing hyperparameter tuning for Random Forest and XGBoost
* Including external features such as holidays and promotions
* Forecasting store-wise sales separately
* Comparing additional models such as Prophet or LSTM
* Building a dashboard for forecast visualization


## Author

Divisha Singh |
M.Sc. Statistics |
University of Delhi
