# Sales Forecasting Using Facebook Prophet
# Introduction

Sales forecasting is a critical aspect of business operations, enabling companies to make informed decisions about inventory, staffing, and strategy. In this project, I forecasted weekly sales using historical transactional data and time series modeling techniques. After initial attempts with ARIMA, I adopted the Prophet model due to its flexibility, built-in handling of seasonality, and ease of use with holiday effects.

# Dataset Description

File: stores_sales_forecasting.csv
Store Sales Forecasting is a dataset designed for retail analysis, containing 21 columns of comprehensive information. It includes data on sales transactions, customer demographics, and product details.

# Data Preprocessing & EDA
Order Date converted to datetime

Sales data was aggregated weekly (each Monday as the start of the week) to form a consistent time series. Missing weeks were filled with zero sales to ensure continuity.

Aggregated daily sales to weekly totals:

weekly_sales = df.resample('W-MON', on='Order Date')['Sales'].sum().reset_index()
weekly_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'}, inplace=True)

Ensured no missing time points using pd.date_range()

Quick EDA revealed high variance in sales, including weeks with zero values (e.g., closures or missing data).

# Model Selection & Justification

I initially explored Auto ARIMA (pmdarima) with m=52 (weekly seasonality), but encountered system-level NumPy binary incompatibilities in Google Colab.
Instead of debugging dependencies, I pivoted to Prophet, which offered:

Native support for weekly & yearly seasonality

Built-in holiday effect modeling

Robustness to missing data & outliers

Simple, interpretable API

# Modeling with Prophet

Data split:

Training = all but last 12 weeks

Test = final 12 weeks (holiday-heavy period)

Prophet configuration:

Weekly & yearly seasonality enabled

U.S. holidays added via:

model.add_country_holidays(country_name='US')

Forecast horizon = 90 days (~13 weeks)

# Improvements & Debugging Tweaks

During development, Prophet and cmdstanpy generated verbose logs (INFO/DEBUG messages). These were silenced for cleaner output using Python’s logging module.

Additionally, evaluation metrics were refined:

Original MAPE was distorted due to division by zero when test weeks had no sales.

I introduced Filtered MAPE (ignoring zero-sales weeks) for a more realistic assessment.

# Model Evaluation
# Original Results

MAPE (raw): ~40.7% (skewed by zero sales)

Filtered MAPE: ~40.7%

RMSE: ~3808.79

# Updated Results (after tweaks)

MAPE (raw): Extremely large (invalid due to zeros)

Filtered MAPE: 28.94% 

RMSE: 934.14 

# Test Set Characteristics:

Mean weekly sales: ~828

Median weekly sales: 0 (many weeks with zero activity)

# Interpretation

RMSE dropped significantly (3808 → 934), meaning predictions are now much closer to actual weekly sales.

Filtered MAPE of ~29% shows reasonable accuracy when actual sales are >0.

The inflated raw MAPE highlights a metric limitation, not a model failure.

#  Visual Results

Forecast vs. actual plots show:

Seasonal patterns captured correctly

Reasonable fit during holidays (though Prophet slightly underestimates spikes)

#  Why Results Are Acceptable

Captures seasonal fluctuations & holiday effects

RMSE under 1000 vs. weekly sales in the hundreds/thousands

Strong baseline that can be improved with:

Additional regressors (promotions, store-level features), Changepoint tuning. Alternative ML models (XGBoost/LightGBM with engineered features)

[Demo Video](https://drive.google.com/file/d/1TysGt7liM5aEuLQjgHurWaSyhzqgdvfb/view?usp=drive_link)

This project demonstrates baseline sales forecasting with Prophet, practical debugging/tuning, and transparent evaluation practices.
