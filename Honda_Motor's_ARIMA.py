#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load your dataset
data = pd.read_csv('Honda Motor’s Corporation.csv')



# In[3]:


# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data.set_index('Date', inplace=True)

# Data exploration and visualization
print("First few rows of the dataset:")
print(data.head())

print("\nData Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())



# In[4]:


# Plotting historical close prices
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.title('Historical Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# In[5]:


# ARIMA model parameters (p, d, q)
p = 3  # AR parameter
d = 1  # Integration order (differencing)
q = 2  # MA parameter

# Create and fit ARIMA model
model = ARIMA(data['Close'], order=(p, d, q))
arima_model = model.fit()

# Forecasting future values
# Forecast for the next 10 years (approx. 252 trading days per year)
future_years = 10  # Number of years to forecast
future_steps = future_years * 252  # Trading days per year
forecast_future = arima_model.get_forecast(steps=future_steps)

# Extract forecasted values and confidence intervals
forecasted_values = forecast_future.predicted_mean
confidence_intervals = forecast_future.conf_int()

# Generate future dates for the forecasted values
last_date = data.index[-1]
forecast_index = pd.date_range(start=last_date, periods=future_steps + 1, freq='B')[1:]  # Business days

# Create a Series for forecasted values with correct index
forecast_series = pd.Series(forecasted_values.values, index=forecast_index)



# In[6]:


# Print forecasted close prices
print(f"\nForecasted Close Prices for Future {future_years} Years:")
print(forecast_series.head(30))  # Print first 30 rows for a quick check

# Plotting actual vs. forecasted
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Actual', color='blue')
plt.plot(forecast_series.index, forecast_series.values, label='Forecast', linestyle='--', color='red')
plt.title('Actual vs. Forecasted Close Price (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error

# Calculate MAE between actual and forecasted values
mae = mean_absolute_error(data['Close'].tail(len(forecast_series)), forecast_series)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

from sklearn.metrics import mean_squared_error

# Calculate MSE between actual and forecasted values
mse = mean_squared_error(data['Close'].tail(len(forecast_series)), forecast_series)
print(f"Mean Squared Error (MSE): {mse:.2f}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# In[ ]:




