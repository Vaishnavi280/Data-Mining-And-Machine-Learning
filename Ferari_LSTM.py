#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
data = pd.read_csv("Ferari.csv")



# In[4]:


# Convert 'Date' to datetime with the correct format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Ensure the data is sorted by date
data.sort_values('Date', inplace=True)

# Exploratory Data Analysis (EDA)

# Plotting Time Series Data
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Closing Price')
plt.title('Stock Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Checking for Missing Values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

print(data.head())  # Print the first few rows to understand the structure of data
print(data.info())



# In[5]:


# Descriptive Statistics
print("\nDescriptive Statistics:\n", data.describe())

# Data Preprocessing

# Ensure the 'Close' prices are scaled using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close']].values)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

# Set the sequence length (number of time steps to look back)
seq_length = 20

# Create input sequences for the LSTM model
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Reshape the input data to be 3-dimensional for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build and Train the LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Generate predictions on the testing data
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values to the original scale
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean((y_test - predictions.flatten())**2))

print(f'MAPE: {mape:.2f}%')
print(f'RMSE: {rmse:.2f}')

# Forecasting for the next 10 years

# Adjust 'last_date' to the last date in the dataset
last_date = data['Date'].max()

# Set future steps for 10 years (10 * 365 = 3650 days, adjusted for business days)
future_steps = 3650  # Approximate number of business days in 10 years

# Generate future dates starting from the next day after the last known date
future_dates = pd.date_range(last_date, periods=future_steps + 1, freq='B')[1:]

# Start with the last sequence from the actual data
last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)

future_predictions = []

for _ in range(future_steps):
    next_pred = model.predict(last_sequence)[0, 0]
    future_predictions.append(next_pred)
    last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred]]], axis=1)

# Inverse transform future predictions to the original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# Plot actual prices and future predictions
plt.figure(figsize=(14, 7))

# Plot actual values up to the last date in the dataset
plt.plot(data['Date'], data['Close'], label='Actual Prices')

# Plot future predictions for the next 10 years
plt.plot(future_dates, future_predictions, linestyle='--', color='orange', linewidth=2, label='10-Year Predictions')

# Highlight the area of future predictions
plt.fill_between(future_dates, future_predictions, color='orange', alpha=0.2)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Add title and labels
plt.title('Actual and 10-Year Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Show plot
plt.show()


# In[ ]:




