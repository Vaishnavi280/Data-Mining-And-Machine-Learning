#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Step 1: Load the dataset
# Replace 'Ferrari.csv' with the actual path to your dataset
df = pd.read_csv('Ferari.csv')



# In[2]:


# Step 2: Explore the dataset (EDA)
print(df.head())  # Print the first few rows to understand the structure of data
print(df.info())  # Get a summary of the dataset including column datatypes
print(df.describe())  # Statistical summary of numerical columns

# Check for missing values
print(df.isnull().sum())

# Step 3: Data Preprocessing
# Convert 'Date' column to datetime format, specifying the correct format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Extracting year, month, and day from Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day



# In[3]:


# Step 4: Visualize the data (Optional)
# Visualize the distribution of 'Close' prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Close'], bins=30, kde=True, color='blue')
plt.title('Distribution of Close Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.show()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()



# In[5]:


# Step 5: Feature Selection
# Selecting features for modeling
features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Year', 'Month', 'Day']
X = df[features]
y = df['Close']  # Target variable is 'Close' price

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Extract 'Year' from the X_test before scaling
years_test = X_test['Year']

# Step 7: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Create a KNN Regressor model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Step 9: Train the model using the training sets
knn_model.fit(X_train_scaled, y_train)

# Step 10: Make predictions using the testing set
y_pred = knn_model.predict(X_test_scaled)

# Step 11: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_error(y_test, y_pred) / np.mean(y_test) * 100  # Calculate MAPE
rmse = np.sqrt(mse)

print('Mean Squared Error:', mse)
print('R-squared:', r2)
print('MAPE:', mape)
print('RMSE:', rmse)

# Step 12: Visualize results (Year-wise Actual vs Predicted Close Price)
result_df = pd.DataFrame({'Year': years_test, 'Actual': y_test, 'Predicted': y_pred})
year_groups = result_df.groupby('Year').mean().reset_index()

plt.figure(figsize=(12, 8))
plt.plot(year_groups['Year'], year_groups['Actual'], linestyle='-', color='b', label='Actual')
plt.plot(year_groups['Year'], year_groups['Predicted'], linestyle='--', color='r', label='Predicted')
plt.title('Year-wise Actual vs Predicted Close Price')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()

# Step 13: Predict future values
# Define the number of future days to predict (e.g., 5 years = 5*252 trading days)
future_years = 5
future_days = future_years * 252

# Generate future dates
last_date = df['Date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

# Prepare future data
future_df = pd.DataFrame({
    'Open': np.repeat(df['Open'].mean(), future_days),
    'High': np.repeat(df['High'].mean(), future_days),
    'Low': np.repeat(df['Low'].mean(), future_days),
    'Adj Close': np.repeat(df['Adj Close'].mean(), future_days),
    'Volume': np.repeat(df['Volume'].mean(), future_days),
    'Year': future_dates.year,
    'Month': future_dates.month,
    'Day': future_dates.day
})

# Standardize future data
future_scaled = scaler.transform(future_df)

# Predict future values
future_predictions = knn_model.predict(future_scaled)
future_df['Predicted_Close'] = future_predictions
future_df['Date'] = future_dates



# In[6]:


# Print future values
print(future_df[['Date', 'Predicted_Close']].head(30))  # Display first 30 future predictions

# Plot actual and future predicted values
plt.figure(figsize=(14, 8))

# Plot actual values using Date column
plt.plot(df['Date'], df['Close'], marker='o', linestyle='--', color='blue', markersize=5, label='Actual Prices')

# Plot future predictions
plt.plot(future_df['Date'], future_df['Predicted_Close'], linestyle='--', color='orange', linewidth=2, label='Future Predictions')

# Highlight the area of future predictions
plt.fill_between(future_df['Date'], future_df['Predicted_Close'], color='orange', alpha=0.2)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Add title and labels
plt.title('Actual and Future Predicted Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Show plot
plt.show()


# In[ ]:




