#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Step 1: Load the dataset
df = pd.read_csv('Toyota Motor’s.csv')



# In[2]:


# Step 2: Explore the dataset (EDA)
print(df.head())  # Print the first few rows to understand the structure of data
print(df.info())  # Get a summary of the dataset including column datatypes
print(df.describe())  # Statistical summary of numerical columns

# Check for missing values
print(df.isnull().sum())



# In[3]:


# Step 3: Data Preprocessing
# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Extracting year from Date
df['Year'] = df['Date'].dt.year

# Step 4: Feature Selection
# Selecting features for modeling
X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Year']]  # Include 'Year' in features
y = df['Close']  # Target variable is 'Close' price

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 6: Create a Linear Regression model
model = LinearRegression()

# Step 7: Train the model using the training sets
model.fit(X_train, y_train)

# Step 8: Make predictions using the testing set
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_error(y_test, y_pred) / np.mean(y_test) * 100  # Calculate MAPE
rmse = np.sqrt(mse)

print('Mean Squared Error:', mse)
print('R-squared:', r2)
print('MAPE:', mape)
print('RMSE:', rmse)

# Step 10: Visualize results (Year-wise Actual vs Predicted Close Price)
result_df = pd.DataFrame({'Year': X_test['Year'], 'Actual': y_test, 'Predicted': y_pred})
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



# In[4]:


# Step 11: Visualize correlation matrix 
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Year']
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()



# In[5]:


# Step 12: Predict future values for the next 10 years
# Define the number of future days to predict (next 10 years)
future_years = np.arange(df['Year'].max() + 1, df['Year'].max() + 11)

# Generate future dates
future_dates = []
for year in future_years:
    future_dates.extend(pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D'))

# Prepare future data
future_data = {
    'Open': np.repeat(df['Open'].mean(), len(future_dates)),
    'High': np.repeat(df['High'].mean(), len(future_dates)),
    'Low': np.repeat(df['Low'].mean(), len(future_dates)),
    'Adj Close': np.repeat(df['Adj Close'].mean(), len(future_dates)),
    'Volume': np.repeat(df['Volume'].mean(), len(future_dates)),
    'Year': [date.year for date in future_dates]
}
future_df = pd.DataFrame(future_data)
future_df['Date'] = future_dates  # Add Date for plotting purposes

# Predict future values
future_predictions = model.predict(future_df[['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Year']])
future_df['Predicted_Close'] = future_predictions

# Print future values
print(future_df[['Date', 'Predicted_Close']].head(30))  # Print first 30 rows for a quick check

# Plot actual and future predicted values
plt.figure(figsize=(14, 8))

# Plot actual values
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




