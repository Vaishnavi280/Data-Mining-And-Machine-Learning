#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.interpolate import make_interp_spline

# Step 1: Load the dataset
df = pd.read_csv('Toyota Motor’s.csv')



# In[6]:


# Step 2: Explore the dataset (EDA)
print(df.head())  # Print the first few rows to understand the structure of data
print(df.info())  # Get a summary of the dataset including column datatypes
print(df.describe())  # Statistical summary of numerical columns

# Check for missing values
print(df.isnull().sum())

# Step 3: Data Preprocessing
# Convert 'Date' column to datetime format, specifying the correct format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Save the last date from the original dataset
last_date = df['Date'].max()

# Extracting year, month, and day from Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day



# In[7]:


# Step 4: Prepare Data for Modeling
# Features for the model
features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Year', 'Month', 'Day']
X = df[features]
y = df['Close']  # Target variable is 'Close' price

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 6: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Create an SVM Regressor model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1)  # Adjust parameters as needed

# Step 8: Train the model using the training sets
svm_model.fit(X_train_scaled, y_train)

# Step 9: Make predictions using the testing set
y_pred = svm_model.predict(X_test_scaled)

# Step 10: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_error(y_test, y_pred) / np.mean(y_test) * 100  # Calculate MAPE
rmse = np.sqrt(mse)

print('Mean Squared Error:', mse)
print('R-squared:', r2)
print('MAPE:', mape)
print('RMSE:', rmse)

# Step 11: Predict future values for the next 10 years
# Create a new dataframe for future predictions (next 10 years)
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365*10, freq='D')
future_data = {
    'Open': np.repeat(df['Open'].mean(), len(future_dates)),
    'High': np.repeat(df['High'].mean(), len(future_dates)),
    'Low': np.repeat(df['Low'].mean(), len(future_dates)),
    'Adj Close': np.repeat(df['Adj Close'].mean(), len(future_dates)),
    'Volume': np.repeat(df['Volume'].mean(), len(future_dates)),
    'Year': future_dates.year,
    'Month': future_dates.month,
    'Day': future_dates.day
}
future_df = pd.DataFrame(future_data)

# Standardize the new data
future_df_scaled = scaler.transform(future_df)

# Predict future values
future_predictions = svm_model.predict(future_df_scaled)

# Add predictions to the future_df
future_df['Predicted_Close'] = future_predictions
future_df['Date'] = future_dates  # Add the Date column back for plotting

# Step 4: Visualize the data 
# Visualize the distribution of 'Close' prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Close'], bins=30, kde=True, color='green')
plt.title('Distribution of Close Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.show()



# In[8]:


# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()




# In[9]:


# Print future values
print(future_df[['Date', 'Predicted_Close']])

# Create smoother lines using spline interpolation for the future predictions
days_smooth = np.linspace(0, len(future_dates) - 1, 300)
spl = make_interp_spline(np.arange(len(future_dates)), future_predictions, k=3)
future_predictions_smooth = spl(days_smooth)



# In[10]:


# Plot actual and future predicted values
plt.figure(figsize=(14, 8))

# Plot actual values
plt.plot(df['Date'], df['Close'], marker='o', linestyle='--', color='blue', markersize=5, label='Actual Prices')

# Plot future predictions with a different style
plt.plot(future_dates, future_predictions, linestyle='--', color='orange', linewidth=2, label='Future Predictions')

# Highlight the area of future predictions
plt.fill_between(future_dates, future_predictions, color='orange', alpha=0.2)

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




