import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Load the dataset (provide a direct link to the dataset)
df = pd.read_csv('/content/AAPL(1).csv')

# Display the first few rows of the dataset
print("Initial dataset:")
print(df.head())

# check for duplicates
duplicate_dates = df[df.duplicated(subset='Date')]
print("Duplicate Dates:")
print(duplicate_dates)

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Convert 'Date' column to datetime format and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Plotting Histogram
plt.figure(figsize=(20, 8))
plt.hist(df['Adj Close'], bins=20, color='LimeGreen')
plt.title('Histogram of Apple Stock Prices')
plt.xlabel('Apple Stocks')
plt.ylabel('Frequency')
plt.show()

# Plotting Scatter Plot
plt.figure(figsize=(20, 8))
sns.scatterplot(df['Adj Close'])
plt.title('Apple Stocks from 2021 to 2022')
plt.ylabel('Stocks')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.show()

# Plotting Time Series Trend
plt.figure(figsize=(20, 8))
plt.plot(df['Adj Close'])
plt.title('Apple Stocks Trend')
plt.show()

# Extract the 'Close' column for further analysis
closing_prices = df.filter(['Close']).values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(closing_prices)

# Create training dataset
train_data = scaled_prices[:int(len(scaled_prices) * 0.95), :]
train_features, train_labels = [], []

# Prepare training data
for i in range(60, len(train_data)):
    train_features.append(train_data[i-60:i, 0])
    train_labels.append(train_data[i, 0])

train_features, train_labels = np.array(train_features), np.array(train_labels)

# Build the DNN model
model = Sequential()
model.add(Dense(units=50, activation='relu', input_dim=train_features.shape[1]))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Initialize lists to store loss and accuracy values for each epoch
loss_values = []
accuracy_values = []

# Fit the model to the training data
for epoch in range(10):
    history = model.fit(train_features, train_labels, epochs=1, batch_size=32, verbose=1)
    loss_values.append(history.history['loss'][0])

    # Evaluate the model on training data
    train_predictions = model.predict(train_features)
    train_predictions = scaler.inverse_transform(train_predictions)
    train_actual = scaler.inverse_transform(train_labels.reshape(-1, 1))
    accuracy = 1 - mean_absolute_error(train_actual, train_predictions) / np.mean(train_actual)
    accuracy_values.append(accuracy)
    print(f'Epoch {epoch+1}: Loss = {loss_values[-1]}, Accuracy = {accuracy_values[-1]}')

# Create the testing dataset
test_data = scaled_prices[int(len(scaled_prices) * 0.95) - 60:, :]
x_test, y_test = [], closing_prices[int(len(closing_prices) * 0.95):, :]

# Prepare testing data
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Create a DataFrame for visualization
validation_set = pd.DataFrame(index=df.index[int(len(closing_prices) * 0.95):])
validation_set['Close'] = y_test.flatten()
validation_set['Predictions'] = predictions.flatten()

# Visualize the predicted prices compared to actual prices
plt.figure(figsize=(16, 8))
plt.title('Stock Price Prediction using DNN')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(validation_set[['Close', 'Predictions']], label=['Actual Prices', 'Predicted Prices'])
plt.legend(loc='lower right')
plt.show()

# Model Evaluation Metrics
mse_dnn = mean_squared_error(validation_set['Close'], validation_set['Predictions'])
rmse_dnn = np.sqrt(mse_dnn)
mae_dnn = mean_absolute_error(validation_set['Close'], validation_set['Predictions'])
mape_dnn = np.mean(np.abs((validation_set['Close'] - validation_set['Predictions']) / validation_set['Close'])) * 100
r2_dnn = r2_score(validation_set['Close'], validation_set['Predictions'])

# Calculate the accuracy of the predictions
accuracy_dnn = 1 - mape_dnn / 100

print('\nDNN Model Evaluation:')
print(f'MSE: {mse_dnn:.2f}')
print(f'RMSE: {rmse_dnn:.2f}')
print(f'MAE: {mae_dnn:.2f}')
print(f'MAPE: {mape_dnn:.2f}%')
print(f'Accuracy: {accuracy_dnn:.2%}')
print(f'R2 Score')
