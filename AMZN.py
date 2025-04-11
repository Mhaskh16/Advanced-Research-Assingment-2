import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
data = pd.read_csv('AMZN.csv', parse_dates=['Date'], index_col='Date')

# Plot the closing prices
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Closing Price', color='blue')
plt.title('Amazon (AMZN) Closing Share Price (2018-2023)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.show()

# Split into train (90%) and test (10%)
train_size = int(len(data) * 0.9)
train = data.iloc[:train_size]
test = data.iloc[train_size:]

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")

# Plot the closing prices
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Close'], label='Closing Price', color='blue')
plt.title('Plot for only training data')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.show()

# Perform ADF test
adf_result = adfuller(train['Close'], autolag='AIC')
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print(f'Critical Values:')
for key, value in adf_result[4].items():
    print(f'   {key}: {value}')
if adf_result[1] <= 0.05:
    print("The series is stationary.")
else:
    print("The series is not stationary.")

# First-order differencing
train['Close_diff'] = train['Close'].diff().dropna()

# Drop NaN/inf values from 'Close_diff'
clean_diff = train['Close_diff'].replace([np.inf, -np.inf], np.nan).dropna()

# Plot differenced data
plt.figure(figsize=(12, 6))
plt.plot(train['Close_diff'], label='First-Order Differencing', color='orange')
plt.title('First-Order Differenced Data')
plt.xlabel('Date')
plt.ylabel('Price Difference ($)')
plt.legend()
plt.grid()
plt.show()

adf_result = adfuller(clean_diff, autolag='AIC')
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print(f'Critical Values:')
for key, value in adf_result[4].items():
    print(f'{key}: {value}')
if adf_result[1] <= 0.05:
    print("The series is stationary.")
else:
    print("The series is not stationary.")

# Plot ACF and PACF to guess p and q
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(clean_diff, lags=20, ax=ax1)
plot_pacf(clean_diff, lags=20, ax=ax2, method='ywm')
plt.show()

# Fit ARIMA(0,1,0)
model = ARIMA(train['Close'], order=(40, 0, 0))
results = model.fit()

# Forecast test period
forecast_steps = len(test['Close'])
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test['Close'], forecast_mean))
print(f"RMSE (Test Forecast): {rmse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(train['Close'].index[-100:], train['Close'][-100:], label='Training Data (Last 100 Obs)')
plt.plot(test['Close'].index, test['Close'], label='Actual Test Data', color='green')
plt.plot(test['Close'].index, forecast_mean, label='ARIMA Forecast', color='red')
plt.fill_between(test['Close'].index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('ARIMA(0,1,0) Forecast vs Actual (Test Set)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Prepare data for RNN
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close']])

# Function to create sequences for RNN
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Parameters
SEQ_LENGTH = 60  # Using 60 days of history to predict next day
TEST_SIZE = len(test)  # Same test set as ARIMA
TOTAL_SIZE = len(data)
TRAIN_SIZE = TOTAL_SIZE - TEST_SIZE

# Create sequences
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split into train and test (same split as ARIMA)
X_train = X[:TRAIN_SIZE-SEQ_LENGTH]
y_train = y[:TRAIN_SIZE-SEQ_LENGTH]
X_test = X[TRAIN_SIZE-SEQ_LENGTH:TRAIN_SIZE-SEQ_LENGTH+TEST_SIZE]
y_test = y[TRAIN_SIZE-SEQ_LENGTH:TRAIN_SIZE-SEQ_LENGTH+TEST_SIZE]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Build RNN model
model = Sequential([
    SimpleRNN(100, activation='relu', input_shape=(SEQ_LENGTH, 1), return_sequences=True),
    Dropout(0.2),
    SimpleRNN(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('RNN Training History')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# Predict on test set
rnn_predictions = model.predict(X_test)
rnn_predictions = scaler.inverse_transform(rnn_predictions).flatten()

actual_values = scaler.inverse_transform(y_test).flatten()

# Calculate RMSE for RNN
rnn_rmse = np.sqrt(mean_squared_error(actual_values, rnn_predictions))
print(f"RNN RMSE (Test Forecast): {rnn_rmse:.2f}")

# Prepare data for 24-month forecast (using last SEQ_LENGTH points from full dataset)
last_sequence = scaled_data[-SEQ_LENGTH:]
future_predictions = []
months_to_predict = 24

for _ in range(months_to_predict):
    # Reshape the last sequence for prediction
    current_sequence = last_sequence.reshape(1, SEQ_LENGTH, 1)
    # Predict next point
    next_pred = model.predict(current_sequence, verbose=0)
    # Append prediction and update sequence
    future_predictions.append(next_pred[0,0])
    last_sequence = np.append(last_sequence[1:], next_pred)

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# Create future dates
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=months_to_predict, freq='M')

# Plot comparison
plt.figure(figsize=(14, 7))
plt.plot(data.index[-100:], data['Close'][-100:], label='Historical Data (Last 100 Obs)', color='blue')
plt.plot(test.index, test['Close'], label='Actual Test Data', color='green')
plt.plot(test.index[:len(rnn_predictions)], rnn_predictions, label='RNN Test Predictions', color='red')
plt.plot(future_dates, future_predictions, label='RNN 24-Month Forecast', color='purple', linestyle='--')
plt.title('Amazon Stock Price: RNN Predictions vs Actual')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Compare both models
print("\nModel Comparison:")
print(f"ARIMA(0,1,0) RMSE: {rmse:.2f}")
print(f"RNN RMSE: {rnn_rmse:.2f}")

if rnn_rmse < rmse:
    print("The RNN model performed better than ARIMA on the test set.")
else:
    print("The ARIMA model performed better than RNN on the test set.")
