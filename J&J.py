import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import math

# Loading data
df = pd.read_csv('jj.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# train-test split (90%-10%)
train_size = int(len(df) * 0.9)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")
print(f"Test period: {test.index[0]} to {test.index[-1]}")

# Plot train
plt.figure(figsize=(12, 6))
plt.plot(df.iloc[:train_size])
plt.title('JJ Training Data')
plt.ylabel('Sales')
plt.xlabel('Year')
plt.grid(True)
plt.show()

adf_test = adfuller(df)
print(f'ADF p-value: {adf_test[1]:.4f}')
if adf_test[1] <= 0.05:
    print("The series is stationary.")
else:
    print("The series is not stationary.")

# Apply transformations to training data only
train_log = np.log(train['data'])
seasonal_diff = train_log.diff(4).dropna()
final_series = seasonal_diff.diff(1).dropna()

# Plot transformed training series
plt.figure(figsize=(12, 6))
final_series.plot()
plt.title('Transformed Training Series')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Check Stationarity test after applying Transformation
adf_test = adfuller(final_series)
print(f'ADF p-value: {adf_test[1]:.4f}')
if adf_test[1] <= 0.05:
    print("The series is stationary.")
else:
    print("The series is not stationary.")

# Plot ACF and PACF to guess p and q
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(final_series, lags=20, ax=ax1)
plot_pacf(final_series, lags=20, ax=ax2, method='ywm')
plt.show()

# ARMA model
model = ARIMA(final_series, order=(1, 1, 1))
result = model.fit()
print(result.summary())

# Calculate how many steps to forecast (length of test set)
forecast_steps = len(test + 24)

# Forecast transformed values
forecast = result.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Reconstruct original scale forecast
# Get last 4 values from training log data for seasonal reconstruction
last_log_values = train_log.iloc[-4:].values

# Inverse transformations
forecast_regdiff = final_series.iloc[-1] + forecast_mean.cumsum()
forecast_seasdiff = np.array([last_log_values[i % 4] for i in range(forecast_steps)]) + forecast_regdiff
final_forecast = np.exp(forecast_seasdiff)

# Create forecast index
forecast_index = pd.date_range(start=train.index[-1] + pd.DateOffset(months=3),
                              periods=forecast_steps,
                              freq='QS')

# Calculate evaluation metrics
mse = mean_squared_error(test['data'], final_forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test['data'] - final_forecast) / test['data'])) * 100

print(f"\nEvaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# Plot comparison
plt.figure(figsize=(15, 6))
df.plot(label='Original')
test['data'].plot(label='Test')
pd.Series(final_forecast, index=forecast_index).plot(label='Forecast', color='red')
plt.fill_between(forecast_index,
                 np.exp(conf_int.iloc[:, 0] + np.array([last_log_values[i % 4] for i in range(forecast_steps)])),
                 np.exp(conf_int.iloc[:, 1] + np.array([last_log_values[i % 4] for i in range(forecast_steps)])),
                 color='red', alpha=0.1)
plt.title('ARMA Model Forecast vs Actual Data')
plt.ylabel('Earnings per Share ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Plot best model forecast
forecast = best_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Reconstruct forecast
forecast_regdiff = final_series.iloc[-1] + forecast_mean.cumsum()
forecast_seasdiff = np.array([last_log_values[i % 4] for i in range(forecast_steps)]) + forecast_regdiff
final_forecast = np.exp(forecast_seasdiff)

plt.figure(figsize=(12, 6))
train['data'].plot(label='Train')
test['data'].plot(label='Test')
pd.Series(final_forecast, index=forecast_index).plot(label=f'ARMA{best_order} Forecast', color='green')
plt.fill_between(forecast_index,
                 np.exp(conf_int.iloc[:, 0] + np.array([last_log_values[i % 4] for i in range(forecast_steps)])),
                 np.exp(conf_int.iloc[:, 1] + np.array([last_log_values[i % 4] for i in range(forecast_steps)])),
                 color='green', alpha=0.1)
plt.title(f'Best ARMA Model (Order 1,1,1) Forecast vs Actual')
plt.ylabel('Earnings per Share ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(df[['data']].values)

train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = 4  # Using 4 quarters (1 year) as lookback
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print(f'Train Score: {trainScore:.2f} RMSE')
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print(f'Test Score: {testScore:.2f} RMSE')

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

plt.figure(figsize=(15,6))
plt.plot(scaler.inverse_transform(dataset), label='Original Data')
plt.plot(trainPredictPlot, label='Training Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
plt.title('LSTM Model Predictions')
plt.ylabel('Earnings per Share ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Forecast future 24 months
def forecast_lstm(model, last_sequence, n_steps):
    forecasts = []
    current_sequence = last_sequence.copy()

    for _ in range(n_steps):
        # Prepare input
        x_input = np.array([current_sequence[-look_back:]])
        x_input = x_input.reshape((1, 1, look_back))

        # Make prediction
        yhat = model.predict(x_input, verbose=0)
        forecasts.append(yhat[0,0])

        # Update sequence
        current_sequence = np.append(current_sequence, yhat)

    return forecasts

last_sequence = dataset[-look_back:]

future_forecast = forecast_lstm(model, last_sequence, 8)
future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

# Create future dates
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=8, freq='QS')

# Plot the future forecast
plt.figure(figsize=(15,6))
plt.plot(df.index, df['data'], label='Historical Data')
plt.plot(future_dates, future_forecast, 'r-', label='24-Month Forecast')
plt.title('LSTM 24-Month Future Forecast')
plt.ylabel('Earnings per Share ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Compare ARIMA and LSTM performance
print("\nModel Comparison:")
print(f"ARIMA {best_order} Test RMSE: {best_rmse:.4f}")
print(f"LSTM Test RMSE: {testScore:.4f}")
