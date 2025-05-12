
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

arima_model = ARIMA(df['Post-Sterilization CFU'], order=(1, 1, 1))
arima_result = arima_model.fit()
df['arima_residuals'] = arima_result.resid

plt.plot(df['arima_residuals'])
plt.title('ARIMA Residuals')
plt.show()

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train Loss', 'Validation Loss'])
plt.show()

arima_forecast = arima_result.forecast(steps=len(X_test))
lstm_forecast = model.predict(X_test)
lstm_forecast_rescaled = scaler.inverse_transform(lstm_forecast)

final_forecast = arima_forecast + lstm_forecast_rescaled.flatten()

plt.plot(df['Post-Sterilization CFU'].iloc[-len(y_test):].values)
plt.plot(final_forecast)
plt.legend(['Actual CFU', 'Combined Forecast'])
plt.title('Combined ARIMA and LSTM Forecast')
plt.show()

mse_combined = mean_squared_error(df['Post-Sterilization CFU'].iloc[-len(y_test):], final_forecast)
print(f'Mean Squared Error (Combined ARIMA + LSTM): {mse_combined}')
