from statsmodels.tsa.arima.model import ARIMA

time_series = df['Post-Sterilization CFU'].dropna()
model = ARIMA(time_series, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=10)
print(forecast)

plt.figure(figsize=(10,6))
plt.plot(time_series.index, time_series)
plt.plot(pd.date_range(time_series.index[-1], periods=10, freq='D'), forecast, color='red')
plt.title('ARIMA Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(['Original', 'Forecast'])
plt.show()

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

df['arima_residuals'] = model_fit.resid

features = ['Temperature (°C)', 'Pressure (kPa)', 'Cycle Time (mins)', 'Load Size (kg)',
            'Moisture Level (%)', 'Initial Contamination (CFU)', 'arima_residuals'] + \
           [col for col in df.columns if 'Sterilization Type' in col or 'Instrument Type' in col]

X = df[features].dropna()
y = df['Post-Sterilization CFU'].loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

gbm_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
gbm_model.fit(X_train, y_train)

y_pred = gbm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'GBM with ARIMA Residuals - Mean Squared Error: {mse}')