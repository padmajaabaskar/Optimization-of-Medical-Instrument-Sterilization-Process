from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

MIN_LENGTH = 10

for column in df.columns:
    time_series = df[column].dropna()
    if time_series.empty or len(time_series) < MIN_LENGTH:
        continue
    result = adfuller(time_series)

numerical_columns = ['Temperature (°C)', 'Pressure (kPa)', 'Cycle Time (mins)',
                     'Load Size (kg)', 'Moisture Level (%)', 'Initial Contamination (CFU)',
                     'Post-Sterilization CFU']

n_columns = len(numerical_columns)
n_rows = n_columns
n_cols = 3

fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axs = axs.flatten()

for i, column in enumerate(numerical_columns):
    time_series = df[column].dropna()
    time_series_diff = time_series.diff().dropna()
    axs[i * n_cols].plot(time_series)
    axs[i * n_cols].set_title(f'{column} - Time Series')
    try:
        plot_acf(time_series_diff, lags=min(len(time_series_diff) // 2, 40), ax=axs[i * n_cols + 1])
        axs[i * n_cols + 1].set_title(f'{column} - ACF')
    except:
        axs[i * n_cols + 1].set_title(f'{column} - ACF (Error)')
    try:
        plot_pacf(time_series_diff, lags=min(len(time_series_diff) // 2, 40), ax=axs[i * n_cols + 2])
        axs[i * n_cols + 2].set_title(f'{column} - PACF')
    except:
        axs[i * n_cols + 2].set_title(f'{column} - PACF (Error)')

plt.tight_layout()
plt.show()
