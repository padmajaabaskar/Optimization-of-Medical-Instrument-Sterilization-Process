pip install pandas openpyxl matplotlib
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd

file_path = '/content/sterilization_process_data.csv'
df = pd.read_csv(file_path)

value_map1 = {'Chemical': 1.0, 'Dry Heat': 2.0, 'Steam': 3.0}
df = df.replace(value_map1)

value_map2 = {'Cloth': 1.0, 'Glassware': 2.0, 'Plastic': 3.0, 'Surgical Tools': 4.0}
df = df.replace(value_map2)

df = df.apply(pd.to_numeric, errors='coerce')
df['Time'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
df.set_index('Time', inplace=True)
print(df.head())