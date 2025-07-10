import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error 
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose


file_path = 'archive/train.csv'
data = pd.read_csv(file_path)
print(data.head())

data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')
data = data.groupby('date').agg({'sales': 'sum'}).reset_index()
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['day_of_week'] = data['date'].dt.dayofweek
data['lag1'] = data['sales'].shift(1)
data['lag2'] = data['sales'].shift(2)
data['rollingMean3'] = data['sales'].rolling(window=3).mean()
data['rollingMean7'] = data['sales'].rolling(window=7).mean()

data = data.dropna()
