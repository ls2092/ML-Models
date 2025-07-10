import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error 
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose


file_path = 'archive/train.csv'
data = pd.read_csv(file_path)
print(data.head())

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data = data.groupby('Date').agg({'Weekly_Sales': 'sum'}).reset_index()
data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year
data['day_of_week'] = data['Date'].dt.dayofweek
data['lag1'] = data['Weekly_Sales'].shift(1)
data['lag2'] = data['Weekly_Sales'].shift(2)
data['rollingMean3'] = data['Weekly_Sales'].rolling(window=3).mean()
data['rollingMean7'] = data['Weekly_Sales'].rolling(window=7).mean()

data = data.dropna()

result = seasonal_decompose(data.set_index('Date')['Weekly_Sales'], model='additive', period=30)
result.plot()
plt.show()

features = data[['day', 'month', 'year', 'day_of_week', 'lag1', 'lag2', 'rollingMean3', 'rollingMean7']]
target = data['Weekly_Sales']

X = features
y = target

split_index = int(len(data) * 0.8) #80% for training, 20% for testing
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

plt.figure(figsize=(14, 6))
plt.plot(data['Date'].iloc[split_index:], y_test, label='Actual', color='green')
plt.plot(data['Date'].iloc[split_index:], y_pred, label='Predicted', color='red')
plt.legend()
plt.title('Weekly Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.show()