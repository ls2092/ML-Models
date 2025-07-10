import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error 
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose


file_path1 = 'archive/train.csv'
file_path2 = 'archive/features.csv'
file_path3 = 'archive/stores.csv'

train_data = pd.read_csv(file_path1)
features = pd.read_csv(file_path2)
store = pd.read_csv(file_path3)
#print(train_data.head())
#print(features.head())

train_data['Date'] = pd.to_datetime(train_data['Date'])
features['Date'] = pd.to_datetime(features['Date'])

#train_data['IsHoliday'] = train_data['IsHoliday'].astype(int)
#features['IsHoliday'] = features['IsHoliday'].astype(int)

data = pd.merge(train_data, features, on=['Store', 'Date'], how='left')
data = pd.merge(data, store, on='Store', how='left')

# one hot encoding of 'Type' column
data = pd.get_dummies(data, columns=['Type'], drop_first=False)

#data['IsHoliday'] = data['IsHoliday_x']
#data.drop(columns=['IsHoliday_x', 'IsHoliday_y'], inplace=True, errors='ignore')

data = data.sort_values('Date')
#data = data.groupby('Date').agg({'Weekly_Sales': 'sum'}).reset_index()
data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year
data['day_of_week'] = data['Date'].dt.dayofweek
data['lag1'] = data['Weekly_Sales'].shift(1)
data['lag2'] = data['Weekly_Sales'].shift(2)
data['rollingMean3'] = data['Weekly_Sales'].rolling(window=3).mean()
data['rollingMean7'] = data['Weekly_Sales'].rolling(window=7).mean()


data = data.dropna(subset=['lag1', 'lag2', 'rollingMean3', 'rollingMean7', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales', 'Size'])

result = seasonal_decompose(data.groupby('Date')['Weekly_Sales'].sum(), model='additive', period=30)
result.plot()
plt.show()

training_features = data[['day', 'month', 'year', 'day_of_week', 'lag1', 'lag2', 'rollingMean3', 'rollingMean7', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Type_A', 'Type_B', 'Type_C']]
target = data['Weekly_Sales']

X = training_features
y = target

split_index = int(len(data) * 0.8) # 80% for training, 20% for testing
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

plt.figure(figsize=(14, 6))
plt.plot(data['Date'].iloc[split_index:], y_test, label='Actual', color='green')
plt.plot(data['Date'].iloc[split_index:], y_pred, label='Predicted', color='red')
plt.legend()
plt.title('Weekly Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.show()