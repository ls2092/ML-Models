import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

file_path1 = 'archive/train.csv'
file_path2 = 'archive/features.csv'
file_path3 = 'archive/stores.csv'
#file_path4 = 'archive/test.csv'

train_data = pd.read_csv(file_path1)
features = pd.read_csv(file_path2)
stores = pd.read_csv(file_path3)
#test = pd.read_csv(file_path4)
#print(train_data.head())
#print(features.head())

train_data['Date'] = pd.to_datetime(train_data['Date'])
features['Date'] = pd.to_datetime(features['Date'])
#test['Date'] = pd.to_datetime(test['Date'])

train_data['IsHoliday'] = train_data['IsHoliday'].astype(int)
#features['IsHoliday'] = features['IsHoliday'].astype(int)
#test['IsHoliday'] = test['IsHoliday'].astype(int)

data = pd.merge(train_data, features, on=['Store', 'Date'], how='left', suffixes=('', '_feat'))
data = pd.merge(data, stores, on='Store', how='left')
#data = pd.merge(data, test, on=['Store', 'Date'], how='left')

data = data.sort_values('Date')
#data = data.groupby('Date').agg({'Weekly_Sales': 'sum'}).reset_index()
data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year
data['day_of_week'] = data['Date'].dt.dayofweek
data['lag1'] = data['Weekly_Sales'].shift(1)
data['lag2'] = data['Weekly_Sales'].shift(2)
data['lag3'] = data['Weekly_Sales'].shift(3)
data['rollingMean3'] = data['Weekly_Sales'].rolling(window=3).mean()
data['rollingMean7'] = data['Weekly_Sales'].rolling(window=7).mean()
data['rollingMean14'] = data['Weekly_Sales'].rolling(window=14).mean()

data = data[data['Weekly_Sales'] > 0]

data['Dept'] = data['Dept'].astype(str)
data = pd.get_dummies(data, columns=['Type', 'Dept'], drop_first=False)

data = data.dropna(subset=['lag1', 'lag2', 'lag3', 'rollingMean3', 'rollingMean7', 'rollingMean14', 'Weekly_Sales', 'Size'])

weeklySalesDate = data.groupby('Date')['Weekly_Sales'].sum()
if len(weeklySalesDate) > 60:
    result = seasonal_decompose(weeklySalesDate, model='additive', period=30)
    result.plot()
    plt.show()
else:
    print("Skipping seasonal decomposition")

training_features = data[['day_of_week', 'lag1', 'lag2', 'lag3','rollingMean3', 'rollingMean7', 'rollingMean14', 'Size', 'IsHoliday']]
training_features = training_features.join(data[[col for col in data.columns if col.startswith('Type_') or col.startswith('Dept_')]])

target = data['Weekly_Sales']

X = training_features
y = target

split_index = int(len(data) * 0.8) # 80% for training, 20% for testing
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,reg_alpha=0.5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

plt.figure(figsize=(14, 6))
plt.plot(data['Date'].iloc[split_index:], y_test, label='Actual', color='yellow')
plt.plot(data['Date'].iloc[split_index:], y_pred, label='Predicted', color='green')
plt.legend()
plt.title('Weekly Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.show()


# Feature importance plot
importances = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(20), importances[sorted_idx][:20])
plt.xticks(range(20), feature_names[sorted_idx][:20], rotation=90)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()
