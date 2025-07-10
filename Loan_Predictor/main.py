import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

loan_data = pd.read_csv('archive/loan_approval_dataset.csv')

print(loan_data.columns)

obj_columns = loan_data.select_dtypes(include=['object']).columns
num_columns = loan_data.select_dtypes(include=['int64', 'float64']).columns

for col in obj_columns:
    loan_data[col] = loan_data[col].fillna(loan_data[col].mode()[0])

for col in num_columns:
    loan_data[col] = loan_data[col].fillna(loan_data[col].median())

label_encoder = LabelEncoder()
for col in obj_columns:
    loan_data[col] = label_encoder.fit_transform(loan_data[col])

X = loan_data.drop(['loan_id', ' loan_status'], axis=1) # predicted variables except loan_id and loan_status
y = loan_data[' loan_status'] #target variable

# Handling class imbalance using SMOTE to not support majority class
print("Before Smote: ", y.value_counts())
smote = SMOTE(random_state=42)
X_resample, y_resample = smote.fit_resample(X, y)
print("After Smote: ", y_resample.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaliuating the model
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Approved", "Approved"]))

important_features = pd.Series(model.feature_importances_, index=X.columns)
important_features.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
