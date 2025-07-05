#exam score prediction based on study hours and attendance

import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path = 'archive/StudentPerformanceFactors.csv'
student_data = pd.read_csv(file_path)
student_data.columns

student_data = student_data.dropna()

plt.scatter(student_data['Hours_Studied'], student_data['Exam_Score'], color='red')
plt.title('No. of hours studied vs Final Grade')
plt.xlabel('Study Hours')
plt.ylabel('Final Grade')
plt.grid(True)

plt.savefig('plot.png')
print("Plot saved as 'plot.png'")

X = student_data[['Hours_Studied', 'Attendance']]
y = student_data['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

examScore_pred = model.predict(X_test)
mse = mean_squared_error(y_test, examScore_pred)
r2 = r2_score(y_test, examScore_pred)

print(f"\nModel Evaluation:")
print(f'Mean Squared Error: {mse: .2f}')
print(f'R² Score: {r2:.2f}')


plt.figure()
plt.scatter(X['Hours_Studied'], y, color='blue', label='Actual Data')
plt.title('Study Hours vs Final Grade (Colored by Attendance)')
plt.xlabel('Study Hours')
plt.ylabel('Final Grade')
plt.grid(True)
plt.savefig('regression_plot.png')
print("Regression plot saved as 'regression_plot.png'")


try:
    user_hours = float(input("\nEnter number of study hours: "))
    user_attendance = float(input("Enter attendance percentage: "))

    user_input = [[user_hours, user_attendance]]
    predicted_score = model.predict(user_input)

    print(f"\nPredicted Final Grade for {user_hours} hours of study and {user_attendance}% attendance: {predicted_score[0]:.2f}")

except ValueError:
    print("Invalid input! Please enter numeric values only.")
