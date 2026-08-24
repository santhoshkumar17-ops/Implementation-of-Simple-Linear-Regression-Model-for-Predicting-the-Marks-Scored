# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable<br>
2. Calculate the mean of the X -values and the mean of the Y -values<br>
3. Find the slope m of the line of best fit using the formula<br>
4. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SANTHOSH KUMAR SS
RegisterNumber:212225230251
*/

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Step 2: Create Dataset (Hours studied vs Marks scored)
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored": [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}

df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df


# Step 3: Split into Features and Target
X = df[["Hours_Studied"]]
y = df["Marks_Scored"]


# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)


# Step 6: Predictions
y_pred = model.predict(X_test)


# Step 7: Model Evaluation
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# Step 8: Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()


# Step 9: Predict Marks for custom input
hours = 7.5
predicted_marks = model.predict([[hours]])

print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")
```

## Output:
<img width="906" height="796" alt="image" src="https://github.com/user-attachments/assets/aed74c48-bc02-480f-9a32-ae31ac474d6f" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
