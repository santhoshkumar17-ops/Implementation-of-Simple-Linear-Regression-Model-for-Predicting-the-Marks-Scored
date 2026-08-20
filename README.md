# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset into a DataFrame and explore its contents to understand the data structure. 
2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3.Create a linear regression model and fit it using the training data.
4.Predict the results for the testing set and plot the training and testing sets with fitted lines.
5.Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance.
## Program:

Program to implement the simple linear regression model for predicting the marks scored.
# Developed by:SANTHOSH KUMAR S S
# RegisterNumber:  212225230251

# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
# 1) Load dataset
```
df = pd.read_csv(r"C:\\Users\\admin\\OneDrive\\Desktop\\ML\\exp_2_dataset_student_scores.csv")   # CSV should have two columns, e.g. "Hours","Scores"
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
```
# 2) Prepare input (X) and output (Y)
# Assume CSV columns: Hours (feature) and Scores (target)
```
X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)
Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)
print("X (features):", X.flatten())
print("Y (targets):", Y)
```

# 3) Split data into training and testing sets
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))
```
# 4) Create and train the model
```
regressor = LinearRegression()
regressor.fit(X_train, Y_train)   # fit on training data
```
# 5) Predict on the test set
```
Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)
```
# 6) Plot training results
```
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```
# 7) Plot testing results (use X_test sorted for a nicer line)
```
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```
# 8) Evaluation metrics
```
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
```
# 9) Example: predict for new students
```
new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
```
## Output:

# Load dataset

<img width="367" height="372" alt="image" src="https://github.com/user-attachments/assets/65a5bf0f-41b6-4945-aaee-93006c5ec426" />

# Prepare input (X) and output (Y)

<img width="927" height="98" alt="image" src="https://github.com/user-attachments/assets/a97d0e13-eab5-4741-ba78-514d4f96c772" />


# Split data into training and testing sets

<img width="497" height="50" alt="image" src="https://github.com/user-attachments/assets/25b6bad8-8446-4c83-b74f-4ab31bd14274" />


# Create and train the model

<img width="372" height="101" alt="image" src="https://github.com/user-attachments/assets/83a87131-700a-4514-a08c-465a515fafc7" />


# Predict on the test set

<img width="795" height="83" alt="image" src="https://github.com/user-attachments/assets/4a7a179d-dde5-432e-b4cc-31ec330dfad9" />


# Plot training results

<img width="855" height="503" alt="image" src="https://github.com/user-attachments/assets/a528b16a-82a8-41be-93a0-fb43f0cdaa04" />


# Plot testing results (use X_test sorted for a nicer line)

<img width="882" height="497" alt="image" src="https://github.com/user-attachments/assets/b5af42b5-b377-4a23-8fb3-d21ccf9093f2" />


# Evaluation metrics

<img width="576" height="102" alt="image" src="https://github.com/user-attachments/assets/c9c38cb5-3adc-4000-8aff-524809954410" />


# Example: predict for new students

<img width="612" height="52" alt="image" src="https://github.com/user-attachments/assets/962a8689-0dd3-42f1-a81e-d913edc218b5" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
