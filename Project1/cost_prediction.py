# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Loading the dataset
data = pd.read_csv('data_insurance.csv')

# Step 2: Exploring the dataset
print(data.head())          
print(data.info())          
print(data.describe())      

# Step 3: Checking for missing values
print(data.isnull().sum())  

# Step 4: Converting categorical variables into numerical values
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data = pd.get_dummies(data, columns=['region'], drop_first=True)  

# Step 5: Defining features (X) and target (y)
X = data.drop('charges', axis=1)  
y = data['charges']               

# Step 6: Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Make predictions using the test dataset
y_pred = model.predict(X_test)

# Step 9: Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Step 10: Analyzing the model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("Model Coefficients:\n", coefficients)

# Step 11: Sorting coefficients by their absolute values to identify the most influential features
coefficients['Absolute Coefficient'] = coefficients['Coefficient'].abs()
sorted_coefficients = coefficients.sort_values(by='Absolute Coefficient', ascending=False)
print("\nSorted Coefficients by Absolute Value:\n", sorted_coefficients)

# Interpretation
print("\nInterpretation of Coefficients:")
for index, row in sorted_coefficients.iterrows():
    feature = index
    coef = row['Coefficient']
    if coef > 0:
        print(f"{feature}: Positive relationship with charges; as {feature} increases, charges tend to increase.")
    else:
        print(f"{feature}: Negative relationship with charges; as {feature} increases, charges tend to decrease.")

#Creating a single figure with two subplots
plt.figure(figsize=(12, 6))

# Subplot 1: Coefficients plot
plt.subplot(1, 2, 1)
plt.bar(sorted_coefficients.index, sorted_coefficients['Coefficient'])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Influence of Features on Insurance Charges')

# Subplot 2: Actual vs Predicted Charges
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Insurance Charges')

# displaying the plot
plt.tight_layout()
plt.show()