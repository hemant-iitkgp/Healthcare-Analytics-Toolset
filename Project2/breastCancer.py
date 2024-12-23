import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz

# Load the dataset
data = pd.read_csv('Breast_Cancer.csv')

# Separate features and target
X = data.drop('Status', axis=1)
y = data['Status']

# Convert categorical variables to numerical
X = pd.get_dummies(X, columns=['Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists to store results
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Train and evaluate the model for different depths
for depth in range(1, 21):
    clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    y_train_prob = clf.predict_proba(X_train)
    y_test_prob = clf.predict_proba(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_loss = log_loss(y_train, y_train_prob)
    test_loss = log_loss(y_test, y_test_prob)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plotting results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, 21), train_losses, label='Train Loss')
plt.plot(range(1, 21), test_losses, label='Validation Loss')
plt.xlabel('Depth')
plt.ylabel('Log Loss')
plt.title('Loss vs Depth')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, 21), train_accuracies, label='Train Accuracy')
plt.plot(range(1, 21), test_accuracies, label='Validation Accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Depth')
plt.legend()

plt.tight_layout()
plt.savefig('performance_metrics.png')
plt.show()

print(f"Final Validation Loss: {test_losses[-1]:.4f}")
print(f"Final Validation Accuracy: {test_accuracies[-1]:.4f}")

def display_decision_tree(clf, feature_names, max_depth=3):
    plt.figure(figsize=(12, 8))
    plot_tree(clf, 
              feature_names=feature_names, 
              class_names=['0', '1'],  # Adjust class names based on your target variable
              filled=True, 
              rounded=True, 
              max_depth=max_depth)
    plt.title(f'Decision Tree (Max Depth = {max_depth})')
    plt.savefig("decision_tree.png")
    plt.show()

# Train the model with a maximum depth of 3
clf_depth_3 = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf_depth_3.fit(X_train, y_train)

# Display the decision tree
display_decision_tree(clf_depth_3, feature_names=X.columns)
