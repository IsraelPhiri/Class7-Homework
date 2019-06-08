
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Loading data from sklearn lib datasets
from sklearn.datasets import load_wine
wine = load_wine()
columns_names = wine.feature_names

#assign viriables to data and target
X = wine.data
y = wine.target

# Splitting features and target datasets into: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.35)

# Training a Linear Regression model with fit()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
lr.fit(X_train, y_train)

# Output of the training is a model: a + b*X0 + c*X1 + d*X2 ...
print(f"Intercept per class: {lr.intercept_}\n")
print(f"Coeficients per class: {lr.coef_}\n")

print(f"Available classes: {lr.classes_}\n")
print(f"Named Coeficients for class 1: {pd.DataFrame(lr.coef_[0], columns_names)}\n")
print(f"Named Coeficients for class 2: {pd.DataFrame(lr.coef_[1], columns_names)}\n")
print(f"Named Coeficients for class 3: {pd.DataFrame(lr.coef_[2], columns_names)}\n")
print(f"Number of iterations generating model: {lr.n_iter_}")

# Predicting the results for test dataset
predicted_values = lr.predict(X_test)

# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, pred: {predicted} {"Ouch!!" if real != predicted else ""}')

# Printing accuracy score(mean accuracy) from 0 - 1
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/ \n')

# Printing the classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print('Classification Report')
print(classification_report(y_test, predicted_values))

# Printing the classification confusion matrix (diagonal is true)
print('Confusion Matrix')
print(confusion_matrix(y_test, predicted_values))

print('Overall f1-score')
print(f1_score(y_test, predicted_values, average="macro"))

# Cross validation using cross_val_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
print(f'Cross validation score:{cross_val_score(lr, X, y, cv=5)}\n')

# Cross validation using shuffle split
cv = ShuffleSplit(n_splits=5)
print(f'ShuffleSplit val_score:{cross_val_score(lr, X, y, cv=cv)}\n')

# Visualizing structure of dataset in 2D
os.makedirs('plots/class7', exist_ok=True)
pca = PCA(n_components=2)
proj = pca.fit_transform(wine.data)
plt.scatter(proj[:, 0], proj[:, 1], c=wine.target, edgecolors='black')
plt.colorbar()
plt.savefig(f'plots/class7/wine_dataset_in_2D.png', format='png')
