import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Loading the Boston Housing dataset
boston_dataset = load_boston()

# Loading the dataset into Pandas' Data Frame
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
features = ['LSTAT', 'RM', 'PTRATIO', 'TAX']


# Normal equation function
def normal_equation(x, y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)


print()
print("----------------------------BOSTON HOUSING DATASET----------------------------")
print(boston.head())

# Correlation matrix
ax, fig = plt.subplots(figsize=(10, 10))
correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

# Scatter plot
f = plt.Figure()
print(pd.plotting.scatter_matrix(boston[features], s=75))

# Biased dataset computing
X_bias = np.c_[np.ones((len(boston[features]), 1)), boston[features].values]
y = boston['MEDV']

# Division into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=5)
print()
print("X TRAIN SIZE: ", X_train.shape)
print("Y TRAIN SIZE: ", y_train.shape)

X_train = np.log1p(X_train)
X_test = np.log1p(X_test)
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

# Computing the Theta value
theta = normal_equation(X_train, y_train)
print()
print("The value of Theta: ", theta)

# Training
predictions = np.dot(X_train, theta)

# Testing
test_predictions = np.dot(X_test, theta)

# Plot constructing
ax, fig = plt.subplots(figsize=(10, 10))
plt.plot(predictions, 'b.', marker='*')
plt.plot(y_train, 'r.')
plt.legend(['predictions', 'true'])
# plt.show()

ax, fig = plt.subplots(figsize=(10, 10))
plt.plot(test_predictions, 'b.', marker='*')
plt.plot(y_test, 'r.')
plt.legend(['test predictions', 'true'])
plt.show()

# MSE results
print()
print("MSE for the Training data: ", mean_squared_error(predictions, y_train))
print("MSE for the Testing data: ", mean_squared_error(test_predictions, y_test))

# Although the test dataset showed the result of MSE equals to ~24,
# it still has inaccuracies in validation process, so that's why
# the model was trained not in a proper way.
