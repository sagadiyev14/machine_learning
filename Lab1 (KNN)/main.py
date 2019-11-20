import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=15)

classifier = KNeighborsClassifier()

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print("Accuracy rate: ", accuracy_score(y_test, predictions))

ax, fig = plt.subplots(figsize=(7.5, 5.5))
plt.plot(predictions, 'b.', marker='*')
plt.plot(y_test, 'r.')
plt.legend(['predictions', 'true'])
plt.show()

neighbors = np.arange(1, 45)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

plt.plot(neighbors, train_accuracy, label='Training dataset accuracy rate')
plt.plot(neighbors, test_accuracy, label='Testing dataset accuracy rate')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy rate')
plt.show()
