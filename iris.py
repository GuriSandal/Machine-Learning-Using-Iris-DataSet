import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors

df = pd.read_csv("Iris_data.csv")

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

measures = np.array([[1,2,4,3]])
prediction = clf.predict(measures)
print(prediction)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

