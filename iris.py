import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

# Read dataset
df = pd.read_csv("Iris_data.csv")

# X=feature columns
X = np.array(df.drop(['class'], 1))

# y=label column
y = np.array(df['class'])

# split the data into two parts. 60% of data for train and 40% of data for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# use classification algorithm
clf = neighbors.KNeighborsClassifier()
# fit data into algo
clf.fit(X_train, y_train)

# find accuracy of our algorithm
accuracy = clf.score(X_test, y_test)
print(accuracy)

measures = np.array([[1,2,4,3]])
prediction = clf.predict(measures)
print(prediction)
