import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Getting the data we're working with
data = pd.read_csv("Skriv hvor du har gemt data'en/train.csv").as_matrix()

# Making the first 21000 rows our training data, but we do not include the first column, because that's our labels
xtrain = data[0:21000, 1:]
# Here we have our labels
train_labels = data[0:21000, 0]

# Test data
xtest = data[21000:, 1:]
test_labels = data[21000:, 0]

# Creating and training
clf = DecisionTreeClassifier()
clf.fit(xtrain, train_labels)

# Using our test to test our model
d = xtest[8]
d.shape = (28, 28)
pt.imshow(255-d, cmap="gray")
print(clf.predict([xtest[8]]))
pt.show()

# Giving us a score of how accurate it is
predictions = clf.predict(xtest)
print(accuracy_score(test_labels, predictions))
