# Supervised classification problem
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# first number: 1 = saw a horror, 0 = didn't
# second number: 1 = has school next day, 0 = don't
# third number: 1 = went to bed, 0 = didn't
data = [[1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1]]

# Getting features and labels
y = []
x = []
for dat in data:
    y.append(dat[2])
    x.append(dat[: 2])

# Splitting training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Creating and training model
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Testing
predictions = clf.predict(x_test)
print(accuracy_score(y_test, predictions))
