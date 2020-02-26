from scipy.spatial import distance


# Calculate the distance between a and b, where a is a point from training data, and b is a point from testing data
def euc(a, b):
    return distance.euclidean(a, b)


class OwnClassifier:
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


from sklearn import datasets
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()

x = iris.data
y = iris.target

# Splits data into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= .5)

# Creating and training classifier
# Removed import to create my own classifier above
# from sklearn.neighbors import KNeighborsClassifier
my_classifier = OwnClassifier()
my_classifier.fit(x_train, y_train)

predictions = my_classifier.predict(x_test)
print(accuracy_score(y_test, predictions))
