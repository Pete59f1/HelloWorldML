# Flat clustering. Making the machine put our data into clusters
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
style.use("ggplot")

# Data we are gonna be using
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

# Show data in graph
plt.scatter(x,y)
plt.show()

# Our training data set. Putting our x and y into a numpy array
X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# Tells the machine learning algorithm how many clusters we want
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Getting the data the algorithm provides
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Showing the new clusters
print(centroids)
print(labels)

colors = ["g.", "r."]

for i in range(len(X)):
    print("Coordinate:", X[i], "Label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker= "x", s=150, linewidths= 5, zorder = 10)
plt.show()
