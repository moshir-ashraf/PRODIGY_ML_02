import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def euclideanDistance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


class KMeans:
    def __init__(self, k=3, maxIter=100):
        self.k = k
        self.maxIter = maxIter
        self.centroids = None
        self.labels = None

    def fitAndPredict(self, data):
        self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        for i in range(self.maxIter):
            distances = np.array([[euclideanDistance(field, centroid) for centroid in self.centroids]
                                  for field in data])
            self.labels = np.argmin(distances, axis=1)
            newCentroids = np.array([data[self.labels == j].mean(axis=0) for j in range(self.k)])
            if np.all(newCentroids == self.centroids):
                break
            self.centroids = newCentroids
        return self.labels

    def plotClusters(self, data):
        uniqueLabels = np.unique(self.labels)
        plt.figure(figsize=(10, 6))

        for clusterLabel in uniqueLabels:
            clusterPoints = data[self.labels == clusterLabel]
            plt.scatter(clusterPoints[:, 0], clusterPoints[:, 1], label=f'Cluster {clusterLabel}', alpha=0.7)

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='X', label='Centroids')
        plt.title('Clusters and Centroids')
        plt.legend()
        plt.show()


data = pd.read_csv(os.path.join('dataset', 'Mall_Customers.csv'))
# data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
scaler = StandardScaler()
scaledData = scaler.fit_transform(data[['Spending Score (1-100)','CustomerID']])
classifier = KMeans(k=9)
clusterLabels = classifier.fitAndPredict(scaledData)
classifier.plotClusters(scaledData)
data['Cluster'] = clusterLabels
print(data)
