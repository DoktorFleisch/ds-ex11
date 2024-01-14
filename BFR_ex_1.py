import numpy as np
from sklearn.cluster import KMeans


class BFR:

    def __init__(self, k, data):
        self.k = k
        self.d = data
        self.kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++')
        self.DS = [[] for i in range(k)]
        self.CS = []
        self.RS = []

        # Sets gehen leider nicht da sie immutable sind

    def select_k_random_points(self, data):
        self.kmeans.fit(data)
        centroids = self.kmeans.cluster_centers_

        for i, centroid in enumerate(centroids):
            self.DS[i].append(centroid)

        #SUM = [np.sum(centroids[i]) for i in range(N)]
        #SUMQ = [np.sum(centroids[i] ** 2) for i in range(N)]

        return self.DS
