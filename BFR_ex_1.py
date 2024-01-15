import numpy as np
from sklearn.cluster import KMeans


class BFR:

    def __init__(self, k, data):
        self.k = k
        self.d = data
        self.kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++')
        self.DS = []
        self.CS = []
        self.RS = []

        # Sets gehen leider nicht da sie immutable sind

    def select_k_random_points(self, data):
        self.kmeans.fit(data)
        centroids = self.kmeans.cluster_centers_

        # dies diente zu Testzwecken
       # centroid_test = np.array([[5, 1], [6, -2], [7, 0]])

        for i, centroid in enumerate(centroids):
            cluster = {'SUM': centroid, 'SUMQ': centroid ** 2, 'N': 1}
            self.DS.append(cluster)

        #SUM = [np.sum(centroids[i]) for i in range(N)]
        #SUMQ = [np.sum(centroids[i] ** 2) for i in range(N)]

        return self.DS
