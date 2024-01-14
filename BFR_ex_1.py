import numpy as np
from sklearn.cluster import KMeans

class BFR:

    def __init__(self, k, data):
        self.k = k
        self.d = data
        self.kmeans = KMeans(n_clusters=k, random_state=0)
        self.DS = {}
        self.CS = {}
        self.RS = {}

    def fit(self, data):
        self.kmeans.fit(data)
        labels = self.kmeans.labels_

        return labels


