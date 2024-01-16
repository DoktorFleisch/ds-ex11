import numpy as np
from sklearn.cluster import KMeans


class BFR:
    def __init__(self, k):
        self.k = k
        self.dimension = 0
        self.kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++')
        self.DS = []
        self.CS = []
        self.RS = []

    def __calculate_variance(self, cluster):
        return (cluster['SUMQ'] / cluster['N']) - (cluster['SUM'] / cluster['N']) ** 2

    def __calculate_standard_deviation(self, variance):
        return np.sqrt(variance)

    def __calculate_normalized_distance(self, centroid, point, standard_deviation):
        normalized_distances = [(p - c) / sd for p, c, sd in zip(point, centroid, standard_deviation)]

        return normalized_distances

    def __calculate_Mahalanobis(self, cluster, point):
        centroid = cluster['SUM'] / cluster['N']
        variance = self.__calculate_variance(cluster)
        standard_deviation = self.__calculate_standard_deviation(variance)
        normalized_distances = self.__calculate_normalized_distance(centroid, point, standard_deviation)

        return np.sqrt(np.sum(np.square(normalized_distances)))

    def __init_dimension(self, centroid):
        self.dimension = len(centroid)

    def select_k_random_points(self, data):
        self.kmeans.fit(data)
        centroids = self.kmeans.cluster_centers_

        self.__init_dimension(centroids[0])

        self.DS = [{'SUM': centroid, 'SUMQ': centroid ** 2, 'N': 1} for centroid in centroids]

        return self.DS

    def fit(self, data_chunk):
        for point in data_chunk:
            distances = [self.__calculate_Mahalanobis(cluster, point) for cluster in self.DS]
            min_distance_index = np.argmin(distances)

            if distances[min_distance_index] < 2 * np.sqrt(self.dimension):
                cluster = self.DS[min_distance_index]
                cluster['SUM'] += point
                cluster['SUMQ'] += point ** 2
                cluster['N'] += 1
            else:
                self.RS.append(point)

        self.kmeans.fit(self.RS)
        cluster_labels = self.kmeans.labels_
        cluster_centers = self.kmeans.cluster_centers_

        self.CS = [{'SUM': centroid, 'SUMQ': centroid ** 2, 'N': 1} for centroid in cluster_centers]
        self.RS = [point for i in range(self.k) if i not in cluster_labels for point in self.RS]

        return self.DS, self.CS, self.RS