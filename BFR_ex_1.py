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
        self.threshold = 0.5

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

            if distances[min_distance_index] <= 3 * np.sqrt(self.dimension):
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

        # merge clusters in CS. Berechne die Vararianz der kombinierten Cluster. Falls diese unterhalb eines Schwellenwertes liegt, füge die Cluster zusammen.

        for i in range(len(self.CS)):
            for j in range(i + 1, len(self.CS)):
                cluster_i = self.CS[i]
                cluster_j = self.CS[j]

                variance_i = self.__calculate_variance(cluster_i)
                variance_j = self.__calculate_variance(cluster_j)

                variance_combined = variance_i + variance_j

                if variance_combined <= self.threshold:
                    cluster_i['SUM'] += cluster_j['SUM']
                    cluster_i['SUMQ'] += cluster_j['SUMQ']
                    cluster_i['N'] += cluster_j['N']

                    self.CS.remove(cluster_j)

        # Merge CS and DS.
        # Calculate the nearest distance between a cluster in CS and a cluster in DS.

        for centroid_cs in self.CS:
            distances = [self.__calculate_Mahalanobis(centroid_ds, centroid_cs) for centroid_ds in self.DS]
            min_distance_index = np.argmin(distances)

            cluster = self.DS[min_distance_index]
            cluster['SUM'] += centroid_cs['SUM']
            cluster['SUMQ'] += centroid_cs['SUMQ']
            cluster['N'] += centroid_cs['N']



        # Merge RS and DS
        # Calculate the nearest distance between a point in RS and a cluster in DS.

        for point in self.RS:
            distances = [self.__calculate_Mahalanobis(cluster_ds, point) for cluster_ds in self.DS]
            min_distance_index = np.argmin(distances)

            cluster = self.DS[min_distance_index]
            cluster['SUM'] += point
            cluster['SUMQ'] += point ** 2
            cluster['N'] += 1

        return self.DS