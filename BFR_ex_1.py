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

        for i, centroid in enumerate(centroids):
            cluster = {'SUM': centroid, 'SUMQ': centroid ** 2, 'N': 1}
            self.DS.append(cluster)

        return self.DS

    def fit(self, data_chunk):
       # Schritt 1: Punkte sind nah einem Cluster von DS: Füge zu DS hinzu und update SUM, SUMQ, N für das jeweilige Cluster.
        for point in data_chunk:
            distances = [self.__calculate_Mahalanobis(cluster, point) for cluster in self.DS]
            min_distance = min(distances)
            min_distance_index = distances.index(min_distance)

            if min_distance < 2*np.sqrt(self.dimension):
                self.DS[min_distance_index]['SUM'] += point
                self.DS[min_distance_index]['SUMQ'] += point ** 2
                self.DS[min_distance_index]['N'] += 1
            else:
                self.RS.append(point)
       # Schritt 2: Keiner der Punkte ist nah einem Cluster von DS: Nutze Kmeans zum Clustern dieser übrig geblieben Punkte, und die
       # aus dem alten RS-Set. Entstehende Cluster gehen in CS, und die outliner in RS.

       # Schritt 3: Ggfs. Merge Cluster in CS.

       # Schritt 4: Merge die Cluster in CS und DS, die nah beieinander liegen. Merge Punkte in RS mit den Clustern in DS.



        return self.DS, self.CS, self.RS