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

    def calculate_variance(self, cluster):
        return (cluster['SUMQ'] / cluster['N']) - (cluster['SUM'] / cluster['N']) ** 2

    def calculate_standard_deviation(self, variance):
        return np.sqrt(variance)

    def calculate_normalized_distance(self, centroid, point, standard_deviation):
        normalized_distances = [(p - c) / sd for p, c, sd in zip(point, centroid, standard_deviation)]

        return normalized_distances

    def calculate_Mahalanobis(self, cluster, point):
        centroid = cluster['SUM'] / cluster['N']
        variance = self.calculate_variance(cluster)
        standard_deviation = self.calculate_standard_deviation(variance)
        normalized_distances = self.calculate_normalized_distance(centroid, point, standard_deviation)

        return np.sqrt(np.sum(np.square(normalized_distances)))

    def select_k_random_points(self, data):
        self.kmeans.fit(data)
        centroids = self.kmeans.cluster_centers_

        for i, centroid in enumerate(centroids):
            cluster = {'SUM': centroid, 'SUMQ': centroid ** 2, 'N': 1}
            self.DS.append(cluster)

        return self.DS

    def fit(self, data):
       # Schritt 1: Punkte sind nah einem Cluster von DS: Füge zu DS hinzu und update SUM, SUMQ, N für das jeweilige Cluster.

       # Schritt 2: Keiner der Punkte ist nah einem Cluster von DS: Nutze Kmeans zum Clustern dieser übrig geblieben Punkte, und die
       # aus dem alten RS-Set. Entstehende Cluster gehen in CS, und die outliner in RS.

       # Schritt 3: Ggfs. Merge Cluster in CS.

       # Schritt 4: Merge die Cluster in DS und CS, die nah beieinander liegen. Merge Punkte in RS mit den Clustern in DS.



        return self.DS, self.CS, self.RS