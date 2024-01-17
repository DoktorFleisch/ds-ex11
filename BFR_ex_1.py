import numpy as np
from sklearn.cluster import KMeans


class BFR:

    # for kmeans später, nutze 3*k
    def __init__(self, k):
        """
        Init wird nur verwendet um die benötigten Variablen zu initialisieren.
        :param k:
        """
        self.k = k
        self.dimension = 0
        self.kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++', n_init=10)
        self.DS = []
        self.CS = []
        self.RS = []
        self.threshold = 0.5

    def __calculate_variance(self, cluster):
        """
        Berechnet die Varianz eines Clusters.
        :param cluster:
        :return:
        """
        return (cluster['SUMQ'] / cluster['N']) - (cluster['SUM'] / cluster['N']) ** 2

    def __calculate_standard_deviation(self, variance):
        """
        Berechnet die Standardabweichung eines Clusters.
        :param variance:
        :return:
        """
        return np.sqrt(variance)

    def __calculate_normalized_distance(self, centroid, point, standard_deviation):
        """
        Berechnet die normalisierten Distanzen zwischen einem Punkt und einem Cluster.
        :param centroid:
        :param point:
        :param standard_deviation:
        :return:
        """
        normalized_distances = [(p - c) / sd for p, c, sd in zip(point, centroid, standard_deviation)]

        return normalized_distances

    def __calculate_Mahalanobis(self, cluster, point):
        """
        Berechnet die Mahalanobis Distanz zwischen einem Punkt und einem Cluster.
        :param cluster:
        :param point:
        :return:
        """
        centroid = cluster['SUM'] / cluster['N']
        variance = self.__calculate_variance(cluster)
        standard_deviation = self.__calculate_standard_deviation(variance)
        normalized_distances = self.__calculate_normalized_distance(centroid, point, standard_deviation)

        return np.sqrt(np.sum(np.square(normalized_distances)))

    def __init_dimension(self, centroid):
        """
        Initialisiert die Dimension der Clusters.
        :param centroid:
        :return:
        """
        self.dimension = len(centroid)

    def select_k_random_points(self, data):
        """
        Wählt k zufällige Punkte aus dem Datensatz aus, und erstellt daraus die initialen Cluster.
        :param data:
        :return:
        """
        self.kmeans.fit(data)
        centroids = self.kmeans.cluster_centers_

        cluster_labels = self.kmeans.labels_

        self.__init_dimension(centroids[0])

        clusters = []
        for i in range(self.k):
            cluster_points = data[cluster_labels == i]
            clusters.append(cluster_points)

        for cluster in clusters:
            sum = np.sum(cluster, axis=0)
            sumq = np.sum(cluster ** 2, axis=0)
            self.DS.append({'SUM': sum, 'SUMQ': sumq, 'N': len(cluster)})

        return
    def fit(self, data_chunk):
        """
        BFR Algorithmus.
        :param data_chunk:
        :return:
        """

        """
        Berechne die Mahalanobis Distanz zwischen jedem Punkt und jedem Cluster in DS. Füge den Punkt dem Cluster hinzu, 
        dessen Distanz am kleinsten ist, und innerhalb des Thresholds ist wie in der VL definiert (3.Wurzel(dimensionen).
        Sonst füge den Punkt zu RS hinzu.
        """
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

        """
        Cluster alle verbleibenden Punkte in RS mit kmeans. Füge die Cluster in CS ein. Outlier werden in RS hinzugefügt.
        """
        if len(self.RS) >= 2:
            self.kmeans.fit(self.RS)
            cluster_labels = self.kmeans.labels_
            cluster_centers = self.kmeans.cluster_centers_

            self.CS = [{'SUM': centroid, 'SUMQ': centroid ** 2, 'N': 1} for centroid in cluster_centers]
            self.RS = [point for i in range(self.k) if i not in cluster_labels for point in self.RS]

        """
        Merge clusters in CS. Berechne die Vararianz der kombinierten Cluster. 
        Falls diese unterhalb eines Schwellenwertes liegt, füge die Cluster zusammen.
        Der Threshold ist hierbei 0.5. Es wurde kein Threshold in der VL definiert.
        """

        for i, cluster_i in enumerate(self.CS):
            for j, cluster_j in enumerate(self.CS[i + 1:]):
                j += i + 1

                variance_i = self.__calculate_variance(cluster_i)
                variance_j = self.__calculate_variance(cluster_j)

                variance_combined = variance_i + variance_j

                if np.any(variance_combined <= self.threshold):
                    self.CS[i]['SUM'] += self.CS[j]['SUM']
                    self.CS[i]['SUMQ'] += self.CS[j]['SUMQ']
                    self.CS[i]['N'] += self.CS[j]['N']

                    del self.CS[j]

        """
        Merge CS and DS.
        Calculate the nearest distance between a cluster in CS and a cluster in DS.
        """

        for centroid_cs in self.CS:
            distances = [self.__calculate_Mahalanobis(centroid_ds, centroid_cs['SUM']) for centroid_ds in self.DS]
            min_distance_index = np.argmin(distances)

            cluster = self.DS[min_distance_index]
            cluster['SUM'] += centroid_cs['SUM']
            cluster['SUMQ'] += centroid_cs['SUMQ']
            cluster['N'] += centroid_cs['N']



        """
        Merge RS and DS
        Calculate the nearest distance between a point in RS and a cluster in DS.
        """

        for point in self.RS:
            distances = [self.__calculate_Mahalanobis(cluster_ds, point) for cluster_ds in self.DS]
            min_distance_index = np.argmin(distances)

            cluster = self.DS[min_distance_index]
            cluster['SUM'] += point
            cluster['SUMQ'] += point ** 2
            cluster['N'] += 1

        return self.DS

if __name__ == '__main__':
    """
    Diese Mainmethode wurde zum Debuggen verwendet.
    """
    data = np.array([[1, 2], [1, 4], [1, 0],
                     [4, 2], [4, 4], [4, 0]])
    # Definieren Sie Mittelwerte und Kovarianzmatrix
    mean = [0, 0]  # Mittelwerte für x und y
    covariance_matrix = [[1, 0.5], [0.5, 1]]  # Kovarianzmatrix

    # Anzahl der Punkte
    num_points = 1000

    # Erstellen Sie die normalverteilten Punkte
    points = np.random.multivariate_normal(mean, covariance_matrix, num_points)

    bfr = BFR(2)
    DS = bfr.select_k_random_points(points)
    DS = bfr.fit(points)
    print(DS)