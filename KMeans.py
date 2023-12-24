from utils import *

class KMeans():

    """
    Description:
        My from scratch implementation of the K Means Clustering Algorithm
    """

    # constructor
    def __init__(self, K, epochs):

        """
        Description:
            Constructor of our KMeans class
        
        Parameters:
            K: number of centroids
            epochs: number of epochs to train on
        
        Returns:
            None
        """

        # number of centroids and epochs
        self.K = K
        self.epochs = epochs

        # initialize clusters for each centroid
        self.clusters = [[] for _ in range(self.K)]

        # centers for each cluster
        self.centroids = []

    
    # predict
    def predict(self, X):

        """
        Description:
            Predicts the labels for our feature vectors X
        
        Parameters:
            X: feature vectors
        
        Returns:
            labels  
        """

        # create a reference for our feature vector to use throughout entire class
        self.X = X
        self.N, self.num_features = X.shape

        # initialize initial samples at random
        random_sample_idxs = np.random.choice(self.N, self.K, replace = False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimize clusters
        for _ in range(self.epochs):

            # assign sample to closest centroid
            self.clusters = self.create_clusters(self.centroids)

            # compute bew centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters)

            # if we are not assigning samples to new groups then break
            if self.is_converged(centroids_old, self.centroids):
                break

        # samples are classified as the index of the cluster
        labels = self.get_cluster_labels(self.clusters)

        # return 
        return self.centroids, labels

    # create clusters
    def create_clusters(self, centroids):

        """
        Description:
            Creates the clusters at every iteration
        
        Parameters:
            centroids: current centroids list
        
        Returns:
            clusters  
        """

        # initialize clusters
        clusters = [[] for _ in range(self.K)]

        # iterate over features
        for idx, sample in enumerate(self.X):

            # fetch the corresponding indices (centroid)
            centroid_idx = self.closest_centroid(sample, centroids)

            # append the sample to its corresponding cluster
            clusters[centroid_idx].append(idx)

        # return 
        return clusters


    # closest centroid
    def closest_centroid(self, sample, centroids):

        """
        Description:
            Finds the closest centroid based on the euclidean distance
        
        Parameters:
            sample: sample to assign to a cluster
            centroids: current centroids list
        
        Returns:
            closest_idx  
        """

        # compute distance using the euclidean distance
        distances = [euclidean_distance(sample, point) for point in centroids]

        # find the closest index
        closest_idx = np.argmin(distances)

        # return
        return closest_idx

    # get centroids
    def get_centroids(self, clusters):

        """
        Description:
            Get the centroids for each cluster
        
        Parameters:
            clusters: current clusters list
        
        Returns:
            centroids  
        """

        # assign the mean value of clusters to centroids
        centroids = np.zeros((self.K, self.num_features))

        # iterate over all clusters
        for cluster_idx, cluster in enumerate(clusters):

            # compute the mean for each cluster
            cluster_mean = np.mean(self.X[cluster], axis = 0)

            # assign each cluster ids to its corresponding mean
            centroids[cluster_idx] = cluster_mean

        # return
        return centroids
    
    # get cluster labels
    def get_cluster_labels(self, clusters):

        """
        Description:
            Get the cluster labels for each cluster
        
        Parameters:
            clusters: current clusters list
        
        Returns:
            labels  
        """

        # each sample get the label of the cluster it was assigned to
        labels = np.empty(self.N)

        # iterate over all clusters
        for cluster_idx, cluster in enumerate(clusters):

            # iterate over current cluster
            for sample_idx in cluster:
                
                # assign labels based on cluster index
                labels[sample_idx] = cluster_idx

        # return
        return labels
    
    # is converged
    def is_converged(self, centroids_old, centroids):

        """
        Description:
            Checks if our K Means Clustering algorithm has converged yet

        Parameters:
            centroids_old: old centroids list
            centroids: new centroids list
        
        Returns:
            converged
        """

        # find the distances between old and new centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]

        # get converged status
        converged = sum(distances) == 0

        # return
        return converged