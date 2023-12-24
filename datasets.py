# sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

# datasets
class Datasets():

    """
    Description:
        Holds different classification datasets
    """

    # constructor
    def __init__(self, random_state):

        """
        Description:
            Constructor for our Datasets class
        
        Parameters:
            random_state: random state chosen for reproducible output
        
        Returns:
            None
        """

        self.random_state = random_state

    # blobs
    def make_blobs(self, K, num_samples, num_features):

        """
        Description:
            Loads toy dataset using sklearn make_blobs
        
        Parameters:
            num_samples: number of samples to generate for toy dataset
            num_features: number of features per training sample
        
        Returns:
            X, y, class_names
        """

        # make blobs of data
        X, y = datasets.make_blobs(n_samples = num_samples, n_features = num_features, 
                                                    centers = K, shuffle = True, random_state = self.random_state)  

        # class names
        class_names = ['class 0', 'class 1', 'class 2']

        # return
        return  X, y, class_names


    # iris
    def load_iris(self):

        """
        Description:
            Loads sklearn's Iris Dataset

        Parameters:
            None
        
        Returns:
            X, y, feature_names, class_names
        """
        
        # load dataset
        data = datasets.load_iris()

        # load features, labels, and class names
        X, y, feature_names, class_names = data.data, data.target, data.feature_names, data.target_names

        # return
        return X, y, feature_names, class_names