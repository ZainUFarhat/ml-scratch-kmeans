# datasets
from datasets import *

# Gaussian Naive Bayes
from KMeans import *

# set numpy random seed
np.random.seed(42)

def main():

    """
    Description:
        Predicts based on our K Means Clustering Algorithm
    
    Parameters:
        None
    
    Returns:
        None
    """

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    random_state = 40
    dataset_name = 'Sklearn Blobs'
    
    # create an instance of Datasets class
    datasets = Datasets(random_state = random_state)

    # K
    K = 3

    # load the Sklearn blobs dataset
    X, y, class_names = datasets.make_blobs(K = K, num_samples = 500, num_features = 2)

    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nK Means Clustering\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # training hyperparameters
    epochs = 1000

    km = KMeans(K = K, epochs = epochs)
    centroids, labels = km.predict(X)

    print('Converged!') 
    print('---------------------------------------------------Testing---------------------------------------------------')
    print('Testing...\n')
    acc = accuracy_fn(y_true = y, y_pred = labels)
    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    print('Plotting...')

    # scatter plot of original data
    title_scatter = f'{dataset_name} - Feature 1 vs. Feature 2'
    save_path_scatter = 'plots/blobs/blobs_scatter.png'
    scatter_plot(X = X, y = y, title = title_scatter, x_label = 'Feature 1', y_label = 'Feature 2', 
                                class_names = class_names, centroids = centroids, savepath = save_path_scatter)


    print('Please refer to plots/blobs directory to view clusters.')
    print('--------------------------------------------------------------------------------------------------------------\n')
    ######################################################################################################################################

    # load the iris dataset
    dataset_name = 'Iris'
    X, y, feature_names, class_names = datasets.load_iris()

    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nK Means Clustering\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # training hyperparameters
    epochs = 150

    km = KMeans(K = K, epochs = epochs)
    centroids, labels = km.predict(X)

    print('Converged!')
    print('---------------------------------------------------Testing---------------------------------------------------')
    print('Testing...\n')
    acc = accuracy_fn(y_true = y, y_pred = labels)
    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    print('Plotting...')

    # scatter plot of original data
    savepath_sepal  = f'plots/iris/iris_sepal.png'
    savepath_petal  = f'plots/iris/iris_petal.png'
    
    # sepal and petal scatterplots
    petal_title = f'{dataset_name} - Petal Length vs. Petal Width'
    sepal_title = f'{dataset_name} - Sepal Length vs. Sepal Width'
    iris_visualize(X, y, feature_names, class_names, centroids_sepal = centroids[:, [0,1]], centroids_petal = centroids[:, [2,3]], 
                   sepal_title = sepal_title, petal_title = petal_title, savepath_sepal = savepath_sepal, savepath_petal = savepath_petal)


    print('Please refer to plots/iris directory to view clusters.')
    print('--------------------------------------------------------------------------------------------------------------\n')

if __name__ == '__main__':

    # run everything
    main()