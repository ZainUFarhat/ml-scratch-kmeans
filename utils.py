# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt

# euclidean distance
def euclidean_distance(x1, x2):

    """
    Description:
        Computes the euclidean distance between two feature vectors
    
    Parameters:
        x1: first feature vector
        x2: second feature vector
    
    Returns:
        euclidean_dist
    """

    # compute euclidean distance
    euclidean_dist = np.sqrt(np.sum((x1 - x2) ** 2))

    # return
    return euclidean_dist

# Calculate accuracy - out of 100 examples, what percentage does our model get right?
def accuracy_fn(y_true, y_pred):
  """
  calculates the accuracies of a given prediction

  Parameters:

    y_true: the true labels
    y_pred: our predicted labels
  
  Returns:

    accuracy
  """
  
  # find the number of correct predictions  
  correct = np.equal(y_true, y_pred).sum()
  # calculate the accuracy
  acc = (correct/len(y_pred))*100
  # return the accuracy
  return round(acc, 2) 

# scatter plot of given data
def scatter_plot(X, y, title, x_label, y_label, class_names, centroids, savepath):

    """
    Description:
        Plots a scatterplot based on X & y data provided

    Parameters:
        X: x-axis datapoints
        y: y-axis datapoints
        title: tite of plot
        x_label: label for x axis
        y_label: label for y axis
        class_names: names of our target classes
        centroids: centroids list
        savepath: path to save our scatterplot to

    Returns:
        None
    """

    # intialize figure
    plt.figure(figsize = (7, 7))

    # set background color to lavender
    ax = plt.axes()
    ax.set_facecolor("lavender")

    # find features corresponding to class labels
    class_0, class_1, class_2 = X[y == 0], X[y == 1], X[y == 2]

    # scatter plots of class features against themselves
    plt.scatter(class_0[:, 0], class_0[:, 1], label = class_names[0], c = 'b')
    plt.scatter(class_1[:, 0], class_1[:, 1], label = class_names[1], c = 'g')
    plt.scatter(class_2[:, 0], class_2[:, 1], label = class_names[2], c = 'darkorange')

    for pt in centroids:
        plt.scatter(*pt, marker = 'x', color = 'black', linewidth = 2)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend()
    plt.savefig(savepath)

    # return
    return None

def iris_visualize(X, y, feature_names, class_names, sepal_title, petal_title, centroids_sepal, centroids_petal, 
                                                                                    savepath_sepal, savepath_petal):
    """
    Decsription:
        Visualize Iris datset from its metadata
    
    Parameters:
        X: features
        y: labels
        feature_names: name of our feaures
        class_names: labels
        centroids_sepal: sepal centroids
        centroids_petal: petal centroids
        savepath_sepal: path to save our sepal scatterplot to
        savepath_petal: path to save our petal scatterplot to
        sepal_title: title of sepal scatter plot
        petal_title: title of petal scatterplot
    
    Returns:
        None
    """

    # feature data
    sepal_lengths, sepal_widths = X[:, 0], X[:, 1]
    petal_lengths, petal_widths = X[:, 2], X[:, 3]
    # targets
    targets = class_names
    # corresponding color (which is also their name)
    colors = y
    # the string title of sepal length and width features
    sepal_length_name, sepal_width_name = feature_names[0], feature_names[1]
    petal_length_name, petal_width_name = feature_names[2], feature_names[3]

    # set figure size
    plt.figure(figsize=(7, 7))

    # set background color to lavender
    ax = plt.axes()
    ax.set_facecolor("lavender")

    # scatterplot
    sc = plt.scatter(sepal_lengths, sepal_widths, c = colors)


    for pt in centroids_sepal:
        plt.scatter(*pt, marker = 'x', color = 'black', linewidth = 2)

    plt.title(sepal_title)
    plt.xlabel(sepal_length_name)
    plt.ylabel(sepal_width_name)
    plt.legend(sc.legend_elements()[0], targets, loc = 'lower right', title = 'Classes')
    plt.grid()
    plt.savefig(savepath_sepal)

    # set figure size
    plt.figure(figsize=(7, 7))

    # set background color to lavender
    ax = plt.axes()
    ax.set_facecolor("lavender")

    # scatterplot
    sc = plt.scatter(petal_lengths, petal_widths, c = colors)

    for pt in centroids_petal:
        plt.scatter(*pt, marker = 'x', color = 'black', linewidth = 2)

    plt.title(petal_title)
    plt.xlabel(petal_length_name)
    plt.ylabel(petal_width_name)
    plt.legend(sc.legend_elements()[0], targets, loc = 'lower right', title = 'Classes')
    plt.grid()
    plt.savefig(savepath_petal)

    # nothing to return, we just want to save plots
    return None