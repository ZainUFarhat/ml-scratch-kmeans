# ml-scratch-kmeans
K Means Clustering Algorithm

## **Description**
The following is my from scratch implementation of the K Means Clustering algorithm.

### **Dataset**

I tested the performance of my model on two datasets: \
\
    &emsp;1. Sklearn Blobs Dataset \
    &emsp;2. Iris Dataset \

### **Walkthrough**

**1.** Need the following packages installed: sklearn, numpy, and matplotlib.

**2.** Once you made sure all these libraries are installed, evrything is simple, just head to main.py and execute it.

**3.** Since code is modular, main.py can easily: \
\
    &emsp;**i.** Load the two datasets \
    &emsp;**ii.** Build a kmeans classifier \
    &emsp;**iii.** Fit the kmeans classifier \
    &emsp;**iv.** Evaluate the classifier \
    &emsp;**v.** Plot scatter plots with centroids for each dataset.

### **Results**

For each dataset I will share the test accuracy and show the scatterplot with centroid predictions.

**Note**: Since this is a naive implementation of an unsupervised algorithms, the test accuracy is not ideal.

**1.** Sklearn Blobs Dataset:

- Numerical Result:
     - Accuracy = 33.4%

- See visualization below:

 ![alt text](https://github.com/ZainUFarhat/ml-scratch-kmeans/blob/main/plots/blobs/blobs_scatter.png?raw=true) 

**2.** Iris Dataset:

- Numerical Result:
     - Accuracy = 9.33%

- See visualizations below:

    - Sepal Visualization:
        ![alt text](https://github.com/ZainUFarhat/ml-scratch-kmeans/blob/main/plots/iris/iris_sepal.png?raw=true)

    - Petal Visualization:
        ![alt text](https://github.com/ZainUFarhat/ml-scratch-kmeans/blob/main/plots/iris/iris_petal.png?raw=true)