# Customers-Segmentation
STP(Abbreviation for Segmentation,Targeting, Positioning) is a fundamental marketing framework. It can be applied to all areas of business and marketing activities. The process of dividing a population of customers into groups that share similar characteristics. Observations within the same group would have comparable purchasing behavior. Observations within the same group would respond similarly to different marketing activities.

The notebook focuses on how to perform customer segmentation, using a hands-on approach. It involves the application of hierarchical and flat clustering techniques for dividing customers into groups. It also features applying the Principal Components Analysis (PCA) to reduce the dimensionality of the problem, as well as combining PCA and K-means for an even more professional customer segmentation.

The dataset has 2000 obervations and 7 features. We applied some techniques to understand data and checked missing values and data types. The final pre-processing technique we applied to the dataset is standardization. Standardization scales, or shifts, the values for each numerical feature in your dataset so that the features have a mean of 0 and standard deviation of 1

## K-means: Expectation–Maximization
Conventional k-means requires only a few steps. The first step is to randomly select k centroids, where k is equal to the number of clusters you choose. Centroids are data points representing the center of a cluster.
Expectation–maximization (E–M) is a powerful algorithm that comes up in a variety of contexts within data science. k-means is a particularly simple and easy-to-understand application of the algorithm, and we will walk through it briefly here. In short, the expectation–maximization approach here consists of the following procedure:

1. Guess some cluster centers
2. Repeat until converged
   - E-Step: assign points to the nearest cluster center
   - M-Step: set the cluster centers to the mean
Here the "E-step" or "Expectation step" is so-named because it involves updating our expectation of which cluster each point belongs to. The "M-step" or "Maximization step" is so-named because it involves maximizing some fitness function that defines the location of the cluster centers—in this case, that maximization is accomplished by taking a simple mean of the data in each cluster.

The quality of the cluster assignments is determined by computing the sum of the squared error (SSE) after the centroids converge, or match the previous iteration’s assignment. The SSE is defined as the sum of the squared Euclidean distances of each point to its closest centroid. Since this is a measure of error, the objective of k-means is to try to minimize this value.

Here are the parameters used in this code:

- **init:** controls the initialization technique. The standard version of the k-means algorithm is implemented by setting init to "random". Setting this to "k-means++" employs an advanced trick to speed up convergence.

- **n_clusters:** sets k for the clustering step. This is the most important parameter for k-means.

- **n_init:** sets the number of initializations to perform. This is important because two runs can converge on different cluster assignments. The default behavior for the scikit-learn algorithm is to perform ten k-means runs and return the results of the one with the lowest SSE.

- **max_iter:** sets the number of maximum iterations for each initialization of the k-means algorithm.
## Choosing the Appropriate Number of Clusters
Ttwo methods that are commonly used to evaluate the appropriate number of clusters:
- The elbow method
- The silhouette coefficient

The number of segments is very challenging in K-means algorithm. The easiest way is the **elbow method**. When you plot SSE as a function of the number of clusters, notice that SSE continues to decrease as you increase k. As more centroids are added, the distance from each point to its closest centroid will decrease. There’s a sweet spot where the SSE curve starts to bend known as the elbow point. The x-value of this point is thought to be a reasonable trade-off between error and number of clusters.  In this code, the elbow might be located at 4: 
<p align="center">
<img src="https://github.com/Hussein-Mazaar/Customers-Segmentation/blob/main/Figures/SSE.jpg"  width="400" height="300">
</p>

Determining the elbow point in the SSE curve isn’t always straightforward. If you’re having trouble choosing the elbow point of the curve, then you could use a Python package, kneed, to identify the elbow point programmatically:

```python
from kneed import KneeLocator
kl = KneeLocator( range(1, 11), wcss, curve="convex", direction="decreasing")
kl.elbow
```
The **silhouette coefficient** is a measure of cluster cohesion and separation. It quantifies how well a data point fits into its assigned cluster based on two factors:
- How close the data point is to other points in the cluster
- How far away the data point is from points in other clusters
Silhouette coefficient values range between -1 and 1. Larger numbers indicate that samples are closer to their clusters than they are to other clusters.

The silhouette coefficient of this model is 0.267, indicating reasonable cluster separation.
<p align="center">
<img src="https://github.com/Hussein-Mazaar/Customers-Segmentation/blob/main/Figures/silhouette.jpg"  width="400" height="300">
</p>
Ultimately, your decision on the number of clusters to use should be guided by a combination of domain knowledge and clustering evaluation metrics.

The elbow method and silhouette coefficient evaluate clustering performance without the use of **ground truth labels**. Ground truth labels categorize data points into groups based on assignment by a human or an existing algorithm. These types of metrics do their best to suggest the correct number of clusters but can be deceiving when used without context.

**Note: In practice, it’s rare to encounter datasets that have ground truth labels.**

Since the ground truth labels are known, it’s possible to use a clustering metric that considers labels in its evaluation. You can use the scikit-learn implementation of a common metric called the **adjusted rand index (ARI)**. Unlike the silhouette coefficient, the ARI uses true cluster assignments to measure the similarity between true and predicted labels. The ARI output values range between -1 and 1. A score close to 0.0 indicates random assignments, and a score close to 1 indicates perfectly labeled clusters.

## Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is one of many dimensionality reduction techniques. PCA transforms the input data by projecting it into a lower number of dimensions called components. The components capture the variability of the input data through a linear combination of the input data’s features.
This below figure shows the segments after running K-means algorithm for age and income which is not clear and well separated. 

<p align="center">
<img src="https://github.com/Hussein-Mazaar/Customers-Segmentation/blob/main/Figures/kmeans_f.jpg"  width="400" height="300">
</p>
Dimensionality reduction or PCA reduces the number of features in the dataset by either removing or combining them. Dimensionality reduction techniques help to address a problem with machine learning algorithms known as the curse of dimensionality. The below figure shows the segments after running PCA and choosing two components only.
<p align="center">
<img src="https://github.com/Hussein-Mazaar/Customers-Segmentation/blob/main/Figures/kmeans_pca.jpg"  width="400" height="300">
</p>


## Setup
1. install Python IDE on your device like Anaconda which includes Jupyter Notebook to easily run the code provided and display visualizations at each step. 
2. Also, make sure to have the following libraries installed — Numpy, Pandas, Matplotlib, Seaborn, Scikit-Learn, Kneed, and Scipy.


