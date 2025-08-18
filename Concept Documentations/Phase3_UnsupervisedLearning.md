# UnsupervisedLearning

## Documentation

```Dim reduction```

### t-SNE(t-distributed Stochastic Neighbor Embedding)

* Non-linear dimensionality reduction
* point as neighbour to another and preserves the pairwise similarities

### t-sne vs pca

* PCA (Principal Component Analysis) is a linear technique
* t-SNE is a nonlinear technique that preserves the pairwise similarities

### Kmeans clustering

* aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares

* The first step chooses the initial centroids, with the most basic method being to choose
samples from the dataset
. After initialization, K-means consists of looping between the two other steps. The first step assigns each sample to its nearest centroid. The second step creates new centroids by taking the mean value of all of the samples assigned to each previous centroid. The difference between the old and the new centroids are computed and the algorithm repeats these last two steps until this value is less than a threshold.
(ref:[https://scikit-learn.org/stable/modules/clustering.html#k-means])

### Hierarchical clustering

* build nested clusters by merging or splitting them successively
* represented as a tree (or dendrogram)
* merge strategy:
'Ward' minimizes the sum of squared differences within all clusters.

### Real-world applications

* Document clustering
similar documents can be grouped and seperated
helpful in news agencies, companies, for researches,etc

* Medical field
to group similar patient symptoms for diagnosis

* Customer analytics
Merchants can seperate buyers based on latest trends and timeless fashions
helps increase production

* Pixel reduction
used to cluster pixels in images

* Traffic in websites
clustering helps identify high-traffic timings in websites
