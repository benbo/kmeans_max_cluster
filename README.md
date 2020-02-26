# kmeans with max cluster size
A simple k-means clustering algorithm implementation with a maximum cluster size constraint. Clusters are initialized via k-means++. The cluster assignment in each iteration is done greedily. That means that we are not trying to perform an assigment of samples to centers minimizing overall within cluster variance in each iteration. 

# usage
```python
from kmeans_max_cluster import kmeans_max_cluster
import numpy as np

X = np.random.rand(100,10)
k = 20
max_size = 10
assignment,centers = kmeans_max_cluster(X,k,max_size)
```
