# kmeans with max cluster size
A simple k-means clustering algorithm implementation with a maximum cluster size constraint. Clusters are initialized via k-means++. The cluster assignment in each iteration is done greedily. That means that we are not trying to perform an assigment of samples to centers minimizing overall within cluster variance in each iteration. 
