from random import shuffle
from sklearn.metrics import pairwise_distances
import numpy as np

def eucl_dist(X,v):
    return np.linalg.norm(X-v,axis=1)


def kmeans_max_cluster(X,k,max_size,maxiter = 100,n_jobs=None,verbose=0):
    '''
    kmeans with kmeans++ initialization and a max cluster constraint.
    This implementation is simple and cluster assignment is greedy, meaning
    we do not attempt to find the assignments respecting constraints that have minimal within
    cluster variance in each iteration.  
    
    Parameters
    ----------
    X : array [n_samples_a, n_features]
    k : number of clusters
    max_size : maximum cluster size
    n_jobs : njobs used to compute euclidean distances
    '''
    
    dist= eucl_dist
    n,m = X.shape
    if k*max_size<n:
        print("max_size too small")
        return None
    idxs = list(range(n))

    # kmeans++ initialization
    idx = np.random.choice(len(idxs))
    center = X[idx]
    center_idxs = [idx]
    centers = [center]

    distsq = dist(X,center)**2
    distances_sq = distsq
    weight = distsq
    weight/=weight.sum()
    weight[center_idxs]=0

    for i in range(k-1):
        idx = np.random.choice(len(idxs),p=weight)
        center_idxs.append(idx)
        center = X[idx]
        centers.append(center)
        distsq = dist(X,center)**2
        distances_sq = np.vstack((distances_sq,distsq))
        # weight by distance to nearest center
        weight = distances_sq.min(0)
        weight/=weight.sum()
        weight[center_idxs]=0

    centers = np.vstack(centers)    
    
    
    # kmeans with max cluster size
    # assign each point to nearest center, conditioned on cluster not being too large yet
    change= True
    iiter = 0
    prev_assignment = np.zeros(n)
    while iiter<maxiter and change:
        if iiter%10==0 and verbose>0:
            print('Iteration: %d'%iiter)
        iiter+=1
        # shuffle points. Assignment is greedy.
        shuffle(idxs)
        distances = pairwise_distances(X,centers,n_jobs=n_jobs)
        assignment = -np.ones(n)
        cluster_sizes = np.zeros(k)
        for i in idxs: 
            possible_assignment = np.argsort(distances[i])
            j = 0           
            while cluster_sizes[possible_assignment[j]]==max_size:
                # if this breaks all clusters are at maximum size. 
                j+=1
            # cluster assignment found
            assignment[i]=possible_assignment[j]
            cluster_sizes[possible_assignment[j]]+=1
        
        # check if cluster assignment changed from previous iteration
        if np.allclose(prev_assignment,assignment):
            change = False
            if verbose>0:
                print('No change')
        else:
            prev_assignment = np.copy(assignment)
            
        # update centers
        toremove = []
        for i in range(k):
            boolidxs = assignment==i
            if boolidxs.sum()==0:
                #remove empty centers
                toremove.append(i)
            else:
                centers[i] = X[assignment==i].mean(0)
        if toremove:
            centers = np.delete(centers, (toremove), axis=0)
            k-=len(toremove)
            if verbose>0:
                print('removed centers: ',toremove)
    if verbose>0 and iiter==maxiter :
        print('maximum number of iterations reached')
    return assignment,centers


