import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator

MAX = 5

def get_kmeans(points, kmax=None, nclusters=None):
    if (nclusters is not None):
        result = KMeans(n_clusters=nclusters).fit(points)
    else:
        sse = []
        if kmax > MAX:
            kmax = MAX
        for k in range(1, kmax+1):
            kmeans = KMeans(n_clusters = k).fit(points)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0
        
            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
            
            sse.append(curr_sse)
        sseRange = range(1,len(sse)+1)
        kn = KneeLocator(sseRange, sse, curve='convex', direction='decreasing')
        result = KMeans(n_clusters=kn.knee).fit(points)
    result.cluster_centers_ = [c[0] for c in result.cluster_centers_]
    return result

def prepare_data(array):
    return np.array([[a,0] for a in array])
