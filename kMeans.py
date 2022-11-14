import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

data = np.loadtxt('dim032.txt')

def tuning_k(lowBound, upBound):
    nClusters = list(range(lowBound, upBound))
    sse = list()
    for k in nClusters:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        sse.append(int(kmeans.inertia_))
    # Plot SSE changes with respect to k to obtain an optimal k using this plot
    plt.plot(nClusters, sse)
    plt.xlabel('K Clusters')
    plt.ylabel('SSE')
    plt.show()
tuning_k(10, 40)
# K choose to be 16

# merge the clusters based on their MIN/MAX/average distances.
min = AgglomerativeClustering(linkage='single', n_clusters=16, compute_distances=True).fit(data)
max = AgglomerativeClustering(linkage='complete', n_clusters=16, compute_distances=True).fit(data)
avg = AgglomerativeClustering(linkage='average', n_clusters=16, compute_distances=True).fit(data)

# hierarchical clustering dendrograms
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot(input, level, title):
    plt.title(title)
    plot_dendrogram(input, truncate_mode="level", p=level)
    plt.xlabel("Cluster")
    plt.show()

plot(min, level=4, title="Min Dendrogram")
plot(max, level=3, title="Max Dendrogram")
plot(avg, level=3, title="Avg Dendrogram")
