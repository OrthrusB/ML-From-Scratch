from __future__ import division, print_function
from sklearn import datasets
from sklearn.cluster import KMeans  # Import scikit-learn's KMeans
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    # Load the dataset
    X, y = datasets.make_blobs(n_samples=300, centers=3, random_state=42)
    
    # Cluster the data using scikit-learn's KMeans
    clf = KMeans(n_clusters=3, max_iter=500, random_state=42)
    y_pred = clf.fit_predict(X)  # fit_predict combines fitting and predicting
    
    # Print some useful information about the clustering
    print("Cluster centers:\n", clf.cluster_centers_)
    print("Number of iterations run:", clf.n_iter_)
    print("Inertia (sum of squared distances to closest centroid):", clf.inertia_)
    
    # Visualize the results
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
    # Add cluster centers to the plot
    plt.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title("K-Means Clustering Results (scikit-learn)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig('kmeans_plot.png')
    print("Plot has been saved as 'kmeans_plot.png'")
    
    return y_pred, clf

if __name__ == "__main__":
    y_pred, clf = main()