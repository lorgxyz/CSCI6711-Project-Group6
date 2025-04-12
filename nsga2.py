import numpy as np
import pickle
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.stats import mode

from collections import Counter

import random

import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from collections import Counter




def plot_2d_projection(features, labels, title="2D Projection"):
    """Plot the 2D projection of the features using PCA or ICA"""
    # Use LabelEncoder to convert labels into numeric values
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # PCA for 2D projection
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Plotting the 2D scatter plot with numeric labels
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels_encoded, cmap='viridis', s=20)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("ICA.png")

def plot_cluster_distribution(cluster_labels):
    """Plot the distribution of data points in each cluster."""
    cluster_counts = Counter(cluster_labels)
    cluster_data = pd.DataFrame(cluster_counts.items(), columns=['Cluster', 'Count'])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y='Count', data=cluster_data, palette='Set2')
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Data Points")
    plt.savefig("clusterdist.png")

def plot_cluster_purity(assigned_clusters):
    """Plot the purity of each cluster."""
    cluster_purities = [p[2] for p in assigned_clusters]
    cluster_ids = [p[0] for p in assigned_clusters]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_ids, y=cluster_purities, palette='coolwarm')
    plt.title("Cluster Purity")
    plt.xlabel("Cluster")
    plt.ylabel("Purity")
    plt.savefig("clusterpurity.png")




class ClusteringProblem(ElementwiseProblem):
    def __init__(self, data, n_clusters, n_dim, n_obj, w1=0.1, w2=0.9):
        self.data = np.array(data)
        self.n_clusters = n_clusters
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.w1 = w1
        self. w2 = w2
        xl=np.min(data, axis=0).repeat(n_clusters)
        xu=np.max(data, axis=0).repeat(n_clusters)
        buffer = 1e-6
        xl -= buffer
        xu += buffer
       
        super().__init__(n_var=n_clusters * n_dim,
                         n_obj=n_obj,  
                         data=data,  
                         xl=xl,
                         xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # Reshape x to interpret it as cluster centroids
        centroids = x.reshape(self.n_clusters, self.n_dim)
        
        # Compute intra-cluster variance
        distances = np.linalg.norm(self.data['data'][:, None, :] - centroids[None, :, :], axis=2)
        closest_clusters = np.argmin(distances, axis=1)
        intra_cluster_var = np.mean([np.mean(distances[closest_clusters == k, k]) 
                                     for k in range(self.n_clusters) if np.any(closest_clusters == k)])

        # Compute inter-cluster separation
        inter_cluster_sep = np.min([np.linalg.norm(c1 - c2) 
                                    for i, c1 in enumerate(centroids) 
                                    for j, c2 in enumerate(centroids) if i != j])
        
        # Penalize empty clusters
        empty_clusters_penalty = sum([1 for k in range(self.n_clusters) if not np.any(closest_clusters == k)])


        # weights
        penalty_weight = 1000
        # Minimize intra-cluster variance and maximize inter-cluster separation
        if self.n_obj==2:
            out["F"] = [self.w1 * intra_cluster_var + (penalty_weight * empty_clusters_penalty), -self.w2 * inter_cluster_sep]
        else:
            out["F"] = [self.w1 * intra_cluster_var + (penalty_weight * empty_clusters_penalty) - (self.w2 * inter_cluster_sep)]



def get_algorithm(algo):
    if algo == 'nsga2':
        return NSGA2(
            pop_size=100,
            sampling=FloatRandomSampling(),
            #crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=PolynomialMutation(),
            eliminate_duplicates=True
        )
    elif algo == 'pso':
        return PSO(
            pop_size=100,
            w=0.72,
            c1=1.49,
            c2=1.49
        )

def main():

    n_clusters = 70
    n_obj = 2
    algo_name = 'nsga2'
    n_gen = 10

    #alter to run on different files as needed
    df = pd.read_csv("dataset/Merged01.csv")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    labels = df['Label'].tolist()


    label_counts = Counter(labels)

    features = df.drop(columns=['Label'])
    scaler = StandardScaler()
    data = scaler.fit_transform(features)
    print("Standardized")

    n_components = 10
    ica = FastICA(n_components=n_components, random_state=42)
    data = ica.fit_transform(features)
    print("ICA done")

    problem = ClusteringProblem(data=data, n_clusters=n_clusters, n_dim=data.shape[1], n_obj=n_obj, w1=0.5, w2=0.5)
    algorithm = get_algorithm(algo_name)

    print("Running NSGA2")
    # Solve the problem
    res = minimize(problem,
                algorithm,
                ('n_gen', n_gen),
                verbose=False)
    
    objective_values = res.F
    if n_obj > 1:
        best_index = np.argmin(objective_values[:, 0])  # Index of the solution with the smallest variance
        optimal_solution = res.X[best_index]
    else:
        optimal_solution = res.X
    
    optimal_centroids = optimal_solution.reshape(n_clusters, data.shape[1])
    # Assign each point to the closest centroid
    distances = np.linalg.norm(data[:, None, :] - optimal_centroids[None, :, :], axis=2)
    cluster_labels = np.argmin(distances, axis=1)

    # Count the number of points in each cluster
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    print("Cluster labels:", unique_labels)
    print("Counts per cluster:", counts)

    df_results = pd.DataFrame({'cluster': cluster_labels, 'true_label': labels})

    cluster_composition = df_results.groupby('cluster')['true_label'].value_counts().unstack(fill_value=0)

    assigned_clusters = []

    # Iterate over each cluster to compute its purity and assign a label if it meets the threshold
    for cluster, counts in cluster_composition.iterrows():
        total = counts.sum()
        majority_label = counts.idxmax()
        majority_count = counts.max()
        purity = majority_count / total
        assigned_clusters.append((cluster, majority_label, purity))
        print(f"Cluster {cluster} assigned to class {majority_label}: {purity*100:.1f}% pure")
        unique_assigned_labels = set([p[1] for p in assigned_clusters])
        diversity = len(unique_assigned_labels)


    # Calculate the average purity of the clusters that were assigned a class label
    if assigned_clusters:
        avg_purity = sum(p[2] for p in assigned_clusters) / len(assigned_clusters)
        print(f"\nAverage purity for assigned clusters: {avg_purity*100:.1f}%")
        print(f"Cluster diversity (number of distinct classes with a pure cluster): {diversity}")

    else:
        print("No clusters meet the purity threshold of 80%.")

        # Plot the 2D projection of the features
    plot_2d_projection(features, labels, title="2D ICA Projection")

    # Plot the cluster distribution
    plot_cluster_distribution(cluster_labels)

    # Plot the cluster purity
    plot_cluster_purity(assigned_clusters)


if __name__ == '__main__':
    main()
   

