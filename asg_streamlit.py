import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import boxcox
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import resample
import skfuzzy as fuzz
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
import os


#----------------------------------------------------- Function declaration --------------------------------------------------------------


# --------------------------------------------- GMM clustering
def perform_gmm_clustering(X, n_clusters):
    gmm_model = GaussianMixture(n_components=n_clusters,
                        covariance_type="full",
                        random_state=42,
                        n_init=1,
                        max_iter=100,
                        init_params="kmeans")

    gmm_model.fit(X)
    labels = gmm_model.predict(X)
    centroids = gmm_model.means_
    return labels, centroids


# --------------------------------------------- Spectral clustering
def perform_spectral_clustering(X, n_clusters):
    sample_size = 3316  # Adjust this number as needed
    sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
    X_sampled = X[sample_indices]
    
    spectral_model = SpectralClustering(n_clusters=n_clusters, random_state=42, n_neighbors=10)  # Adjust n_neighbors as needed
    labels = spectral_model.fit_predict(X_sampled)
    # Convert X_sampled to a DataFrame with original column names
    X_sampled_df = pd.DataFrame(X_sampled, columns=nutrition_table.columns)

    # Add cluster labels to the DataFrame
    X_sampled_df['Cluster'] = labels
    centroids = X_sampled_df.groupby('Cluster').mean()
    return labels, centroids


# --------------------------------------------- KMeans clustering
def perform_kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids


# --------------------------------------------- Hierarchical clustering
def perform_hierarchical_clustering(X, n_clusters):
    
    # Perform Agglomerative Clustering on the sampled data
    sampled_X = resample(X, n_samples=3316, random_state=42)  # Adjust the number of samples as needed
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    y_hc = hc.fit_predict(sampled_X)
    labels = y_hc
    
    # Calculate centroids
    centroids = []
    for i in range(n_clusters):
        cluster_points = sampled_X[y_hc == i]  # Get data points belonging to the cluster
        centroid = np.mean(cluster_points, axis=0)  # Calculate mean along each attribute
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    return labels, centroids


# --------------------------------------------- Fuzzy C Mean clustering
def perform_fuzzy_c_mean_clustering(X, n_clusters):
    sampled_X = resample(X, n_samples=3316, random_state=42)  # Adjust the number of samples as needed
    
    # Transpose the data for fuzzy clustering
    data = X.T.values
    X_array = X.to_numpy()

    # Define parameters for fuzzy c-means clustering
    m = 2.0  # Fuzziness parameter (usually between 1.1 and 2.0)
    error = 0.005  # Stopping criterion (sensitivity to changes in cluster centers)

    # Perform fuzzy c-means clustering
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data, n_clusters, m, error, maxiter=1000, init=None)

    # Get the cluster centers
    cluster_centers = cntr.T

    # Assign samples to clusters based on maximum membership
    cluster_membership = np.argmax(u, axis=0)

    # Convert X array to DataFrame with feature names as columns
    X_df = pd.DataFrame(X_array, columns=X.columns)

    # Add cluster membership information to the DataFrame
    X_df['Cluster'] = cluster_membership

    
    # Plot fuzzy c-means clustering graph
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    parallel_coordinates(X_df, 'Cluster', colormap='viridis',ax=ax1)
    ax1.set_xlabel('Attributes')
    ax1.set_ylabel('Feature Value')
    ax1.set_title('Parallel Coordinates Plot with Clusters')
    ax1.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    ax1.legend(loc='upper right')
    ax1.grid(True)



    # Plot cluster centers
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i in range(n_clusters):
        ax2.plot(X.columns, cluster_centers[:, i], label='Cluster {}'.format(i+1))
    ax2.set_xlabel('Attributes')
    ax2.set_ylabel('Membership Value')
    ax2.set_title('Fuzzy C-Means Cluster Centers')
    ax2.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    ax2.legend()
    ax2.grid(True)

    return fig1, fig2
    

# ---------------------------------------------------------------------- Parallel coordinate graph of Cluster
def plot_parallel_coordinates_labels(X, labels):
    # Convert data to DataFrame
    # Convert X to a DataFrame with original column names
    X_df = pd.DataFrame(X, columns=nutrition_table.columns)

    # Convert labels to a pandas Series
    labels_series = pd.Series(labels, name='cluster')

    # Add the cluster labels to your data
    X_labeled = pd.concat([X_df, labels_series], axis=1)

    # Drop rows with NaN values
    X_labeled.dropna(inplace=True)
    
    # Plot parallel coordinates plot
    fig, ax = plt.subplots(figsize=(10, 6))
    parallel_coordinates(X_labeled,'cluster', colormap='viridis', ax=ax)
    ax.set_title('Parallel Coordinates Plot with Clusters')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Value')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper right')

    st.pyplot(fig)

# ---------------------------------------------------------------------- Parallel coordinate graph of Centroids
def plot_parallel_coordinates_centroids(X, centroids):
    # Convert X to a DataFrame with original column names
    X_df = pd.DataFrame(X, columns=nutrition_table.columns)
    
    # Plot centroids separately
    # Convert centroids to a DataFrame
    centroids_df = pd.DataFrame(centroids, columns=nutrition_table.columns)

    # Add the cluster labels to your centroids (assuming each centroid corresponds to a cluster)
    centroids_df['Cluster'] = centroids_df.index

    # Concatenate centroids with the original data
    combined_data = pd.concat([X_df, centroids_df], axis=0)

    # Drop rows with missing values
    combined_data = combined_data.dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    parallel_coordinates(combined_data, 'Cluster', colormap='viridis', ax=ax)
    ax.set_title('Parallel Coordinates Plot of Centroids')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Value')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper right')

    st.pyplot(fig)



#---------------------------------------------------------- Main Code --------------------------------------------------------------------


st.title('Clustering Models for nutritions')

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the path to the CSV file relative to the script's directory
csv_file_path = os.path.join(script_dir, 'transformedData.csv')

# Read the CSV file
nutrition_table = pd.read_csv(csv_file_path)



# Generate unique keys for the slider and radio button widgets
slider_key = 'slider_unique_key'

# Create the slider widget with a unique key
n_clusters = st.slider('Select number of clusters', 2, 10, 2, step=1, key=slider_key)

# Define the options for clustering algorithms
clustering_options = ['Select clustering algorithm', 'Gaussian Mixture Model (GMM)', 'Spectral Clustering', 'K-Means Clustering', 'Hierarchical Clustering', 'Fuzzy C Mean']

# Let the user select the clustering algorithm using a select box
selected_algorithm = st.selectbox('Select clustering algorithm', clustering_options)

# Perform clustering based on the selected algorithm
if selected_algorithm == 'Gaussian Mixture Model (GMM)':
    # Convert data to numpy array
    X = nutrition_table.values
    
    # Perform GMM clustering
    labels, centroids = perform_gmm_clustering(X, n_clusters)
    
    # Plot parallel coordinates plot
    plot_parallel_coordinates_labels(X, labels)
    plot_parallel_coordinates_centroids(X, centroids)
    

elif selected_algorithm == 'Spectral Clustering':
    # Convert data to numpy array
    X = nutrition_table.values
    
    # Perform Spectral clustering
    labels, centroids = perform_spectral_clustering(X, n_clusters)

    # Plot parallel coordinates plot
    plot_parallel_coordinates_labels(X, labels)
    plot_parallel_coordinates_centroids(X, centroids)   
    

elif selected_algorithm == 'K-Means Clustering':
    # Convert data to numpy array
    X = nutrition_table.values
    
    # Perform K-Means clustering
    labels, centroids = perform_kmeans_clustering(X, n_clusters)
    
    # Plot parallel coordinates plot
    plot_parallel_coordinates_labels(X, labels)
    plot_parallel_coordinates_centroids(X, centroids)
    

elif selected_algorithm == 'Hierarchical Clustering':
    X = nutrition_table
    
    # Perform Hierarchical clustering
    labels, centroids = perform_hierarchical_clustering(X, n_clusters)

    # Plot parallel coordinates plot
    plot_parallel_coordinates_labels(X, labels)
    plot_parallel_coordinates_centroids(X, centroids)
    

elif selected_algorithm == 'Fuzzy C Mean':
    X = nutrition_table
    
    # Perform Fuzzy C Mean clustering
    fig1, fig2 = perform_fuzzy_c_mean_clustering(X, n_clusters)
    st.pyplot(fig1)
    st.pyplot(fig2)
