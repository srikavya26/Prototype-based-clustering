#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 3 08:56:19 2025

@author: vasalasrikavya
"""

#1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import skfuzzy as fuzz
from prettytable import PrettyTable
from scipy.stats import zscore

# Set Seaborn style
sns.set(style="whitegrid", palette="pastel")

#2. Load and Read Dataset
df = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')
print("Shape:", df.shape)
df.head()

#3. Preprocessing
sales_data = df.iloc[:, 1:53].copy()
sales_data.index = df['Product_Code']
sales_data = sales_data.replace('?', np.nan).astype(float)
sales_data.dropna(inplace=True)

# Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sales_data)

# Handling outliers using Z-score
z_scores = np.abs(zscore(scaled_data))
filtered_data = scaled_data[(z_scores < 3).all(axis=1)]

print(f"Filtered data shape after outlier removal: {filtered_data.shape}")

# Elbow Method to find best K 
# Range of k values to try
k_values = range(1, 11)

#Initialize sum of square error(SSE)
sse = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(filtered_data)  
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, sse, marker='o', color='b', linestyle='-', linewidth=2)
plt.title('Elbow Plot for K-Means Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.xticks(k_values)  
plt.grid(True)
plt.show()

#4. Apply K-Means Clustering
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(filtered_data)
kmeans_centroids = kmeans.cluster_centers_

#5. Apply Fuzzy C-Means Clustering
cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
    filtered_data.T, c=k, m=2, error=0.005, maxiter=1000, init=None
)
fcm_labels = np.argmax(u, axis=0)

#6. Evaluation Metrics
sil_kmeans = silhouette_score(filtered_data, kmeans_labels)
sil_fcm = silhouette_score(filtered_data, fcm_labels)
db_kmeans = davies_bouldin_score(filtered_data, kmeans_labels)
db_fcm = davies_bouldin_score(filtered_data, fcm_labels)

#7. Comparison Table
table = PrettyTable(["Metric", "K-Means", "Fuzzy C-Means"])
table.add_row(["Silhouette Score", f"{sil_kmeans:.3f}", f"{sil_fcm:.3f}"])
table.add_row(["Davies-Bouldin Index", f"{db_kmeans:.3f}", f"{db_fcm:.3f}"])
table.add_row(["Fuzzy Partition Coefficient", "-", f"{fpc:.3f}"])
print(table)

#8. Visualization: PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(filtered_data)
dark_palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)

num_colors = len(set(kmeans_labels))
colors = [dark_palette(i / num_colors) for i in range(num_colors)] 

# FCM with PCA
plt.subplot(1, 2, 2)
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=fcm_labels, s=60)
plt.title("FCM with PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

#9. Visualization: t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_data = tsne.fit_transform(filtered_data)

kmeans_palette = sns.color_palette("cool", n_colors=len(set(kmeans_labels)))  
fcm_palette = sns.color_palette("viridis", n_colors=len(set(fcm_labels)))

# Plot t-SNE with K-Means clustering
plt.figure(figsize=(10, 5))

# K-Means with t-SNE
plt.subplot(1, 2, 1)
sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=kmeans_labels, palette=kmeans_palette, s=60)
plt.title("K-Means with t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# FCM with t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=fcm_labels, palette=fcm_palette, s=60)
plt.title("FCM with t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# Adjust layout
plt.tight_layout()
plt.show()


# Group products by cluster, check total/avg sales, trends etc.
filtered_index = sales_data.index[(z_scores < 3).all(axis=1)]

# Assign cluster labels
df['Cluster_KMeans'] = pd.Series(kmeans_labels, index=filtered_index)
df['Cluster_FCM'] = pd.Series(fcm_labels, index=filtered_index)

# Group by cluster
mean_sales_by_kmeans = df.groupby('Cluster_KMeans').mean(numeric_only=True)
mean_sales_by_fcm = df.groupby('Cluster_FCM').mean(numeric_only=True)

print("Average Weekly Sales by KMeans Cluster:\n", mean_sales_by_kmeans.head())
print("Average Weekly Sales by FCM Cluster:\n", mean_sales_by_fcm.head())
