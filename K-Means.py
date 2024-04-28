from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from data_utils import *

X, y, df = import_data_normalized_split('dataset/cleaned_dataset.csv')

# What is for loops
kmeans_1 = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
kmeans_2 = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
kmeans_3 = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X)
kmeans_4 = KMeans(n_clusters=9, random_state=0, n_init="auto").fit(X)
kmeans_5 = KMeans(n_clusters=12, random_state=0, n_init="auto").fit(X)
kmeans_6 = KMeans(n_clusters=18, random_state=0, n_init="auto").fit(X)

silhouette_score_1 = silhouette_score(X, kmeans_1.labels_)
silhouette_score_2 = silhouette_score(X, kmeans_2.labels_)
silhouette_score_3 = silhouette_score(X, kmeans_3.labels_)
silhouette_score_4 = silhouette_score(X, kmeans_4.labels_)
silhouette_score_5 = silhouette_score(X, kmeans_5.labels_)
silhouette_score_6 = silhouette_score(X, kmeans_6.labels_)

davies_boulding_score_1 = davies_bouldin_score(X, kmeans_1.labels_)
davies_boulding_score_2 = davies_bouldin_score(X, kmeans_2.labels_)
davies_boulding_score_3 = davies_bouldin_score(X, kmeans_3.labels_)
davies_boulding_score_4 = davies_bouldin_score(X, kmeans_4.labels_)
davies_boulding_score_5 = davies_bouldin_score(X, kmeans_5.labels_)
davies_boulding_score_6 = davies_bouldin_score(X, kmeans_6.labels_)

# Plot Silhouette Scores
k_values = [2, 3, 5, 9, 12, 18]
silhouette_scores = [silhouette_score_1, silhouette_score_2, silhouette_score_3, silhouette_score_4, silhouette_score_5, silhouette_score_6]

max_silhouette_score = max(silhouette_scores)
min_silhouette_score = min(silhouette_scores)
y_axis_buffer = (max_silhouette_score - min_silhouette_score) * 0.05  # 5% buffer

plt.figure(figsize=(8, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.ylim(min_silhouette_score - y_axis_buffer, max_silhouette_score + y_axis_buffer)
plt.yticks(silhouette_scores)
plt.grid(True)
plt.show()

# Plot davies bouldin scores
davies_bouldin_scores = [davies_boulding_score_1, davies_boulding_score_2, davies_boulding_score_3, davies_boulding_score_4, davies_boulding_score_5, davies_boulding_score_6]

max_davies_score = max(davies_bouldin_scores)
min_davies_score = min(davies_bouldin_scores)
y_axis_buffer_davies = (max_davies_score - min_davies_score) * 0.05  # 5% buffer

plt.figure(figsize=(8, 6))
plt.plot(k_values, davies_bouldin_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin scores')
plt.title('Davies-Bouldin scores vs. Number of Clusters')
plt.ylim(min_davies_score-y_axis_buffer_davies, max_davies_score+y_axis_buffer_davies)
plt.yticks(davies_bouldin_scores)
plt.grid(True)
plt.show()

# Plot clustering plots
kmeans_models = [kmeans_1, kmeans_2, kmeans_3, kmeans_4, kmeans_5, kmeans_6]
# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Loop over each KMeans model and subplot (using for loops this time hehe)
for i, (k, kmeans) in enumerate(zip(k_values, kmeans_models)):
    # Calculate subplot indices
    row = i // 2
    col = i % 2

    # Scatter plot for the first two features
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans.labels_, palette='viridis', ax=axes[row, col], legend='full')
    axes[row, col].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='s', s=20, c='red',
                           label='Centroids')
    axes[row, col].set_xlabel('Area')
    axes[row, col].set_ylabel('Perimeter')
    axes[row, col].legend()

    # Title for the subplot
    axes[row, col].set_title(f'KMeans with {k} clusters')

# Adjust layout
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, (k, kmeans) in enumerate(zip(k_values, kmeans_models)):
    # Calculate subplot indices
    row = i // 2
    col = i % 2

    # Scatter plot for the first two features
    sns.scatterplot(x=X[:, 2], y=X[:, 4], hue=kmeans.labels_, palette='viridis', ax=axes[row, col], legend='full')
    axes[row, col].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='s', s=20, c='red',
                           label='Centroids')
    plt.xlabel('Major_Axis_length')
    plt.ylabel('Eccentricity')
    axes[row, col].legend()

    # Title for the subplot
    axes[row, col].set_title(f'KMeans with {k} clusters')

# Adjust layout
plt.tight_layout()
plt.show()
