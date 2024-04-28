from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import *

X, y, df = import_data_normalized_split('dataset/cleaned_dataset.csv')

# Perform DBScan tests
dbscan_1 = DBSCAN(eps=0.6, min_samples=10).fit(X)
dbscan_2 = DBSCAN(eps=1, min_samples=5, algorithm='brute').fit(X)
dbscan_3 = DBSCAN(eps=0.1, min_samples=3, algorithm='kd_tree').fit(X)

silhouette_score_1 = silhouette_score(X, dbscan_1.labels_)
silhouette_score_2 = silhouette_score(X, dbscan_2.labels_)
silhouette_score_3 = silhouette_score(X, dbscan_3.labels_)

# Create subplots for DBSCAN
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot clustering results for dbscan_1
sns.scatterplot(x=X[:, 0], y=X[:, 2], hue=dbscan_1.labels_, palette='viridis', ax=axes[0], legend='full')
axes[0].set_title('DBScan 1')
axes[0].set_xlabel('Area')
axes[0].set_ylabel('Major Axis Length')
axes[0].legend()

# Plot clustering results for dbscan_2
sns.scatterplot(x=X[:, 0], y=X[:, 2], hue=dbscan_2.labels_, palette='viridis', ax=axes[1], legend='full')
axes[1].set_title('DBScan 2')
axes[1].set_xlabel('Area')
axes[1].set_ylabel('Major Axis Length')
axes[1].legend()

# Plot clustering results for dbscan_3
sns.scatterplot(x=X[:, 0], y=X[:, 2], hue=dbscan_3.labels_, palette='viridis', ax=axes[2], legend='full')
axes[2].set_title('DBScan 3')
axes[2].set_xlabel('Area')
axes[2].set_ylabel('Major Axis Length')
axes[2].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Silhouette scores and corresponding labels
silhouette_scores = [silhouette_score_1, silhouette_score_2, silhouette_score_3]
labels = ['DBSCAN (eps=0.5, min_samples=20)',
          'DBSCAN (eps=1, min_samples=5, algorithm="brute")',
          'DBSCAN (eps=0.1, min_samples=3, algorithm="kd_tree")']

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(labels, silhouette_scores, color='skyblue')
plt.xlabel('')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different DBSCAN Configurations')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
