import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Create a directory to save plots, embeddings, and related files
project_dir = '/home/vivora/vanity_plates_final_clustering_v2'
if not os.path.exists(project_dir):
    os.makedirs(project_dir)

# Load and merge the datasets
print("Loading datasets...")
cali_df = pd.read_csv('/home/vivora/data/cali.csv')
t5_df = pd.read_csv('/home/vivora/data/cali_vanity_plates_meanings_T5_llama_rc.csv')
merged_df = pd.merge(cali_df, t5_df, on='plate')

# Combine textual features
print("Preprocessing data...")
merged_df['text_data'] = merged_df['customer_meaning'].fillna('') + ' ' + \
                        merged_df['reviewer_comments'].fillna('') + ' ' + \
                        merged_df['predicted_meaning'].fillna('')

# Vectorize the text data
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=100)
X_text = vectorizer.fit_transform(merged_df['text_data'])

# Convert review_reason_code to numeric and handle NaN values
merged_df['review_reason_code'] = pd.to_numeric(merged_df['review_reason_code'], errors='coerce')

# One-hot encode the reason codes
reason_encoded = pd.get_dummies(merged_df['review_reason_code'], prefix='reason')

# Combine text features and encoded reason categories
X_combined = np.hstack((X_text.toarray(), reason_encoded.values))

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Perform K-means clustering
print("Performing K-means clustering...")
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
merged_df['cluster'] = clusters

# Predefined cluster topics
cluster_topics = [
    "Phonetic Connotations",
    "Number and Letter Modifications",
    "Derogatory References",
    "Breaking DMV Rules",
    "Location References",
    "Playful References",
    "Slang Words"
]

# Visualize clusters using t-SNE
print("Applying t-SNE for visualization...")
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_scaled)

# Save the t-SNE embeddings
np.save(os.path.join(project_dir, 'vanity_plate_tsne_embeddings.npy'), X_tsne)
# X_tsne = np.load(os.path.join(project_dir, 'vanity_plate_tsne_embeddings.npy'))

# Create and save the t-SNE scatter plot with an adjusted legend position
print("Creating visualization with adjusted legend position...")
plt.figure(figsize=(16, 10))  # Increase the plot size
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.7)

# Add legend with predefined topics, positioned outside the plot
for i, topic in enumerate(cluster_topics):
    plt.scatter([], [], label=f"Cluster {i}: {topic}", c=[plt.cm.viridis(i / n_clusters)])

plt.legend(loc='best', fontsize=12)  # Increase font size of legend text
plt.colorbar(scatter, label='Cluster')
plt.title('t-SNE Visualization of Vanity License Plate Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Adjust x and y axis limits if needed to focus on data and avoid overcrowding
plt.xlim(-150, 250)
plt.ylim(-120, 120)

# Save the updated plot
plt.savefig(os.path.join(project_dir, 'final_clusters.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save the cluster assignments
merged_df[['plate', 'cluster']].to_csv(os.path.join(project_dir, 'vanity_plate_clusters.csv'), index=False)

# Save cluster analysis
cluster_analysis = merged_df.groupby('cluster').agg({
    'review_reason_code': lambda x: x.value_counts().index[0],
    'plate': 'count'
}).reset_index()
cluster_analysis.columns = ['Cluster', 'Most Common Reason Code', 'Number of Plates']
cluster_analysis['Topic'] = cluster_topics
cluster_analysis.to_csv(os.path.join(project_dir, 'cluster_analysis.csv'), index=False)

print("\nCluster Analysis:")
print(cluster_analysis.to_string(index=False))

print("\nAnalysis complete. Results saved in the updated directory.")
