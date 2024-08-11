import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load data
data = pd.read_csv("data/data.csv", delimiter=',', encoding='unicode_escape')

# Filter out cancelled transactions and those starting with 'A'
data = data[~data['InvoiceNo'].str.startswith(('C', 'A'))]

# Create a flag for returns
data['IsReturn'] = data['StockCode'].str.startswith('C')


# Calculate the adjusted Quantity
data['AdjustedQuantity'] = data.apply(
    lambda row: -row['Quantity'] if row['IsReturn'] else row['Quantity'], axis=1
)

data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
# Feature Engineering
# Aggregating data on a customer level (assuming CustomerID is available)
customer_data = data.groupby('CustomerID').agg({
    'TotalPrice': 'sum',
    'InvoiceNo': 'nunique',  # Number of distinct invoices
    'AdjustedQuantity': 'sum',
    'StockCode': 'nunique'  # Number of distinct products bought
}).reset_index()

# Rename columns for clarity
customer_data.rename(columns={
    'AdjustedQuantity': 'Quantity'
}, inplace=True)

# Drop rows with missing CustomerID (if any)
customer_data.dropna(subset=['CustomerID'], inplace=True)

# Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['TotalPrice', 'InvoiceNo', 'Quantity', 'StockCode']])

# Dimensionality Reduction (optional)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pca_data)
customer_data['Cluster'] = kmeans.labels_

# Evaluate Clustering
sil_score = silhouette_score(pca_data, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

# Visualization
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=customer_data['Cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segments')
plt.colorbar(label='Cluster')
plt.show()

# Elbow Method to Determine Optimal Number of Clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()