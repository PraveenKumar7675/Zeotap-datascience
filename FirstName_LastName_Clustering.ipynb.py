#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Data
customers = pd.read_csv('Customers.csv')  # Replace with actual path to Customers.csv
transactions = pd.read_csv('Transactions.csv')  # Replace with actual path to Transactions.csv

# Step 2: Data Preprocessing
# Merge datasets on CustomerID
data = transactions.merge(customers, on='CustomerID')

# Create features for clustering
customer_features = data.groupby('CustomerID').agg({
    'TotalValue': 'sum',  # Total spend per customer
    'TransactionID': 'count',  # Number of transactions per customer
    'ProductID': lambda x: x.nunique()  # Number of unique products purchased
}).reset_index()

customer_features.columns = ['CustomerID', 'TotalSpend', 'PurchaseFrequency', 'UniqueProducts']

# Check for NaN values and handle them if necessary
if customer_features.isnull().values.any():
    print("NaN values found in customer features. Filling NaN values with zeros.")
    customer_features.fillna(0, inplace=True)

# Step 3: Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features[['TotalSpend', 'PurchaseFrequency', 'UniqueProducts']])

# Step 4: Determine Optimal Number of Clusters (Elbow Method)
inertia = []
k_range = range(2, 11)  # Number of clusters from 2 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    try:
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    except Exception as e:
        print(f"Error during KMeans fitting for k={k}: {e}")

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid()
plt.show()

# Step 5: Apply KMeans Clustering (Choose optimal k based on Elbow Curve)
optimal_k = 4  # Replace with the optimal number of clusters from the elbow method based on your findings
kmeans = KMeans(n_clusters=optimal_k, random_state=42)

try:
    clusters = kmeans.fit_predict(scaled_features)  # Generate cluster labels
except Exception as e:
    print(f"Error during KMeans fitting: {e}")

# Add cluster labels to the customer features DataFrame
customer_features['Cluster'] = clusters

# Step 6: Calculate Davies-Bouldin Index (DB Index)
db_index = davies_bouldin_score(scaled_features, clusters)
print(f"Davies-Bouldin Index (DB Index): {db_index}")

# Step 7: Visualize Clusters (2D Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=customer_features['TotalSpend'],
    y=customer_features['PurchaseFrequency'],
    hue=customer_features['Cluster'],
    palette='viridis',
    s=100,
)
plt.title('Customer Segmentation Clusters')
plt.xlabel('Total Spend')
plt.ylabel('Purchase Frequency')
plt.legend(title='Cluster')
plt.show()

# Step 8: Save Results to CSV
customer_features.to_csv('Customer_Segmentation.csv', index=False)
print("Customer segmentation results saved to Customer_Segmentation.csv.")

