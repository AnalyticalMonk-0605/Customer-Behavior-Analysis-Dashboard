import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Creating a synthetic customer dataset
np.random.seed(42)

# Simulating customer data
n = 500  # Number of customers
age = np.random.randint(18, 70, size=n)  # Age of customers
annual_income = np.random.randint(20000, 100000, size=n)  # Annual income in USD
spending_score = np.random.randint(1, 100, size=n)  # Spending score (1 to 100)

# Creating the dataframe
df = pd.DataFrame({
    'Age': age,
    'Annual_Income': annual_income,
    'Spending_Score': spending_score
})

# Display the first few rows of the dataset
df.head()

# Check for missing values
df.isnull().sum()

# Scaling the features (important for KMeans clustering)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert scaled data back to a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Display the first few rows of scaled data
df_scaled.head()

# EDA: Visualizing distributions of features
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Annual_Income'], kde=True)
plt.title('Annual Income Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Spending_Score'], kde=True)
plt.title('Spending Score Distribution')
plt.show()

sns.pairplot(df)
plt.show()

# Apply KMeans clustering (assuming 4 clusters for this example)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualize the clustering result using Plotly (interactive)
fig = px.scatter(df, x='Age', y='Annual_Income', color='Cluster', title='Customer Segmentation')
fig.show()

# AI-generated insights based on the clustering results
cluster_summary = df.groupby('Cluster').agg(
    avg_age=('Age', 'mean'),
    avg_income=('Annual_Income', 'mean'),
    avg_spending_score=('Spending_Score', 'mean')
).reset_index()

st.write("**Customer Segments Summary**")
st.write(cluster_summary)

# Example insight generation
for _, row in cluster_summary.iterrows():
    if row['avg_spending_score'] > 80:
        st.success(f"Cluster {row['Cluster']} is a high spender with an average spending score of {row['avg_spending_score']}.")
    else:
        st.warning(f"Cluster {row['Cluster']} tends to spend less with an average spending score of {row['avg_spending_score']}.")

# Streamlit Dashboard
st.title("Customer Behavior Analysis")

# Select a cluster
selected_cluster = st.selectbox("Select Customer Segment", cluster_summary['Cluster'].values)

# Display the details for the selected cluster
cluster_details = cluster_summary[cluster_summary['Cluster'] == selected_cluster].iloc[0]
st.subheader(f"Details for Cluster {selected_cluster}")
st.write(f"Average Age: {cluster_details['avg_age']:.2f}")
st.write(f"Average Annual Income: ${cluster_details['avg_income']:.2f}")
st.write(f"Average Spending Score: {cluster_details['avg_spending_score']:.2f}")

# Visualizing the distribution of age for the selected cluster
selected_cluster_data = df[df['Cluster'] == selected_cluster]
fig = px.histogram(selected_cluster_data, x='Age', title=f'Age Distribution of Cluster {selected_cluster}')
st.plotly_chart(fig)
