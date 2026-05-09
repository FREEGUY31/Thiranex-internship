import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


df = pd.read_csv("B:\\thiranex\\week 2\\Mall_Customers.csv")

X = df.iloc[:, [3, 4]].values 

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)


plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[y_kmeans == 0, 0], y=X[y_kmeans == 0, 1], label='Sensible (Mid-Income/Mid-Spend)', s=100)
sns.scatterplot(x=X[y_kmeans == 1, 0], y=X[y_kmeans == 1, 1], label='Careless (Low-Income/High-Spend)', s=100)
sns.scatterplot(x=X[y_kmeans == 2, 0], y=X[y_kmeans == 2, 1], label='Target (High-Income/High-Spend)', s=100)
sns.scatterplot(x=X[y_kmeans == 3, 0], y=X[y_kmeans == 3, 1], label='Frugal (High-Income/Low-Spend)', s=100)
sns.scatterplot(x=X[y_kmeans == 4, 0], y=X[y_kmeans == 4, 1], label='Impulsive (Low-Income/Low-Spend)', s=100)

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

df['Cluster'] = y_kmeans
df.to_csv('segmented_customers.csv', index=False)
print("Project Complete! Results saved to segmented_customers.csv")