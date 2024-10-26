# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preprocessing
2. Initialize Centroids
3. Assign Clusters
4. Update Centroids

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: monisha.L
RegisterNumber:  2305001019
*/
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
plt.figure(figsize=(4,4))
plt.scatter(X['Annual Income (k$)'],X['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
k=3
Kmeans=KMeans(n_clusters=k)
Kmeans.fit(X)
centroids=Kmeans.cluster_centers_
labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b']
for i in range(k):
cluster_points=X[labels==i]
plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (
distances=euclidean_distances(cluster_points,[centroids[i]])
radius=np.max(distances)
circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroi
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/4a5ec042-e00c-43e2-985c-c044761a92cb)
![image](https://github.com/user-attachments/assets/08705ba7-9bbe-40a4-a08b-8721fd71e0af)
![image](https://github.com/user-attachments/assets/6fa1f24a-4f4e-4157-ae12-5fcdbd9becd5)
![image](https://github.com/user-attachments/assets/0f944fbd-34c6-4945-bd10-57397058d703)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
