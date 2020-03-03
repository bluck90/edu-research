# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 00:59:13 2020

@author: btgl1e14

Doing some clustering on a dataset 

"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('PISA etc.csv')
X = dataset.iloc[:, 4:].values
y = dataset.iloc[:, 3].values
countrynames = dataset.iloc[:, 0].values
X_col_names = list(dataset.columns.values)
X_col_names = X_col_names[4:]

# Dealing with missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

# Scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Using "elbow method" to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('W.C.S.S.')
plt.show()

"""
2 clusters looks optimal, but going for 3 to get a bit more granularity.
"""

# Fitting KMeans to country data
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

# Making dataframe to inspect clusters
clusterdata = pd.DataFrame(list(zip(countrynames,y,y_kmeans)), columns = ['Country', 'PISA Score', 'Cluster'])

# Using decision tree on clusters to help suggest interpretation of the clusters
# Fitting classifier to the Training set
import sklearn.tree as sktree
classifier = sktree.DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X, y_kmeans)
sktree.plot_tree(classifier, feature_names = X_col_names)
sktree.export_text(classifier, feature_names = X_col_names)