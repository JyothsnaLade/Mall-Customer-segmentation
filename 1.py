#!/usr/bin/env python


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




# Reading the data
data=pd.read_csv('./Mall_Customers.csv')
data.columns




# Dropping unnecessary attributes
data.drop(['Gender','CustomerID'],axis=1,inplace=True)
data.head()




# Scaling the data
log_data=np.log(data)
good_data=log_data.drop([128,65,66,75,154])
good_data[:10]




# Determining number of components for PCA.
from sklearn.decomposition import PCA
pca=PCA().fit(good_data)
print(pca.explained_variance_ratio_)
print()
print(good_data.columns.values.tolist())
print(pca.components_)




cumulative=np.cumsum(pca.explained_variance_ratio_)
plt.step([i for i in range(len(cumulative))],cumulative)
plt.show()



#Performing PCA
pca=PCA(n_components=2)
pca.fit(good_data)
reduced_data=pca.transform(good_data)
inverse_data=pca.inverse_transform(reduced_data)
plt.scatter(reduced_data[:,0],reduced_data[:,1],label='reduced')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()




#Visualizing the data.
reduced_data=pd.DataFrame(reduced_data,columns=['Dim1','Dim2'])
reduced_data[:10]







