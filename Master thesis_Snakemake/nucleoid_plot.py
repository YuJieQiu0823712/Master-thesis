# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:27:17 2023

@author: Bart Steemans. Govers Lab.
"""



import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def load_meshdata(processed_file):
    
    with open(processed_file, "rb") as file:
        # Load the data from the file using pickle.load()
        processed_dataframe = pickle.load(file)
        
    return processed_dataframe
  


# ###### Standard Scaling (mean = 0 and variance = 1)


#PCA preserves the variance in the data, whereas t-SNE preserves the relationships 
#between data points in a lower-dimensional space


def PCA_plot(processed_dataframe):
    
    
    #standardize the feature matrix
    X= StandardScaler().fit_transform(processed_dataframe)
    # print(X.shape)
    # print(X)
        
    #create a PCA that will retain 85% of the variance
    pca = PCA(n_components=0.85,whiten=True)
    
    #conduct PCA 
    X_pca = pca.fit_transform(X)
    # print(X_pca.shape)
    # print(X_pca) #n_component=3
    
    #create a PCA with 2 components
    pca = PCA(n_components=2,whiten=True)
    X_pca = pca.fit_transform(X)
    # print(X_pca.shape)
    # print(X_pca)
    
    pca_df = pd.DataFrame(data = X_pca, columns = ['PC1','PC2'])
    pca_df #397x2
    
    # Visualization
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize = (10,6))
    c_map = plt.cm.get_cmap('jet', 10)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s = 15, cmap = c_map )
    plt.xlabel('PC-1') , plt.ylabel('PC-2')
    




#processed_dataframe.object_total_volume.value_counts().plot(kind='bar')

# ### t-SNE = non-linear ###
# from sklearn.manifold import TSNE
# # plot t-SNE clusters
# from bioinfokit.visuz import cluster

# def t_SNE_plot(processed_dataframe):
    

#     pca_score= PCA().fit_transform(processed_dataframe)
#     pca_df = pd.DataFrame(pca_score)
    
#     #By default, TSNE() function uses the Barnes-Hut approximation, which is computationally less intensive.
#     tsne = TSNE(n_components = 2, perplexity = 4, early_exaggeration = 12, 
#                     n_iter = 1000, learning_rate = 33, verbose = 1).fit_transform(pca_df.loc[:,0:12])
#     #perplexity is the most important parameter in t-SNE, and it measures the effective number of neighbors.
#     #(standard range 10-100)
#     #In case of large, datasets, keeping large perplexity parameter (n/100; where n is the number of observations) is helpful for preserving the global geometry.
#     #In addition to the perplexity parameter, other parameters such as the number of iterations (n_iter), 
#     #learning rate (set n/12 or 200 whichever is greater), and early exaggeration factor (early_exaggeration) 
#     #can also affect the visualization and should be optimized for larger datasets (Kobak et al., 2019).
#     #https://www.reneshbedre.com/blog/tsne.html
#     ##https://www.nature.com/articles/s41467-019-13056-x    
   
    
#     cluster.tsneplot(score=tsne)
#     #plot will be saved in same directory (tsne_2d.png) 






# UPGMA


if __name__ == "__main__":

    input_featrue_data = sys.argv[1]
    output_plot = sys.argv[2]
    
    processed_dataframe = load_meshdata(input_featrue_data)
    PCA_plot(processed_dataframe)
    plt.savefig(output_plot)
   
    



