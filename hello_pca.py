import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# data preprocessed elsewhere.  .. 
df = pd.read_csv("https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in/mfcc_13_labels.csv")
X = df.iloc[:,:-1].to_numpy()   # coefficients 

data = X

# Perform PCA
pca = PCA()
pca.fit(data)

# Principal component coefficients (eigenvectors)
coeff = pca.components_

# Principal component scores (transformed data)
score = pca.transform(data)

# Eigenvalues of the covariance matrix
latent = pca.explained_variance_

# Percentage of variance "explained" by each principal component
explained = pca.explained_variance_ratio_ * 100



# Plot first two principal components
plt.scatter(x=score[:, 0], y=score[:, 1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.draw()

plt.show()


# Plot percentage of variance explained
plt.bar(range(1, len(explained) + 1), explained)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.draw()
plt.show()
