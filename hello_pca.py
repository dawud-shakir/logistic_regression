import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample data
data = np.random.randn(100, 3) # Example 100 samples with 3 features

# Perform PCA
pca = PCA()
pca.fit(data)

# Principal component coefficients (eigenvectors)
coeff = pca.components_

# Principal component scores (transformed data)
score = pca.transform(data)

# Eigenvalues of the covariance matrix
latent = pca.explained_variance_

# Percentage of variance explained by each principal component
explained = pca.explained_variance_ratio_ * 100

# Plot percentage of variance explained
plt.bar(range(1, len(explained) + 1), explained)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.show()

# Plot first two principal components
plt.scatter(score[:, 0], score[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
