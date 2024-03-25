# caleb

import numpy as np
import pandas as pd


from sklearn.preprocessing import OneHotEncoder


''' Sorry, this is what I meant about cols/rows, Caleb... '''
# === before === preprocessing: 
# each row is a feature (a coeff)
# axis=1

df = pd.read_csv("https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/mfcc_13_labels.csv")
X = df.iloc[:,:-1].to_numpy()   # columns zero to next to last
Y = df.iloc[:,-1].to_numpy()    # last column


# === after === preprocessing: 
# each column is a feature (a coeff)
# use axis=0 
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)     


ones_column = np.ones((X.shape[0], 1))
new_array = np.hstack((ones_column, X))
Xval = new_array
print(pd.DataFrame(Xval))   

'''
Convert labels ("blues", "classical", ...) to binary arrays.
'''

Y_one_column = Y

Y = OneHotEncoder(sparse_output=False).fit_transform(pd.DataFrame(Y))   # expects a 2-D container, not a 1-D series
Y = Y.T # to make size(Y)=(k,m)=(10,900)
print(Y)


print(pd.DataFrame(Y))

print("X's shape", X.shape)
print("Y's shape", Y.shape)


#initializing weights matrix
rows, cols = 10, 14
W = np.random.normal(0, 1, (rows, cols)) 
print(pd.DataFrame(W))

'''W.shape=[K,N+1], X.shape=[M,N+1], Y.shape=[K,M]'''

X_t = Xval.T                              
#W_Xt = np.dot(Xval, W.T)    # (X)(W^T)  
W_Xt = np.dot(W,X_t)         # (W)(X^T)   


exp_W_Xt = np.exp(W_Xt)
log_term = np.log(1 + exp_W_Xt)

# W_Xt = PY_X
# W_new = W + (Y - PY_X)*X

print(np.shape(np.dot(Y-W_Xt, X))) 
print(np.shape(log_term))

# Compute P
#P = np.dot(Y, W_Xt) - log_term  #  size(Y) * size(W_Xt) = (10,900) * (10,900)
