# demo

def label2num(label):
    classes = {
        "blues": 1,
        "classical": 2,
        "country": 3,
        "disco": 4,
        "hiphop": 5,
        "jazz": 6,
        "metal": 7,
        "pop": 8,
        "reggae": 9,
        "rock": 10
    }
    return classes.get(label, -1)  # return -1 if label is not found


import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

# data preprocessed elsewhere.  .. 
df = pd.read_csv("https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in/mfcc_13_labels.csv")
X = df.iloc[:,:-1].to_numpy()   # coefficients 
Y = df.iloc[:,-1].to_numpy()    # label "blues", "classical"


# === after === preprocessing: 
# each column is a feature (a coeff)
# use axis=0 
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)     


ones_column = np.ones((X.shape[0], 1))
new_array = np.hstack((ones_column, X))
Xval = new_array
print(pd.DataFrame(Xval))   

'''
convert labels ("blues", "classical", ...) to binary vectors 
'''

Y_one_column = Y

Y = OneHotEncoder(sparse_output=False).fit_transform(pd.DataFrame(Y))   # expects a 2-D container, not a 1-D series
Y = Y.T # to make size(Y) = (k,m) = (10, 900)
print(Y)


print(pd.DataFrame(Y))

print("X's shape", X.shape)
print("Y's shape", Y.shape)


#initializing weights matrix
rows, cols = 10, 14     # k,m
W = np.random.normal(0, 1, (rows, cols)) 
print(pd.DataFrame(W))

'''W.shape=[K,N+1], X.shape=[M,N+1], Y.shape=[K,M]'''

X_t = Xval.T                              

for i in range(1000):
    
    W_Xt = np.dot(W,X_t)         # PY = (W)(X')   

    print(i)
    #print(pd.DataFrame(W_Xt))
    
    exp_W_Xt = np.exp(W_Xt)
    log_term = np.log(1 + exp_W_Xt)

    # update with gradient
    W = W + 0.001*(np.dot((Y - log_term), Xval) - 0.001*W)     

np.random.seed(0)

# guessing these samples with W:
test = np.random.randint(low=0, high=899, size=50)

 
X_test = Xval[test]   # already standardized

y_predict = 1 / (1 + np.exp( np.dot( X_test, W.T ) ))
y_predict = np.argmax(y_predict, axis=1)

y_test = Y_one_column[test]
y_test = list(map(label2num,y_test))

accuracy = sum(y_predict==y_test) / len(y_test)

print("accuracy: ", accuracy)



