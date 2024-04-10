# demo



'''
Machine Learning

Logistic Regression Classifier

'''

import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA




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

def num2label(number):
    classes = {
        1: "blues",
        2: "classical",
        3: "country",
        4: "disco",
        5: "hiphop",
        6: "jazz",
        7: "metal",
        8: "pop",
        9: "reggae",
        10: "rock"
    }
    return classes.get(number, "not found") # return "not found" if number not found




# 
# Files expected in root directory: 
#        mfcc_13_labels.csv
#        kaggle_mfcc_13.csv       
#        list_test.txt
#
root_path = "https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in/"



np.random.seed(0)   # seed

# 0. Load data
df = pd.read_csv(root_path + "mfcc_13_labels.csv")


# 1. One hot encode labels and X data
Y_labels = df.iloc[:,-1]   # labels "blues", "classical", etc.
Y_encoded = OneHotEncoder(sparse_output=False).fit_transform(pd.DataFrame(Y_labels))
X = df.iloc[:,:-1]   # coefficients

# 4. Make training and validation sets
# For split, X is (900 x 13) and Y_encoded is (900 x 10) 
x_train, x_val, y_train, y_val = train_test_split(X, Y_encoded, test_size=0.2, random_state=30)

# 2. Standardize X  
#X = (X - X.mean(axis=0)) / X.std(axis=0)
scaler = StandardScaler()         
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# 3. Implement PCA
pca = PCA(0.95)
x_train_red = pca.fit_transform(x_train)
X_val_red = pca.transform(x_val)

# Transpose y_train and y_test to be (10 x 900) 
y_train, y_val = y_train.T, y_val.T  

# 3. Add column of ones to X
#X.insert(0, "X0", 1)

ones_column = np.ones((x_train_red.shape[0], 1))
ones_column_2 = np.ones((X_val_red.shape[0], 1))
x_train_red = np.hstack((ones_column, x_train_red))
X_val_red = np.hstack((ones_column_2, X_val_red))

# 5. Make W 
'''
W's rows = Y's rows  
W's cols = X's cols 
'''
W = np.random.normal(0, 1, (y_train.shape[0], x_train_red.shape[1]))
W[:,0] = 0  # set column of zeros


print("x_train size is ", x_train_red.shape)
print("x_validate size is ", X_val_red.shape)
print("y_train size is ", y_train.shape)
print("y_validate size is ", y_val.shape)

print("W's size is ", W.shape)

print()


threshold = 1e-6  # Convergence threshold
max_iterations = 4000  # Maximum number of iterations
learning_rate = 0.001  # Learning rate
regularization_strength = 0.00001  # Regularization strength

def sigmoid(x):
    y = 1 / (1 + np.exp( -x ))
    return y

X_t = x_train_red.T 

for i in range(max_iterations):
    #W_Xt = sigmoid(np.dot(W, X_t))

    PY = sigmoid(np.dot(W, X_t))
    W_new = W + learning_rate * (np.dot((y_train - PY), x_train_red) - regularization_strength * W)
    

    #print(abs(np.mean(np.mean(y_train - PY))))

    W = W_new

# Check for convergence
if np.linalg.norm(y_train - PY) < threshold:
    print(f'Convergence reached after {i+1} iterations.')
else:
    print('Maximum iterations reached without convergence.')
    


# guess for each sample
y_test = sigmoid(np.dot(W, X_val_red.T))  
guesses = np.argmax(y_test, axis=0)   # one guess per row

score = 0
for i in range(len(guesses)):
    
   
    if y_val[guesses[i]][i] == 1:   # Is guess where the "1" is?
        score = score + 1


accuracy = score / len(guesses)

print("accuracy: ", accuracy)
print('The end??')

if 1:    # test W on Kaggle
        
    df = pd.read_csv(root_path + "kaggle_mfcc_13.csv")    

    x_test = df.iloc[:,0:]  
    #x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)    
    x_test = scaler.transform(x_test)                # standardize
    x_test_red = pca.transform(x_test)
    #x_test_red = np.insert(x_test, 0, "X0", 1)   # column of ones
    ones_column_3 = np.ones((x_test_red.shape[0], 1))
    x_test_red = np.hstack((ones_column_3, x_test_red))

    # guess for each sample
    y_test = sigmoid(np.dot(W, x_test_red.T))
    guesses = np.argmax(y_test, axis=0)   # best guess per row

    files_in_test_dir = pd.read_csv(root_path + "list_test.txt", header=None)   # from data/test/
    guess_labels = list(map(num2label, guesses + 1))

    out_kaggle = pd.DataFrame()
    out_kaggle.insert(0, "id", files_in_test_dir)
    out_kaggle.insert(1, "class", guess_labels)


    print(out_kaggle)

    out_path = os.getcwd() + "/out_kaggle.csv"
    out_kaggle.to_csv(out_path, index=False)  # no index for submission file
    print("wrote kaggle guesses:", out_path)

    exit()


