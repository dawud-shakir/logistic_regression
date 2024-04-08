# no pca (python) version 

import numpy as np
import pandas as pd
import os


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split




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
# Need these files in root directory: 
#        mfcc_13_labels.csv
#        kaggle_mfcc_13.csv         
#        list_test.txt              (from data/test)
#
root_path = "https://raw.githubusercontent.com/dawud-shakir/logistic_regression/main/in/"



np.random.seed(0)   # seed

# 1. Load training data
df = pd.read_csv(root_path + "mfcc_13_labels.csv")


# 2. One hot encode labels
Y_labels = df.iloc[:,-1]   # labels "blues", "classical", etc.
Y_encoded = OneHotEncoder(sparse_output=False).fit_transform(pd.DataFrame(Y_labels))


# 3. Standardize X 
X = df.iloc[:,:-1]   # coefficients 
X = (X - X.mean(axis=0)) / X.std(axis=0)


# 4. Add column of ones to X
X.insert(0, "X0", 1)


# 4. Split X and Y

# X is (900 x 13) and Y_encoded is (900 x 10) 
x_train, x_validate, y_train, y_validate = train_test_split(X, Y_encoded, train_size=0.8, random_state=0)

# Transpose y_train and y_test to be (10 x 900) 
y_train = y_train.T
y_validate = y_validate.T  

# 5. W 
W = np.random.normal(0, 1, (y_train.shape[0], x_train.shape[1]))
W[:,0] = 0  # set column of zeros


print("x_train size is ", x_train.shape)
print("x_validate size is ", x_validate.shape)
print("y_train size is ", y_train.shape)
print("y_validate size is ", y_validate.shape)

print("W's size is ", W.shape)

print()

# 6. Hyperparameters 

threshold = 1e-6  # Convergence threshold
max_iterations = 2500  # Maximum number of iterations
learning_rate = 0.001  # Learning rate
penalty = 0.0001  # Regularization strength

def sigmoid(x):
    y = 1 / (1 + np.exp( -x ))
    return y

# 7. Train

for i in range(max_iterations):
    # Predict p_y for x_train and y_train
    py = sigmoid(np.dot(W, x_train.T))

    # Update W 
    W = W + learning_rate * (np.dot((y_train - py), x_train) - penalty * W)
    
    print(abs(np.mean(np.mean(y_train - py))))

else:
    print('Maximum iterations reached without convergence.')



# 8. Test
    
y_test = sigmoid(np.dot(W, x_validate.T))  
guesses = np.argmax(y_test, axis=0)   # best guess per row

score = 0
for i in range(len(guesses)):
    
   
    if y_validate[guesses[i]][i] == 1:   # Is guess where the "1" is?
        score = score + 1


accuracy = score / len(guesses)
print("accuracy: ", accuracy)

if 1:    
    # Apply W to Kaggle test data 
        
    df = pd.read_csv(root_path + "kaggle_mfcc_13.csv")    

    x_test = df.iloc[:,0:]  
    x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)    # standardize
    x_test.insert(0, "X0", 1)   # column of ones

    # guess for each sample
    y_test = sigmoid(np.dot(W, x_test.T))
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