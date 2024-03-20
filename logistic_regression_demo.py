'''
Machine Learning

Preprocess audio files for logistic regression.

Needs the "data" folder from Prof. Trilce's "Google Drive > Projects > Project 2 > data"

'''

import numpy as np
import pandas as pd
import os


'''
    > python -m pip install librosa 
    librosa documentation: https://librosa.org/doc/0.10.1/index.html
'''
import librosa

from sklearn.preprocessing import OneHotEncoder


'''
These folders are our classes. 
'''
all_folders = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

just_two_folders = [
    "blues",
    "classical"
]

#folders = all_folders
folders = just_two_folders


    
'''
Extract coefficients from audio file.

Return a 1-d array. 

Each column in the 1-d array is the mean of coefficients at that time step. 


n_coefficients: number of coefficients 

n_samples_in_frame: number of samples in a frame (hop length)

frame_length: the length of each frame (win_length) 

'''

def read_audio_file_mfcc(path, n_coefficients=40, n_samples_in_frame=512, frame_length=None) -> np.array:

    y, sr = librosa.load(path)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_coefficients, hop_length=n_samples_in_frame, win_length=frame_length)

    # mean of rows
    return np.mean(mfccs, axis=0)   

    
#dir = os.getcwd() + "/data/test/"
dir = os.getcwd() + "/data/train/"

X = []  # coefficients
Y = []  # folder label

'''
Preprocess audio folders. 

There are 10 folders. Each folder has 90 audio files.
'''
for c in folders:
    folder_dir = os.path.join(dir, c) 
    
    if not os.path.isdir(folder_dir):
        exit(f"{dir} missing folder \"{c}\"")


    '''
    Find audio files for folder k. 
    '''
    files_in_folder = librosa.util.find_files(folder_dir, ext='au', recurse=False)
    
    '''
    MFCCs
    '''

    mfcc_lengths = []

    for i in range(len(files_in_folder)):
        mfccs = read_audio_file_mfcc(files_in_folder[i])

        X.append(mfccs)
        Y.append(c)
    
        mfcc_lengths.append(len(mfccs))
    
    print(":"*15, c, 15*":")
    print(c, "has", len(files_in_folder), ".au files")
    
#    print(c, "has", len(mfcc_lengths), "mfccs")
#    print("min mfcc length: ", pd.Series(mfcc_lengths).min())
#    print("max mfcc length: ", pd.Series(mfcc_lengths).max())

'''
Truncate X because some audio files are longer than others.
'''
column_lengths = [len(row) for row in X]
min_length = np.min(column_lengths) 
for i in range(len(X)):
    X[i] = X[i][0:min_length]



X = np.array(X)
Y = np.array(Y)

print("X's size is ", X.shape)
print("Y's size is ", Y.shape)


'''
Standardize each column in X. 
'''
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

'''
Convert labels ("blues", "classical", ...) to binary arrays.
'''
Y_one_column = Y
Y = OneHotEncoder().fit_transform(Y.reshape(-1,1))
Y = Y.toarray()

print("X's shape", X.shape)
print("Y's shape", Y.shape)




'''
This chapter will compare our Logistic Model's accuracy to sklean's SVC/GaussianNB/RandomForest 
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Y_one_column are the labels associated with each audio file
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y_one_column, 
                                                    test_size=0.2,
                                                    random_state=0)

'''
Support Vector Classifier (SVC)
'''
from sklearn.svm import SVC

clf = SVC(kernel='linear', C=1.0, random_state=0)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("SVC accuracy:", accuracy)


'''
Gaussian Naive Bayes' Classifier
'''
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Gaussian Naive Bayes accuracy:", accuracy)


'''
Random Forest Classifier
'''
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=0)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("RandomForest accuracy:", accuracy)

exit()


'''
Works for boolean logical regression (Y <- {0,1}) 

def logistic_regression(X : pd.DataFrame, Y : pd.Series):
    N = len(X.columns)

    # array of N + 1 zeros for weights
    weights = [0] * (N + 1)

    Y_0 = sum(Y == 0) / len(Y)  # P(Y = 0)
    Y_1 = sum(Y == 1) / len(Y)  # P(Y = 1)

    weights[0] = np.log(Y_0 / Y_1)  # bias (intercept) weight

    # bias weights with mean and std of columns

    for i,col in enumerate(X.columns):
        weights[0] += (X[col][Y == 0].mean()**2 - X[col][Y == 1].mean()**2) / (2 * X[col].std()**2)

        weights[i + 1] = (X[col][Y == 0].mean() - X[col][Y == 1].mean()) / X[col].std()**2

    #print("weights=", weights)

    # linear combination Z = W0 + W1*X[0] + W2*X[1] + ...  

    Z = weights[0]  # intercept

    for i, col in enumerate(X.columns):
        Z += weights[i + 1] * X[col]

    expected_value_when_Y_1 = 1 / (1 + np.exp(-Z))

    X.index = expected_value_when_Y_1
    print(X)

#########################################################
'''

