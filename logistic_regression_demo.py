'''
Machine Learning

Preprocess audio files for logistic regression.

Needs the "data" folder from Prof. Trilce's "Google Drive > Projects > Project 2 > data"

************** Output at bottom ****
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


folders = all_folders
print("if time is an issue, set variable folders=just_two_folders")
#folders = just_two_folders


    
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
print("extracting coefficients from audio file...")

for c in folders:
    folder_dir = os.path.join(dir, c) 
    
    if not os.path.isdir(folder_dir):
        exit(f"{dir} missing folder \"{c}\"")


    '''
    Find audio files for folder k. 
    '''
    files_in_folder = librosa.util.find_files(folder_dir, ext='au', recurse=False)

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


print('X is the coefficients matrix from audio file in folder')
print('Y is the audio folder (blues, classical, country')
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
print(X)

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



# all_folders
'''

if time is an issue, set variable folders=just_two_folders
extracting coefficients from audio file...
::::::::::::::: blues :::::::::::::::
blues has 90 .au files
::::::::::::::: classical :::::::::::::::
classical has 90 .au files
::::::::::::::: country :::::::::::::::
country has 90 .au files
::::::::::::::: disco :::::::::::::::
disco has 90 .au files
::::::::::::::: hiphop :::::::::::::::
hiphop has 90 .au files
::::::::::::::: jazz :::::::::::::::
jazz has 90 .au files
::::::::::::::: metal :::::::::::::::
metal has 90 .au files
::::::::::::::: pop :::::::::::::::
pop has 90 .au files
::::::::::::::: reggae :::::::::::::::
reggae has 90 .au files
::::::::::::::: rock :::::::::::::::
rock has 90 .au files
X is the coefficients matrix from audio file in folder
Y is the audio folder (blues, classical, country
X's size is  (900, 1290)
Y's size is  (900,)
[[ 1.4668637e-03 -3.8113907e-02  9.5596351e-03 ...  2.8269383e-01
   2.1083282e-01  2.2434103e-01]
 [-4.3218166e-01  1.8818676e-01  1.0549458e+00 ...  8.3892208e-01
   9.2070884e-01  1.0053785e+00]
 [ 3.7053797e-01  2.0519267e-01  1.8060875e-01 ...  6.3767773e-01
   7.5352615e-01  8.1290799e-01]
 ...
 [ 8.9658087e-01  7.6986706e-01  6.7155725e-01 ...  4.8403674e-01
   3.9009428e-01  3.7656170e-01]
 [-2.3531650e-01 -8.4036820e-02 -4.6797141e-02 ... -8.9093977e-01
  -1.6940556e-01  3.6492574e-01]
 [-1.2876230e+00 -1.5448823e+00 -1.5353658e+00 ... -2.5888512e-02
   1.5623225e-01  9.3924932e-02]]
X's shape (900, 1290)
Y's shape (900, 10)
SVC accuracy: 0.22777777777777777
Gaussian Naive Bayes accuracy: 0.24444444444444444
RandomForest accuracy: 0.3111111111111111
'''

# just_two_folders
'''


::::::::::::::: blues :::::::::::::::
blues has 90 .au files
blues has 90 mfccs
min mfcc length:  1293
max mfcc length:  1293
::::::::::::::: classical :::::::::::::::
classical has 90 .au files
classical has 90 mfccs
min mfcc length:  1292
max mfcc length:  1314
X's size is  (180, 1292)
Y's size is  (180,)
X's shape (180, 1292)
Y's shape (180, 2)
SVC accuracy: 0.6388888888888888
Gaussian Naive Bayes accuracy: 0.8888888888888888
RandomForest accuracy: 0.8888888888888888

'''
