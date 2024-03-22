'''
Machine Learning

Preprocess audio files for logistic regression.

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

rows = 0
cols = 1

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
print("if time is an issue, set folders=just_two_folders")
#folders = just_two_folders


'''
This chapter preprocesses the audio files.
'''
    
'''
Extract coefficients from audio file.

Return a 1-d array. 
'''

def read_audio_file_mfcc(path) -> np.array:

    y, sr = librosa.load(path)

    # mfcc_matrix 
    mfcc_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=512, win_length=None)
    

    # mean of cols
    cols = 1
    return np.mean(mfcc_matrix, axis=cols)    

    
#dir = os.getcwd() + "/data/test/"
dir = os.getcwd() + "/data/train/"

X = []  # coefficients
Y = []  # folder label

'''
Preprocess audio folders. 

There are 10 folders. Each folder has 90 audio files.
'''
print("extracting coefficients from audio file...")

for folder in folders:
    folder_dir = os.path.join(dir, folder) 
    
    if not os.path.isdir(folder_dir):
        exit(f"{dir} missing folder \"{folder}\"")


    '''
    Find audio files for folder k. 
    '''
    files_in_folder = librosa.util.find_files(folder_dir, ext='au', recurse=False)

    for i in range(len(files_in_folder)):
        coefficients = read_audio_file_mfcc(files_in_folder[i])

        X.append(coefficients)
        Y.append(folder)
    

    print(":"*15, folder, 15*":")
    print(folder, "has", len(files_in_folder), ".au files")


print('X is the coefficients matrix from audio file in folder')
print('Y is the audio folder (blues, classical, country')

print("X=", pd.DataFrame(X))
print("Y=", pd.DataFrame(Y))

print("X's size is ", len(X), " by ", len(X[0]))
print("Y's size is ", len(Y))


'''
Standardize
'''

X = (X - np.mean(X, axis=rows)) / np.std(X, axis=rows)

   


'''
One Hop Encoding 
'''
# Skip for now
#Y_one_column = Y
#Y = OneHotEncoder().fit_transform(Y.reshape(-1,1))
#Y = Y.toarray()


#exit("done with preprocessing")


'''
Chapter II ... find P(Y_k | X_k, W_k)

class handout
X = samples
Y = observations 



W = ((X**2)-np.mean(X)^2) / np.std(X)**2

X = (X - np.mean(X, axis=rows)) / np.std(X, axis=rows)
X[:,0] = 1

print(pd.DataFrame(X))


    
c = 1
for k in np.unique(Y):
    Y[Y==k] = c
    c += 1

argmax = -1
for k in range(len(folders)):
    pass    
'''

'''
Chapter IV. Our logistic regression model versus sklean
'''

Y_one_column = Y


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







# all_folders
'''
if time is an issue, set folders=just_two_folders
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
X=              0           1          2          3          4          5          6   ...        33        34        35        36        37        38        39
0   -113.598824  121.570671 -19.162262  42.363941  -6.362266  18.621931 -13.699734  ... -2.999698  4.476317 -0.476855  6.006285 -0.059690 -3.458585 -1.841832
1   -207.523834  123.985138   8.947020  35.867149   2.909594  21.519472  -8.556513  ... -2.451906  5.834808  3.544988  4.897320 -0.415597 -1.995414 -0.465218
2    -90.757164  140.440872 -29.084547  31.686693 -13.976547  25.753752 -13.664991  ...  0.493469  0.447271 -4.162672 -4.815749 -6.703234 -4.425333 -0.981519
3   -199.575134  150.086105   5.663404  26.855278   1.770071  14.232647  -4.827845  ... -3.845604 -2.524410 -4.935610 -5.954977 -6.616996 -6.396001 -1.501189
4   -160.354172  126.209480 -35.581394  22.139256 -32.473549  10.850701 -23.350071  ... -6.110803 -6.951970 -4.070553 -1.137216 -0.491605 -4.786619 -3.221128
..          ...         ...        ...        ...        ...        ...        ...  ...       ...       ...       ...       ...       ...       ...       ...
895 -185.500031   98.925781 -36.442879  44.427540 -17.759830  21.284361 -20.293684  ... -3.955510 -3.703457 -6.930363 -1.700546 -2.314521  0.727262  2.184963
896 -160.157043   88.741447 -35.476883  47.494301 -12.603620  17.301561 -19.144144  ... -2.667592 -2.553800 -2.767049 -1.134696 -2.076545  0.316649 -2.790205
897  -42.662823  102.988686 -36.877899  44.516605 -11.996202  25.576914 -22.668953  ...  1.308880  3.513731 -3.199157 -1.240945 -2.604901 -4.466717 -2.298241
898 -117.974052   84.058403 -50.267879  46.049637 -17.698416  20.855793 -21.641119  ... -2.901361 -1.249006 -3.046344 -1.682882 -2.913632 -0.135997 -0.648021
899 -156.298126  130.789597 -25.229616  43.485470 -12.138994  25.392202  -7.940168  ... -0.646797 -0.067076  0.996034  0.362248 -4.546776 -4.545654 -3.438572

[900 rows x 40 columns]
Y=          0
0    blues
1    blues
2    blues
3    blues
4    blues
..     ...
895   rock
896   rock
897   rock
898   rock
899   rock

[900 rows x 1 columns]
X's size is  900  by  40
Y's size is  900
SVC accuracy: 0.5166666666666667
Gaussian Naive Bayes accuracy: 0.4
RandomForest accuracy: 0.5333333333333333
'''
