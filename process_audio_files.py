'''
Machine Learning

Process audio files for logistic regression. Prints and writes X and Y to out_file.
'''

import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt


#dir = os.getcwd() + "/data/test/"
dir = os.getcwd() + "/data/train/"

X = []  # coefficients
Y = []  # folder label


num_coefficients = 13 
   

n_mfcc=num_coefficients
hop_length=512
win_length=None

out_file = os.getcwd() + f"/mfcc_{n_mfcc}_ids.csv"


def dataframe_X_Y():
    df_Y = pd.DataFrame({"Y":Y})
    
    df_X = pd.DataFrame(X)
    df_X.columns = ["X" + str(col+1) for col in df_X.columns]
    
    df = pd.concat([df_X, df_Y], axis=cols)
    return df

def print_X_Y():
    print("printing X and Y...")
    print(dataframe_X_Y().to_string())

def write_X_Y():
    print(f"writing X and Y to {out_file}...")
    dataframe_X_Y().to_csv(out_file, index=False)


def read_X_Y(in_file):
    df = pd.read_csv(in_file)
    return df.iloc[:,0], df.iloc[:,1:]


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
print("if time is an issue, set folders=just_two_folders", "\n\n")
#folders = just_two_folders


'''
Extract coefficients from audio file.

Return a 1-d array. 
'''

n_mfcc=13
def read_audio_file_mfcc(path) -> np.array:

    y, sr = librosa.load(path)

    # mfcc_matrix 
    mfcc_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length)
    

    # mean of cols
    cols = 1
    return np.mean(mfcc_matrix, axis=cols)    

    

'''
Preprocess audio folders. 

There are 10 folders. Each folder has 90 audio files.
'''
print("extracting coefficients from each audio file...\n\n")

for folder_id,folder in enumerate(folders):
    folder_dir = os.path.join(dir, folder) 
    
    if not os.path.isdir(folder_dir):
        exit(f"{dir} missing folder \"{folder}\"")


    '''
    Find audio files for folder k. 
    '''
    files_in_folder = librosa.util.find_files(folder_dir, ext='au', recurse=False)

    
    coefficients_count = 0
    for i in range(len(files_in_folder)):
        
        coefficients = read_audio_file_mfcc(files_in_folder[i])
        
        X.append(coefficients)
        coefficients_count += 1

        Y.append(folder_id+1)   #  ids are 1..10
    

    print(":"*15, folder, 15*":")
    print(f"folder '{folder}' has {len(files_in_folder)} .au files")
    print(f"folder '{folder}' has {coefficients_count} x {len(X[0])} coefficients")


print()

print(f"{len(X)} by {len(X[0])} ({len(X)*len(X[0])}) total coefficients loaded from audio files")

#print(f"{len(Y)} by {len(Y[0])} class labels ('blues', 'classical', 'country', ...)")

print("X's size is ", len(X), " by ", len(X[0]))
print("Y's size is ", len(Y))


write_X_Y()


print_X_Y()




exit()

'''
Standardize
'''
X = (X - np.mean(X, axis=rows)) / np.std(X, axis=rows)

print("standardized")

print_X_Y()


'''
Convert Y to spare matrix (a single 1 per row)
'''

'''
Y_one_hop_encoded = np.array(Y)
Y_one_hop_encoded = OneHotEncoder().fit_transform(Y_one_hop_encoded.reshape(-1,1))
Y_one_hop_encoded = Y_one_hop_encoded.toarray()

print("one hop encoded Y", pd.DataFrame(Y_one_hop_encoded))
'''




# parameters ready to go!

X = np.array(X)
Y = np.array(Y) 




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

