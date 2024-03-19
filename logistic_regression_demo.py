'''
Machine Learning
Demo for logistic regression that extracts audio file features (mfccs) 


Expects the data folder from Prof. Trilce's google drive -> Projects -> Project 2 -> data
'''

import numpy as np
import pandas as pd
import os


'''
    > python -m pip install librosa 
    librosa documentation: https://librosa.org/doc/0.10.1/index.html
'''
import librosa



# directory "train" has 10 folders ..

# .. each folder has 90 audio files

folders = [
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


def read_audio_file_mfcc(path, n_mfcc=40, hop_length=512):
    '''
    y: audio time series
    sr: sampling rate (e.g., 22,050 samples per second)

    sampling rate = len(y)/librosa.get_duration(path=filepath)
    
    '''
    y, sr = librosa.load(path)

    
    '''
    n_mfcc: number of coefficients
    hop_length: number of samples between frames
    win_length: the length each frame of audio is windowed by     
    '''
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    '''
    mfccs is an np.array, with shape [n_mfcc, round(len(y) / hop_length)]
    mean(mfccs) produces one vector (up-down means)
    '''
    return np.mean(mfccs, axis=0)   
    

#dir = os.getcwd() + "/data/test/"
dir = os.getcwd() + "/data/train/"

X = []
Y = []

'''
k is the class (the folder's name)
'''

for k in folders:
    folder_dir = os.path.join(dir, k) 
    
    if not os.path.isdir(folder_dir):
        exit(f"{dir} missing folder \"{k}\"")


    '''
    Find audio files for folder k
    '''
    files_in_folder = librosa.util.find_files(folder_dir, ext='au', recurse=False)
    
    '''
    MFCCs
    '''

    mfcc_lengths = []

    for i in range(len(files_in_folder)):
        mfccs = read_audio_file_mfcc(files_in_folder[i])

        X.append(mfccs)
        Y.append(k)
    
        mfcc_lengths.append(len(mfccs))
    
    print(":"*15, k, 15*":")
    print(k, "has", len(files_in_folder), ".au files")
    
    print(k, "has", len(mfcc_lengths), "mfccs")
    print("min mfcc length: ", pd.Series(mfcc_lengths).min())
    print("max mfcc length: ", pd.Series(mfcc_lengths).max())

'''
Truncate matrix of mfccs because some audio files are longer than others 
'''
column_lengths = [len(row) for row in X]
min_length = np.min(column_lengths) 
for i in range(len(X)):
    X[i] = X[i][0:min_length]



X_np = np.array(X)
Y_np = np.array(Y)

print("X's size is ", X_np.shape)
print("Y's size is ", Y_np.shape)


'''
            ✰✰✰✰✰✰✰✰✰✰✰✰✰✰✰ first file stats for each folder ✰✰✰✰✰✰✰✰✰✰✰✰✰✰✰
                    blues  classical   country     disco    hiphop      jazz     metal       pop    reggae      rock
            X_1     0.703420   0.701523  0.608531  0.479831  0.568229  0.731900  0.541029  0.735052  0.635830  0.702455
            X_10    0.462661   0.677816  0.415386  0.137065  0.256116  0.706626  0.548116  0.830835  0.652842  0.443482
            X_100   0.544752   0.658246  0.569525  0.201122  0.474191  0.628309  0.517817  0.701178  0.677881  0.215750
            X_1000  0.461137   0.582427  0.169560  0.495315  0.608102  0.729812  0.459332  0.409259  0.580456  0.643486
            X_1001  0.293657   0.600244  0.169043  0.471940  0.319406  0.731631  0.507238  0.470019  0.612739  0.633723
            ...          ...        ...       ...       ...       ...       ...       ...       ...       ...       ...
            X_995   0.454218   0.571575  0.332156  0.529729  0.690678  0.746002  0.543152  0.591831  0.566443  0.648446
            X_996   0.458582   0.568542  0.386990  0.382775  0.605958  0.752240  0.538136  0.567777  0.616360  0.652182
            X_997   0.501597   0.566279  0.390437  0.144551  0.612386  0.753069  0.535918  0.568598  0.571460  0.656321
            X_998   0.521433   0.581031  0.380135  0.174165  0.698357  0.754550  0.559912  0.609452  0.546842  0.646221
            X_999   0.536049   0.579963  0.274374  0.435200  0.720420  0.741862  0.498093  0.581725  0.565388  0.645676

            [1298 rows x 10 columns]

                        blues    classical      country        disco       hiphop         jazz        metal          pop       reggae         rock
            count  1293.000000  1293.000000  1296.000000  1298.000000  1293.000000  1293.000000  1293.000000  1293.000000  1293.000000  1293.000000
            mean      0.467257     0.651950     0.314330     0.512766     0.486264     0.610844     0.496691     0.576609     0.585336     0.470690
            std       0.120775     0.042661     0.101040     0.164918     0.142193     0.107923     0.075952     0.168046     0.098075     0.168306
            min       0.159499     0.475634     0.132712     0.116199     0.170401     0.127411     0.176629     0.142276     0.209380     0.153916
            25%       0.390433     0.630592     0.236386     0.448723     0.376731     0.560414     0.466499     0.478607     0.532833     0.306408
            50%       0.483170     0.657403     0.287197     0.540686     0.479392     0.639916     0.506569     0.618071     0.606196     0.484847
            75%       0.558213     0.679427     0.382908     0.617238     0.590588     0.681311     0.541659     0.706032     0.657383     0.617007
            max       0.739320     0.767573     0.649253     0.889780     0.891265     0.834926     0.710777     0.865869     0.808952     0.786995
'''



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

