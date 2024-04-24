'''
Machine Learning (UNM)

Process train and test files for logistic regression. (version 2)

Write X (mfccs) and Y (genres) to out_file.

Option to write all MFCC information.
'''


import numpy as np
import pandas as pd
import os               # for getcwd()

import pickle # for writing bytes 


# options: 128.512.02 -> {128x1293 per file


n_mfcc = 128          # aka D
hop_length = 512                
win_length_percent = 2            # 2 percent



# use file name as metadata

s = "%03d" % (win_length_percent,)  # python 2+
#print(s)
s = "{:03d}".format(win_length_percent) # python 3+
#print(s)
s = f"{win_length_percent:03d}" # python 3.6+
#print(s)

X_Y_file = f"{n_mfcc}.{hop_length}.{s}.csv"
print("outfile: ", X_Y_file)


'''
    > python -m pip install librosa 

    librosa documentation: https://librosa.org/doc/0.10.1/index.html
'''
import librosa

# Find paths to all .au files in /data

audio_paths = librosa.util.find_files(os.getcwd() + "/data", ext='au', recurse=True)
assert(len(audio_paths)==1000)  # for project, expect 1000 (train + test)


### handle Y
Y = []
for file_path in audio_paths:
    
    # format: blues.00027.au
    file_name = os.path.basename(file_path)
    genre = file_name.split(".")[0]     # genre
    id = file_name.split(".")[1]    # unique identifier

    Y = Y + [genre]


### handle X
X = []

for file_path in audio_paths:

    # read audio data and sample rate 
    y, sr = librosa.load(file_path)     
            
    win_length = int(sr * (win_length_percent/100))    # overlap window for sampling

    mfcc_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length)
    
    if 0:
        # reduce X to a 1xD by taking the column mean, min, max, etc. 
        cols = 1
        mfcc_matrix = mfcc_matrix.mean(axis=cols)

    # append the new row to the existing DataFrame X_Y
    X = X + [mfcc_matrix] 



X_out = X
Y_out = Y

# Convert list to bytes
bytes_data = pickle.dumps((X_out,Y_out)) # 662_390_923 bytes

# Write bytes to a file
with open('data.bin', 'wb') as f:
    f.write(bytes_data)

# Read bytes from the file
with open('data.bin', 'rb') as f:
    loaded_bytes_data = f.read()

# Convert bytes back to list
(X_in,Y_in) = pickle.loads(loaded_bytes_data)


depth = len(X_in)
rows = len(X_in[0])
cols = len(X_in[0][0])



print("depth=", depth)
print("rows=", rows)
print("cols=", cols)
print("Y=", len(Y_in))


