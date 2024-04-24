'''
Machine Learning (UNM)

Process train and test files for logistic regression. (new version)

Write X (mfccs) and Y (genres) to out_file.

Utility functions for very large write and read (e.g., to port data to matlab)
'''

import numpy as np
import pandas as pd
import os               # for getcwd()



import pickle # for writing and reading bytes

# utility functions for write and reads too large for text/csv
def write_very_large_data(file_name, X, Y):
        # convert X,Y to bytes
        bytes_data = pickle.dumps((X,Y)) # 662_390_923 bytes for all .au files

        ### write bytes to a file
        with open('data.bin', 'wb') as f:
            f.write(bytes_data)

def read_very_large_data(file_name):
        ### read bytes from the file
        with open('data.bin', 'rb') as f:
            loaded_bytes_data = f.read()

        # convert bytes back to X,Y
        (X_in,Y_in) = pickle.loads(loaded_bytes_data)


        depth = len(X_in)
        rows = len(X_in[0])
        cols = len(X_in[0][0])

        print("depth=", depth)
        print("rows=", rows)
        print("cols=", cols)
        print("Y=", len(Y_in))


      
'''
    > python -m pip install librosa 

    librosa documentation: https://librosa.org/doc/0.10.1/index.html
'''
import librosa
import time # let's time performance

tic = time.time()

# extract metadata from filename
def filename_to_metadata(file_name):

    # extract variables
    file_name_parts = out_file_X_Y.split(".")[0].split("-")
    
    n_mfcc = int(file_name_parts[1])
    hop_length = int(file_name_parts[2])
    win_length_percent = int(file_name_parts[3])

    return (n_mfcc, hop_length, win_length_percent)




n_mfcc = 128            # number of mfccs to extract
hop_length = 512                
win_length_percent = 2            # 2 percent

# X,Y will be a csv, use file name as metadata
# example = "mfcc-128-512-002.csv"

out_file_X_Y = f"mfcc-{n_mfcc}-{hop_length}-{win_length_percent:03d}.csv"
print("outfile: ", out_file_X_Y)

print("metadata:")
a,b,c = filename_to_metadata(out_file_X_Y)
print("n_mfcc:", a)
print("hop_length:", b)
print("win_length_percent:", c)

# find paths to all .au files in /data

audio_paths = librosa.util.find_files(os.getcwd() + "/data", ext='au', recurse=True)
assert(len(audio_paths)==1000)  # for project, expect 1000 (train + test)


### handle Y
Y = []
for file_path in audio_paths:
    
    # read genre (and id) from file name
    # example = blues.00027.au
    file_name = os.path.basename(file_path)
    genre = file_name.split(".")[0]     # genre
    id = file_name.split(".")[1]        # unique identifier (not used)

    Y = Y + [genre]


### handle X
X = []

for file_path in audio_paths:

    y, sr = librosa.load(file_path)         # read audio data and sample rate 
    win_length = int(sr * (win_length_percent/100))    # overlap window for sampling
    
    # extract coefficients from file
    mfcc_matrix = librosa.feature.mfcc(
                                       y=y,
                                       sr=sr, 
                                       n_mfcc=n_mfcc, 
                                       hop_length=hop_length, 
                                       win_length=win_length
                                       )
    
    
    if 1:

        # reduce X to 1 x D by taking the column mean, std, min, max, etc. 
        reduce_with = "mean"

        cols = 1
        if reduce_with == "mean":
            mfcc_matrix = mfcc_matrix.mean(axis=cols)
        elif reduce_with == "std":
            mfcc_matrix = mfcc_matrix.std(axis=cols)

    # append the new row to the existing DataFrame X_Y
    X = X + [mfcc_matrix] 

pd.DataFrame({"X":X, "Y":Y}).to_csv(out_file_X_Y)
print("wrote X and Y:", out_file_X_Y)

toc = time.time()
print("time to extract and write mfccs: %.2d secs" % abs(tic-toc))
