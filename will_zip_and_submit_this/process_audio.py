'''
Machine Learning (UNM)

Process train and test files for logistic regression. (new version)

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



'''

    audio, sr = librosa.load(path)


    # Calculate win_length based on the sampling rate
    win_length = int(sr * win_length_ratio) 
    
    # mfcc_matrix 
    mfcc_matrix = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_coefficients, hop_length=hop_length, win_length=win_length)
    

    # mean of cols
    cols = 1
    return np.std(mfcc_matrix, axis=cols)    
       coefficients = read_audio_file_mfcc(files_in_folder[i])
            
            X.append(coefficients)
            coefficients_count += 1

            Y.append(audio_label)   
            
            
            file_id = files_in_folder[i].split(".")[1]
            
            
'''


'''
Preprocess audio folders. 
'''




exit("early exit")


'''

def dataframe_X_Y(X, Y):
    df_X = pd.DataFrame(X)
    df_X.columns = ["X" + str(col+1) for col in df_X.columns]

    df_Y = pd.DataFrame({"Y":Y})
    df = pd.concat([df_X, df_Y], axis=cols)
    return df

def print_X_Y(X,Y):
    print("printing X and Y...")
    print(dataframe_X_Y(X,Y).to_string())

def write_X_Y(X,Y, csv_out):
    print(f"writing X and Y to {csv_out}...")
    X_Y = dataframe_X_Y(X,Y)
    if len(Y)==0:
        X_Y.iloc[:,:-1].to_csv(csv_out, index=False)
    else:
        X_Y.to_csv(csv_out, index=False)
   
def read_X_Y(in_file):
    df = pd.read_csv(in_file)
    return df.iloc[:,0], df.iloc[:,1:]





num_mfccs = 128
hop_length = 512
win_length_ratio = 0.02



rows = 0
cols = 1

#These folders are our classes. 

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

no_folders = [""] 

folders = no_folders    # because there's no folders in test
#folders = all_folders
print("if time is an issue, set folders=just_two_folders", "\n\n")
#folders = just_two_folders




# Extract coefficients from audio file.

# Return a 1-d array. 

def read_audio_file_mfcc(path, num_coefficients=128,  hop_length=512, win_length_ratio=0.02) -> np.array:

    audio, sr = librosa.load(path)


    # Calculate win_length based on the sampling rate
    win_length = int(sr * win_length_ratio) 
    
    # mfcc_matrix 
    mfcc_matrix = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_coefficients, hop_length=hop_length, win_length=win_length)
    

    # mean of cols
    cols = 1
    return np.std(mfcc_matrix, axis=cols)    


def extract_mfccs(dir, folders, csv_out):    


  

    # Preprocess audio folders. 

    # There are 10 folders. Each folder has 90 audio files.
    
    print(f"extracting coefficients from each audio file in {dir}...\n\n")

    X = []  # coefficients
    Y = []  # folder label

    for folder in folders:
        folder_dir = os.path.join(dir, folder) 
        
        if not os.path.isdir(folder_dir):
            exit(f"{dir} missing folder \"{folder}\"")


        #Find audio files for folder k. 
        
        files_in_folder = librosa.util.find_files(folder_dir, ext='au', recurse=False)

        
        audio_label = folder    # "blues", "classical", "country", etc.
        coefficients_count = 0
        for i in range(len(files_in_folder)):
            
            coefficients = read_audio_file_mfcc(files_in_folder[i])
            
            X.append(coefficients)
            coefficients_count += 1

            Y.append(audio_label)   
            
            
            file_id = files_in_folder[i].split(".")[1]
            
            
        

        print(":"*15, folder, 15*":")
        print(f"folder '{folder}' has {len(files_in_folder)} .au files")
        print(f"folder '{folder}' has {coefficients_count} x {len(X[0])} coefficients")


    print()

    print(f"{len(X)} by {len(X[0])} ({len(X)*len(X[0])}) total coefficients loaded from audio files")

    print("X's size is ", len(X), " by ", len(X[0]))
    print("Y's size is ", len(Y))


    write_X_Y(X, Y, csv_out)


    #print_X_Y(X, Y)


out_train_file = os.getcwd() + f"/train_mfcc_{num_mfccs}_std.csv"
out_test_file = os.getcwd() + f"/test_mfcc_{num_mfccs}_std.csv"

root_path = os.getcwd()

extract_mfccs(root_path + "/data/train/", all_folders, out_train_file)
extract_mfccs(root_path + "/data/test/", ["."], out_test_file)


  







print(f"extracting coefficients from each audio file in {dir}...\n\n")

    X = []  # coefficients
    Y = []  # folder label

    for folder in folders:
        folder_dir = os.path.join(dir, folder) 
        
        if not os.path.isdir(folder_dir):
            exit(f"{dir} missing folder \"{folder}\"")


        # Find audio files for folder k. 
        
        files_in_folder = librosa.util.find_files(folder_dir, ext='au', recurse=False)

        
        audio_label = folder    # "blues", "classical", "country", etc.
        coefficients_count = 0
        for i in range(len(files_in_folder)):
            
            coefficients = read_audio_file_mfcc(files_in_folder[i])
            
            X.append(coefficients)
            coefficients_count += 1

            Y.append(audio_label)   
            
            
            file_id = files_in_folder[i].split(".")[1]
            
            
        

        print(":"*15, folder, 15*":")
        print(f"folder '{folder}' has {len(files_in_folder)} .au files")
        print(f"folder '{folder}' has {coefficients_count} x {len(X[0])} coefficients")


    print()

    print(f"{len(X)} by {len(X[0])} ({len(X)*len(X[0])}) total coefficients loaded from audio files")

    print("X's size is ", len(X), " by ", len(X[0]))
    print("Y's size is ", len(Y))


    write_X_Y(X, Y, csv_out)


    #print_X_Y(X, Y)
'''