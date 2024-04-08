import os
import librosa
import numpy as np
import pandas as pd

data_path = os.getcwd() + os.sep + "data"

def extract_stats(audio_path):
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration
    rms = librosa.feature.rms(y=y)
    mean_rms = np.mean(rms)
    max_rms = np.max(rms)
    min_rms = np.min(rms)
    return duration, mean_rms, max_rms, min_rms

def compile_stats(root_folder):
    stats = {}
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            folder_stats = []
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path) and file_path.endswith(".au"):
                    #duration, mean_rms, max_rms, min_rms = extract_stats(file_path)
                    duration = extract_stats(file_path)
                    folder_stats.append(duration)
                    #folder_stats.append((file_name, duration, mean_rms, max_rms, min_rms))
            stats[folder_name] = folder_stats
    return stats

train_stats = compile_stats(os.path.join(data_path, "train"))
#test_stats = compile_stats(os.path.join(data_path, "test"))

print("Train Stats:")
print(pd.DataFrame(train_stats).describe())
print("\nTest Stats:")
#print(pd.DataFrame(test_stats).describe())
