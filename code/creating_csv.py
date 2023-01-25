import csv
import numpy as np
import pandas as pd
import librosa as lb
import glob
from librosa import feature

fn_list_i = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_contrast,
    feature.spectral_rolloff  # tempogram as an alternative
]

fn_list_ii = [
    feature.zero_crossing_rate,
    feature.spectral_flatness
]


def get_feature_vector(y, s):
    fv_i_m = [np.mean(funct(y, s)) for funct in fn_list_i]
    fv_ii_m = [np.mean(funct(y)) for funct in fn_list_ii]
    fv_i_s = [np.std(funct(y, s)) for funct in fn_list_i]
    fv_ii_s = [np.std(funct(y)) for funct in fn_list_ii]
    f_v = fv_i_m + fv_ii_m + fv_i_s + fv_ii_s
    return f_v


participants = pd.read_excel("PsychiatricDiscourse_participant_data.xlsx")
schizophrenia_only = participants.loc[
    (participants['depression.symptoms'] == 0.) &
    (participants['thought.disorder.symptoms'] != 0.)
]

wav_files_dir = '**/*.wav'
wav_files = glob.glob(wav_files_dir)  # how to drop 'wav_files/'?

feature_vector = []
for file in wav_files:
    if file[10:16] in list(schizophrenia_only.ID):
        y, sr = lb.load(file, sr=None)
        fv = get_feature_vector(y, sr)
        fv.insert(0, file[10:16])
        fv.insert(1, file.split("-")[2])
        fv.insert(2, schizophrenia_only.loc[(schizophrenia_only['ID'] == file[10:16]), 'thought.disorder.symptoms'].item())
        feature_vector.append(fv)


norm_output = 'voice_features.csv'
header = ["id", "stimulus", "symptoms_score", "chroma_stft_mean", "spectral_centroid_mean", "spectral_bandwidth_mean",
          "spectral_contrast_mean", "spectral_rolloff_mean", "zero_crossing_rate_mean", "spectral_flatness_mean", "chroma_stft_std",
          "spectral_centroid_std", "spectral_bandwidth_std", "spectral_contrast_std", "spectral_rolloff_std",
          "zero_crossing_rate_std", "spectral_flatness_std"]

with open(norm_output, "+w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(header)
    csv_writer.writerows(feature_vector)

df = pd.read_csv("voice_features.csv")
print(df)
