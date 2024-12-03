import librosa
import os
import numpy as np
import pandas as pd


def feature_extraction(file, duration_goal=4, sr_goal=16000, n_mfcc=30):
    y, sr = librosa.load(file, sr=sr_goal)
    numberofsamples = int(sr_goal * duration_goal)
    
    # Handle length adjustments
    if len(y) > numberofsamples:
        y = y[:numberofsamples]
    else:
        repeats_needed = numberofsamples // len(y)
        remainder_samples = numberofsamples % len(y)
        y_repeated = np.tile(y, repeats_needed)
        y_repeated = np.concatenate([y_repeated, y[:remainder_samples]]) if remainder_samples > 0 else y_repeated
        y = np.pad(y_repeated, (0, numberofsamples - len(y_repeated)), mode='constant')
    
    y = librosa.util.normalize(y)
    
    # Extract features
    features = {}
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(1, n_mfcc + 1):
        features[f'mfcc_{i}'] = np.mean(mfccs[i - 1])
    features['mel_spectrogram'] = np.mean(librosa.feature.melspectrogram(y=y, sr=sr))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))
    features['rms'] = np.mean(librosa.feature.rms(y=y))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y=y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    features['tempogram'] = np.mean(librosa.feature.tempogram(y=y, sr=sr))
    
    return features


def load_metadata(annotations_file):
    metadata = pd.read_csv(annotations_file)
    filename_to_label = dict(zip(metadata['slice_file_name'], metadata['class']))
    return filename_to_label


def create_dataset(root_dir, annotations_file):
    filename_to_label = load_metadata(annotations_file)
    
    for folder_name in os.listdir(root_dir):  # Iterate over the 10 folders (fold1, fold2, ...)
        folder_path = os.path.join(root_dir, folder_name)
        
        if os.path.isdir(folder_path):
            features_list = []
            label_list = []
            
            for file_name in os.listdir(folder_path):  # Iterate over each WAV file in the folder
                file_path = os.path.join(folder_path, file_name)
                
                if file_path.endswith('.wav'):  # Process only WAV files
                    label = filename_to_label.get(file_name, None)  # Get the label from the metadata
                    
                    if label is not None:  # Proceed only if the label exists
                        try:
                            features = feature_extraction(file_path)
                            features_list.append(features)
                            label_list.append(label)
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
            
            # Save features and labels to a CSV for the current folder
            df = pd.DataFrame(features_list)
            df['Label'] = label_list
            df.to_csv(f"sound_features_{folder_name}.csv", index=False)
            print(f"Processed {folder_name}: {len(features_list)} files.")


def main():
    root_dir = "C:/Users/fafer/Desktop/AC2/UrbanSound8K/audio"
    annotations_file = "C:/Users/fafer/Desktop/AC2/UrbanSound8K/metadata/UrbanSound8K.csv"
    create_dataset(root_dir, annotations_file)


if __name__ == '__main__':
    main()


    
