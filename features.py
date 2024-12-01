import librosa
import os
import numpy as np
import pandas as pd

def feature_extraction(file, duration_goal=4, sr_goal=16000, n_mfcc=30):
    y, sr = librosa.load(file, sr)
    
    if sr != sr_goal:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_goal)
    
    numberofsamples = int(sr_goal * duration_goal)  
    
    if len(y) > numberofsamples:
        y = y[:numberofsamples]
        
    else:
        repeats_needed = numberofsamples // len(y) 
        remainder_samples = numberofsamples % len(y)  
        
        y_repeated = np.tile(y, repeats_needed)  
        
        if remainder_samples > 0:
            y_repeated = np.concatenate([y_repeated, y[:remainder_samples]])  
        
        y = y_repeated  
    
    
    y = librosa.util.normalize(y)



    # Extract features
    features = {}
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(1, n_mfcc + 1):
        features[f'mcffs_{i}'] = np.mean(mfccs[i - 1])
    
    # Mel-Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    features['mel_spectrogram'] = np.mean(mel_spectrogram, axis=1)
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid'] = np.mean(spectral_centroid, axis=1)
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth'] = np.mean(spectral_bandwidth, axis=1)
    
    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast'] = np.mean(spectral_contrast, axis=1)
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff'] = np.mean(spectral_rolloff, axis=1)
    
    # Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    features['spectral_flatness'] = np.mean(spectral_flatness, axis=1)
    
    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms'] = np.mean(rms, axis=1)
    
    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    features['zero_crossing_rate'] = np.mean(zero_crossing_rate, axis=1)
    
    # Tempo (beats per minute)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    
    # Tempogram
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    features['tempogram'] = np.mean(tempogram, axis=1)
    
    return features



    

def create_dataset(root_dir,output_csv="sound_features.csv"):
    
    features_list=[]
    label_list=[]
    
    for folder_name in os.listdir(root_dir):
        folder_path=os.path.join(root_dir,folder_name)
        
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path=os.path.join(folder_path,file_name)
                
                if file_path.endswith('.wav'):
                    
                    try:
                        
                        
                        features=feature_extraction(file_path)
                        label=file_name
                        
                        
                        
                        features_list.append(features)
                        label_list.append(label)
                    
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
    df=pd.DataFrame(features_list)
    df['Label'] = label_list
    
    df.to_csv(output_csv,index=False)
    
    print(df.head())
    
def main():
    root_dir = "C:/Users/fafer/Desktop/AC2/UrbanSound8K/audio"
    create_dataset(root_dir)


if __name__ == '__main__':
    main()

    