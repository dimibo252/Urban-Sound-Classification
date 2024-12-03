import librosa
import os
import numpy as np
import pandas as pd



def feature_extraction(file, duration_goal=4, sr_goal=16000, n_mfcc=30):
    y, sr = librosa.load(file, sr=sr_goal)
    
    numberofsamples = int(sr_goal * duration_goal)  
    
    if len(y) > numberofsamples:
        y = y[:numberofsamples]
        
    else:
        repeats_needed = numberofsamples // len(y) 
        remainder_samples = numberofsamples % len(y)  
        
        y_repeated = np.tile(y, repeats_needed)  
        
        if remainder_samples > 0:
            y_repeated = np.concatenate([y_repeated, y[:remainder_samples]])  
        
        if len(y_repeated) < numberofsamples:
            y_repeated = np.pad(y_repeated, (0, numberofsamples - len(y_repeated)), mode='constant')    
        
        y = y_repeated  
    
    
    y = librosa.util.normalize(y)
    


    features = {}
    
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(1, n_mfcc + 1):
        features[f'mcffs_{i}'] = np.mean(mfccs[i - 1])
    
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    features['mel_spectrogram'] = np.mean(mel_spectrogram, axis=1)
    

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid'] = np.mean(spectral_centroid)
    

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
    

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast'] = np.mean(spectral_contrast)
    

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff'] = np.mean(spectral_rolloff)
    

    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    features['spectral_flatness'] = np.mean(spectral_flatness)
    

    rms = librosa.feature.rms(y=y)
    features['rms'] = np.mean(rms)
    

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    features['zero_crossing_rate'] = np.mean(zero_crossing_rate)


    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    
 
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    features['tempogram'] = np.mean(tempogram)
    

    return features


def load_metadata(annotations_file):
    
    metadata=pd.read_csv(annotations_file)
    filename_to_label = dict(zip(metadata['slice_file_name'], metadata['class']))
    return filename_to_label    
    

def create_dataset(root_dir,annotations_file):
    
    filename_to_label = load_metadata(annotations_file)
    
    features_list=[]
    label_list=[]
    
    for folder_name in os.listdir(root_dir):
        folder_path=os.path.join(root_dir,folder_name)
        
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path=os.path.join(folder_path,file_name)
                
                if file_path.endswith('.wav'):
                    
                    label = filename_to_label.get(file_name, None)
                    
                    try:
                        
                        
                        features=feature_extraction(file_path)
                        label=file_name
                        
                        
                        
                        features_list.append(features)
                        label_list.append(label)
                    
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
        df=pd.DataFrame(features_list)
        df['Label'] = label_list
    
        df.to_csv("sound_features" + folder_name + '.csv',index=False)
    
    
    
        print(df.head())

    
def main():
    root_dir = "C:/Users/fafer/Desktop/AC2/UrbanSound8K/audio"
    annotations_file="C:/Users/fafer/Desktop/AC2/UrbanSound8K/metadata/UrbanSound8K.csv"
    
    create_dataset(root_dir,annotations_file)


if __name__ == '__main__':
    main()


    
