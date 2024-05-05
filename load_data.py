"""
This file contains 2 functions to read raw gdf data. In order to use those functions, you need to download BCI competition IV dataset 2a and put them in BCI2a/ folder in the same folder with this folder. You can modify the load_single_subject_data() function to read .mat file if you have dataset in .mat format. 
"""
import mne
import numpy as np

def load_single_subject_data(path: str):
    """
    Load raw gdf file data
    Parameters:
        path (str): path to .gdf file

    Returns:
        features (numpy.ndarray): features of subject
        labels (numpy.ndarray): labels of subject
    """
    raw = mne.io.read_raw_gdf(
        input_fname=path,
        eog=['EOG-left', 'EOG-central', 'EOG-right'],
        preload=True,
        verbose=None
    )
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    raw.set_eeg_reference()
    events = mne.events_from_annotations(
        raw=raw,
        verbose=None
    )
    event_id = {
        'left': 7, 
        'right': 8, 
        'foot': 9, 
        'tongue': 10
    }
    epochs = mne.Epochs(
        raw=raw,
        events=events[0],
        event_id=event_id,
        on_missing='warn',
        verbose=None
    )
    labels = epochs.events[:, -1]
    features = epochs.get_data()
    return features, labels

def load_data(train=True):
    """
    Load all subjects' datas
    Parameters:
        train (bool): True if subject is a training subject, False if subject is a evaluating subject

    Return:
        features (list): data features
        labels (list): labels (4 classes)
    """
    features = []
    labels = []
    if train == True:
        print('Training data')
        for i in range(1, 10):
            print(f"Subject no {i} ...")
            data_path = f'BCI2a/A0{i}T.gdf'
            feature, label = load_single_subject_data(data_path)
            print("-------------------------")
            features.append(feature)
            labels.append(label)
    else:
        print('Evaluation data')
        for i in range(1, 10):
            print(f"Subject no {i} ...")
            data_path = f'BCI2a/A0{i}E.gdf'
            feature, label = load_single_subject_data(data_path)
            print("-------------------------")
            features.append(feature)
            labels.append(label)
    return features, labels 
    
if __name__ == '__main__':
    print('Loading data...')
    features, labels = load_data()
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    print("-------------------------")
    print(f"Feature's shape: {features.shape}")
    print(f"Label's shape: {labels.shape}")
    print()
    
    print(f"Number of 'not a number' value in features: {np.isnan(features).sum()}")
    print(f"Number of 'not a number' value in labels: {np.isnan(labels).sum()}")
    print()
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Classes in labels: {unique}")
    print(f"Number of each classes in labels: {counts}")