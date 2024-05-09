import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

def get_data(
    path,
    subject,
    is_standard,
    is_shuffle
):
    '''
    Load and divide dataset based on subject-specific. 
    Training: 288 x 9 trials in session 1
    Testing: 288 x 9 trials in session 2

    Parameters:
    ----------
    path (str):
        dataset path
    subject (int):
        no of subject in [1, 2, ..., 9]
    classes_labels (list):
        a list of labels
    is_standard (bool):
        if True, standardize training and testing data
    is_shuffle (bool):
        if True, shuffle the dataset
    '''
    path = path + '/s{:}/'.format(subject + 1)
    X_train, y_train = load_data(path, subject + 1, True)
    X_test, y_test = load_data(path, subject + 1, False)

    # shuffle dataset
    if is_shuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=44)
        X_test, y_test = shuffle(X_test, y_test, random_state=44)

    # prepare training data
    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)

    # prepare test data
    N_tr, N_ch, T = X_test.shape
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)

    # standardize data
    if is_standard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

def standardize_data(X_train, X_test, num_channels):
    for channel in range(num_channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, channel, :])
        X_train[:, 0, channel, :] = scaler.transform(X_train[:, 0, channel, :])
        X_test[:, 0, channel, :] = scaler.transform(X_test[:, 0, channel, :])
    return X_train, X_test

def load_data(
    path, 
    subject,
    training,
    all_trials=True
):
    '''
    Load and divide dataset based on subject-specific.
    Training: 288 x 9 trials
    Testing: 288 x 9 trials

    Parameters:
    ----------
        data_path (str): 
            dataset path
        subject (int): 
            no of subject in [1, 2, ..., 9]
        training (bool): 
            if True, load training data, 
            if False, load test data
        all_trials (bool):
            if True, load all trials
            if False, ignore trials with artifacts
    '''

    # define MI-trials parameters
    n_channels = 22          # no of channels
    n_tests = 6*48           # no of sample points
    window_length = 7*250    # window size

    # define MI trial window
    fs = 250                 # sampling frequency
    t1 = int(1.5*fs)         # start time point
    t2 = int(6*fs)           # end time point

    class_return = np.zeros(n_tests) # return labels
    data_return = np.zeros((n_tests, n_channels, window_length)) # return data

    no_valid_trials = 0

    if training:
        a = sio.loadmat(path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(path + 'A0' + str(subject) + 'E.mat')

    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if (a_artifacts[trial] != 0 and not all_trials):
                continue
            data_return[no_valid_trials, :, :] = np.transpose(
                a_X[int(a_trial[trial]):(int(a_trial[trial])+window_length),:22]
            )
            class_return[no_valid_trials] = int(a_y[trial])
            no_valid_trials += 1

    data_return = data_return[0:no_valid_trials, :, t1:t2]
    class_return = class_return[0:no_valid_trials]
    class_return = (class_return - 1).astype(int)

    return data_return, class_return

if __name__ == '__main__':
    X_train = [] 
    y_train = []
    y_train_onehot = []
    X_test = []
    y_test = []
    y_test_onehot = []

    path = 'BCI2a-mat'
    for subject in range(0, 9):
        X_train_subject, y_train_subject, y_train_onehot_subject, X_test_subject, y_test_subject, y_test_onehot_subject = get_data(
            path=path,
            subject=subject,
            is_shuffle=True,
            is_standard=True
        )
        X_train.append(X_train_subject)
        y_train.append(y_train_subject)
        y_train_onehot.append(y_train_onehot_subject)
        X_test.append(X_test_subject)
        y_test.append(y_test_subject)
        y_test_onehot.append(y_test_onehot_subject)
