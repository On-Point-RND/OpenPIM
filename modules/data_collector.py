import random
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

def standardize_complex_array(complex_array):
       
    real_part = np.real(complex_array)
    imaginary_part = np.imag(complex_array)

    real_mean, real_std = np.mean(real_part), np.std(real_part)
    imaginary_mean, imaginary_std = np.mean(imaginary_part), np.std(imaginary_part)

   
    real_std = real_std if real_std != 0 else 1
    imaginary_std = imaginary_std if imaginary_std != 0 else 1

   
    real_standardized = (real_part - real_mean) / real_std
    imaginary_standardized = (imaginary_part - imaginary_mean) / imaginary_std

    standardized_complex_array = real_standardized + 1j * imaginary_standardized

    return standardized_complex_array

def to2Dreal(x):
    return np.row_stack((x.real, x.imag)).T

def prepare_data_for_predict(data_path):
    data = loadmat(data_path)

    all_txa = data["txa"]
    all_rxa = data["rxa"]
    all_nfa = data["nfa"]

    rxa = []
    txa = []
    nfa = []
    
    for x_id in range(all_txa.shape[0]):
        rxa.append(to2Dreal(all_rxa[x_id]))
        txa.append(to2Dreal(all_txa[x_id]))
        nfa.append(to2Dreal(all_nfa[x_id]))


    # txa = to2Dreal(data["txa"])
    # rxa = to2Dreal(data["rxa"])
    # nfa = to2Dreal(data["nfa"])
    
    FC_TX = data['BANDS_DL'][0][0][0][0][0] / 10**6
    FS = data['Fs'][0][0] / 10**6
    return {'X': txa, 'Y': rxa, 'noise': nfa, 'FC_TX': FC_TX, 'FS': FS}
    
    
def prepare_data(data_path, filter_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):

    fil = loadmat(filter_path)['flt_coeff']
    data = loadmat(data_path)
    
    all_txa = data["txa"]
    all_rxa = data["rxa"]
    all_nfa = data["nfa"]

    rxa = []
    txa = []
    nfa = []
    
    for x_id in range(all_txa.shape[0]):
        rxa.append(to2Dreal(all_rxa[x_id]))
        txa.append(to2Dreal(all_txa[x_id]))
        nfa.append(to2Dreal(all_nfa[x_id]))
        
    FC_TX = data['BANDS_DL'][0][0][0][0][0] / 10**6
    FC_RX = data['BANDS_UL'][0][0][0][0][0] / 10**6
    FS = data['Fs'][0][0] / 10**6
    PIM_SFT = data['PIM_sft'][0][0] / 10**6
    PIM_BW = data['BANDS_TX'][0][0][1][0][0] / 10**6
    PIM_total_BW = data['BANDS_TX'][0][0][3][0][0] / 10**6

    spec_dictionary = {
        "FC_TX": FC_TX,
        "FC_RX": FC_RX,
        "FS": FS, 
        "PIM_SFT": PIM_SFT,
        "PIM_BW": PIM_BW,
        "PIM_total_BW": PIM_total_BW,
        "nperseg": 1536,
    }
   
    total_samples = all_txa.shape[1]
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    return {
    'X': {'Train': [x[:train_end, :] for x in txa], 'Val': [x[train_end:val_end, :] for x in txa], 'Test': [x[val_end:, :] for x in txa]},
    'Y': {'Train': [y[:train_end, :] for y in rxa], 'Val': [y[train_end:val_end, :] for y in rxa], 'Test': [y[val_end:, :] for y in rxa]},
    'N': {'Train': [n[:train_end, :] for n in nfa], 'Val': [n[train_end:val_end, :] for n in nfa], 'Test': [n[val_end:, :] for n in nfa]},
    'specs': spec_dictionary,
    'filter': fil
}

def back_fwd_feature_prepare(list_sequence_x, sequence_t, n_back, n_fwd, n_iterations):
    sequence_x = list_sequence_x[0]
    for id in range(len(list_sequence_x)-1):
        sequence_x = np.row_stack((sequence_x.T, list_sequence_x[id].T)).T

    win_len = n_back + n_fwd + 1
    num_samples = min(sequence_x.shape[0] - win_len + 1, n_iterations)
    
    segments_x = np.zeros((num_samples, win_len, sequence_x.shape[1]), dtype=float)
    segments_y = np.zeros((num_samples, sequence_t.shape[1]), dtype=float)
    
    for step in range(num_samples):
        segments_x[step,:] = sequence_x[step:win_len+step,:]
        segments_y[step,:] = sequence_t[win_len+step-n_fwd-1,:]

    return segments_x, segments_y

class InfiniteIQSegmentDataset(IterableDataset):
    def __init__(self, features, targets, n_back, n_fwd, n_iterations,shuffle=True):
        segments_x, segments_y = back_fwd_feature_prepare(features, targets, n_back, n_fwd,n_iterations)
        
        self.features = torch.Tensor(segments_x)
        self.targets = torch.Tensor(segments_y)
        self.shuffle = shuffle
        self.actual_length = len(segments_x)
        self.indices = list(range(self.actual_length))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        indices = self.indices.copy()

        # Split indices across workers
        if worker_info is not None:
            per_worker = len(indices) // worker_info.num_workers + 1
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(indices))
            indices = indices[start:end]

        while True:
            if self.shuffle:
                random.shuffle(indices)  # Internal shuffle handling
            
            for idx in indices:
                yield self.features[idx], self.targets[idx]

    def __len__(self):
        return self.actual_length

class IQSegmentDataset(Dataset):
    def __init__(self, features, targets, n_back, n_fwd):     
        segments_x, segments_y = back_fwd_feature_prepare(features,targets, n_back=n_back, n_fwd=n_fwd)
       
        self.features = torch.Tensor(segments_x)
        self.targets = torch.Tensor(segments_y)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx, ...]
        targets = self.targets[idx, ...]
        return features, targets

def data_prepare(X, y, frame_length, degree):
    Input = []
    Output = []
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    for k in range(X.shape[0]):
        Complex_In = torch.complex(X[k, :, 0], X[k, :, 1])
        Complex_Out = torch.complex(y[k, :, 0], y[k, :, 1])
        ulength = len(Complex_In) - frame_length
        Input_matrix = torch.complex(torch.zeros(ulength, frame_length),
                                     torch.zeros(ulength, frame_length))
        degree_matrix = torch.complex(torch.zeros(ulength - frame_length, frame_length * frame_length * degree),
                                      torch.zeros(ulength - frame_length, frame_length * frame_length * degree))
        for i in range(ulength):
            Input_matrix[i, :] = Complex_In[i:i + frame_length]
        for j in range(1, degree):
            for h in range(frame_length):
                degree_matrix[:,
                (j - 1) * frame_length * frame_length + h * frame_length:(j - 1) * frame_length * frame_length + (
                        h + 1) * frame_length] = Input_matrix[:ulength - frame_length] * torch.pow(
                    abs(Input_matrix[h:h + ulength - frame_length, :]), j)
        Input_matrix = torch.cat((Input_matrix[:ulength - frame_length], degree_matrix), dim=1)
        b_output = np.array(Complex_Out[:len(Complex_In) - 2 * frame_length])
        b_input = np.array(Input_matrix)
        Input.append(b_input)
        Output.append(b_output)

    return Input, Output