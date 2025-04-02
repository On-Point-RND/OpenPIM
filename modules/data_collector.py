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

def prepare_data(data_path, filter_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):

    fil = loadmat(filter_path)['flt_coeff']
    data = loadmat(data_path)
    rxa = to2Dreal(data["rxa"])
    txa = to2Dreal(data["txa"])
    nfa = to2Dreal(data["nfa"])
        
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
   
    total_samples = txa.shape[0]
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    return {
    'X': {'Train': txa[:train_end, :], 'Val': txa[train_end:val_end, :], 'Test': txa[val_end:, :]},
    'Y': {'Train': rxa[:train_end, :], 'Val': rxa[train_end:val_end, :], 'Test': rxa[val_end:, :]},
    'N': {'Train': nfa[:train_end, :], 'Val': nfa[train_end:val_end, :], 'Test': nfa[val_end:, :]},
    'specs': spec_dictionary,
    'filter': fil
}

def back_fwd_feature_prepare(sequence_x, sequence_t, n_back, n_fwd):
    win_len = n_back + n_fwd + 1
    num_samples = sequence_x.shape[0] - win_len + 1
    segments_x = np.zeros((num_samples, win_len, 2), dtype=float)
    segments_y = np.zeros((num_samples, 2), dtype=float)
    
    for step in range(num_samples):
        segments_x[step,:] = sequence_x[step:win_len+step,:]
        segments_y[step,:] = sequence_t[win_len+step-n_fwd-1,:]
    return segments_x, segments_y




class InfiniteIQSegmentDataset(IterableDataset):
    def __init__(self, features, targets, n_back, n_fwd, shuffle=True):
        segments_x, segments_y = back_fwd_feature_prepare(features, targets, n_back, n_fwd)
        
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


# class IQFrameDataset(Dataset):
#     def __init__(self, features, targets, frame_length, stride, n_back, n_fwd):

#         features = back_fwd_feature_prepare(features, n_back=n_back, n_fwd=n_fwd)
#         targets = back_fwd_target_prepare(targets, n_back=n_back, n_fwd=n_fwd)
        
#         self.features = torch.Tensor(self.get_frames(features, frame_length, stride))
#         self.targets = torch.Tensor(self.get_frames(targets, frame_length, stride))
        
#     @staticmethod
#     def get_frames(sequence, frame_length, stride_length):
#             frames = []
#             sequence_length = len(sequence)
#             num_frames = (sequence_length - frame_length) // stride_length + 1
#             for i in range(num_frames):
#                 frame = sequence[i * stride_length: i * stride_length + frame_length]
#                 frames.append(frame)
#             return np.stack(frames)
        
#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.targets[idx]


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


# NOTE: why do we need that?

# class IQFrameDataset_gmp(Dataset):
#     def __init__(self, segment_dataset, frame_length, degree, stride_length=1):
#         """
#         Initialize the frame dataset using a subset of IQSegmentDataset.

#         Args:
#         - segment_dataset (IQSegmentDataset): The pre-split segment dataset.
#         - seq_len (int): The length of each frame.
#         - stride_length (int, optional): The step between frames. Default is 1.
#         """

#         # Extract segments from the segment_dataset
#         IQ_in_segments = [item[0] for item in segment_dataset]
#         IQ_out_segments = [item[1] for item in segment_dataset]

#         # Convert the list of tensors to numpy arrays
#         IQ_in_segments = torch.stack(IQ_in_segments).numpy()
#         IQ_out_segments = torch.stack(IQ_out_segments).numpy()

#         self.IQ_in_frames, self.IQ_out_frames = data_prepare(IQ_in_segments, IQ_out_segments, frame_length, degree)

#     def __len__(self):
#         return len(self.IQ_in_frames)

#     def __getitem__(self, idx):
#         return self.IQ_in_frames[idx], self.IQ_out_frames[idx]
