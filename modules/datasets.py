import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset


class InfiniteSequentialDataset(IterableDataset):
    """
    Infinite dataset for sequential chains of length seq_len.
    Splits data into non-overlapping sequential chains of length seq_len.
    Each sequence has shape (seq_len, channels, 2).
    """
    def __init__(self, features, targets, seq_len, shuffle=True):
        """
        Args:
            features: numpy array shape (num_samples, channels, 2)
            targets: numpy array shape (num_samples, channels, 2)
            seq_len: length of the signal sequence
            shuffle: shuffle the sequences
        """
        self.seq_len = seq_len
        num_samples = features.shape[0]
        self.num_sequences = num_samples // seq_len
        if self.num_sequences == 0:
            raise ValueError(
                f"seq_len ({seq_len}) is greater than num_samples ({num_samples}). "
            )

        end_idx = self.num_sequences * seq_len
        features_cut = features[:end_idx]
        targets_cut = targets[:end_idx]
        feature_sequences = features_cut.reshape(
            self.num_sequences, seq_len, features.shape[1], features.shape[2]
        )
        target_sequences = targets_cut.reshape(
            self.num_sequences, seq_len, targets.shape[1], targets.shape[2]
        )

        self.feature_sequences = torch.Tensor(feature_sequences)
        self.target_sequences = torch.Tensor(target_sequences)

        self.shuffle = shuffle
        self.sequence_indices = list(range(self.num_sequences))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        indices = self.sequence_indices.copy()

        if worker_info is not None:
            per_worker = len(indices) // worker_info.num_workers + 1
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(indices))
            indices = indices[start:end]

        while True:
            if self.shuffle:
                random.shuffle(indices)
            for seq_idx in indices:
                yield (
                    self.feature_sequences[seq_idx],
                    self.target_sequences[seq_idx]
                )

    def __len__(self):
        return self.num_sequences


class SequentialDataset(Dataset):
    """
    Dataset for signal sequences of length seq_len.
    Splits data into non-overlapping sequential sequences of length seq_len.
    Each sequence has shape (seq_len, channels, 2).
    Used for test and eval datasets.
    """
    def __init__(self, features, targets, seq_len):
        """
        Args:
            features: numpy array shape (num_samples, channels, 2)
            targets: numpy array shape (num_samples, channels, 2)
            seq_len: length of the signal sequence
        """
        self.seq_len = seq_len
        num_samples = features.shape[0]
        self.num_sequences = num_samples // seq_len
        if self.num_sequences == 0:
            raise ValueError(
                f"seq_len ({seq_len}) is greater than num_samples ({num_samples}). "
            )

        end_idx = self.num_sequences * seq_len
        features_cut = features[:end_idx]
        targets_cut = targets[:end_idx]

        feature_sequences = features_cut.reshape(
            self.num_sequences, seq_len, features.shape[1], features.shape[2]
        )
        target_sequences = targets_cut.reshape(
            self.num_sequences, seq_len, targets.shape[1], targets.shape[2]
        )

        self.feature_sequences = torch.Tensor(feature_sequences)
        self.target_sequences = torch.Tensor(target_sequences)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Returns:
            features: (seq_len, channels, 2)
            targets: (seq_len, channels, 2)
        """
        return self.feature_sequences[idx], self.target_sequences[idx]
