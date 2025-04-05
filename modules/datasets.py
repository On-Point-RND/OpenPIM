import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from modules.data_utils import back_fwd_feature_prepare


class InfiniteIQSegmentDataset(IterableDataset):
    def __init__(self, features, targets, n_back, n_fwd, shuffle=True):
        segments_x, segments_y = back_fwd_feature_prepare(
            features, targets, n_back, n_fwd
        )

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
        segments_x, segments_y = back_fwd_feature_prepare(
            features, targets, n_back=n_back, n_fwd=n_fwd
        )

        self.features = torch.Tensor(segments_x)
        self.targets = torch.Tensor(segments_y)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx, ...]
        targets = self.targets[idx, ...]
        return features, targets
