import os
import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader
from modules.datasets import InfiniteIQSegmentDataset, IQSegmentDataset
from modules.data_utils import ComplexScaler, to2Dreal
from modules.loggers import make_logger

logger = make_logger()

def extract_predictions(
    data: dict, 
    preds: dict,
    n_back: int,
    n_fwd: int,
    path_dir_save: str,
    ):

    
    СScaler = ComplexScaler(data, path_dir_save)
    for data_part in ["Train", "Val", "Test"]:

        # print('data[Y][data_part]: ', data["Y"][data_part])
        # print('СScaler.normalize(data[Y][data_part], key=Y): ', СScaler.normalize(data["Y"][data_part], key="Y"))
        # print('preds[data_part]: ', preds[data_part])
        # print('СScaler.normalize(preds[data_part], key=Y): ', СScaler.normalize(preds[data_part], key="Y"))
        
        total_samples = data["X"][data_part].shape[0]
        
        data["X"][data_part] = data["X"][data_part][n_back: total_samples-n_fwd]
        data["Y"][data_part] = data["Y"][data_part][n_back: total_samples-n_fwd]

        # print('data[Y][data_part]: ', data["Y"][data_part])
        # print('preds[data_part]: ', СScaler.rescale(preds[data_part], key="Y"))
        data["Y"][data_part] = data["Y"][data_part] - СScaler.rescale(preds[data_part], key="Y")
        # data["Y"][data_part] = data["Y"][data_part] - preds[data_part]
    return data
    

# INFO: This is the main script to load resources used in RUNNER, it runs at the begginign of training once
def prepare_dataloaders(
    data: dict,
    n_back: int,
    n_fwd: int,
    batch_size: int,
    batch_size_eval: int,
    path_dir_save: str,
    specific_channels,
):
    input_size = 1 + n_back + n_fwd
    n_channels = data["X"]["Train"].shape[1]

    # Calculate normalization parameters
    СScaler = ComplexScaler(data, path_dir_save)

    # Apply normalization and slice data
    for data_part in ["Train", "Val", "Test"]:
        # data["X"][data_part] = СScaler.normalize(data["X"][data_part], key="X")
        # data["Y"][data_part] = СScaler.normalize(data["Y"][data_part], key="Y")
        data["N"][data_part] = data["N"][data_part][n_back:-n_fwd, :]

    train_set = InfiniteIQSegmentDataset(
        СScaler.normalize(data["X"]["Train"], key="X"),
        СScaler.normalize(data["Y"]["Train"], key="Y"),
        n_back=n_back,
        n_fwd=n_fwd,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    # Train for pred set/loader
    train_pred_set = IQSegmentDataset(
        СScaler.normalize(data["X"]["Train"], key="X"), 
        СScaler.normalize(data["Y"]["Train"], key="Y"), 
        n_back=n_back, n_fwd=n_fwd
    )
    train_pred_loader = DataLoader(train_pred_set, batch_size=batch_size_eval, shuffle=False)

    # Validation set/loader
    val_set = IQSegmentDataset(
        СScaler.normalize(data["X"]["Val"], key="X"), 
        СScaler.normalize(data["Y"]["Val"], key="Y"), 
        n_back=n_back, n_fwd=n_fwd
    )
    val_loader = DataLoader(val_set, batch_size=batch_size_eval, shuffle=False)

    # Test set/loader
    test_set = IQSegmentDataset(
        СScaler.normalize(data["X"]["Test"], key="X"), 
        СScaler.normalize(data["Y"]["Test"], key="Y"), 
        n_back=n_back, n_fwd=n_fwd
    )
    test_loader = DataLoader(test_set, batch_size=batch_size_eval, shuffle=False)

    logger.success(f"Dataloaders were created")
    return (
        (train_loader, train_pred_loader, val_loader, test_loader),
        input_size,
        n_channels,
        data["N"],
        data["filter"],
        СScaler,
        data["specs"],
    )

