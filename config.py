from dataclasses import dataclass
import pyrallis


@dataclass
class Config:
    """Configuration class for PIM model training"""

    # Dataset & Log
    dataset_path: str = (
        "/home/dev/public-datasets/e.shvetsov/PIM/REAL/Real_data/16TR/"  # "/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/"  #
    )
    dataset_name: str = "data_16TR_3"  # "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L"  #

    log_out_dir: str = "./results"
    log_precision: int = 8
    filter_path: str = (
        "/home/dev/work_main/2025/OpenPIM/data/filter_real.mat"  # "/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/rx_filter.mat"
    )

    # PIM Model Settings
    PIM_backbone: str = "linear"
    PIM_hidden_size: int = 8
    PIM_num_layers: int = 1
    specific_channels = "all"

    # Training Process
    step: str = "train_pim_single"
    n_back: int = 128
    n_fwd: int = 3
    accelerator: str = "cuda"
    devices: int = 2
    re_level: str = "soft"
    exp_name: str = "exp"

    # General Hyperparameters
    seed: int = 0
    loss_type: str = "l2"
    # opt_type: str = "adabound"
    opt_type: str = "adabound"
    batch_size: int = 64
    batch_size_eval: int = 64
    n_iterations: int = 80e3
    n_log_steps: int = 2e3
    lr_schedule: int = 1
    lr: float = 1e-4
    lr_end: float = 1e-6
    decay_factor: float = 0.0
    patience: float = 10.0
    grad_clip_val: float = 200.0
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    save_results: bool = True


def main(config: Config):
    """Main function using the configuration"""
    print("Loaded configuration:")
    print(pyrallis.dump(config))

    # Example of accessing config values
    print(f"\nTraining {config.PIM_backbone} model on {config.dataset_name}")
    print(f"Using batch size: {config.batch_size}, learning rate: {config.lr}")


if __name__ == "__main__":
    # Parse configuration from command line or config file
    config = pyrallis.parse(config_class=Config)

    # Run main function with the parsed config
    main(config)
