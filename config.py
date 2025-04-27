from dataclasses import dataclass
import pyrallis


@dataclass
class Config:
    """Configuration class for PIM model training"""

    # dataset_path: str = "./Data/FOR_COOPERATION/"
    dataset_path: str = (
        # "/home/dev/public-datasets/e.shvetsov/PIM/REAL/Real_data/16TR/"
        # "/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/"
        # "/home/dev/public-datasets/e.shvetsov/PIM/data_cooperation_21.04.25/artificial_data_cooperation/"
        "/home/dev/public-datasets/e.shvetsov/PIM/data_cooperation_21.04.25/real_data_cooperation/1TR"
        # "../../../Data/FOR_COOPERATION/"
        # "../../../Data/data_cooperation/artificial_data_cooperation/"
    )
    dataset_name: str = (
        "data_B"
        # "data_16TR_3"
        #  "data_16TR_0"
        # "16TR_C22Nc8CD_OTX_CL_E20Ne1CD_20250421_16L"
        # "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L"
        # "16TR_C22Nc8CD_OTX_CL_E20Ne1CD_20250421_1L"
    )

    log_out_dir: str = "./results"
    log_precision: int = 8
    filter_path: str = (
        "/home/dev/work_main/2025/OpenPIM/data/filter_real.mat"
        # "/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/rx_filter.mat"
        # "../../../Data/FOR_COOPERATION/rx_filter.mat"
    )

    # PIM Model Settings
    PIM_backbone: str = "leaklinpoly"
    PIM_hidden_size: int = 8
    PIM_num_layers: int = 1
    # PIM Type options: "total", "cond", "leak", "ext"
    PIM_type: str = "total"
    specific_channels = "all"
    out_filtration: bool = True

    # Training Process
    step: str = "train_pim_single"
    n_back: int = 30
    n_fwd: int = 10
    accelerator: str = "cuda"
    devices: int = 1
    re_level: str = "soft"

    # General Hyperparameters
    seed: int = 0
    loss_type: str = "l2"
    # pim_type: str = "total"
    opt_type: str = "adabound"
    batch_size: int = 512
    batch_size_eval: int = 512
    n_iterations: int = 50e3
    n_log_steps: int = 1e3
    lr_schedule: int = 1
    lr: float = 1e-4
    lr_end: float = 1e-6
    decay_factor: float = 0.001
    patience: float = 10.0
    grad_clip_val: float = 5.0
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    save_results: bool = True
    exp_name: str = "test"
    specific_channels: str = "all"

    # GMP Hyperparameters
    K: int = 4

    # Parameters for cascaded model
    ext_PIM_backbone: str = "extlinpoly"
    leak_PIM_backbone: str = "leaklinpoly"
    int_PIM_backbone: str = "intlinpoly"


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
