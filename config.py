from dataclasses import dataclass
import pyrallis


@dataclass
class Config:
    """Configuration class for PIM model training"""

    dataset_path: str = (
        # "../../../Data/FOR_COOPERATION/"
        # "../../../Data/data_cooperation/artificial_data_cooperation/"
        # "./data/"
        # "./data/real_data/16TR/"
    )
    dataset_name: str = (
        # "data_16TR_0"
        # "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L"  # "data_16TR_0"  #   #  #
        # "16TR_C22Nc8CD_OTX_CL_E20Ne1CD_20250421_1L"
    )

    log_out_dir: str = "./results_v2"
    log_precision: int = 8
    filter_same: bool = True
    out_filtration: bool = True
    filter_path: str = (
        # "/home/dev/work_main/2025/OpenPIM/data/filter_real.mat"
        #  "../../../Data/FOR_COOPERATION/rx_filter.mat"
        # "./data/rx_filter.mat"
        # "./data/real_data/filter_real.mat"
    )

    # PIM Model Settings
    PIM_backbone: str = "cond_linear"
    PIM_hidden_size: int = 8
    PIM_num_layers: int = 1
    # PIM Type options: "total", "cond", "leak", "ext"
    PIM_type: str = "total"  # "cond"

    # Training Process
    # step options: "train_pim_single", "train_pim_cascaded"
    step: str = "train_pim_single"
    n_back: int = 68
    n_fwd: int = 10
    out_window: int = 30
    medium_sim_size: int = 5
    accelerator: str = "cuda"
    devices: int = 1
    re_level: str = "soft"

    # General Hyperparameters
    seed: int = 0
    loss_type: str = "l2"
    opt_type: str = "adam"
    batch_size: int = 64
    batch_size_eval: int = 64
    n_iterations: int = 5e4
    n_log_steps: int = 5e3
    lr_schedule: int = 1
    lr: float = 1e-4
    lr_end: float = 1e-6
    decay_factor: float = 0.001
    patience: float = 10.0
    grad_clip_val: float = 200.0
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    save_results: bool = True
    exp_name: str = "test"

    # Parameters for cascaded model
    ext_PIM_backbone: str = "ext_linear"
    leak_PIM_backbone: str = "leak_linear"
    int_PIM_backbone: str = "int_linear"


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
