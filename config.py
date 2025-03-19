from dataclasses import dataclass
import pyrallis

@dataclass
class Config:
    """Configuration class for PIM model training"""
    # Dataset & Log
    dataset_path: str = "/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/"
    dataset_name: str = "1TR_C20Nc1CD_E20Ne1CD_20250117_5m"
    log_out_dir: str = "./results"
    log_precision: int = 8
    filter_path: str = "/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/rx_filter.mat"

    # PIM Model Settings
    PIM_backbone: str = "linear"
    PIM_hidden_size: int = 8
    PIM_num_layers: int = 1

    # Training Process
    step: str = "train_pim"
    n_back: int = 28
    n_fwd: int = 1
    eval_val: int = 1
    eval_test: int = 1
    accelerator: str = "cpu"
    devices: int = 0
    re_level: str = "soft"
    use_segments: bool = False

    # # Feature Extraction
    # frame_length: int = 50
    # frame_stride: int = 1

    # General Hyperparameters
    seed: int = 0
    loss_type: str = "l2"
    opt_type: str = "adamw"
    batch_size: int = 32
    batch_size_eval: int = 256
    n_epochs: int = 1
    lr_schedule: int = 1
    lr: float = 1e-3
    lr_end: float = 1e-6
    decay_factor: float = 0.5
    patience: float = 10.0
    grad_clip_val: float = 200.0
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # GMP Hyperparameters
    K: int = 4

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
