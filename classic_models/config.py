from dataclasses import dataclass
import pyrallis


@dataclass
class Config:
    """Configuration class for classic PIM models"""

    # Dataset Settings
    dataset_path: str = "/home/sandbox/datasets/syth_01_shared_data/1TR/"
    dataset_name: str = "1TR_C20Nc1CD_E20Ne1CD_20250331_0.5m"
    log_out_dir: str = "/home/sandbox/project_dir/results/classic_models"
    filter_path: str = "/home/sandbox/project_dir/data/rx_filter.mat"

    # Model Settings
    model: str = "volterra_second_order_full"
    poly: str = "cheb"
    PIM_type: str = "total"  # Options: "total", "cond", "leak", "ext"

    # Training Process
    n_back: int = 8
    n_fwd: int = 2
    accelerator: str = "cpu"
    devices: int = 0

    # General Hyperparameters
    seed: int = 0   
    train_ratio: float = 0.6
    test_ratio: float = 0.6


def main(config: Config):
    """Main function using the configuration"""
    print("Loaded configuration:")
    print(pyrallis.dump(config))

    # Example of accessing config values
    print(f"\nTraining {config.model} model on {config.dataset_name}")


if __name__ == "__main__":
    # Parse configuration from command line or config file
    config = pyrallis.parse(config_class=Config)

    # Run main function with the parsed config
    main(config) 
