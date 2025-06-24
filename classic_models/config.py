from dataclasses import dataclass
import pyrallis


@dataclass
class Config:
    """Configuration class for classic PIM models"""

    # Dataset Settings
    dataset_path: str = "../data/"
    dataset_name: str = "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L"
    log_out_dir: str = "../results/classic_models_long/"
    filter_path: str = "../data/rx_filter.mat"

    # Model Settings
    model: str = "utd_nlin_mult_infl_fix_pwr"
    poly: str = "cheb"
    PIM_type: str = "total"  # Options: "total", "cond", "leak", "ext"

    # Training Process
    n_back: int = 68
    n_fwd: int = 10
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
