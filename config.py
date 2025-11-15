from dataclasses import dataclass
import pyrallis


@dataclass
class Config:
    """Configuration class for PIM model training"""

    dataset_path: str = (
        # ./data/
        # "./data/real_data/16TR/"
        # "./data/synth_data/"
    )

    dataset_name: str = (
        # "data_A"
        # "data_B"
        # "data_16TR_0"
        # "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L"
    )

    data_type: str = (
        # 'real'
        # 'synth'
    )

    log_out_dir: str = "./results"
    log_precision: int = 8
    out_filtration: bool = False
    filter_path: str = (
        "./data/rx_filter.mat"
        # "./data/filter_real.mat"
    )

    # PIM Model Settings
    PIM_backbone: str = "m_mlp"
    PIM_hidden_size: int = 8
    # PIM Type options: "total", "cond", "leak", "ext"
    PIM_type: str = "total"
    use_aux_loss_if_present: bool = False

    # Training Process
    step: str = "train_pim_single"
    n_back: int = 33
    n_fwd: int = 8
    out_window: int = 30
    accelerator: str = "cuda"
    devices: int = 2
    re_level: str = "soft"

    # General Hyperparameters
    seed: int = 0
    loss_type: str = "l2"
    opt_type: str = "adam"
    batch_size: int = 2048
    batch_size_eval: int = 2048
    n_iterations: int = 2e5
    n_log_steps: int = 5e3
    # n_lr_steps we can begin experiments from 1e3 if n_iterations is 2e5
    n_lr_steps: int = 1e3
    schedule_lr: bool = False
    # lr_scheduler_type options: "rop" (reduce on plateau), "cosine"
    lr_scheduler_type : str = "rop"
    lr: float = 1e-2
    lr_end: float = 1e-6
    decay_factor: float = 0.001
    patience: float = 10.0
    grad_clip_val: float = 200.0
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    save_results: bool = True
    exp_name: str = "test"
    load_experiment: str = '/home/dev/work_main/2025/OpenPIM/results/m_mlp/data_16TR_0/mmlp_real_for_pca/training_config.json'


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
