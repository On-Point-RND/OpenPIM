from dataclasses import dataclass
import pyrallis


@dataclass
class Config:
    """Configuration class for PIM model training"""

    dataset_path: str = (
        # "./data/real_data/16TR/"
        # "./data/"
    )

    dataset_name: str = (
        # "data_A"
        # "data_16TR_0"
        # "data_32TR_A0"
        # "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L"
        # "16TR_C22Nc8CD_CL_E20Ne1CD_20250331_1L"
        # "1TR_C20Nc1CD_E20Ne1CD_20250331_0.5m"
        # "2TR"   # ./data/2TR.pt
    )

    data_type: str = (
        # 'real'
        # 'synth'
    )

    log_out_dir: str = "./results"
    log_precision: int = 8
    out_filtration: bool = False
    filter_path: str = (
        # "./data/filter_synth.mat"
        # "./data/rx_filter.mat"
        # "./data/filter_real.mat"
    )

    # PIM Model Settings
    PIM_backbone: str = "mcp"
    # Dataset structure:
    # "sequential" or "sliding"
    dataset_mode: str = "sequential"
    PIM_hidden_size: int = 8
    # PIM type options: "total", "cond", "leak", "ext"
    # For synthetic datasets with separation available
    PIM_type: str = "total"
    use_aux_loss_if_present: bool = False

    # Training Process
    step: str = "train_pim_single"
    seq_len: int = 2048
    tx_window: int = 48
    rx_window: int = 30
    accelerator: str = "cuda"
    devices: int = 0
    re_level: str = "soft"

    # General Hyperparameters
    seed: int = 0
    loss_type: str = "l2"
    opt_type: str = "adam"
    batch_size: int = 1
    batch_size_eval: int = 1
    n_iterations: int = 6e4
    n_log_steps: int = 5e3
    # Dense-then-sparse logging: first phase logs every n_log_steps_dense
    n_log_steps_dense: int = 200
    dense_phase_end_iter: int = 0  # 0 = no dense phase, use n_log_steps all the way

    # lr_scheduler_type options: "none", "rop" (reduce on plateau), "cosine"
    lr_scheduler_type : str = "none"
    n_lr_steps: int = 100

    lr: float = 1e-3
    lr_end: float = 1e-6
    decay_factor: float = 0.001
    patience: float = 10.0
    grad_clip_val: float = 200.0
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    save_results: bool = True
    # to plot spectrum plots each logging step
    plot_per_step_spectrums: bool = False
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
