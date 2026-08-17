from dataclasses import dataclass
import pyrallis


@dataclass
class Config:
    """Configuration class for PIM model training"""

    dataset_path: str = (
        "./data/"
    )

    dataset_name: str = (
        # "2TR"  # ./data/2TR.pt
        # "16T16R_1_EXT"  # ./data/16TR.pt
        # "16T16R_3_EXT"  # ./data/16TR.pt
    )

    data_type: str = (
        # 'real'
        # 'synth'
    )

    log_out_dir: str = "./results"
    out_filtration: bool = False
    filter_path: str = (
        "./data/filter_synth.mat"
    )

    # PIM Model Settings
    PIM_backbone: str = "mcp"
    PIM_hidden_size: int = 8
    # PIM type options: "total", "cond", "leak", "ext"
    # For synthetic datasets with separation available
    PIM_type: str = "total"
    use_aux_loss_if_present: bool = False

    # Training Process
    step: str = "train_pim_single"
    seq_len: int = 2048
    tx_window: int = 20
    rx_window: int = 20
    accelerator: str = "cuda"
    devices: int = 0
    re_level: str = "soft"

    # General Hyperparameters
    seed: int = 0
    loss_type: str = "l2"
    opt_type: str = "adam"
    batch_size: int = 1
    batch_size_eval: int = 1
    n_iterations: int = 2e4
    n_log_steps: int = 2e3

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
    # Optional per-channel / grid PSD PNGs at each log step (data always in spectra.npz)
    plot_per_step_spectrums: bool = True
    exp_name: str = "2e4_seq2048_tx20rx20"
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
