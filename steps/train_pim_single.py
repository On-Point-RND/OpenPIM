import models as model
from runner import Runner
from modules.metrics import count_net_params
from modules.loggers import make_logger


def main(exp: Runner):

    exp.set_device()

    # Build Dataloaders (shared across seeds)
    (
        (train_loader, val_loader, test_loader),
        n_channels,
        noise,
        filter,
        CScaler,
        specs,
    ) = exp.load_resources()

    logger = make_logger()
    base_seed = exp.args.seed
    n_seeds = int(exp.args.n_seeds)
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")

    for run_idx in range(n_seeds):
        seed = base_seed + run_idx
        primary = run_idx == 0
        exp.args.seed = seed
        exp.reproducible()
        logger.info(f"::: Seed run {run_idx + 1}/{n_seeds} (seed={seed})")

        net = model.CoreModel(
            n_channels=n_channels,
            seq_len=exp.args.seq_len,
            tx_window=exp.args.tx_window,
            rx_window=exp.args.rx_window,
            hidden_size=exp.args.PIM_hidden_size,
            backbone_type=exp.args.PIM_backbone,
            batch_size=exp.args.batch_size,
            out_filtration=exp.args.out_filtration,
            filter_path=exp.args.filter_path,
            aux_loss_present=exp.args.use_aux_loss_if_present,
        )

        n_net_pim_params = count_net_params(net)
        if primary:
            logger.info(f"::: Number of PIM Model Parameters:   {n_net_pim_params}")

        pim_model_id = exp.gen_model_id(n_net_pim_params)
        checkpoint = exp.build_logger(pim_model_id)

        net = net.to(exp.device)
        criterion = exp.build_criterion()
        optimizer, lr_scheduler = exp.build_optimizer(net=net)

        exp.train(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            noise={
                "train": noise["train"],
                "val": noise["val"],
                "test": noise["test"],
            },
            filter=filter,
            CScaler=CScaler,
            spec_dictionary=specs,
            writer=checkpoint,
            data_type=exp.args.data_type,
            primary=primary,
        )

    exp.args.seed = base_seed
