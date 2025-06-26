import models as model
from runner import Runner
from utils.util import count_net_params
from modules.loggers import make_logger


def main(exp: Runner):

    exp.set_device()

    # Build Dataloaders
    (
        (train_loader, val_loader, test_loader),
        input_size,
        n_channels,
        noise,
        filter,
        CScaler,
        specs,
    ) = exp.load_resources()

    net = model.CoreModel(
        n_channels=n_channels,
        input_size=input_size,
        out_window=exp.args.out_window,
        medium_sim_size=exp.args.medium_sim_size,
        hidden_size=exp.args.PIM_hidden_size,
        num_layers=exp.args.PIM_num_layers,
        backbone_type=exp.args.PIM_backbone,
        batch_size=exp.args.batch_size,
        out_filtration=exp.args.out_filtration,
        filter_path=exp.args.filter_path,
        filter_same=exp.args.filter_same,
    )

    logger = make_logger()
    n_net_pim_params = count_net_params(net)
    logger.info(f"::: Number of PIM Model Parameters:   {n_net_pim_params}")

    pim_model_id = exp.gen_model_id(n_net_pim_params)

    PandasWriter = exp.build_logger(pim_model_id)

    net = net.to(exp.device)
    criterion = exp.build_criterion()

    # Create Optimizer and Learning Rate Scheduler
    optimizer, lr_scheduler = exp.build_optimizer(net=net)

    train_loader = train_loader
    val_loader = val_loader
    test_loader = test_loader

    exp.train(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        best_model_metric="Reduction_level",
        noise={"Train": noise["Train"], "Val": noise["Val"], "Test": noise["Test"]},
        filter=filter,
        CScaler=CScaler,
        n_channel_id=0,
        spec_dictionary=specs,
        writer=PandasWriter,
        data_type = exp.args.data_type
    )
