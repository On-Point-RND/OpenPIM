import models as model
from runner import Runner
from utils.metrics import count_net_params
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
        hidden_size=exp.args.PIM_hidden_size,
        num_layers=exp.args.PIM_num_layers,
        backbone_type=exp.args.PIM_backbone,
        batch_size=exp.args.batch_size,
        out_filtration=exp.args.out_filtration,
        filter_path=exp.args.filter_path,
        aux_loss_present=exp.args.use_aux_loss_if_present,
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
        noise={"train": noise["train"], "val": noise["val"], "test": noise["test"]},
        filter=filter,
        CScaler=CScaler,
        spec_dictionary=specs,
        writer=PandasWriter,
        data_type=exp.args.data_type,
    )
