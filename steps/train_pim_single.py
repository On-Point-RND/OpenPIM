import models as model
from runner import Runner
from utils.util import count_net_params


def main(epx: Runner):

    epx.set_device()

    # Build Dataloaders
    (
        (train_loader, val_loader, test_loader),
        input_size,
        n_channels,
        noise,
        filter,
        CScaler,
        specs,
    ) = epx.load_resources()

    net = model.CoreModel(
        n_channels=n_channels,
        input_size=input_size,
        hidden_size=epx.args.PIM_hidden_size,
        num_layers=epx.args.PIM_num_layers,
        backbone_type=epx.args.PIM_backbone,
        batch_size=epx.args.batch_size,
    )

    n_net_pim_params = count_net_params(net)
    print("::: Number of PIM Model Parameters: ", n_net_pim_params)

    pim_model_id = epx.gen_model_id(n_net_pim_params)

    net = net.to(epx.device)
    criterion = epx.build_criterion()

    # Create Optimizer and Learning Rate Scheduler
    optimizer, lr_scheduler = epx.build_optimizer(net=net)

    train_loader = train_loader
    val_loader = val_loader
    test_loader = test_loader

    epx.train(
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
        pim_model_id=pim_model_id,
    )
