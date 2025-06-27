import os
import models as model
from runner import Runner
from utils.metrics import count_net_params
from modules.loggers import make_logger
from modules.train_funcs import net_eval
from modules.data_cascaded import extract_predictions

from modules.data_utils import ComplexScaler

def main(exp: Runner):

    exp.set_device()
    data = exp.load_and_split_data()

    ### First model iteration (external PIM) ###
    # TODO: runner creates directories with PIM_backbone name,
    # however in cascaded we do not use PIM_backbone config field
    # This is a temporary solution
    dir = exp.path_dir_save
    dir_without_backbone = dir.split('/')[:-1]
    dir_for_cascaded = os.path.join(*dir_without_backbone, 'cascaded_moe')
    exp.path_dir_save = dir_for_cascaded + '/first_model_ext'
    exp.args.PIM_backbone = exp.args.ext_PIM_backbone
    
    # Build Dataloaders
    
    (
        (train_loader, train_pred_loader, val_loader, test_loader),
        input_size,
        n_channels,
        noise,
        filter,
        CScaler,
        specs,
    ) = exp.prepare_dataloaders(data)

    net = model.CoreModel(
        n_channels=n_channels,
        input_size=input_size,
        out_window=exp.args.out_window,
        hidden_size=exp.args.PIM_hidden_size,
        num_layers=exp.args.PIM_num_layers,
        backbone_type=exp.args.PIM_backbone,
        batch_size=exp.args.batch_size,
        out_filtration=exp.args.out_filtration,
    )

    logger = make_logger()
    n_net_pim_params = count_net_params(net)
    logger.info(f"::: Number of External PIM Model Parameters:   {n_net_pim_params}")

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
        noise={"Train": noise["Train"], "Val": noise["Val"], "Test": noise["Test"]},
        filter=filter,
        CScaler=CScaler,
        spec_dictionary=specs,
        writer=PandasWriter,
    )

    ## Second model iteration (leakage PIM) ###

    exp.args.PIM_backbone = exp.args.leak_PIM_backbone
    
    logs = dict()
    _, pred_train, _ = net_eval(logs, net, train_pred_loader, criterion, exp.device)
    _, pred_val, _ = net_eval(logs, net, val_loader, criterion, exp.device)
    _, pred_test, _ = net_eval(logs, net, test_loader, criterion, exp.device)

    preds = {'Train': pred_train, 'Val': pred_val, 'Test': pred_test}
    data = extract_predictions(data, preds, exp.args.n_back, exp.args.n_fwd, exp.path_dir_save)

    exp.path_dir_save = dir_for_cascaded + '/second_model_int'
    (
        (train_loader, train_pred_loader, val_loader, test_loader),
        input_size,
        n_channels,
        noise,
        filter,
        CScaler,
        specs,
    ) = exp.prepare_dataloaders(data)

    net = model.CoreModel(
        n_channels=n_channels,
        input_size=input_size,
        out_window=exp.args.out_window,
        hidden_size=exp.args.PIM_hidden_size,
        num_layers=exp.args.PIM_num_layers,
        backbone_type=exp.args.PIM_backbone,
        batch_size=exp.args.batch_size,
        out_filtration=exp.args.out_filtration,
    )

    logger = make_logger()
    n_net_pim_params = count_net_params(net)
    logger.info(f"::: Number of Leakage PIM Model Parameters:   {n_net_pim_params}")

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
        noise={"Train": noise["Train"], "Val": noise["Val"], "Test": noise["Test"]},
        filter=filter,
        CScaler=CScaler,
        spec_dictionary=specs,
        writer=PandasWriter,
    )

    ## Third model iteration (internal PIM) ###

    exp.args.PIM_backbone = exp.args.int_PIM_backbone
    
    logs = dict()
    _, pred_train, _ = net_eval(logs, net, train_pred_loader, criterion, exp.device)
    _, pred_val, _ = net_eval(logs, net, val_loader, criterion, exp.device)
    _, pred_test, _ = net_eval(logs, net, test_loader, criterion, exp.device)

    preds = {'Train': pred_train, 'Val': pred_val, 'Test': pred_test}
    data = extract_predictions(data, preds, exp.args.n_back, exp.args.n_fwd, exp.path_dir_save)

    exp.path_dir_save = dir_for_cascaded + '/third_model_int'
    (
        (train_loader, train_pred_loader, val_loader, test_loader),
        input_size,
        n_channels,
        noise,
        filter,
        CScaler,
        specs,
    ) = exp.prepare_dataloaders(data)

    net = model.CoreModel(
        n_channels=n_channels,
        input_size=input_size,
        out_window=exp.args.out_window,
        hidden_size=exp.args.PIM_hidden_size,
        num_layers=exp.args.PIM_num_layers,
        backbone_type=exp.args.PIM_backbone,
        batch_size=exp.args.batch_size,
        out_filtration=exp.args.out_filtration,
    )

    logger = make_logger()
    n_net_pim_params = count_net_params(net)
    logger.info(f"::: Number of Internal PIM Model Parameters:   {n_net_pim_params}")

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
        noise={"Train": noise["Train"], "Val": noise["Val"], "Test": noise["Test"]},
        filter=filter,
        CScaler=CScaler,
        spec_dictionary=specs,
        writer=PandasWriter,
    )
    
