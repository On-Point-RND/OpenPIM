import models as model
from runner import Runner
from utils.metrics import count_net_params
import pandas as pd
import os


def main(epx: Runner):

    epx.set_device()

    # Build Dataloaders
    (_, _, _), input_size, n_channels, _, filter, _, _ = epx.load_resources()

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

    for ch in range(n_channels):
        # path_file_pim_in = './results/save/1TR_C20Nc1CD_E20Ne1CD_20250117_5m/linear/PIM_B_128_F_1_S_0_M_LINEAR_H_8_P_34068.csv'
        path_file_pim_in = os.path.join(
            epx.path_dir_save,
            "CH_" + str(ch),
            "PIM_B_128_F_1_S_0_M_LINEAR_H_8_P_34068" + ".csv",
        )

        epx.path_dir_save = epx.path_dir_save + "/train_res"
        epx.path_dir_log_hist = epx.path_dir_log_hist + "/train_res"
        epx.path_dir_log_best = epx.path_dir_log_best + "/train_res"
        dir_paths = [epx.path_dir_save, epx.path_dir_log_hist, epx.path_dir_log_best]
        [os.makedirs(p, exist_ok=True) for p in dir_paths]

        new_data = pd.read_csv(path_file_pim_in)
        pim_model_id = (
            epx.gen_model_id(n_net_pim_params) + "_STEP_" + str(epx.args.step)
        )

        print("pim_model_id: ", pim_model_id)

        (train_loader, train_val, test_loader), noise, initial_rxa, means, sd = (
            epx.prepare_residuals(new_data)
        )

        epx.build_logger(model_id=pim_model_id)

        criterion = epx.build_criterion()

        # Create Optimizer and Learning Rate Scheduler
        optimizer, lr_scheduler = epx.build_optimizer(net=net)

        ###########################################################################################################
        # Training
        ###########################################################################################################
        epx.train(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            train_val=train_val,
            test_loader=test_loader,
            best_model_metric="Reduction_level",
            noise=noise,
            filter=filter,
            means=means,
            sd=sd,
            n_channel=ch,
            model_id=pim_model_id,
            initial_rxa=initial_rxa,
        )
