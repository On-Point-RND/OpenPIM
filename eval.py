import os
import json
import torch
import numpy as np
import pandas as pd
import models as model
from runner import Runner
from modules.loggers import make_logger
from modules.train_funcs import net_eval
from modules.data_utils import toComplex
from modules.data_utils import convert_to_serializable

from utils.metrics import (count_net_params, 
                           plot_final_spectrums,
                           plot_spectrums,
                           calculate_metrics, 
                           calculate_mean_red,
                           compute_power,
                           plot_total_perf)





def run_evaluation(net, 
                   test_loader, 
                   criterion, 
                   device,
                   filter,
                   data_type,
                   data_name,
                   CScaler,
                   FS,
                   FC_TX,
                   PIM_SFT,
                   PIM_BW,
                   noise,
                   logs,
                   step_logger,
                   path_dir_save):
    
    _, pred, gt = net_eval(logs, net, test_loader, criterion, device)
    logs = calculate_metrics(
                pred,
                gt,
                noise,
                filter,
                data_type,
                data_name,
                CScaler,
                FS,
                PIM_SFT,
                PIM_BW,
                logs,
            )
    

    mean_reduction = sum(
        logs["Reduction_level"].values()
    ) / len(logs["Reduction_level"])

    step_logger.success(
        f"Mean Reduction_level EVAL: {mean_reduction}"
    )
    step_logger.success(
        f"Reduction_level EVAL: {convert_to_serializable(logs['Reduction_level'])}"
    )


    pred = CScaler.rescale(pred, key="Y")
    gt = CScaler.rescale(gt, key="Y")

    plot_spectrums(
        toComplex(pred),
        toComplex(gt),
        FS,
        FC_TX,
        PIM_SFT,
        PIM_BW,
        0,
        logs["Reduction_level"],
        data_type,
        path_dir_save,
        cut=False,
        phase_name='EVAL',
    )

    plot_final_spectrums(
        toComplex(pred),  
        toComplex(gt), 
        toComplex(noise),
        FS,
        FC_TX,
        PIM_SFT,
        PIM_BW,
        0,
        data_type,
        path_dir_save,
        phase_name='EVAL',
    )

   


    red_levels = logs["Reduction_level"]
    
    powers = dict()
    for key, value in (("gt", gt), ("err", gt - pred), ("noise", noise)):
        compl = toComplex(value)
        powers[key] = [
            compute_power(compl[:, id], data_type, FS, PIM_SFT, PIM_BW, data_name)
            for id in range(compl.shape[1])
        ]

    mean_red_level = calculate_mean_red(list(red_levels.values()))
    max_red_level = max(red_levels.values())

    plot_total_perf(
        powers,
        max_red_level,
        mean_red_level,
        path_dir_save,
    )
    
    logs['MEAN_REDUCTION'] = mean_reduction
    return logs




if __name__ == "__main__":
    step_logger = make_logger()
    exp = Runner(load_exp=True)
    loaded_config = exp.load_experiment()
    exp.set_device()
    
    exp.args.path_dir_save = os.path.join(exp.args.path_dir_save,'evaluation')

    os.makedirs(exp.args.path_dir_save,exist_ok=True)
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
        backbone_type=exp.args.PIM_backbone,
        batch_size=exp.args.batch_size,
        out_filtration=exp.args.out_filtration,
        filter_path=exp.args.filter_path,
        aux_loss_present=exp.args.use_aux_loss_if_present,
    )

    weights = torch.load(loaded_config['path_save_file_best'], map_location='cpu')
    net.load_state_dict(weights, strict=True)
    net = net.to(exp.device)
    logger = make_logger()
    n_net_pim_params = count_net_params(net)

    logger.info(f"::: Number of PIM Model Parameters:   {n_net_pim_params}")

    pim_model_id = exp.gen_model_id(n_net_pim_params)

    PandasWriter = exp.build_logger(pim_model_id)
    
    exp.build_logger(model_id=pim_model_id)

    criterion = exp.build_criterion()

    ###########################################################################################################
    # EVALUTION
    ###########################################################################################################

    logs = {}
    results = run_evaluation(
            net=net,
            test_loader=test_loader,
            criterion=criterion,
            device=exp.device,
            filter=filter,
            data_type=exp.args.data_type,
            data_name=exp.args.dataset_name,
            CScaler=CScaler,
            FS=loaded_config['FS'],
            FC_TX=loaded_config['FC_TX'],
            PIM_SFT=loaded_config['PIM_SFT'],
            PIM_BW=loaded_config['PIM_BW'],
            noise=noise['test'],
            logs=logs,
            step_logger=step_logger,
            path_dir_save =exp.args.path_dir_save
            
        )

    results = convert_to_serializable(results)
    print(results)
    with open(exp.args.path_dir_save+'/results.json', "w") as f:
        json.dump(results, f, indent=4)
   

    