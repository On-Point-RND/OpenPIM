import models as model
from runner import Runner
from utils.util import count_net_params

def main(epx: Runner):
   
    epx.set_device()
    
    # Build Dataloaders
    (train_loader, val_loader, test_loader), input_size, noise, filter, means, sd = epx.load_resources()
   
    net = model.CoreModel(input_size=input_size,
                          hidden_size=epx.args.PIM_hidden_size,
                          num_layers=epx.args.PIM_num_layers,
                          backbone_type=epx.args.PIM_backbone,
                          batch_size=epx.args.batch_size)
    n_net_pim_params = count_net_params(net)
    print("::: Number of PIM Model Parameters: ", n_net_pim_params)
    pim_model_id = epx.gen_model_id(n_net_pim_params)

    net = net.to(epx.device)
   
    epx.build_logger(model_id=pim_model_id)

    criterion = epx.build_criterion()

    # Create Optimizer and Learning Rate Scheduler
    optimizer, lr_scheduler = epx.build_optimizer(net=net)

    ###########################################################################################################
    # Training
    ###########################################################################################################
    epx.train(net=net,
               criterion=criterion,
               optimizer=optimizer,
               lr_scheduler=lr_scheduler,
               train_loader=train_loader,
               val_loader=val_loader,
               test_loader=test_loader,
               best_model_metric='Reduction_level',
               noise = noise, 
               filter = filter, 
               means = means, 
               sd = sd
              )
