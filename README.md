## Run examples:

Use default parameters from config.py: <br>
> python main.py <br>

Update default parameters from a command line: <br>

> python main.py --lr 0.0001 --batch_size 64 <br>

Update default parameters from a config file: <br>

> python main.py --config_path exp.yaml <br>


#### TODO:

- [ ]  Infinite dataloader
- [ ]  Save Image and Intermediate metrics - check how PIM outside of the signal changes
- [ ]  Try other, convolution like models
- [ ]  Rewrite linear model with dot product instead of linear layer