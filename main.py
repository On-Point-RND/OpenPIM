from steps import train_pim_single
from runner import Runner

if __name__ == "__main__":
    exp = Runner()
    if exp.args.step == 'train_pim_single':
        train_pim_single.main(exp)
    else:
        raise ValueError(f"The step '{exp.args.step}' is not supported.")
        
    
