from steps import train_pim, run_pim
from runner import Runner

if __name__ == '__main__':
    exp = Runner()

    if exp.args.step == 'train_pim':
        print("####################################################################################################")
        print("# Step: Train PIM model                                                                            #")
        print("####################################################################################################")
        train_pim.main(exp)

    elif exp.args.step == 'run_pim':
        print("####################################################################################################")
        print("# Step: Run PIM model to get predictions                                                           #")
        print("####################################################################################################")
        run_pim.main(exp)

    else:
        raise ValueError(f"The step '{exp.step}' is not supported.")