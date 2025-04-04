from steps import train_pim_batch, train_pim_single, train_res
from runner import Runner

if __name__ == "__main__":
    exp = Runner()

    if exp.args.step == "train_pim_single":
        print(
            "####################################################################################################"
        )
        print(
            "# Step: Train PIM model                                                                            #"
        )
        print(
            "####################################################################################################"
        )
        train_pim_single.main(exp)

    elif exp.args.step == "train_pim_batch":
        print(
            "####################################################################################################"
        )
        print(
            "# Step: Run PIM model to get predictions                                                           #"
        )
        print(
            "####################################################################################################"
        )
        train_pim_batch.main(exp)

    elif exp.args.step == "train_res":
        print(
            "####################################################################################################"
        )
        print(
            "# Step: Train PIM model on residuals                                                               #"
        )
        print(
            "####################################################################################################"
        )
        train_res.main(exp)

    else:
        raise ValueError(f"The step '{exp.args.step}' is not supported.")
