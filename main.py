from steps import train_pim_batch, train_pim_single, train_res
from runner import Runner

if __name__ == "__main__":
    exp = Runner()

    if exp.args.step == "train_pim_single":  # NOTE: Only this works
        train_pim_single.main(exp)

    elif exp.args.step == "train_pim_batch":
        train_pim_batch.main(exp)

    elif exp.args.step == "train_res":
        train_res.main(exp)

    else:
        raise ValueError(f"The step '{exp.args.step}' is not supported.")
