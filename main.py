from steps import train_pim_single, train_pim_cascaded
from runner import Runner

if __name__ == "__main__":
    exp = Runner()
    train_pim_cascaded.main(exp)
    # train_pim_single.main(exp)
