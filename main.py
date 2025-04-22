from steps import train_pim_single
from runner import Runner

if __name__ == "__main__":
    exp = Runner()
    train_pim_single.main(exp)
