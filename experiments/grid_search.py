import sys
sys.path.append('../')
import run_experiments 
from runner import Runner

for n_back in range(25, 31):
    for n_fwd in range(0, 5):
        exp = Runner()
        try:
            run_experiments.main(exp, n_back = n_back, n_fwd = n_fwd)
        except:
            pass


