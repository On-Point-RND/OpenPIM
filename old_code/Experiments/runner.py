#!/usr/bin/env python
import sys
import pandas as pd

from run_experiments import run_polynomial_experiment


if __name__ == '__main__':
    experiment_i = int(sys.argv[1])
    data_path = sys.argv[2]
    if experiment_i == 0:
        RESULTS_PATH = '../Results/Polynomial_experiments/Simple/'
        train_metrics = []
        test_metrics = []
        params = []
        for experiment_n_back in range(20):
            for experiment_n_fwd in range(5):
                train_metric_value, test_metric_value = run_polynomial_experiment(experiment_mode = 'simple', experiment_degree = 0, 
                                  experiment_n_back = experiment_n_back, 
                                  experiment_n_fwd = experiment_n_fwd, experiment_n_points = 0, 
                                  experiment_name = 'simple_experiment__back_' + str(experiment_n_back) + '_fwd_' + str(experiment_n_fwd), 
                                  output_dir = RESULTS_PATH, save_results = False, data_path=data_path)
            
                train_metrics.append(train_metric_value)
                test_metrics.append(test_metric_value)
                params.append([experiment_n_back, experiment_n_fwd])
                
        pd.DataFrame({'Train_metric': train_metrics, 'Test_metric': test_metrics, 
                      'Back' : [v[0] for v in  params], 'Forward' : [v[1] for v in params]}).to_csv(RESULTS_PATH + data_path + '_simple_metrics.tsv')


    if experiment_i == 1:
        RESULTS_PATH = '../Results/Polynomial_experiments/Polynomial/'
        train_metrics = []
        test_metrics = []
        params = []
        for experiment_degree in range(10):
            for experiment_n_back in range(20):
                for experiment_n_fwd in range(20):
                    train_metric_value, test_metric_value = run_polynomial_experiment(experiment_mode = 'polynomial', 
                                  experiment_degree = experiment_degree, experiment_n_back = experiment_n_back, 
                                  experiment_n_fwd = experiment_n_fwd, experiment_n_points = 0, 
                                  experiment_name = 'polynomial_experiment__back_' + str(experiment_n_back) + '_fwd_' + str(experiment_n_fwd), 
                                  output_dir = RESULTS_PATH, save_results = False, data_path=data_path)
            
                    train_metrics.append(train_metric_value)
                    test_metrics.append(test_metric_value)
                    params.append([experiment_n_back, experiment_n_fwd, experiment_degree])
                
        pd.DataFrame({'Train_metric': train_metrics, 'Test_metric': test_metrics, 
                      'Back' : [v[0] for v in  params], 'Forward' : [v[1] for v in params], 
                     'Degree': [v[2] for v in  params]}).to_csv(RESULTS_PATH + data_path +'_polynomial_metrics.tsv')


    if experiment_i == 2:
        RESULTS_PATH = '../Results/Polynomial_experiments/Odd_polynomial/'
        train_metrics = []
        test_metrics = []
        params = []
        for experiment_degree in range(10):
            for experiment_n_back in range(20):
                for experiment_n_fwd in range(20):
                    train_metric_value, test_metric_value = run_polynomial_experiment(experiment_mode = 'odd_polynomial', 
                                  experiment_degree = experiment_degree, experiment_n_back = experiment_n_back, 
                                  experiment_n_fwd = experiment_n_fwd, experiment_n_points = 0, 
                                  experiment_name = 'odd_polynomial_experiment__back_' + str(experiment_n_back) + '_fwd_' + str(experiment_n_fwd), 
                                  output_dir = RESULTS_PATH, save_results = False, data_path=data_path)
            
                    train_metrics.append(train_metric_value)
                    test_metrics.append(test_metric_value)
                    params.append([experiment_n_back, experiment_n_fwd, experiment_degree])
                
        pd.DataFrame({'Train_metric': train_metrics, 'Test_metric': test_metrics, 
                      'Back' : [v[0] for v in  params], 'Forward' : [v[1] for v in params], 
                     'Degree': [v[2] for v in  params]}).to_csv(RESULTS_PATH + data_path +'_odd_polynomial_metrics.tsv')


    if experiment_i == 3:
        RESULTS_PATH = '../Results/Polynomial_experiments/Abs_polynomial/'
        train_metrics = []
        test_metrics = []
        params = []
        for experiment_degree in range(10):
            for experiment_n_back in range(20):
                for experiment_n_fwd in range(20):
                    train_metric_value, test_metric_value = run_polynomial_experiment(experiment_mode = 'abs_polynomial', 
                                  experiment_degree = experiment_degree, experiment_n_back = experiment_n_back, 
                                  experiment_n_fwd = experiment_n_fwd, experiment_n_points = 0, 
                                  experiment_name = 'abs_polynomial_experiment__back_' + str(experiment_n_back) + '_fwd_' + str(experiment_n_fwd), 
                                  output_dir = RESULTS_PATH, save_results = False, data_path=data_path)
            
                    train_metrics.append(train_metric_value)
                    test_metrics.append(test_metric_value)
                    params.append([experiment_n_back, experiment_n_fwd, experiment_degree])
                
        pd.DataFrame({'Train_metric': train_metrics, 'Test_metric': test_metrics, 
                      'Back' : [v[0] for v in  params], 'Forward' : [v[1] for v in params], 
                     'Degree': [v[2] for v in  params]}).to_csv(RESULTS_PATH + data_path +'_abs_polynomial_metrics.tsv')

    if experiment_i == 4:
        RESULTS_PATH = '../Results/Polynomial_experiments/Simple_polynomial/'
        train_metrics = []
        test_metrics = []
        params = []
        for experiment_degree in range(10):
            for experiment_n_back in range(20):
                for experiment_n_fwd in range(20):
                    train_metric_value, test_metric_value = run_polynomial_experiment(experiment_mode = 'simple_polynomial', 
                                  experiment_degree = experiment_degree, experiment_n_back = experiment_n_back, 
                                  experiment_n_fwd = experiment_n_fwd, experiment_n_points = 0, 
                                  experiment_name = 'simple_polynomial_experiment__back_' + str(experiment_n_back) + '_fwd_' + str(experiment_n_fwd), 
                                  output_dir = RESULTS_PATH, save_results = False, data_path=data_path)
            
                    train_metrics.append(train_metric_value)
                    test_metrics.append(test_metric_value)
                    params.append([experiment_n_back, experiment_n_fwd, experiment_degree])
                
        pd.DataFrame({'Train_metric': train_metrics, 'Test_metric': test_metrics, 
                      'Back' : [v[0] for v in  params], 'Forward' : [v[1] for v in params], 
                     'Degree': [v[2] for v in  params]}).to_csv(RESULTS_PATH + data_path +'_simple_polynomial_metrics.tsv')


    if experiment_i == 5:
        RESULTS_PATH = '../Results/Polynomial_experiments/Qasi-linear/'
        train_metrics = []
        test_metrics = []
        params = []
        for experiment_degree in range(1):
            for experiment_n_back in range(20):
                for experiment_n_fwd in range(20):
                    train_metric_value, test_metric_value = run_polynomial_experiment(experiment_mode = 'qasi-linear', 
                                  experiment_degree = experiment_degree, experiment_n_back = experiment_n_back, 
                                  experiment_n_fwd = experiment_n_fwd, experiment_n_points = 0, 
                                  experiment_name = 'qasi-linear_experiment__back_' + str(experiment_n_back) + '_fwd_' + str(experiment_n_fwd), 
                                  output_dir = RESULTS_PATH, save_results = False, data_path=data_path)
            
                    train_metrics.append(train_metric_value)
                    test_metrics.append(test_metric_value)
                    params.append([experiment_n_back, experiment_n_fwd, experiment_degree])
                
        pd.DataFrame({'Train_metric': train_metrics, 'Test_metric': test_metrics, 
                      'Back' : [v[0] for v in  params], 'Forward' : [v[1] for v in params], 
                     'Degree': [v[2] for v in  params]}).to_csv(RESULTS_PATH + data_path +'_qasi-linear_metrics.tsv')
