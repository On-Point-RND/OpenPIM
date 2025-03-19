import numpy as np
import matplotlib.pyplot as plt
import sys 

from scipy.signal import convolve
import random

# import optuna
# optuna.logging.set_verbosity(optuna.logging.WARNING)


def abs2(x):
    return np.array([i**2 for i in x])


def create_basis_functions(arr_x, mode, degree, n_back, n_fwd):
    """
        Build list of basis functions to predict PIM

        Args:
            arr_x: array of signals to build basis functions with
            mode: function type for modeling
            - 'simple' for x * |x|**2
            - 'polynomial' for simple polynomes
            - 'odd_polynomial' for polynomes with odd degrees
            degree: maximal degree of polynome
            n_back: number of countdowns back (for each countdown basis function is built)
            n_fwd: number of countdowns forward (for each countdown basis function is built)

        Returns:
            total_list_functions: total list of basis functions

    """
    
    j = 1j
    
    if mode == 'simple':
        list_functions = []
        for x_id in range(arr_x.shape[0]):
            x = arr_x[x_id]
            list_functions.append(x*abs(x)**2)
            
    elif mode == 'polynomial':
        list_functions = []
        for x_id in range(arr_x.shape[0]):
            x = arr_x[x_id]
            for i in range(1, (degree+1)):
                list_functions.append(x**i)
                
    elif mode == 'odd_polynomial':
        list_functions = []
        for x_id in range(arr_x.shape[0]):
            x = arr_x[x_id]
            for i in range(degree//2+1):
                list_functions.append(x**(2*i + 1))
                
    elif mode == 'abs_polynomial':
        list_functions = []
        for x_id in range(arr_x.shape[0]):
            x = arr_x[x_id]
            for i in range(1, (degree+1)):
                list_functions.append(x*abs(x)**i)
                
    elif mode == 'simple_polynomial':
        list_functions = []
        for x_id in range(arr_x.shape[0]):
            x = arr_x[x_id]
            for i in range(1, (degree+1)):
                list_functions.append((x*abs(x)**2)**i)

    elif mode == 'qasi-linear':
        frac = 10**(-3)
        list_functions = []
        for x_id in range(arr_x.shape[0]):
            x = arr_x[x_id]
            list_modules = np.abs(x).tolist()
            list_modules.sort()
            
            smallest_value = list_modules[0]
            biggest_value = list_modules[:int(len(list_modules)*(1-frac))][-1]
    
            list_thresholds = np.linspace(smallest_value, biggest_value, degree+1)
    
            
            for i in range(degree):
                function = x
                function[abs(function)<=list_thresholds[i]] = list_thresholds[i]
                function[abs(function)>=list_thresholds[i+1]] = list_thresholds[i+1]
                list_functions.append(function)
            
            
    win_len = n_back + n_fwd + 1
    n_pts = len(x) - win_len + 1
    total_list_functions = []
    for function in list_functions:
        for step in range(win_len):
            total_list_functions.append(function[step:step+n_pts])
    if mode != 'simple':
        total_list_functions.append((arr_x[0][:n_pts])**0)
    return total_list_functions


def create_model_matrix(arr_x, fil, mode = 'polynomial', degree = 10, n_back = 0, n_fwd = 0, convolved = True, exact = True):
    """
        Build model matrix to predict PIM

        Args:
            arr_x: array of signals to build basis functions with
            fil: signal filter 
            mode: function type for modeling
            - 'simple' for x * |x|**2
            - 'polynomial' for simple polynomes
            - 'odd_polynomial' for polynomes with odd degrees
            - 'abs_polynomial' for x * {polynomial of |x|}
            - 'simple_polynomial' for polynomial of x * |x|**2
            - 'qasi-linear' for qasi-linear functions
            degree: maximal degree of polynome
            n_back: number of countdowns back (for each countdown basis function is built)
            n_fwd: number of countdowns forward (for each countdown basis function is built)
            convolved: create matrix for convolved functions
            exact: find solution based on each signal sample (if no, random subset is sampled from the signal and solution is calculated based on this subsample)

        Returns:
            mmat: model matrix with basis functions
            list_idx: list with ids of sampled elements from signal x
            basis_functions_convolved: list of basis functions 

    """    
    basis_functions = create_basis_functions(arr_x, mode, degree, n_back, n_fwd)
    basis_functions_convolved = []
    
    if convolved:
        for function in basis_functions:
            basis_functions_convolved.append(convolve(function, fil['flt_coeff'].flatten()))
    else:
        for function in basis_functions:
            basis_functions_convolved.append(function)
            
    len_functions = basis_functions_convolved[0].shape[0]

    if exact:
        list_idx = [i for i in range(basis_functions_convolved[0].shape[0])]
        gen_mmat = np.zeros((len_functions,len(basis_functions_convolved)), dtype=np.complex_)
        for n in range(len(basis_functions_convolved)):
            gen_mmat[:,n] = basis_functions_convolved[n]
        mmat = gen_mmat
    else:
        sample_frac = 10**7 / (len_functions * len(basis_functions_convolved))
        list_idx = random.sample(range(len_functions), int(len_functions*sample_frac))
        list_idx.sort()
        gen_mmat = np.zeros((len(list_idx),len(basis_functions_convolved)), dtype=np.complex_)
        for n in range(len(basis_functions_convolved)):
            added_function = basis_functions_convolved[n]
            gen_mmat[:,n] = [added_function[i] for i in list_idx]
        mmat = gen_mmat
    return mmat, list_idx, basis_functions_convolved


def pim_from_regression(arr_x, arr_y, fil, mode = 'polynomial', degree = 10, n_back = 0, n_fwd = 0, verbose = False, FS = 0, 
                        FC_TX = 0, PIM_SFT = 0, PIM_total_BW = 0, arr_noise = [], convolved = True, exact = True, n_points = 0):
    """
        Model PIM based on polynomial models with solution based on pseudoinverse matrix

        Args:
            arr_x: array of signals to build basis functions with
            arr_y: array of signals before PIM filtering
            fil: signal filter 
            mode: function type for modeling
            - 'simple' for x * |x|**2
            - 'polynomial' for simple polynomes
            - 'odd_polynomial' for polynomes with odd degrees
            - 'abs_polynomial' for x * {polynomial of |x|}
            - 'simple_polynomial' for polynomial of x * |x|**2
            - 'qasi-linear' for qasi-linear functions
            degree: maximal degree of polynome
            n_back: number of countdowns back (for each countdown basis function is built)
            n_fwd: number of countdowns forward (for each countdown basis function is built)
            verbose: print calculated final loss
            FS: sampling frequency (samples per time unit)
            FC_TX: center frequency
            PIM_SFT: PIM shift
            PIM_total_BW: PIM total bandwidth
            convolved: create matrix for convolved functions
            exact: find solution based on each signal sample (if no, random subset is sampled from the signal and solution is calculated based on this subsample)
            n_points: number of signal points from initial moment of timeto be filtered from PIM

        Returns:
            total_pred_pim: list of predicted PIMs
            list_w: list of coefficients for basis functions for each PIM channel

    """        
    win_len = n_back + n_fwd + 1
    if n_points>0:
        arr_x = arr_x[:, : n_points + win_len - 1]
        arr_y = arr_y[:, : n_points + win_len - 1]
        arr_noise = arr_noise[:, : n_points + win_len - 1]
        
    n_pts = len(arr_y[0]) - win_len + 1
    arr_y = arr_y[:, n_back : n_pts + n_back]
    arr_noise = arr_noise[:, n_back : n_pts + n_back]

    A, list_idx, basis_functions_convolved = create_model_matrix(arr_x, fil, mode = mode, degree = degree, 
                                                                 n_back = n_back, n_fwd = n_fwd, convolved = convolved)

    total_pred_pim = []
    for id in range(arr_y.shape[0]):
        y = arr_y[id]
        noise = arr_noise[id]
        losses = []
        list_w = []
        if convolved:
            y_convolved = convolve(y, fil['flt_coeff'].flatten())
        else:
            y_convolved = y
            y_convolved_sampled = np.array([y_convolved[i] for i in list_idx])

        inverted = (A.conj().T @ A)
        if np.linalg.cond(inverted)>10**5:
            I = np.eye(inverted.shape[0], dtype=np.complex_)
            # psi = 10**(-11)
            psi = 10**(-10)
            w = np.linalg.inv(inverted + I*psi) @ A.conj().T @ y_convolved_sampled 
        else:
            w = np.linalg.inv(inverted) @ A.conj().T @ y_convolved_sampled 
        # pred_pim = A @ w
        list_w.append(w)
        
        pred_pim = np.zeros(basis_functions_convolved[0].shape[0], dtype=np.complex_)
        for idx in range(len(basis_functions_convolved)):
            pred_pim += basis_functions_convolved[idx]*w[idx]
            
        total_pred_pim.append(pred_pim)
    
        if verbose:
            if convolved:
                residual =  pred_pim + convolve(noise, fil['flt_coeff'].flatten() ) - convolve( y, fil['flt_coeff'].flatten() )
            else:
                residual =  convolve(pred_pim, fil['flt_coeff'].flatten() ) + convolve(noise, fil['flt_coeff'].flatten() ) - convolve( y, fil['flt_coeff'].flatten() )
                
            ax = plt.subplot(1,1,1)
            plt.close()
            
            psd_residual,f = ax.psd(residual, Fs = FS, Fc = FC_TX, NFFT = 2048, window = np.kaiser(2048,10), 
                                  noverlap = 1, pad_to = 2048, label='Residual')
            left = FC_TX + PIM_SFT - PIM_total_BW/2 
            right = FC_TX + PIM_SFT + PIM_total_BW/2
            mask = (f>left) & (f<right)
            losses.append(abs2( np.log(psd_residual[mask]) ).mean())
        if verbose:
            print( 'Final loss: ', np.mean(losses) )
    
    return total_pred_pim, list_w


def predict_pim_with_regression(arr_x, list_w, fil, mode = 'polynomial', degree = 10, n_back = 0, n_fwd = 0, convolved = True):
    """
        Predict PIM based on polynomial models with already known coefficients list_w

        Args:
            arr_x: array of signals to build basis functions with
            list_w: list of coefficients for basis functions for each PIM channel
            fil: signal filter 
            mode: function type for modeling
            - 'simple' for x * |x|**2
            - 'polynomial' for simple polynomes
            - 'odd_polynomial' for polynomes with odd degrees
            - 'abs_polynomial' for x * {polynomial of |x|}
            - 'simple_polynomial' for polynomial of x * |x|**2
            - 'qasi-linear' for qasi-linear functions
            degree: maximal degree of polynome
            n_back: number of countdowns back (for each countdown basis function is built)
            n_fwd: number of countdowns forward (for each countdown basis function is built)
            convolved: create matrix for convolved functions
        Returns:
            total_pred_pim: list of predicted PIMs

    """            
    basis_functions = create_basis_functions(arr_x, mode, degree, n_back, n_fwd)
    basis_functions_convolved = []
    total_pred_pim = []
    
    if convolved:
        for function in basis_functions:
            basis_functions_convolved.append(convolve(function, fil['flt_coeff'].flatten()))
    else:
        for function in basis_functions:
            basis_functions_convolved.append(function)

    for w in list_w:
        pred_pim = np.zeros(basis_functions_convolved[0].shape[0], dtype=np.complex_)
        for idx in range(len(basis_functions_convolved)):
            pred_pim += basis_functions_convolved[idx]*w[idx]
            
        total_pred_pim.append(pred_pim)

    return total_pred_pim
