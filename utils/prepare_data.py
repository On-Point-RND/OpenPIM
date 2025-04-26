import sys, os
from scipy.io import loadmat
import pandas as pd
import json

def prepare_data(data_path = '5m', train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, n_iterations=-1):
                            
    # Ensure the ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios should sum to 1"

    if data_path == '5m':
        data = loadmat("../../../Data/FOR_COOPERATION/1TR_C20Nc1CD_E20Ne1CD_20250117_5m/1TR_C20Nc1CD_E20Ne1CD_20250117_5m.mat")
        output_dir = '../../../Data/FOR_COOPERATION/Prepared/1TR_C20Nc1CD_E20Ne1CD_20250117_5m/'
        os.makedirs(output_dir, exist_ok=True)
        
    elif data_path == '0.5m':
        data = loadmat("../../../Data/FOR_COOPERATION/1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m/1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m.mat")
        output_dir = '../../../Data/FOR_COOPERATION/Prepared/1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m/'
        os.makedirs(output_dir, exist_ok=True)
    
    elif data_path == '1L':
        data = loadmat("../../../Data/FOR_COOPERATION/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L.mat")
        output_dir = '../../../Data/FOR_COOPERATION/Prepared/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L/'
        os.makedirs(output_dir, exist_ok=True)
    
    elif data_path == '16L':
        data = loadmat("../../../Data/FOR_COOPERATION/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L.mat")
        output_dir = '../../../Data/FOR_COOPERATION/Prepared/16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L/'
        os.makedirs(output_dir, exist_ok=True)

    else:
        raise ValueError(f"Data path '{data_path}' is not supported. Please choose data path among '5m', '0.5m', '1L', '16L'.")
    
    # if not os.path.exists('../../../Data/FOR_COOPERATION/Prepared/filter.csv'):
    #     fil = loadmat("../../../Data/FOR_COOPERATION/rx_filter.mat")
    #     pd.DataFrame(fil['flt_coeff'][0], columns=['Filter_coeffs']).to_csv('../../../Data/FOR_COOPERATION/Prepared/filter.csv', index = False)
    
    fil = loadmat("../../../Data/FOR_COOPERATION/rx_filter.mat")
    pd.DataFrame(fil['flt_coeff'][0], columns=['Filter_coeffs']).to_csv(output_dir + '/filter.csv', index = False)

    rxa = data["rxa"][:n_iterations*3] # multiply by 3 for validations and paddings etc, neet to cut due to memory limits
    txa = data["txa"][:n_iterations*3]
    nfa = data["nfa"][:n_iterations*3]
        
    FC_TX = data['BANDS_DL'][0][0][0][0][0] / 10**6
    FC_RX = data['BANDS_UL'][0][0][0][0][0] / 10**6
    FS = data['Fs'][0][0] / 10**6
    PIM_SFT = data['PIM_sft'][0][0] / 10**6
    PIM_BW = data['BANDS_TX'][0][0][1][0][0] / 10**6
    PIM_total_BW = data['BANDS_TX'][0][0][3][0][0] / 10**6

    spec_dictionary = {
        "FC_TX": FC_TX,
        "FC_RX": FC_RX,
        "FS": FS, 
        "PIM_SFT": PIM_SFT,
        "PIM_BW": PIM_BW,
        "PIM_total_BW": PIM_total_BW,
        "nperseg": 1536,
    }
    json_object = json.dumps(spec_dictionary)
    with open(output_dir + "spec.json", "w") as outfile:
        outfile.write(json_object)

    total_samples = txa.shape[1]
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    # Split the datasets
    train_input = txa[:, :train_end]
    train_output = rxa[:, : train_end]
    train_noise = nfa[:, : train_end]

    val_input = txa[:, train_end:val_end]
    val_output = rxa[:, train_end:val_end]
    val_noise = nfa[:, train_end:val_end]

    test_input = txa[:, val_end:]
    test_output = rxa[:, val_end:]
    test_noise = nfa[:, val_end:]
    
    for id in range(train_input.shape[0]):
        os.makedirs(output_dir + '/CH' + str(id) + '/', exist_ok=True)
        pd.DataFrame({'I': train_input[id].real, 'Q': train_input[id].imag}).to_csv(output_dir + '/CH' + str(id) + '/' + "train_input.csv", index=False)
        pd.DataFrame({'I': train_output[id].real, 'Q': train_output[id].imag}).to_csv(output_dir + '/CH' + str(id) + '/' + "train_output.csv", index=False)
        pd.DataFrame({'I': train_noise[id].real, 'Q': train_noise[id].imag}).to_csv(output_dir + '/CH' + str(id) + '/' + "train_noise.csv", index=False)
        
        pd.DataFrame({'I': val_input[id].real, 'Q': val_input[id].imag}).to_csv(output_dir + '/CH' + str(id) + '/' + "val_input.csv", index=False)
        pd.DataFrame({'I': val_output[id].real, 'Q': val_output[id].imag}).to_csv(output_dir + '/CH' + str(id) + '/' + "val_output.csv", index=False)
        pd.DataFrame({'I': val_noise[id].real, 'Q': val_noise[id].imag}).to_csv(output_dir + '/CH' + str(id) + '/' + "val_noise.csv", index=False)
        
        pd.DataFrame({'I': test_input[id].real, 'Q': test_input[id].imag}).to_csv(output_dir + '/CH' + str(id) + '/' + "test_input.csv", index=False)
        pd.DataFrame({'I': test_output[id].real, 'Q': test_output[id].imag}).to_csv(output_dir + '/CH' + str(id) + '/' + "test_output.csv", index=False)
        pd.DataFrame({'I': test_noise[id].real, 'Q': test_noise[id].imag}).to_csv(output_dir + '/CH' + str(id) + '/' + "test_noise.csv", index=False)

    # return train_input, train_output, val_input, val_output, test_input, test_output
