'''
generate synthetic data with correlated demand and random lead time
'''

import pandas as pd
import numpy as np
import os

"""configuration"""
# data save path
data_path = "data/syn_data_corr_random/"

# basic parameters
dc_num = 20
sku_num = 50
total_len = 360  # after data processing, we'll use the intermediate [30, 330] as the planning horizon

for seed in range(20):

    np.random.seed(seed)

    # loc parameter for observable features for each dc_sku pair
    com_loc = np.random.uniform(0, 1)
    dc_loc = np.random.uniform(0, 1, size=dc_num)
    sku_loc = np.random.uniform(0, 1, size=sku_num)
    dc_sku_loc = np.random.uniform(0, 1, size=(dc_num, sku_num))
    scale_factor = 0.6

    # all the features follow the AR(1) process, with a zero-mean noise also following AR(1) process
    phi_x = 0.8
    epsilon_std_x = 1 * np.sqrt(1 - phi_x**2)

    # assuming basic lead time is AR(1) process with mean 3
    phi_L = 0.8
    epsilon_std_L = 1  # 1.0
    bias_L = 3 * (1 - phi_L)

    # save loc parameters
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.savetxt(data_path + f"dc_loc_seed{seed}.txt", dc_loc)
    np.savetxt(data_path + f"sku_loc_seed{seed}.txt", sku_loc)

    # observable features for demand: common level, dc level, sku level, and dc-sku level, with AR(1) process
    com_fea = np.zeros(total_len)
    com_fea[0] = np.random.normal(com_loc, scale_factor*com_loc)
    for t in range(1, total_len):
        com_fea[t] = com_loc * (1 - phi_x) + phi_x * com_fea[t-1] + np.random.normal(0, scale_factor*com_loc*np.sqrt(1-phi_x**2))

    dc_fea = np.zeros((dc_num, total_len))
    for dc_idx in range(dc_num):
        dc_fea[dc_idx, 0] = np.random.normal(dc_loc[dc_idx], scale_factor*dc_loc[dc_idx])
        for t in range(1, total_len):
            dc_fea[dc_idx, t] = dc_loc[dc_idx] * (1 - phi_x) + phi_x * dc_fea[dc_idx, t-1] + np.random.normal(0, scale_factor*dc_loc[dc_idx]*np.sqrt(1-phi_x**2))

    sku_fea = np.zeros((sku_num, total_len))
    for sku_idx in range(sku_num):
        sku_fea[sku_idx, 0] = np.random.normal(sku_loc[sku_idx], scale_factor*sku_loc[sku_idx])
        for t in range(1, total_len):
            sku_fea[sku_idx, t] = sku_loc[sku_idx] * (1 - phi_x) + phi_x * sku_fea[sku_idx, t-1] + np.random.normal(0, scale_factor*sku_loc[sku_idx]*np.sqrt(1-phi_x**2))

    dc_sku_fea = np.zeros((dc_num, sku_num, total_len))
    for dc_idx in range(dc_num):
        for sku_idx in range(sku_num):
            dc_sku_fea[dc_idx, sku_idx, 0] = np.random.normal(dc_sku_loc[dc_idx, sku_idx], scale_factor*dc_sku_loc[dc_idx, sku_idx])
            for t in range(1, total_len):
                dc_sku_fea[dc_idx, sku_idx, t] = dc_sku_loc[dc_idx, sku_idx] * (1 - phi_x) + phi_x * dc_sku_fea[dc_idx, sku_idx, t-1] + np.random.normal(0, scale_factor*dc_sku_loc[dc_idx, sku_idx]*np.sqrt(1-phi_x**2))
    
    # unobservable noise for demand
    noise_D = np.zeros((dc_num, sku_num, total_len))
    noise_D[:, :, 0] = np.random.normal(0, 1, size=(dc_num, sku_num))
    for t in range(1, total_len):
        epsilon = np.random.normal(0, epsilon_std_x, size=(dc_num, sku_num))
        noise_D[:, :, t] = phi_x * noise_D[:, :, t-1] + epsilon
    # noise_D = np.random.normal(0, epsilon_std, size=(dc_num, sku_num, total_len))

    # basic lead time
    leadtime = np.zeros((dc_num, sku_num, total_len))
    leadtime[:, :, 0] = np.random.randint(1, 10, size=(dc_num, sku_num))
    for t in range(1, total_len):
        epsilon = np.random.normal(0, epsilon_std_L, size=(dc_num, sku_num))
        leadtime[:, :, t] = bias_L + phi_L * leadtime[:, :, t-1] + epsilon
        
    # data generating process
    date = pd.date_range(start="2020-01-01", periods=total_len, freq="D")
    demand = np.zeros((dc_num, sku_num, total_len))
    fea_records = []
    for dc_idx in range(dc_num):
        for sku_idx in range(sku_num):
            for t in range(total_len):
                demand[dc_idx, sku_idx, t] = np.exp(dc_sku_fea[dc_idx, sku_idx, t] - 0.5) + 2 * (dc_fea[dc_idx, t] + sku_fea[sku_idx, t] - 1)**2 + np.abs(com_fea[t] - 0.5) + noise_D[dc_idx, sku_idx, t]

                record = {
                    "dc_sku_idx": f"DC{str(dc_idx+1).zfill(3)}_SKU{str(sku_idx+1).zfill(3)}",
                    "date": date[t],
                    "com_fea": com_fea[t],
                    "dc_fea": dc_fea[dc_idx, t],
                    "sku_fea": sku_fea[sku_idx, t],
                    "dc_sku_fea": dc_sku_fea[dc_idx, sku_idx, t]
                }
                fea_records.append(record)

    # save the data: pivoted dataframe, index: date, columns: "DCXXX_SKUXXX"
    dc_sku = [f"DC{str(i).zfill(3)}_SKU{str(j).zfill(3)}" for i in range(1, dc_num+1) for j in range(1, sku_num+1)]
    sale_pivoted = pd.DataFrame(np.maximum(demand.reshape(-1, total_len).T, 0), index=date, columns=dc_sku)
    lead_pivoted = pd.DataFrame(np.clip(np.round(leadtime[:, :, 30:330].reshape(-1, total_len-60).T), 1, 9), index=date[30:330], columns=dc_sku)
    sale_pivoted.to_csv(data_path + f"sales_seed{seed}.csv")
    lead_pivoted.to_csv(data_path + f"lead_seed{seed}.csv")

    # save the observable features: columns: "dc_sku_idx", "date", "com_fea", "dc_fea", "sku_fea", "dc_sku_fea"
    feas_df = pd.DataFrame(fea_records, columns=["dc_sku_idx", "date", "com_fea", "dc_fea", "sku_fea", "dc_sku_fea"])
    feas_df.to_csv(data_path + f"sale_covariate_seed{seed}.csv")