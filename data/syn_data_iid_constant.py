'''
generate synthetic data with iid demand and constant lead time
'''

import pandas as pd
import numpy as np
import os

"""configuration"""
# data save path
data_path = "data/syn_data_iid_constant/"

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

    # all the features follow normal distribution
    epsilon_std = 1.0

    # assuming basic lead time is constant
    bias_L = 3

    # save loc parameters
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.savetxt(data_path + f"dc_loc_seed{seed}.txt", dc_loc)
    np.savetxt(data_path + f"sku_loc_seed{seed}.txt", sku_loc)

    # observable features for demand: common level, dc level, sku level, and dc-sku level, with normal distribution
    com_fea = np.random.normal(com_loc, scale_factor*com_loc, size=total_len)
    dc_fea = np.zeros((dc_num, total_len))
    for dc_idx in range(dc_num):
        dc_fea[dc_idx] = np.random.normal(dc_loc[dc_idx], scale_factor*dc_loc[dc_idx], size=total_len)
    sku_fea = np.zeros((sku_num, total_len))
    for sku_idx in range(sku_num):
        sku_fea[sku_idx] = np.random.normal(sku_loc[sku_idx], scale_factor*sku_loc[sku_idx], size=total_len)
    dc_sku_fea = np.zeros((dc_num, sku_num, total_len))
    for dc_idx in range(dc_num):
        for sku_idx in range(sku_num):
            dc_sku_fea[dc_idx, sku_idx, :] = np.random.normal(dc_sku_loc[dc_idx, sku_idx], scale_factor*dc_sku_loc[dc_idx, sku_idx], size=total_len)

    # unobservable noise for demand
    noise_D = np.random.normal(0, epsilon_std, size=(dc_num, sku_num, total_len))

    # basic lead time
    leadtime = np.ones((dc_num, sku_num, total_len)) * bias_L
        
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

    # save the observable features: columns: "dc_sku_idx", "date", "com_fea", "dc_fea", "sku_fea"
    feas_df = pd.DataFrame(fea_records, columns=["dc_sku_idx", "date", "com_fea", "dc_fea", "sku_fea", "dc_sku_fea"])
    feas_df.to_csv(data_path + f"sale_covariate_seed{seed}.csv")