'''
Simulation has two functions: 
1. simulate to get path data for ETE training input
2. simulate to calculate the cost for different methods
'''

import numpy as np
import pandas as pd
import torch
import time
import os 
import pickle
import gzip
from utils.data_loader import *

class simulator:

    def __init__(self, args, initial_stock_path=True):

        self.save_data_path = args['data_save_path']
        self.sale_pivoted = pd.read_csv(args['data_path'] + f"sales_seed{args['data_seed']}.csv", index_col=0)
        self.sale_pivoted.index = pd.to_datetime(self.sale_pivoted.index)
        self.lead_pivoted = pd.read_csv(args['data_path'] + f"lead_seed{args['data_seed']}.csv", index_col=0)
        self.lead_pivoted.index = pd.to_datetime(self.lead_pivoted.index)
        self.initial_stock = pd.read_csv(args['data_path'] + f"instock_seed{args['data_seed']}.csv", index_col=0) if initial_stock_path and os.path.exists(args['data_path'] + f"instock_seed{args['data_seed']}.csv") else []
        self.dcsku_num = len(self.sale_pivoted.columns)
        self.MAX_LEAD = args['MAX_LEAD']
        self.MAX_PERISH = args['MAX_PERISH']
        self.noise_samnum = args['noise_sample_num']

        self.period = args['period']
        self.d_histlen = args['d_histlen']
        self.l_histlen = args['l_histlen']
        self.perish = args['perish']
        self.b = args['b']
        self.h = args['h']
        self.theta = args['theta']

        self.lead_date = self.lead_pivoted.index
        self.sale_date = self.sale_pivoted.index
        self.date_start, self.date_end = self.l_histlen, len(self.lead_date) - self.period
        self.using_date = self.lead_date[self.date_start:self.date_end]
        self.phase = args['phase']
        if args['phase'] == "train":
            self.measure_date = self.using_date[:int(len(self.using_date) - args['test_len'])]  # for saving the path data
        else:
            self.measure_date = self.using_date[int(len(self.using_date) - args['test_len']):]  # for measuring the performance (ordering decision is made)
        self.running_date_sale = self.sale_date[self.sale_date>=self.measure_date[0]]  # for simulation state transition (T)
        self.running_date_lead = self.lead_date[self.lead_date>=self.measure_date[0]]

        self.model_name = args['model_name']

        self.train = True if self.phase == "train" else False
        self.test = True if self.phase == "test" else False
        self.PB = True if "PB" in self.model_name else False   # include PPB, PB_tuning
        self.ETE = True if ("ETE" in self.model_name and self.ETE_SA==False) else False  # include ETE, ETE_tuning, not include ETE_SA
        self.target = True if "target" in self.model_name else False
        self.tuning = True if "tuning" in self.model_name else False

        if self.PB:
            self.beta_adjust = args['beta_adjust']
            self.dl_samnum = args['dl_sample_num']
            self.noresid = False
        if self.ETE:
            self.boosting_gamma = args['boosting_gamma']

        self.train_seed = args['train_seed']
        np.random.seed(self.train_seed)
        torch.manual_seed(self.train_seed)


    def get_D_tilde_PIL(self, instock, L_sample, D_sample):

        if len(D_sample.shape) == 1:
            D_sample = D_sample.reshape(1, -1)
        if type(L_sample) != np.ndarray:
            L_sample = np.array([[L_sample]])
        # L_sample shape: (sample_num,1), D_sample shape: (sample_num, MAX_LEAD+perish)
        # output shape: (sample_num, perish)
        assert L_sample.shape[0] == D_sample.shape[0]

        # calculating Bx: cumulative outdating
        Bx = np.zeros((D_sample.shape[0], D_sample.shape[1]))
        D_cum = np.cumsum(D_sample, axis=1)
        instock_cum = instock.cumsum().reshape(1, -1)
        for delta_s in range(D_sample.shape[1]-1):
            if delta_s == 0:
                Bx[:, 0] = np.maximum(instock_cum[:, 0] - D_cum[:, 0], 0)
            else:
                Bx[:, delta_s] = np.maximum(instock_cum[:, delta_s] - D_cum[:, delta_s], Bx[:, delta_s-1])
        
        indices = np.round(L_sample + np.arange(self.perish).reshape(1, -1)).astype(int)  # (sample_num, perish)
        # calculating \tilde{D} and PIL: from arrival point to perish point
        lead_D_cum = np.take_along_axis(D_cum, indices, axis=1)  # (sample_num, perish), same as below
        lead_D1_cum = np.take_along_axis(D_cum, indices-1, axis=1)
        lead_Bx = np.take_along_axis(Bx, indices-1, axis=1)
        D_tilde = lead_D_cum + lead_Bx - instock_cum[:, -1:]  # current demand is included in D_tilde
        pil = instock_cum[:, -1:] - lead_D1_cum - lead_Bx    # not including the current demand

        return D_tilde, pil
    

    def get_PB_order(self, instock, demand_pred, L_pred, beta_adjust):
        '''
        get the order quantity of Proportional Balance policy
        the demand_pred/L_pred distribution is using prediction model to sample, with shape (SAA_num, MAX_LEAD+perish)/(SAA_num, 2)
        the future cost is calculated using SAA by the sample demand and leadtime
        '''
        
        SAA_num = demand_pred.shape[0]  
        D_tilde_matrix, _ = self.get_D_tilde_PIL(instock, L_pred[:, :1], demand_pred)  # shape (SAA_num, perish)
        L_mean = np.mean(L_pred)

        # for unfixed length summation caculation in B_cost
        effective_lengths = self.period + L_pred[:, 1] - L_pred[:, 0]  # shape (SAA_num,)
        indices = np.arange(D_tilde_matrix.shape[1]).reshape(1, -1)  # shape (1, MAX_LEAD+perish)
        mask = indices < effective_lengths.reshape(-1, 1)  # shape (SAA_num, max_length)

        beta = 1 + beta_adjust if self.perish == 1 else ((self.perish)*self.h+self.theta)/(((2*self.perish+L_mean-2)*self.h+self.theta)) + beta_adjust
        lb, ub = 0, max(np.sum(demand_pred, axis=1))
        for i in range(1000): # bisearch the order quantity

            order_i = (lb + ub) / 2
            H_cost, O_cost, B_cost = 0, 0, 0
            
            H_cost = self.h * np.sum(np.maximum(order_i-np.maximum(D_tilde_matrix,0), 0)) / SAA_num
            O_cost = 0 if D_tilde_matrix.shape[1] < self.perish else np.sum(self.theta * np.maximum(order_i-np.maximum(D_tilde_matrix[:, -1],0),0)) / SAA_num
            D_tilde_selected = np.where(mask, D_tilde_matrix, order_i)  # effective part is those before the next arrival 
                                                                        # (as well as before perish, which naturally holds when we calculate D_tilde).
                                                                        # ineffective part set to order_i to eliminate the cost cal.
            B_cost = self.b * np.sum(np.maximum(D_tilde_selected-order_i, 0)) / SAA_num

            # print(f"iteration {i}: {beta * (H_cost + O_cost)} vs {B_cost}")

            if np.abs(beta * (H_cost + O_cost) - B_cost) <= 1.0:
                break
            elif beta * (H_cost + O_cost) > B_cost:
                ub = order_i
            else:
                lb = order_i

        PB_order = order_i

        return PB_order
    

    def initialize_path(self, T_measure):

        instock_train = np.zeros((self.dcsku_num, T_measure, self.perish+self.MAX_LEAD-1))
        tildeD_train = np.zeros((self.dcsku_num, T_measure, 1+self.noise_samnum, self.perish))
        PIL_train = np.zeros((self.dcsku_num, T_measure, 1+self.noise_samnum, self.perish))
        Lnoise_train = np.zeros((self.dcsku_num, T_measure, 1+self.noise_samnum, 2))

        return instock_train, tildeD_train, PIL_train, Lnoise_train


    def get_DL_pred(self, dataset, demand_model, lead_model, dc_sku_idx):

        # shape: (date_len, sample_num, MAX_LEAD+perish)
        demand_pred = np.maximum(demand_model.sample(dataset, dc_sku_idx, self.phase, self.dl_samnum, self.noresid) * (dataset.normalizer["D"]["max"] - dataset.normalizer["D"]["min"]) + dataset.normalizer["D"]["min"], 0)[:, :, :self.MAX_LEAD+self.perish]
        if dataset.L_max != dataset.L_min:
            lead_pred = lead_model.sample(dataset, dc_sku_idx, self.phase, self.dl_samnum, self.noresid) * (dataset.L_max - dataset.L_min) + dataset.L_min
        else:
            lead_pred = lead_model.sample(dataset, dc_sku_idx, self.phase, self.dl_samnum, self.noresid) * dataset.L_max
        # restrict the leadtime prediction to be in the range of [1, MAX_LEAD], and be integer
        lead_pred = np.round(np.clip(lead_pred, 1, self.MAX_LEAD))

        return demand_pred, lead_pred
    

    def get_DL_noise(self, D_future, Lm, Lm1, demand_model, lead_model, dc_sku_idx, dataset):

        demand_noise_cov = demand_model.cov[dc_sku_idx, :self.MAX_LEAD+self.perish, :self.MAX_LEAD+self.perish] * (dataset.normalizer["D"]["max"] - dataset.normalizer["D"]["min"])**2
        lead_noise_cov = lead_model.cov[dc_sku_idx] * (dataset.L_max - dataset.L_min)**2
        D_noise = np.maximum(D_future.reshape(1,-1) + np.random.multivariate_normal(np.zeros(len(D_future)), demand_noise_cov, size=self.noise_samnum), 0)
        L_noise = np.round(np.clip(np.array([[Lm, Lm1]]) + np.random.multivariate_normal(np.zeros(2), lead_noise_cov, size=self.noise_samnum), 1, self.MAX_LEAD))

        return D_noise, L_noise


    def simulate(self, dataset, pto_model=None, ete_model=None, test_save=False):

        T = len(self.running_date_sale)
        T_measure = len(self.measure_date)

        # initial setup: train phase to save the path data, test phase to measure the performance
        if self.train:
            assert dataset.train_set["sale"].shape[1] == T_measure  # The length of basic data should be the same as the measure date
            if self.tuning == True:  # tuning the beta parameter of PPB or boosting gamma of ETE
                TC, HC, BC, OC, SO, OD = 0, 0, 0, 0, 0, 0
                start_time = time.time()
            else:
                instock_train, tildeD_train, PIL_train, Lnoise_train = self.initialize_path(T_measure)
        else:
            assert dataset.test_set["sale"].shape[1] == T_measure
            TC, HC, BC, OC, SO, OD = 0, 0, 0, 0, 0, 0
            start_time = time.time()

        if test_save:  # save the testing on-hand inventory input data
            instock_test = np.zeros((self.dcsku_num, T_measure, self.perish+self.MAX_LEAD-1))
            pil_test = np.zeros((self.dcsku_num, T_measure, 1))

        # model input
        if self.ETE:
            order_model = ete_model
        elif self.PB:
            demand_model, lead_model = pto_model[0], pto_model[1]

        order_output_save = np.zeros((self.dcsku_num, self.period, T_measure))
        total_instock_save = np.zeros((self.dcsku_num, self.period, T_measure))
        onhand_instock_save = np.zeros((self.dcsku_num, self.period, T_measure))

        print("simulation started with method: " + self.model_name)
        for dc_sku_idx in range(self.dcsku_num):

            dc_sku = self.sale_pivoted.columns[dc_sku_idx]
            sales_dc_sku = np.array(self.sale_pivoted.loc[self.running_date_sale, dc_sku])
            leadtime_dc_sku = np.array(self.lead_pivoted.loc[self.running_date_lead, dc_sku])

            if self.PB:
                demand_pred, lead_pred = self.get_DL_pred(dataset, demand_model, lead_model, dc_sku_idx)

            # for periodical ordering, we divide each product's horizon into several shifts
            for shift in range(self.period):
                
                stock_path = np.zeros((T_measure+1, self.perish+self.MAX_LEAD-1))
                # if no input, the initial stock is set to satisfy the first leadtime demand as much as possible
                if len(self.initial_stock) == 0:
                    stock_path[0, self.perish-1] = sum(sales_dc_sku[:min(self.perish,shift+int(leadtime_dc_sku[shift]))])
                else:
                    stock_path[0, self.perish-1] = self.initial_stock.loc[dc_sku]
                
                if self.test or self.tuning:
                    total_cost, holding_cost, backorder_cost, outdating_cost, stockout, outdate  = 0, 0, 0, 0, 0, 0
                
                # assuming the max ordering period is 8, then max shift is 7
                for t in range(T_measure):

                    # system dynamics, demand comsuming in this period
                    # demand consuming
                    stock_path[t+1, :self.perish-1] = np.maximum(stock_path[t, 1:self.perish]-np.maximum(sales_dc_sku[t]-np.cumsum(stock_path[t, :self.perish-1]), 0), 0)
                    stock_path[t+1, self.perish-1] = stock_path[t, self.perish] - np.max((sales_dc_sku[t] - np.sum(stock_path[t, :self.perish]), 0))  # backorder

                    # pipeline update
                    stock_path[t+1, self.perish:self.perish+self.MAX_LEAD-2] = stock_path[t, self.perish+1:]

                    # periodic ordering
                    order_quantity = 0
                    if (t - shift) % self.period == 0:

                        if (self.train and self.tuning == False) or test_save:
                            Lm, Lm1 = int(leadtime_dc_sku[t]), int(leadtime_dc_sku[t+self.period])
                            D_future = sales_dc_sku[t:min(t+self.MAX_LEAD+self.perish, T)]
                            D_tilde_unflat, pil_unflat = self.get_D_tilde_PIL(stock_path[t], Lm, D_future)
                            D_tilde, pil_true = D_tilde_unflat.flatten(), pil_unflat.flatten()
                        
                        if self.PB:
                            order_quantity = self.get_PB_order(stock_path[t], demand_pred[t], lead_pred[t], self.beta_adjust)
                        elif self.ETE:
                            instock_fea = torch.tensor((stock_path[t:t+1, :].copy() - dataset.normalizer["D"]["min"]) / (dataset.normalizer["D"]["max"] - dataset.normalizer["D"]["min"]), dtype=torch.float32)
                            oq, pil, _, _ = order_model.determine(dataset, dc_sku_idx, self.phase, t, instock_fea)
                            oq_scale = oq * (dataset.normalizer["D"]["max"] - dataset.normalizer["D"]["min"]) + dataset.normalizer["D"]["min"]
                            if self.target:
                                pil_scale = pil * (dataset.normalizer["D"]["max"] - dataset.normalizer["D"]["min"]) + dataset.normalizer["D"]["min"]
                                order_quantity = max(max(oq_scale[0,0], 0) * self.boosting_gamma - pil_scale[0,0], 0)

                            else:
                                order_quantity = max(max(oq_scale[0,0], 0) * self.boosting_gamma, 0)
                        else:
                            raise ValueError("No such model!")

                        stock_path[t+1, self.perish + int(leadtime_dc_sku[t])-2] += order_quantity
                    
                        if self.train and self.tuning == False:
                            # get the noise data
                            D_noise, L_noise = self.get_DL_noise(D_future, Lm, Lm1, demand_model, lead_model, dc_sku_idx, dataset)
                            D_tilde_noise, pil_noise = self.get_D_tilde_PIL(stock_path[t], L_noise[:, :1], D_noise)
                            D_tilde_noise = np.concatenate((D_tilde_unflat, D_tilde_noise), axis=0)
                            pil_noise = np.concatenate((pil_unflat, pil_noise), axis=0)
                            L_noise = np.concatenate((np.array([[Lm, Lm1]]), L_noise), axis=0)
                            # save the path data
                            instock_train[dc_sku_idx, t, :] = stock_path[t, :].copy()
                            tildeD_train[dc_sku_idx, t, :, :] = D_tilde_noise.copy()
                            PIL_train[dc_sku_idx, t, :, :] = pil_noise.copy()
                            Lnoise_train[dc_sku_idx, t, :, :] = L_noise.copy()
                        if test_save:
                            pil_test[dc_sku_idx, t, 0] = pil_true[0]
                            instock_test[dc_sku_idx, t, :] = stock_path[t, :].copy()

                    order_output_save[dc_sku_idx, shift, t] = order_quantity
                    total_instock_save[dc_sku_idx, shift, t] = sum(stock_path[t])
                    onhand_instock_save[dc_sku_idx, shift, t] = sum(stock_path[t, :self.perish])
                            
                    if self.test or self.tuning:
                        # cost calculation: only calculate the cost after the first arrival (before the first arrival, all methods are the same for the same initial stock)
                        if t >= shift+int(leadtime_dc_sku[shift]):
                            unmet_demand = sales_dc_sku[t] - sum(stock_path[t, :self.perish])  # unmet demand in this period
                            holding_cost += self.h * np.maximum(-unmet_demand, 0)  # can be negative if overstock
                            backorder_cost += self.b * np.maximum(unmet_demand, 0)
                            outdating_cost += self.theta * np.maximum(stock_path[t, 0] - sales_dc_sku[t], 0)
                            if unmet_demand > 1e-10:
                                stockout += 1
                            if stock_path[t, 0] > sales_dc_sku[t] + 1e-10:
                                outdate += 1

                        if t == T_measure - 1:  # the last period
                            stockout = stockout
                            outdate = outdate
                            total_cost = (holding_cost + backorder_cost + outdating_cost)
                            TC += total_cost / (T_measure - shift - int(leadtime_dc_sku[shift]))
                            HC += holding_cost / (T_measure - shift - int(leadtime_dc_sku[shift]))
                            BC += backorder_cost / (T_measure - shift - int(leadtime_dc_sku[shift]))
                            OC += outdating_cost / (T_measure - shift - int(leadtime_dc_sku[shift]))
                            SO += stockout / (T_measure - shift - int(leadtime_dc_sku[shift]))
                            OD += outdate / (T_measure - shift - int(leadtime_dc_sku[shift]))

            if (dc_sku_idx+1) % 50 == 0: print(f"{dc_sku_idx+1} dc_sku finished!")

        print("simulation finished!")
        # np.save(self.save_data_path + f"order_output_save_{self.model_name}_period{self.period}_perish{self.perish}_b{self.b}_theta{self.theta}_seed{self.train_seed}.txt", order_output_save)
        # np.save(self.save_data_path + f"total_instock_save_{self.model_name}_period{self.period}_perish{self.perish}_b{self.b}_theta{self.theta}_seed{self.train_seed}.txt", total_instock_save)
        # np.save(self.save_data_path + f"onhand_instock_save_{self.model_name}_period{self.period}_perish{self.perish}_b{self.b}_theta{self.theta}_seed{self.train_seed}.txt", onhand_instock_save)

        if self.train and self.tuning == False:
            return instock_train, tildeD_train, PIL_train, Lnoise_train

        if self.test or self.tuning:
            TC = TC / (self.dcsku_num * self.period)
            HC = HC / (self.dcsku_num * self.period)
            BC = BC / (self.dcsku_num * self.period)
            OC = OC / (self.dcsku_num * self.period)
            SO = SO / (self.dcsku_num * self.period)
            OD = OD / (self.dcsku_num * self.period)
            end_time = time.time()
            if test_save:
                return TC, HC, BC, OC, SO, OD, end_time - start_time, instock_test, pil_test
            else:
                return TC, HC, BC, OC, SO, OD, end_time - start_time         
    

    def generate_path_data(self, instock_train, tildeD_train, PIL_train, Lnoise_train, save=False):
        
        if self.train == False:
            raise ValueError("This function is only for training phase!")
        
        if self.h != 0:
            scaled_coeff = self.h * 10  # scaling the cost coefficient to 0.1 for nn training
        else:
            scaled_coeff = 1.0

        # Shape: (dc_sku_num*T_train*(noise_sample_num+1), perish)
        tildeD_train_flat = tildeD_train.reshape(-1, self.perish)
        Lnoise_train_flat = Lnoise_train.reshape(-1, 2)
        
        Dh_train = np.maximum(tildeD_train_flat, 0)
        h_train = np.ones_like(tildeD_train_flat) * self.h / scaled_coeff
        theta_train = np.ones((tildeD_train_flat.shape[0], 1)) * self.theta / scaled_coeff

        # for unfixed length summation (caused by random leadtime) caculation in B_cost, same technique as in PB_order
        effective_lengths = self.period + Lnoise_train_flat[:, 1] - Lnoise_train_flat[:, 0]
        indices = np.arange(tildeD_train_flat.shape[1]).reshape(1, -1)
        mask = indices < effective_lengths.reshape(-1, 1)
        last_valid_indices = np.argmax(mask[:, ::-1], axis=1)
        last_valid_indices = mask.shape[1] - 1 - last_valid_indices
        last_valid_values = tildeD_train_flat[np.arange(mask.shape[0]), last_valid_indices]  # shape (dc_sku_num*T_train*(noise_sample_num+1),)
        Db_train = np.where(mask, tildeD_train_flat, last_valid_values.reshape(-1, 1))  # use the last effective element to fill the ineffective part

        mask_ahead = indices < (effective_lengths.reshape(-1, 1)-1)
        invalid_counts = np.sum(~mask, axis=1)
        invalid_b_replacement = self.b / ((1 + invalid_counts) * scaled_coeff)
        b_train = np.ones_like(tildeD_train_flat) * self.b / scaled_coeff
        b_train = np.where(mask_ahead, b_train, invalid_b_replacement.reshape(-1, 1))  # averaging the tail sequence to achieve fixed length

        # shape back
        Dh_train = Dh_train.reshape(tildeD_train.shape)
        Db_train = Db_train.reshape(tildeD_train.shape)
        h_train = h_train.reshape(tildeD_train.shape)
        b_train = b_train.reshape(tildeD_train.shape)
        theta_train = theta_train.reshape(*tildeD_train.shape[:3], 1)

        # use a dictionary to store the path data
        path_train = {"instock": instock_train, "PIL": PIL_train, "Dh": Dh_train, "Db": Db_train, "h": h_train, "b": b_train, "theta": theta_train}
        for key in path_train.keys():
            path_train[key] = torch.tensor(path_train[key], dtype=torch.float32)

        print("path data generated successfully!")

        # save the path data
        if self.save_data_path is not None and save:
            if not os.path.exists(self.save_data_path):
                os.makedirs(self.save_data_path)
            with gzip.open(self.save_data_path + f"path_data_period{self.period}_perish{self.perish}_b{self.b}_theta{self.theta}_seed{self.train_seed}.pt", "wb") as f:
                pickle.dump(path_train, f)
            # torch.save(path_train, self.save_data_path + f"path_data_period{self.period}_perish{self.perish}_b{self.b}_theta{self.theta}_seed{self.train_seed}.pt")

        return path_train