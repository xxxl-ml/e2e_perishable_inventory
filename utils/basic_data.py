import numpy as np
import pandas as pd
import os
import torch
import pickle
import gzip
from scipy.stats import norm, gamma

class basic_data_parser:

    def __init__(self, args):

        self.save_data_path = args['data_save_path']
        self.sale_pivoted = pd.read_csv(args['data_path'] + f"sales_seed{args['data_seed']}.csv", index_col=0)
        self.sale_pivoted.index = pd.to_datetime(self.sale_pivoted.index)
        self.lead_pivoted = pd.read_csv(args['data_path'] + f"lead_seed{args['data_seed']}.csv", index_col=0)
        self.lead_pivoted.index = pd.to_datetime(self.lead_pivoted.index)
        if args['extra_sale_cvt']:
            self.sale_cvt = pd.read_csv(args['data_path'] + f"sale_covariate_seed{args['data_seed']}.csv", index_col=0)
            self.sale_cvt["date"] = pd.to_datetime(self.sale_cvt["date"])
            self.sale_cvtlen = len(self.sale_cvt.columns) - 2  # except for dc_sku_idx and date columns
        else:
            self.sale_cvt = None
            self.sale_cvtlen = 0
        if args['extra_lead_cvt']:
            self.lead_cvt = pd.read_csv(args['data_path'] + f"lead_covariate_seed{args['data_seed']}.csv", index_col=0)
            self.lead_cvt["date"] = pd.to_datetime(self.lead_cvt["date"])
            self.lead_cvtlen = len(self.lead_cvt.columns) - 2
        else:
            self.lead_cvt = None
            self.lead_cvtlen = 0
        self.DL_corr = args['DL_corr']

        self.dcsku_num = len(self.sale_pivoted.columns)
        self.MAX_LEAD = args['MAX_LEAD']
        self.MAX_PERISH = args['MAX_PERISH']

        self.period = args['period']
        self.l_histlen = args['l_histlen']
        self.d_histlen = args['d_histlen']
        self.cross_histlen = args['cross_histlen']
        self.perish = args['perish']
        self.b = args['b']
        self.h = args['h']
        self.theta = args['theta']

        self.lead_date = self.lead_pivoted.index
        self.sale_date = self.sale_pivoted.index
        self.date_start, self.date_end = self.l_histlen, len(self.lead_date) - self.period
        self.using_date = self.lead_date[self.date_start:self.date_end]
        self.sale_start = np.where(self.sale_date == self.using_date[0])[0][0]
        self.lead_start = np.where(self.lead_date == self.using_date[0])[0][0]

        self.data_seed = args['data_seed']

    def generate_data(self, save=False):   # basic feature data: history sales and leadtime, weekday, dc and sku indicators and basestock benchmarks

        using_len = len(self.using_date)
        if self.DL_corr:  # the demand and leadtime are correlated
            sale_data = np.zeros((self.dcsku_num, using_len, self.d_histlen, 2+self.sale_cvtlen))
            lead_data = np.zeros((self.dcsku_num, using_len, self.l_histlen, 2+self.lead_cvtlen))
        else:
            sale_data = np.zeros((self.dcsku_num, using_len, self.d_histlen, 1+self.sale_cvtlen))
            lead_data = np.zeros((self.dcsku_num, using_len, self.l_histlen, 1+self.lead_cvtlen))
        if self.cross_histlen == 1:
            cross_data = np.zeros((self.dcsku_num, using_len, (min(self.d_histlen,self.cross_histlen)+self.sale_cvtlen)*(min(self.l_histlen,self.cross_histlen)+self.lead_cvtlen)))
        else:
            cross_data = np.zeros((self.dcsku_num, using_len, (min(self.d_histlen,self.cross_histlen)+self.sale_cvtlen+7)*(min(self.l_histlen,self.cross_histlen)+self.lead_cvtlen+7)))
        oth_data = np.zeros((self.dcsku_num, using_len, 3+7))
        # oth_data = np.zeros((self.dcsku_num, using_len, 3)) 
        dc_data = np.zeros((self.dcsku_num, using_len, 1), dtype=int)
        sku_data = np.zeros((self.dcsku_num, using_len, 1), dtype=int)
        D_data = np.zeros((self.dcsku_num, using_len, self.MAX_PERISH+self.MAX_LEAD))
        L_data = np.zeros((self.dcsku_num, using_len, 2))

        print("Start generating basic data...")
        for dc_sku_idx in range(self.dcsku_num):

            dc_sku = self.sale_pivoted.columns[dc_sku_idx]
            dc_data[dc_sku_idx, :, 0] = (int(dc_sku[2:5]) - 1) * np.ones(using_len)
            sku_data[dc_sku_idx, :, 0] = (int(dc_sku[-3:]) - 1) * np.ones(using_len)

            # dc_sku data
            sale_dc_sku = np.array(self.sale_pivoted.loc[:, dc_sku])
            lead_dc_sku = np.array(self.lead_pivoted.loc[:, dc_sku])
            if self.sale_cvt is not None:
                sale_cvt_dc_sku = np.array(self.sale_cvt.loc[self.sale_cvt["dc_sku_idx"]==dc_sku, self.sale_cvt.columns[2:]])
            if self.lead_cvt is not None:
                lead_cvt_dc_sku = np.array(self.lead_cvt.loc[self.lead_cvt["dc_sku_idx"]==dc_sku, self.lead_cvt.columns[2:]])

            for t in range(len(self.using_date)):

                # history sale in recent d_histlen day, length: d_histlen
                # history leadtime in recent l_histlen day, length: l_histlen
                his_sale = sale_dc_sku[self.sale_start+t-self.d_histlen:self.sale_start+t]
                his_lead = lead_dc_sku[self.lead_start+t-self.l_histlen:self.lead_start+t]

                # feature
                if self.sale_cvt is not None:
                    sale_cvtfea = sale_cvt_dc_sku[self.sale_start+t-self.d_histlen+1:self.sale_start+t+1, :]  # today's feature can be observed
                if self.lead_cvt is not None:
                    lead_cvtfea = lead_cvt_dc_sku[self.lead_start+t-self.l_histlen+1:self.lead_start+t+1, :]

                # cross term of sale and leadtime features
                if self.cross_histlen == 1:
                    his_sale_cross = his_sale[-min(self.d_histlen,self.cross_histlen):].tolist()
                    his_lead_cross = his_lead[-min(self.l_histlen,self.cross_histlen):].tolist()
                else:
                    his_sale_cross = his_sale[-min(self.d_histlen,self.cross_histlen):].tolist() + [his_sale.max(), his_sale.min(), his_sale.mean(), np.median(his_sale), his_sale.std(), \
                                    np.count_nonzero(his_sale), (his_sale.max() - his_sale.min()) + (his_sale[1:]-his_sale[:-1]).mean() if self.d_histlen != 1 else 0]
                    his_lead_cross = his_lead[-min(self.l_histlen,self.cross_histlen):].tolist() + [his_lead.max(), his_lead.min(), his_lead.mean(), np.median(his_lead), his_lead.std(), \
                                    np.count_nonzero(his_lead), (his_lead.max() - his_lead.min()) + (his_lead[1:]-his_lead[:-1]).mean() if self.l_histlen != 1 else 0]
                if self.sale_cvt is not None:
                    his_sale_cross  = his_sale_cross + sale_cvtfea[-1, :].tolist()
                if self.lead_cvt is not None:
                    his_lead_cross  = his_lead_cross + lead_cvtfea[-1, :].tolist()
                
                # cross term feature: for order_decision ~ leadtime * demand
                his_cross = [s * l for s in his_sale_cross for l in his_lead_cross]

                # benchmark decision feature: base-stock level for i.i.d demand and ignoring age effect
                D_mean, D_std, VLT_mean, VLT_std = sale_dc_sku[:self.sale_start+t].mean(), sale_dc_sku[:self.sale_start+t].std(), lead_dc_sku[:self.lead_start+t].mean(), lead_dc_sku[:self.lead_start+t].std()
                theta = (D_std+0.00001)**2/(D_mean+0.00001)
                k = (min(self.period, self.perish) + VLT_mean) * (D_mean+0.00001)/(theta+0.00001)
                bm_norm = D_mean * (min(self.period, self.perish) + VLT_mean) + norm.ppf(self.b/(self.b+self.h+self.theta/self.perish)) * np.sqrt((min(self.period, self.perish) + VLT_mean)*(D_std**2) + (D_mean**2) * (VLT_std**2))
                bm_gamma = gamma.ppf(self.b/(self.b+self.h+self.theta/self.perish), a=k, scale=theta)
                bm_emperi = (min(self.period, self.perish) + VLT_mean) * np.quantile(sale_dc_sku, self.b/(self.b+self.h+self.theta/self.perish))

                # weekday feature
                weekday = np.zeros(7)
                weekday[self.using_date[t-self.date_start].weekday()] = 1

                # future leadtime and demand
                Lm = int(lead_dc_sku[self.lead_start+t])
                Lm1 = int(lead_dc_sku[self.lead_start+t+self.period])
                D_future = sale_dc_sku[self.sale_start+t:self.sale_start+t+self.MAX_PERISH+self.MAX_LEAD]
                
                sale_data[dc_sku_idx, t, :, 0] = his_sale.copy()
                lead_data[dc_sku_idx, t, :, 0] = his_lead.copy()
                if self.DL_corr:
                    sale_data[dc_sku_idx, t, -min(self.l_histlen, self.d_histlen):, 1] = his_lead[-min(self.l_histlen, self.d_histlen):].copy()
                    lead_data[dc_sku_idx, t, -min(self.l_histlen, self.d_histlen):, 1] = his_sale[-min(self.l_histlen, self.d_histlen):].copy()
                    if self.sale_cvt is not None:
                        sale_data[dc_sku_idx, t, :, 2:] = sale_cvtfea.copy()
                    if self.lead_cvt is not None:
                        lead_data[dc_sku_idx, t, :, 2:] = lead_cvtfea.copy()
                else:
                    if self.sale_cvt is not None:
                        sale_data[dc_sku_idx, t, :, 1:] = sale_cvtfea.copy()
                    if self.lead_cvt is not None:
                        lead_data[dc_sku_idx, t, :, 1:] = lead_cvtfea.copy()
                cross_data[dc_sku_idx, t, :] = his_cross.copy()
                oth_data[dc_sku_idx, t, :] = np.array([bm_norm, bm_gamma, bm_emperi]+weekday.tolist()).copy()
                # oth_data[dc_sku_idx, t, :] = np.array([bm_norm, bm_gamma, bm_emperi]).copy()
                L_data[dc_sku_idx, t, :] = np.array([Lm,Lm1]).copy()
                D_data[dc_sku_idx, t, :] = np.array(D_future).copy()

            if (dc_sku_idx+1) % 50 == 0:
                print(f"{dc_sku_idx+1} dc_sku finished!")

        # use a dictionary to store the basic data
        basic_data = {"sale": sale_data, "lead": lead_data, "cross": cross_data, "oth": oth_data, "dc": dc_data, "sku": sku_data, "D": D_data, "L": L_data}
        for key in basic_data.keys():
            basic_data[key] = torch.from_numpy(basic_data[key]).type(torch.float32)

        print("Basic data generated successfully!")

        # save the processed data
        if self.save_data_path is not None and save:
            if not os.path.exists(self.save_data_path):
                os.makedirs(self.save_data_path)
            with gzip.open(self.save_data_path + f"basic_data_period{self.period}_perish{self.perish}_b{self.b}_theta{self.theta}_seed{self.data_seed}.pt", "wb") as f:
                pickle.dump(basic_data, f)

        return basic_data