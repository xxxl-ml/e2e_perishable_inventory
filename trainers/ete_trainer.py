'''
The ETE models include:
1. ETE_order_DLreg: output order quantity with regularization on demand and leadtime prediction
2. ETE_target_DLreg: output target level with regularization on demand and leadtime prediction
3. ETE_target_PILreg: output target level with regularization on PIL prediction
4. ETE_target_PILreg_noinstock: (3) with instock feature removed from the learning layers (i.e., only in PIL calculation)
5/6/7/8. same as 1/2/3/4, but loss function with noise regularization
'''

from model.ETE import *
from model.loss_func import MNVLoss
from utils.data_loader import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os


class ETE_decision:
    def __init__(self, args):

        self.hidden_size = args['hidden_size_ete']
        self.embedding_size = args['embedding_size']
        self.num_dc = args['num_dc']
        self.num_sku = args['num_sku']
        self.num_layers = args['num_layers']
        self.weight_decay = args['weight_decay']
        self.learning_rate = args['learning_rate']
        self.decay_step = args['decay_step']
        self.lr_decay = args['lr_decay']
        self.batch_size = args['batch_size']

        self.lmd_noise = args['lmd_noise']
        self.lmd_d = args['lmd_d']
        self.lmd_l = args['lmd_l']
        self.lmd_pil0 = args['lmd_pil0']
        self.lmd_pil1 = args['lmd_pil1']

        self.epochs = args['epochs']
        self.model_name = args['model_name']
        self.device = args['device']
        self.early_stop = args['early_stop']
        self.log_step = args['log_step']
        self.save_step = args['save_step']
        self.patience = args['patience']
        self.delta = args['delta']
        self.model_save_path = args['model_save_path']
        self.train_seed = args['train_seed']

        self.MAX_LEAD = args['MAX_LEAD']
        self.MAX_PERISH = args['MAX_PERISH']
        self.period = args['period']
        self.perish = args['perish']
        self.h = args['h']
        self.b = args['b']
        self.theta = args['theta']

        self.fea = ["sale", "lead", "cross", "oth", "instock", "dc", "sku"]
        self.label = ["D", "L", "PIL", "Dh", "Db", "h", "b", "theta"]

    def train(self, dataset, existing_model=None, train_ratio=1.0):

        # prepare the data and network structure
        sizes = {"salefea_size": dataset.train_set["sale"].shape[-1], "leadfea_size": dataset.train_set["lead"].shape[-1],
                 "cross_size": dataset.train_set["cross"].shape[-1], "oth_size": dataset.train_set["oth"].shape[-1],
                 "instock_size": self.perish+self.MAX_LEAD-1, "hidden_size": self.hidden_size, "output_size": 1}
        str_args = {"embedding_size": self.embedding_size, "num_dc": self.num_dc, "num_sku": self.num_sku, 
                    "num_layers": self.num_layers,"model_name": self.model_name, "MAX_LEAD": self.MAX_LEAD, "perish": self.perish}
        
        torch.manual_seed(self.train_seed)
        model = ETE(str_args, sizes, dataset.L_min, dataset.L_max)

        if existing_model is None:
            model.to(self.device)
            loss_func = MNVLoss(self.model_name, self.lmd_d, self.lmd_l, self.lmd_pil0, self.lmd_pil1)
            loss_func_val = MNVLoss(self.model_name, val=True)  # no regularization in validation
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma=self.lr_decay)

            if self.early_stop:
                train_data, val_data = dataset.get_dataloader(keys=self.fea+self.label, batch_size=self.batch_size, have_val=True, train_ratio=train_ratio)
            else:
                train_data = dataset.get_dataloader(keys=self.fea+self.label, batch_size=self.batch_size, have_val=False, train_ratio=train_ratio)

            loss_name = ["train_loss", "order_loss", "demand_loss", "lead_loss", "pil_loss0", "pil_loss1"]
            start_time = time.time()
            best_model, best_val_loss, patience = None, float("inf"), 0
            for epoch in range(self.epochs):
                model.train()
                train_loss, order_loss, demand_loss, lead_loss, pil_loss0, pil_loss1 = 0, 0, 0, 0, 0, 0

                for fea_label_batch in train_data:

                    optimizer.zero_grad()
                    out = model(*fea_label_batch[:len(self.fea)])
                    loss, o_loss, d_loss, l_loss, p_loss0, p_loss1 = loss_func(*out, *fea_label_batch[len(self.fea):])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    order_loss += o_loss.item()
                    demand_loss += d_loss.item()
                    lead_loss += l_loss.item()
                    pil_loss0 += p_loss0.item()
                    pil_loss1 += p_loss1.item()
                
                    if torch.isnan(loss):
                        raise ValueError("NaN loss!")

                scheduler.step()
                # only print those loss term larger than 0, to save space
                loss_all = [train_loss, order_loss, demand_loss, lead_loss, pil_loss0, pil_loss1]
                loss_all = [loss_name[i]+": "+str(round(loss_all[i], 4)) for i in range(len(loss_all)) if loss_all[i] > 0]
                print(f"[{self.model_name}_decision] Epoch {epoch+1}/{self.epochs}, {'; '.join(loss_all)}")

                if self.early_stop and (epoch+1) % self.log_step == 0:
                    val_loss = 0
                    model.eval()
                    for fea_label_batch in val_data:
                        with torch.no_grad():
                            val_out = model(*fea_label_batch[:len(self.fea)])
                            loss, _, _, _, _, _ = loss_func_val(*val_out, *fea_label_batch[len(self.fea):])
                            val_loss += loss.item()
                    val_loss /= len(val_data)
                    print(f"[{self.model_name}_decision] Validation loss: {val_loss}")

                    if (epoch+1) % self.save_step == 0:
                        # save the log
                        if not os.path.exists(self.model_save_path + "logs/"):
                            os.makedirs(self.model_save_path + "logs/")
                        torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss}, \
                                self.model_save_path + f"logs/{self.model_name}_period{self.period}_perish{self.perish}_b{self.b}_theta{self.theta}_seed{self.train_seed}_epoch{epoch+1}.pt")
                        
                    if val_loss < best_val_loss - self.delta:
                        best_val_loss, best_model, patience = val_loss, model.state_dict(), 0
                    else:
                        patience += 1
                        if patience == self.patience:
                            break

            if self.early_stop==False:
                best_val_loss = 0
                best_model = model.state_dict()
            model.load_state_dict(best_model)
            self.model = model.cpu()
            torch.cuda.empty_cache()
            # save the model
            if self.save_step <= self.epochs:  # not in tuning
                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)
                torch.save(best_model, self.model_save_path + f"{self.model_name}_model_period{self.period}_perish{self.perish}_b{self.b}_theta{self.theta}_seed{self.train_seed}.pt")
            end_time = time.time()
            return best_val_loss, end_time - start_time
        
        else:
            model.load_state_dict(existing_model)
            self.model = model.cpu()
            return 0, 0


    def determine(self, dataset, dc_sku_idx, phase, t, instock):

        # prepare the input data
        sale, lead, cross, oth, dc, sku = dataset.get_data(["sale", "lead", "cross", "oth", "dc", "sku"], dc_sku_idx=dc_sku_idx, phase=phase, device="cpu")       
        instock = torch.tensor(instock).reshape(1, -1).float()
        
        # model prediction
        self.model.eval()
        with torch.no_grad():
            target_o, pil_o, demand_o, lead_o = self.model(sale[t:t+1], lead[t:t+1], cross[t:t+1], oth[t:t+1], instock, dc[t:t+1], sku[t:t+1])
        
        return target_o, pil_o, demand_o, lead_o
    

    def cross_validation(self, dataset, k):

        val_loss = 0
        count = 0
        for train_data, val_data in dataset.get_k_fold(k, keys=self.fea+self.label, device=self.device):
            print(f"Fold {count+1}/{k}: training...")
            sizes = {"salefea_size": train_data[0].shape[-1], "leadfea_size": train_data[1].shape[-1],
                     "cross_size": train_data[2].shape[-1], "oth_size": train_data[3].shape[-1],
                     "instock_size": train_data[4].shape[-1], "hidden_size": self.hidden_size,
                     "output_size": 1}
            str_args = {"embedding_size": self.embedding_size, "num_dc": self.num_dc, "num_sku": self.num_sku, 
                        "num_layers": self.num_layers,"model_name": self.model_name, "MAX_LEAD": self.MAX_LEAD, "perish": self.perish}
            torch.manual_seed(self.train_seed)
            model = ETE(str_args, sizes, dataset.L_min, dataset.L_max)
            model.to(self.device)
            loss_func = MNVLoss(self.model_name, self.lmd_noise, self.lmd_d, self.lmd_l, self.lmd_pil0, self.lmd_pil1)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma=self.lr_decay)
            train_data = DataLoader(TensorDataset(*train_data), batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.epochs):
                model.train()
                train_loss = 0

                for fea_label_batch in train_data:
                    optimizer.zero_grad()
                    out = model(*fea_label_batch[:len(self.fea)])
                    loss, _, _, _, _, _ = loss_func(*out, *fea_label_batch[len(self.fea):])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                    if torch.isnan(loss):
                        raise ValueError("NaN loss!")
                    
                scheduler.step()

            model.eval()
            loss_func_val = MNVLoss(self.model_name, 0, 0, 0, 0, 0)  # no regularization in validation
            with torch.no_grad():
                val_out = model(*val_data[:len(self.fea)])
                loss, _, _, _, _, _ = loss_func_val(*val_out, *val_data[len(self.fea):])
                print(f"Fold {count+1}/{k}: validation loss: {loss.item()}")
                val_loss += loss.item()

            count += 1
            torch.cuda.empty_cache()

        return val_loss / k