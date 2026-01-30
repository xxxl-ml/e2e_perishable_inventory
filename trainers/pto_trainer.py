'''
Prediction part in PTO model: demand/leadtime forcasting
'''

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model.DL_pred import DL_LSTM
from utils.data_loader import *
from torch.utils.data import DataLoader, TensorDataset
from model.loss_func import QuantileLoss
import time
import os


class PTO_predict:
    def __init__(self, args, label="D", loss="MSE"):
        
        if label == "D":
            self.label = ["D"]
            self.fea = ["sale", "dc", "sku"]
        elif label == "L":
            self.label = ["L"]
            self.fea = ["lead", "dc", "sku"]
        else:
            raise ValueError("The label should be 'D' or 'L'.")

        self.hidden_size = args['hidden_size_pto']
        self.embedding_size = args['embedding_size']
        self.num_dc = args['num_dc']
        self.num_sku = args['num_sku']
        self.num_layers = args['num_layers']
        self.weight_decay = args['weight_decay']
        self.learning_rate = args['learning_rate']
        self.decay_step = args['decay_step']
        self.lr_decay = args['lr_decay']
        self.batch_size = args['batch_size']

        self.epochs = args['epochs']
        self.device = args['device']
        self.early_stop = args['early_stop']
        self.model_save_path = args['model_save_path']
        self.train_seed = args['train_seed']
        self.log_step = args['log_step']
        self.save_step = args['save_step']
        self.patience = args['patience']
        self.delta = args['delta']

        self.period = args['period']
        self.perish = args['perish']
        self.b = args['b']
        self.theta = args['theta']
        self.MAX_LEAD = args['MAX_LEAD']
        self.MAX_PERISH = args['MAX_PERISH']

        self.model_name = args['model_name']
        if loss == "MSE":
            self.loss = nn.MSELoss()
        elif loss == "quantile":
            self.loss = QuantileLoss(args['b']/(args['b']+args['h']+args['theta']/args['perish']))
        else:
            raise ValueError("The loss function should be 'MSE' or 'quantile'.")


    def train(self, dataset, existing_model=None):

        # prepare the data and network structure, the model is trained for all dc_sku
        original_shape = dataset.train_set[self.label[0]].shape
        fea_size = dataset.train_set[self.fea[0]].shape[-1]
        if self.label[0] == "D": output_size = self.MAX_LEAD + self.MAX_PERISH
        elif self.label[0] == "L": output_size = 2
        else: output_size = 1

        torch.manual_seed(self.train_seed)
        model = DL_LSTM(fea_size, self.hidden_size, output_size, self.embedding_size, self.num_dc, self.num_sku, self.num_layers)
        
        start_time = 0
        if existing_model is None:
            model.to(self.device)
            loss_func = self.loss
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma=self.lr_decay)

            if self.early_stop:
                train_data, val_data = dataset.get_dataloader(keys=self.fea+self.label, batch_size=self.batch_size, have_val=True)
            else:
                train_data = dataset.get_dataloader(keys=self.fea+self.label, batch_size=self.batch_size, have_val=False)

            start_time = time.time()
            best_model, best_val_loss, patience = None, float("inf"), 0
            for epoch in range(self.epochs):
                model.train()
                train_loss = 0

                for fea_label_batch in train_data:
                    optimizer.zero_grad()
                    output, _ = model(*fea_label_batch[:-1])
                    loss = loss_func(output, fea_label_batch[-1])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                scheduler.step()

                print(f"[{self.model_name}_predict] Epoch {epoch+1}/{self.epochs}, Train_loss: {train_loss:.10f}")

                if self.early_stop and (epoch+1) % self.log_step == 0:
                    val_loss = 0
                    model.eval()
                    for fea_label_batch in val_data:
                        with torch.no_grad():
                            output, _ = model(*fea_label_batch[:-1])
                            loss = loss_func(output, fea_label_batch[-1])
                            val_loss += loss.item()
                    val_loss /= len(val_data)
                    print(f"[{self.model_name}_predict] Epoch {epoch+1}/{self.epochs}, Val_loss: {val_loss:.10f}")

                    if (epoch+1) % self.save_step == 0:
                        # save the log
                        if not os.path.exists(self.model_save_path + "logs/"):
                            os.makedirs(self.model_save_path + "logs/")
                            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss}, \
                                self.model_save_path + f"logs/{self.label[0]}_model_seed{self.train_seed}_epoch{epoch+1}.pt")
                        else:
                            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss}, \
                                self.model_save_path + f"logs/{self.label[0]}_ETEmodel_period{self.period}_perish{self.perish}_b{self.b}_theta{self.theta}_seed{self.train_seed}_epoch{epoch+1}.pt")
                        
                    if val_loss < best_val_loss - self.delta:
                        best_model, best_val_loss, patience = model.state_dict(), val_loss, 0
                    else:
                        patience += 1
                        if patience == self.patience:
                            break

            if self.early_stop == False:
                best_val_loss = 0
                best_model = model.state_dict()
            self.model = model.to("cpu")
            torch.cuda.empty_cache()
            # save the model
            if self.save_step <= self.epochs:  # not in tuning
                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)
                torch.save(best_model, self.model_save_path + f"{self.label[0]}_model_seed{self.train_seed}.pt")
            end_time = time.time()
        
        else:
            model.load_state_dict(existing_model)
            self.model = model.to("cpu")
            best_val_loss, end_time = 0, 0
        
        # train data fit
        model.eval()
        with torch.no_grad():
            label_fitted, _ = model(*dataset.get_data(self.fea, device="cpu"))

        self.residuals = (dataset.get_data(self.label, device="cpu")[0] - label_fitted).numpy().reshape(original_shape)
        self.mu = np.zeros((original_shape[0], original_shape[2]))
        self.cov = np.zeros((original_shape[0], original_shape[2], original_shape[2]))
        for dc_sku_idx in range(original_shape[0]):
            self.mu[dc_sku_idx] = np.mean(self.residuals[dc_sku_idx], axis=0)
            self.cov[dc_sku_idx] = np.cov(self.residuals[dc_sku_idx], rowvar=False)
        
        return best_val_loss, end_time - start_time
    

    def sample(self, dataset, dc_sku_idx, phase="train", sample_num=1, is_noresid=False):

        model = self.model
        model.eval()
        with torch.no_grad():
            label_pred, _ = model(*dataset.get_data(self.fea, dc_sku_idx=dc_sku_idx, phase=phase, device="cpu"))

        if is_noresid:
            return np.maximum(np.expand_dims(label_pred, axis=1), 0)

        residuals_sample = np.random.multivariate_normal(self.mu[dc_sku_idx], self.cov[dc_sku_idx], size=(label_pred.shape[0], sample_num))
        label_sample = np.maximum(np.expand_dims(label_pred.cpu().numpy(), axis=1).repeat(sample_num, axis=1) + residuals_sample, 0)

        return label_sample
    

    def cross_validation(self, dataset, k):  # k-fold cross validation

        val_loss = 0
        count = 0
        for train_data, val_data in dataset.get_k_fold(k, keys=self.fea+self.label, device=self.device):
            print(f"Fold {count+1}/{k}: training...")
            torch.manual_seed(self.train_seed)
            model = DL_LSTM(train_data[0].shape[-1], self.hidden_size, train_data[-1].shape[-1], self.embedding_size, self.num_dc, self.num_sku, self.num_layers)
            model.to(self.device)
            loss_func = self.loss
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma=self.lr_decay)
            train_data = DataLoader(TensorDataset(*train_data), batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.epochs):
                model.train()
                train_loss = 0

                for fea_label_batch in train_data:
                    optimizer.zero_grad()
                    output, _ = model(*fea_label_batch[:-1])
                    loss = loss_func(output, fea_label_batch[-1])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                scheduler.step()

            model.eval()
            with torch.no_grad():
                val_label_pred, _ = model(*val_data[:-1])
                val_loss += loss_func(val_label_pred, val_data[-1]).item()
            count += 1
            torch.cuda.empty_cache()

        return val_loss / k