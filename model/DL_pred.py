'''
LSTM models for demand/leadtime forcasting
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# model setup: DL_LSTM
class DL_LSTM(nn.Module):
    def __init__(self, fea_size, hidden_size, output_size, embedding_size, num_dc, num_sku, num_layers):
        super(DL_LSTM, self).__init__()

        # embedding the dc label and sku label
        self.embedding_size = embedding_size
        if embedding_size > 0:
            self.dc_embedding = nn.Embedding(num_dc, embedding_size)
            self.sku_embedding = nn.Embedding(num_sku, embedding_size)

        # LSTM
        self.lstm = nn.LSTM(fea_size, hidden_size, num_layers=num_layers, dropout=0.2, batch_first=True)

        # concate linear
        self.linear1 = nn.Linear(hidden_size+2*embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        self._initialize_weights()  # initialize all the parameters

    def _initialize_weights(self):
        # Initialize embedding layers
        if self.embedding_size > 0:
            nn.init.xavier_uniform_(self.dc_embedding.weight)
            nn.init.xavier_uniform_(self.sku_embedding.weight)

        for param in self.lstm.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)
        for param in self.linear1.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)
        for param in self.linear2.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)

    def forward(self, fea, dc, sku):

        # lstm layer
        lstm_out, _ = self.lstm(fea)
        lstm_out = lstm_out[:, -1, :].squeeze(1)

        # embedding layer
        if self.embedding_size > 0:
            dc_embed = self.dc_embedding(dc)
            sku_embed = self.sku_embedding(sku)
            fea_dcsku = torch.cat((lstm_out, dc_embed.squeeze(1), sku_embed.squeeze(1)), dim=1)
        else:
            fea_dcsku = lstm_out

        # concatenate linear layer
        hidden = self.linear1(fea_dcsku)
        hidden = F.relu(hidden)
        out = self.linear2(hidden)

        return out, hidden