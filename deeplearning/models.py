import torch

import lightning.pytorch as pl
from torch import nn
from torch.nn import functional as F


class StockPredictModel(nn.Module):

    def __init__(self):
        super(StockPredictModel, self).__init__()

        self.daily_rnn = nn.LSTM(
            input_size=10,
            hidden_size=256,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )  # 日线行情的特征编码器。


        self.predict_linear = nn.Linear(256, 1)


    def forward(self, inputs):
        """"""
        """
        outputs：每一个时序输出的最后一层的hidden_state，因此shape为(batch_size, sequence_num, hidden_size)
        hidden_states: 最后一个时序输出的所有层hidden_states，因此shape为(num_layers, batch_size, hidden_size),
        cell_states: 最后一个时序输出的所有层cell_states，因此shape为(num_layers, batch_size, hidden_size)
        
        其中, outputs[:, -1, :] ==  hidden_states[-1, :, :]，因为它们都是最后一个时序的最后一层的hidden_state
        """
        outputs, (hidden_states, cell_states) = self.daily_rnn(inputs, None)

        # 最后一个时序的最后一层hidden_state，作为抽取出的特征
        daily_features = outputs[:, -1, :]

        return torch.sigmoid(self.predict_linear(daily_features))


class StockPredictTask(pl.LightningModule):

    def __init__(self):
        super(StockPredictTask, self).__init__()

        self.model = StockPredictModel()

        self.loss_fnt = nn.BCELoss()

    def forward(self, inputs):
        return self.model(inputs)

    def compute_loss(self, outputs, targets):
        return self.loss_fnt(outputs.view(-1), targets)

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)

        loss = self.compute_loss(outputs, targets)
        acc = (((outputs.view(-1) >= 0.5) == targets.bool()).sum() / len(targets)).item()

        self.log("t_loss", loss.item(), prog_bar=True)
        self.log("t_acc", acc, prog_bar=True)

        return {
            "loss": loss,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=2e-4)
