import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import lightning.pytorch as pl

from deeplearning.datasets import create_dataloader
from deeplearning.models import StockPredictTask


class Trainer(object):

    def __init__(self):
        self.train_loader = create_dataloader()

        self.model = StockPredictTask()
        self.trainer = pl.Trainer(
            max_epochs=100,
        )

    def train(self):
        self.trainer.fit(self.model,
                         train_dataloaders=self.train_loader,
                         )


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
