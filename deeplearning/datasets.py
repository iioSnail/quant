import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.data_process import format_tushare
from utils.data_reader import TuShareDataReader
from utils.utils import load_obj, save_obj

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class DailyDataset(Dataset):

    def __init__(self,
                 window_size=240,  # 窗口大预小。用多少个交易日的数据去测未来股价
                 future_day=5,  # 预测未来多少天的股价
                 offset=120,  # 偏移量，窗口一次往前移动多少个交易日
                 regression=False,  # 回归数据。False：标签为未来第n个交易日股票相比的涨或跌，True：标签为未来第n个交易日的收盘价
                 market='all',  # 板块
                 dir_path=ROOT / 'data',  # 数据存放目录
                 use_cache=False,  # 是否使用缓存
                 ):
        super(DailyDataset, self).__init__()

        self.window_size = window_size
        self.future_day = future_day
        self.offset = offset
        self.regression = regression
        self.market = market
        self.dir_path = dir_path

        self.stock_list = self.read_stock_list()  # 所有的股票情况

        cache_dir = ROOT / 'cache'
        cache_filename = f"daily_w={window_size}_f={future_day}_o={offset}_r={'T' if regression else 'F'}_m={market}.pkl"

        if use_cache and os.path.exists(cache_dir / cache_filename):
            print("Load daily dataset from cache file.")
            self.data = load_obj(cache_dir / cache_filename)
        else:
            self.data = self.read_data()
            os.makedirs(cache_dir, exist_ok=True)
            save_obj(self.data, cache_dir / cache_filename)

    def read_stock_list(self):
        return TuShareDataReader.get_stock_list()


    def read_data(self):
        filenames = os.listdir(self.dir_path)

        data = []

        for filename in tqdm(filenames, desc="Load Dataset"):
            stock_code = filename.split(".")[0]

            stock_list = self.stock_list[self.stock_list['symbol'] == stock_code]
            if len(stock_list) != 1:
                continue

            stock = stock_list.iloc[0]

            if self.market != 'all' and stock['market'] != self.market:
                continue

            stock_data = pd.read_csv(self.dir_path / filename)
            stock_data = format_tushare(stock_data)

            # 构造数据
            for i in range(0, len(stock_data), self.offset):
                if i + self.window_size + self.future_day - 1 >= len(stock_data):
                    break

                X = stock_data.iloc[i:i + self.window_size]

                # 未来第n天的收盘价
                y = stock_data.iloc[i + self.window_size + self.future_day - 1]['close']

                X = self.process_X(X)

                if not self.regression:
                    # 窗口内最后一天的收盘价
                    X_close = stock_data.iloc[i + self.window_size - 1]['close']
                    y = y > X_close

                data.append((X, y))

                # debug code
                # if len(data) > 20:
                #     return data

        return data

    def process_X(self, X):
        """
        对数据进行预处理
        """
        X = X.copy()
        del X['ts_code']

        # 将日期改为每年的第多少天。（每个月按31天算）
        trade_date = X.index.astype(str)
        day_index = (trade_date.str[5:7].astype(int) - 1) * 31 + trade_date.str[8:10].astype(int)
        X["day_index"] = day_index
        del X['index']

        return X.to_numpy()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def default_collate_fn(batch):
    X, y = zip(*batch)

    # inputs.shape = (batch_size, day_num, feature_num)
    inputs = np.array(X)

    # y.shape = (batch_size, )。For example: [1,1,0,1,...]
    targets = np.array(y).astype(int)

    return torch.from_numpy(inputs).to(torch.float32), torch.from_numpy(targets).to(torch.float32)


def create_dataloader():
    dataset = DailyDataset(use_cache=True)

    train_loader = DataLoader(dataset,
                              batch_size=4,
                              shuffle=True,
                              collate_fn=default_collate_fn,
                              num_workers=0)

    return train_loader


if __name__ == '__main__':
    dataset = DailyDataset()
    print(dataset.__getitem__(0))
