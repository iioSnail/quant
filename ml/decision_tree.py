"""
使用决策树预测股价
"""
import os
import sys
from pathlib import Path
from typing import Callable

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn import tree
from tqdm import tqdm

from utils.data_reader import TuShareDataReader
from utils.utils import save_obj, load_obj


class DecisionTreeModel(object):

    def __init__(self, limit=-1):
        self.limit = limit

        self.model = tree.DecisionTreeClassifier()
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.get_data()

    def get_data(self):
        cache_file = ROOT / 'cache' / 'combine_data.pkl'

        key = '1UOD'
        feature_day = 1

        if os.path.exists(cache_file):
            train_data, test_data = load_obj(cache_file)
        else:
            train_data_list = []
            test_data_list = []

            stock_list = TuShareDataReader.get_stock_list()
            for stock_code in tqdm(stock_list.index, desc="Prepare Data"):
                reader = TuShareDataReader(stock_code).set_date_range('2010-01-01', '2023-12-31')
                data = reader.get_combine(require_data=('daily', 'daily_extra', 'daily_basic', 'index_daily'))
                uod = data['close'].iloc[feature_day:].array > data['close'].iloc[:-feature_day].array
                data = data.iloc[:-feature_day]
                data[key] = uod

                train_data_list.append(data.loc['2010-01-01':'2017-12-31'])
                test_data_list.append(data.loc['2018-01-01':'2023-12-31'])

                if self.limit > 0 and len(train_data_list) >= self.limit:
                    break

            train_data = pd.concat(train_data_list)
            test_data = pd.concat(test_data_list)
            save_obj((train_data, test_data), cache_file)

        features = ['change', 'pct_chg', 'RPY1', 'RPM1', 'RPM2', 'gap', 'upper_wick', 'lower_wick', 'amp', 'FAR',
                    'RAF', 'FPY1', 'FPM1', 'FPM2', 'QRR5', 'QRR20', 'turnover_rate_f', 'pe_ttm', 'circ_mv', 'index_close',
                    'index_change', 'index_pct_chg', 'index_vol']

        train_data = train_data[features + [key]].dropna()
        test_data = test_data[features + [key]].dropna()

        X_train = train_data[features]
        Y_train = train_data[key]

        X_test = test_data[features]
        Y_test = test_data[key]

        return X_train, Y_train, X_test, Y_test

    def train(self):
        self.model.fit(self.X_train, self.Y_train)

    def test(self):
        Y_predict = self.model.predict(self.X_test)
        Y_test = np.array(self.Y_test)

        acc = (Y_predict == Y_test).sum() / len(Y_predict) * 100

        print("acc:", round(acc, 2))

    def predict(self):
        pass



if __name__ == '__main__':
    model = DecisionTreeModel(limit=300)
    model.train()
    model.test()
