from typing import Tuple

import numpy as np
from pandas import DataFrame

from data_reader.index.index_daily import IndexDailyDataReader
from src.analysis import get_results
from src.backtest import Backtest
from utils.date_utils import date_add
from utils.utils import is_null


class IndexBacktest(Backtest):
    buy_min_sample = 15  # 最少样本数。若低于该样本数，则继续扩大搜索范围
    buy_win_rate = 55  # 买入胜率。当胜率大于该值时，买入。

    sell_min_sample = 15
    sell_win_rate = 35  # 卖出胜率。当胜率低于该值时，卖出。

    data = IndexDailyDataReader("沪深300", index_prefix=False).get_data()
    data['index'] = list(range(1, len(data) + 1))
    data['open'] = data['open'] / 1000
    data['close'] = data['close'] / 1000

    def index_volume_analyse(
            self,
            vol_range,
            k_range,
            pct_range,
            pos_range,
            data,
    ):
        index_code = "沪深300"

        data = data.copy()
        data_bak = data.copy()

        # 当天成交量与近一年成交量的百分比
        amount_mean = data['amount'].rolling(250).mean()
        data['vol_pos'] = (data['amount'] - amount_mean) / amount_mean * 100
        # 当天的K线情况
        data['k_line'] = (data['close'] - data['open']) / data['open'] * 100
        # 当日上涨情况
        # data['pct_chg']

        data = data[(pos_range[0] < data['close']) & (data['close'] < pos_range[1])]
        data = data[(vol_range[0] < data['vol_pos']) & (data['vol_pos'] < vol_range[1])]
        data = data[(k_range[0] < data['k_line']) & (data['k_line'] < k_range[1])]
        data = data[(pct_range[0] < data['pct_chg']) & (data['pct_chg'] < pct_range[1])]

        # print("vol_range:", vol_range, "; k_range:", k_range, "; pct_range:", pct_range, "; pos_range:", pos_range)
        results = get_results(index_code,
                              data,
                              data_bak,
                              future_days=[1, 5, 10, 20, 40, 60],
                              print_result=False,
                              remove_continual=False,
                              remove_2015=False,
                              )

        return results

    def _read_data(self,
                   stock_code: str,
                   curr_date: str,
                   ) -> Tuple[DataFrame, DataFrame]:
        data = self.data[:curr_date]
        return data, data

    def compute_win_rate(self, data, curr_date, curr_data, min_sample):
        vol_mean = data['vol'][date_add(curr_date, -365): curr_date].mean()
        vol = (curr_data['vol'] - vol_mean) / vol_mean * 100
        k = (curr_data['close'] - curr_data['open']) / curr_data['open'] * 100

        for i in range(0, 10):
            vol_range = (vol - (10 + i * 1), vol + (15 + i * 1.2))
            k_range = (k - (0.5 + 0.1 * i), k + (0.5 + 0.1 * i))
            pct_range = (curr_data['pct_chg'] - (0.5 + 0.1 * i), curr_data['pct_chg'] + (0.5 + 0.1 * i))
            pos_range = (curr_data['close'] - (0.1 + 0.02 * i), curr_data['close'] + (0.1 + 0.02 * i))

            results = self.index_volume_analyse(vol_range, k_range, pct_range, pos_range, data)

            if is_null(results):
                continue

            total_num = results[0]['total_num']
            if total_num < min_sample:
                continue

            win_rates = []
            for result in results:
                win_rates.append(result['rise_num'] / total_num * 100)

            win_rate = np.mean(win_rates)
            return win_rate

        return 50

    def buy_strategy(self, data, curr_date, prev_data, curr_data, dto, stock_code) -> float:
        win_rate = self.compute_win_rate(data, curr_date, curr_data, self.buy_min_sample)

        if win_rate >= self.buy_win_rate:
            return 1.
        else:
            return 0.

    def sell_strategy(self, data, curr_data, dto) -> float:
        win_rate = self.compute_win_rate(data, curr_data.name, curr_data, self.sell_min_sample)

        if win_rate < self.sell_win_rate:
            return 1.
        else:
            return 0.


if __name__ == '__main__':
    IndexBacktest(
        init_money=10000,
        start_date='2017-01-01',
        days=1500,
        buy_timing='close',
        stock_code_list=['沪深300'],
    ).do()
