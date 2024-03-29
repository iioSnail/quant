# -*- coding: UTF-8 -*-

import sys
from pathlib import Path
from typing import Dict

from utils.data_reader import get_require_data_list

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

from pandas import Series, DataFrame
from src.backtest import BacktestDto, backtest, multi_backtest
from src.strategy import sell_by_day, debug_buy_strategy, sell_by_drawdown, kelly_buy_ratio_strategy, \
    common_buy_strategy


def 近一年超跌():
    factors_range = {
        'FPY1': (-100, -60),
        'FAR': (-100, -6.25),
        'turnover_rate_f': (1, 7),
        'upper_wick': (0, 2.5),
        'pe_ttm': (3, 75),
    }

    multi_backtest(
        years='2018-2023',
        init_money=100000,
        buy_strategy=common_buy_strategy(factors_range, buy_ratio=0.1),
        sell_strategy=sell_by_day(20),
        require_data=get_require_data_list(factors_range.keys()),
        buy_timing='open',
        random_stock_list=False,
        debug_limit=-1,
    )


if __name__ == '__main__':
    近一年超跌()
