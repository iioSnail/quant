# -*- coding: UTF-8 -*-

import sys
from pathlib import Path
from typing import Dict

from utils import date_utils

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

from src.statistics import CommonStatistics
from pandas import Series, DataFrame
from src.backtest import BacktestDto, backtest, multi_backtest
from src.strategy import sell_by_day, debug_buy_strategy, sell_by_drawdown, kelly_buy_ratio_strategy

stat = CommonStatistics(cache=True)

def LowPeTTM():
    """
    低估值股票
    """

    require_data = ()

    def buy_strategy(
            data: Dict[str, DataFrame],
            curr_date: str,  # 当天日期
            prev_data: Series,  # 昨日的数据
            curr_data: Series,  # 当天的数据
            dto: BacktestDto,  # 当前的持仓情况
            stock_code,
            *args,
            **kwargs,
    ) -> float:

        if dto.n_shares > 0:  # 如果该股票已经持仓，则不再买入
            return 0.

        stock_code_list = stat.low_avg_pe_ttm(date_utils.date_add(curr_date, -365), curr_date, 50)['stock_code'].tolist()

        if stock_code in stock_code_list:
            return 1 / 50
        else:
            return 0.

    sell_strategy = sell_by_day(20)
    # sell_strategy = sell_by_drawdown(0.1)

    return require_data, buy_strategy, sell_strategy


if __name__ == '__main__':
    require_data, buy, sell = LowPeTTM()

    multi_backtest(
        years='2018-2018',
        init_money=500000,  # 初始资金
        buy_strategy=buy,  # 买入策略
        # buy_strategy=debug_buy,
        sell_strategy=sell,  # 卖出策略
        # sell_strategy=debug_sell,  # 卖出策略
        require_data=require_data,
        buy_timing='open',  # 买入时机；开盘买入：open，收盘买入：close
        random_stock_list=False,
        debug_limit=-1,
    )
