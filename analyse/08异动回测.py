# -*- coding: UTF-8 -*-

import sys
from pathlib import Path
from typing import Dict



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

from pandas import Series, DataFrame
from src.backtest import BacktestDto, backtest, multi_backtest
from src.strategy import sell_by_day, debug_buy_strategy, sell_by_drawdown, kelly_buy_ratio_strategy


def DayangXianLowTurnover():
    """
    大阳线，低换手率
    """

    require_data = ('daily', )

    def buy_strategy(
            data: Dict[str, DataFrame],
            curr_date: str,  # 当天日期
            prev_data: Series,  # 昨日的数据
            curr_data: Series,  # 当天的数据
            dto: BacktestDto,  # 当前的持仓情况
            *args,
            **kwargs,
    ) -> float:
        if dto.n_shares > 0:  # 如果该股票已经持仓，则不再买入
            return 0.

        # 当天涨幅 >= 6%
        is_DaYangXian = (curr_data['close'] - curr_data['open']) / curr_data['open'] * 100 >= 6
        # 换手率低
        low_turnover = curr_data['turnover_rate'] <= 0.5

        if is_DaYangXian and low_turnover:
            # return 0.5 # 半仓
            return kelly_buy_ratio_strategy(0.65, 0.2, 0.1)  # 凯利公式
        else:
            return 0.

    # sell_strategy = sell_by_day(10)
    sell_strategy = sell_by_drawdown(0.1)

    return require_data, buy_strategy, sell_strategy


if __name__ == '__main__':
    require_data, buy, sell = DayangXianLowTurnover()

    # debug_buy = debug_buy_strategy()
    # debug_sell = sell_by_drawdown()

    multi_backtest(
        years='2018-2023',
        init_money=100000,  # 初始资金
        buy_strategy=buy,  # 买入策略
        # buy_strategy=debug_buy,
        sell_strategy=sell,  # 卖出策略
        # sell_strategy=debug_sell,  # 卖出策略
        require_data=require_data,
        buy_timing='open',  # 买入时机；开盘买入：open，收盘买入：close
        random_stock_list=False,
        debug_limit=-1,
    )
