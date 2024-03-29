# -*- coding: UTF-8 -*-

import sys
from pathlib import Path
from typing import Dict

from pandas import Series, DataFrame

from src.backtest import BacktestDto, backtest, multi_backtest
from src.strategy import sell_by_day, debug_buy_strategy, sell_by_drawdown

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))


def XuRiDongSheng():
    """
    旭日东升回测
    """

    require_data = ('daily', 'daily_extra', 'cyq_extra')

    def buy_strategy(
            data: Dict[str, DataFrame],
            curr_date: str,  # 当天日期
            prev_data: Series,  # 昨日的数据
            curr_data: Series,  # 当天的数据
            dto: BacktestDto,  # 当前的持仓情况
            *args,
            **kwargs,
    ) -> float:
        if prev_data['index'] <= 1:
            return 0.

        if dto.n_shares > 0:  # 如果该股票已经持仓，则不再买入
            return 0.

        daily_data = data['daily']
        daily_extra_data = data['daily_extra']
        cyq_extra_data = data['cyq_extra']

        data_1 = daily_data[daily_data['index'] == int(prev_data['index'] - 1)].iloc[0]
        data_2 = prev_data

        data_2_extra = daily_extra_data.loc[prev_data.name]

        if prev_data.name not in cyq_extra_data.index:
            return 0.

        data_2_cyq_extra = cyq_extra_data.loc[prev_data.name]

        # 当天为中阳线或大阳线：涨跌幅>=3%
        if not data_2['pct_chg'] >= 3.:
            return 0.

        # 前一天为中阴线或大阴线：涨跌幅<=-3.%
        if not data_1['pct_chg'] <= -3.:
            return 0.

        # 当天的阳线收盘价比前一天阴线最高点还高：当日最高价>前一天最高价
        if not data_2['close'] > data_1['high']:
            return 0.

        # 股票经过长期下跌
        if not data_2_extra['RPY1'] <= 25:
            return 0.

        # ====================探索指标====================
        # if not data_2['turnover_rate'] <= 0.5:
        #     return 0.

        # 近一个月回撤
        if not data_2_extra['FPM1'] <= -20:
            return 0.

        # 开盘跳空
        if not data_2_extra['gap'] >= 0.5:
            return 0.

        if not data_2_cyq_extra['ASR'] <= 15:
            return 0.

        return 0.5  # 半仓

    # sell_strategy = sell_by_day(10)
    sell_strategy = sell_by_drawdown(0.1)

    return require_data, buy_strategy, sell_strategy


if __name__ == '__main__':
    require_data, buy, sell = XuRiDongSheng()

    # debug_buy = debug_buy_strategy()
    # debug_sell = sell_by_drawdown()

    multi_backtest(
        years='2018-2022',
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
