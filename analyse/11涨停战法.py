# -*- coding: UTF-8 -*-

import sys
from pathlib import Path

from pandas import DataFrame, Series

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

from typing import Dict

from src.strategy import sell_by_day
from src.backtest import BacktestDto, multi_backtest
from src.analysis import AnalysisOne, analysis_A_share
from utils.data_process import add_prev_n

"""
尝试使用涨停战法，看看能不能赚钱。

涨停战法定义：
买入点：若一只股票连续两天涨停，则买入。
卖出点：若一只股票涨停板解开，即当天未成功封板，则卖出。

涨停定义：主板涨幅>=9.5%。创业板涨幅>=19%。科创板、北交所不玩。若不满足涨停定义，则认定未封板
"""


class LimitUpAnalysisOne(AnalysisOne):
    """
    | n个交易日后 | 样本总数 | 上涨样本数 | 上涨概率 | 平均涨跌幅 | 上涨时平均涨幅 | 下跌时平均跌幅 | 样本集中度 | 胜率标准差 |
|:-----------:|:--------:|:----------:|:--------:|:----------:|:--------------:|:--------------:|:----------:|:----------:|
|      5      |   192    |     90     |  46.87%  |   3.94%    |     17.98%     |     -8.46%     |   17.19%   |   26.24%   |
|      10     |   192    |     87     |  45.31%  |   3.65%    |     21.18%     |    -10.87%     |   17.19%   |   21.77%   |
|      20     |   192    |     78     |  40.62%  |   2.24%    |     26.27%     |     -14.2%     |   17.19%   |   21.86%   |
|      40     |   192    |     71     |  36.98%  |   -0.73%   |     27.02%     |    -17.45%     |   17.46%   |   24.76%   |
|      60     |   192    |     78     |  40.62%  |   1.27%    |     30.91%     |    -20.84%     |   18.13%   |   28.75%   |

    结论：小样本结果下结果不理想（多次都是这样
    """

    def filter_data(self, data: DataFrame, *args, **kwargs):
        market = self.stock_properties['market']
        limit_up = None
        if market == '主板':
            limit_up = 9.5
        else:  # todo 先不玩创业板
            return None

        data = add_prev_n(data, n=1)
        data = add_prev_n(data, n=2)
        data = add_prev_n(data, n=3)
        # 当天涨停
        data = data[data['pct_chg'] > limit_up]
        # 前一天也涨停
        data = data[data['prev_1_pct_chg'] > limit_up]
        # 但往前推2天，不是涨停
        data = data[data['prev_2_pct_chg'] < limit_up]
        # 但往前推3天，不是涨停
        data = data[data['prev_3_pct_chg'] < limit_up]

        return data


def LimitUpBacktest():
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

        c1 = curr_data['pct_chg'] >= 9.5
        c2 = prev_data['pct_chg'] >= 9.5

        if c1 and c2:
            return 0.2

        return 0.

    def sell_strategy(
            data: Dict[str, DataFrame],
            curr_data: Series,
            dto: BacktestDto,
            trade_days: int,
            *args,
            **kwargs,
    ) -> float:
        # 如果当天没有涨停，则卖出
        if curr_data['pct_chg'] < 9:
            return 1.

        return 0.

    return require_data, buy_strategy, sell_strategy


if __name__ == '__main__':
    # LimitUpAnalysisOne("000001").analysis()

    # analysis_A_share(LimitUpAnalysisOne, limit=100, limit_random=True)

    require_data, buy, sell = LimitUpBacktest()
    multi_backtest(
        years='2018-2018',
        init_money=100000,  # 初始资金
        buy_strategy=buy,  # 买入策略
        # buy_strategy=debug_buy,
        sell_strategy=sell,  # 卖出策略
        # sell_strategy=debug_sell,  # 卖出策略
        require_data=require_data,
        buy_timing='open',  # 买入时机；开盘买入：open，收盘买入：close
        random_stock_list=True,
        debug_limit=-1,
    )
