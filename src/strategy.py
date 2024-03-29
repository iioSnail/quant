"""
回测的通用买入、卖出策略
"""
import random
from typing import Dict

from pandas import DataFrame, Series

from src.backtest import BacktestDto


def debug_buy_strategy():
    """
    用于debug的买入策略。仅仅当日上涨>=3%。保证有数据
    """

    def buy_strategy(
            data: Dict[str, DataFrame],  # 所有历史数据，例如 data['daily']为日线数据，具体参考_read_data(..)
            curr_date: str,  # 当天日期
            prev_data: Series,  # 昨日的日线数据
            curr_data: Series,  # 当天的日线数据
            dto: BacktestDto,  # 当前的持仓情况
            *args,
            **kwargs,
    ) -> float:
        # 涨幅>=6%
        if not prev_data['pct_chg'] >= 3.:
            return 0.

        return random.uniform(0.1, 0.3)  # 随机买入 0%~30%的仓位

    return buy_strategy


def common_buy_strategy(factors_range: dict,
                        buy_ratio=0.5,  # 买入比例，默认半仓
                        only_once=True,  # 是否仅买入一次。即，若某股票已经持仓，不管持仓多少，再次符合条件时，也不再买入。
                        ):
    """
    公用的买入策略

    :param factor_range: 买入策略。例如：{
                                        'pct_chg': (6, 100),
                                        'turnover_rate': (3, 100)
                                    }
                         即，上涨幅度大于等于6%，且换手率大于等于3%时，符合买入条件
    """

    def buy_strategy(
            data: Dict[str, DataFrame],
            curr_date: str,  # 当天日期
            prev_data: Series,  # 昨日的数据
            curr_data: Series,  # 当天的数据
            dto: BacktestDto,  # 当前的持仓情况
            *args,
            **kwargs,
    ) -> float:
        if only_once and dto.n_shares > 0:  # 如果该股票已经持仓，则不再买入
            return 0.

        for factor, (min_factor, max_factor) in factors_range.items():
            if min_factor > max_factor:
                raise RuntimeError(f"{factor}指标的范围({min_factor}, {max_factor})异常！")

            meet_condition = (min_factor <= curr_data[factor]) and (curr_data[factor] <= max_factor)

            if not meet_condition:
                return 0.  # 不满足买入条件

        # 满足买入条件，
        return buy_ratio

    return buy_strategy


def kelly_buy_ratio_strategy(
        p: float,  # 获胜概率
        W: float,  # 获胜时的盈利，或止盈点。例如：当上涨时预计盈利为20%，则rW=0.2
        L: float,  # 失败时的亏损，或止损点。例如：当下跌10%时止损，则rL=0.1
):
    """
    根据“凯利公式”决定买入的仓位比例

    原理请参考：https://blog.csdn.net/zhaohongfei_358/article/details/135400940
    """

    buy_ratio = (p * W - (1 - p) * L) / (W * L)

    if buy_ratio > 1.:  # 不上杠杆
        buy_ratio = 1.

    if buy_ratio < 0:
        buy_ratio = 0

    return buy_ratio


def sell_by_day(
        holding_days=5,  # 持有天数。一只股票持有这么多天后立刻卖出
):
    """
    返回按天卖出的策略函数。即买入股票后，过{holding_days}个交易日后，就卖出股票（不考虑涨跌）。

    例如：
    1. 假设我1月1日买入100股茅台股票，则1月6日卖出这100股股票。（这里假设每天都是交易日）
    2. 假设我分两次买入了200股茅台股票，但今天只有第一次买入的100股超过了5个交易日，那么这次只卖出50%。
    """

    def sell_strategy(
            data: Dict[str, DataFrame],
            curr_data: Series,
            dto: BacktestDto,
            trade_days: int,
            *args,
            **kwargs,
    ) -> float:
        # 卖出持有超过了{holding_days}的股票数量。
        return dto.get_ratio_by_trade_days(holding_days, trade_days)

    return sell_strategy


def sell_by_drawdown(
        drawdown: float = 0.1,
):
    """
    若股票从买入后的最高点(最高价)回撤{drawdown}，则卖出股票。

    例如：
    1. 我以10元每股买入了100股，该股票持续上涨，最高涨到了20元，那么当股票从20元跌到18元时（回撤10%），则卖出股票。
    2. 我以10元每股买入了100股，该股票持续下跌，最高价就是我买入的10元，那么当股票跌到9元时（回撤10%），则卖出股票。
    """

    if drawdown >= 1:
        raise RuntimeError("drawdown必须小于1")

    def sell_strategy(
            data: Dict[str, DataFrame],
            curr_data: Series,
            dto: BacktestDto,
            trade_days: int,
            *args,
            **kwargs,
    ) -> float:
        if dto.n_shares <= 0:
            return 0.

        buy_trade_day = dto.get_min_holding_trade_day()  # 该股票最早买入日期
        daily_data = data.iloc[buy_trade_day - 1:trade_days]  # 最早买入日期到今天所有的日线数据

        high = daily_data['high'].max()
        close = curr_data['close']

        if (close - high) / high <= -drawdown:
            # 回撤超过{drawdown}，则全仓卖出股票
            return 1.

        return 0.

    return sell_strategy


if __name__ == '__main__':
    # print(kelly_buy_ratio_strategy(1, 0.1, 0.1))
    print(kelly_buy_ratio_strategy(0.55, 0.1, 0.1))
    # print(kelly_buy_ratio_strategy(0.65, 0.2, 0.1))
