# -*- coding: UTF-8 -*-

"""
回测模块：当发现好的买入卖出时机时，需要用近几年的数据进行模拟，看看收益率是多少。
        主要用于对买入卖出时机的模拟交易验证。
"""

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import copy
import math
import random
import re
from typing import Dict, Tuple

from pandas import DataFrame, Series
from tqdm import tqdm

import utils.data_reader
from src.analysis import analysis_index
from utils.data_process import get_all_china_stock_code_list
from utils.data_reader import TuShareDataReader, is_trading_day
from utils.date_utils import get_future_date_list, date_diff, get_now
from utils.log_utils import get_fprint, table_print

log_filename = "backtest_" + get_now(format='%Y%m%d_%H%M%S')
fprint = get_fprint(log_filename)

print(f"日志文件:{log_filename}.log", )


class BacktestDto(object):
    """
    记录回测过程中某只股票的当前持仓情况
    """

    def __init__(self, stock_code: str):
        self.stock_code = stock_code

        self.n_shares = 0  # number of shares 持股数量
        self.cost_price = 0.  # 平均成本价。若只买入一次，则为买入价格，若买入多次，则为加权平均成本价格。即 总花费的钱/总持股数

        # 用于记录持有天数。key为买入日期（在哪个交易日买入的），value为该交易买入的股票数量（当卖出后，该值会减少）
        self.holding_days: Dict[int, int] = {}

    def buy(self,
            price: float,  # 当前价格
            number: int,  # 买入数量
            trade_days: int,  # 当天是第几个交易日。可以理解为交易日期
            ):
        """
        买入股票
        :return: 花了多少钱
        """
        if number <= 0:
            return 0.

        cost = self.n_shares * self.cost_price  # 计算之前花费的总金额
        self.n_shares += number  # 本次买入的股数

        curr_cost = price * number  # 本次买入花费的金额

        self.cost_price = (cost + curr_cost) / self.n_shares  # 更新平均成本价

        fprint(f"买入 {self.stock_code} 股票{number}股，每股{price}元，共花费{curr_cost}元")

        # 记录本次买入股票的时间和数量
        self.holding_days[trade_days] = self.holding_days.get(trade_days, 0) + number

        return curr_cost  # 返回本次花的钱

    def sell(self, price: float, number: int):
        """
        卖出股票
        :param price: 当前股价
        :param number: 卖出数量
        :return: 卖了多少钱
        """
        if number <= 0:
            return 0.

        if self.n_shares - number < 0:
            raise RuntimeError(f"n_shares小于0, n_share={self.n_shares}, number={number}")

        sell_ratio = number / self.n_shares * 100
        profit_rate = (price - self.cost_price) / self.cost_price * 100

        self.n_shares -= number

        earning = price * number
        earning = earning * (1 - 0.001)  # 要收印花税，印花税千分之一

        fprint(f"卖出 {self.stock_code} 股票{number}股（{sell_ratio}%仓位），每股{price}元（盈利{profit_rate}%），收入{earning}元")

        # 修改self.holding_days，优先卖出最早买入的股票
        sell_num = number
        holding_keys = list(self.holding_days.keys())
        holding_keys.sort()
        for key in holding_keys:
            holding_num = self.holding_days[key]
            if holding_num <= 0:
                del self.holding_days[key]
                continue

            if sell_num <= 0:
                break

            if sell_num <= holding_num:
                self.holding_days[key] -= sell_num
            else:
                sell_num -= self.holding_days[key]
                self.holding_days[key] = 0

        return earning, profit_rate

    def get_ratio_by_trade_days(self,
                                days: int,
                                curr_trade_days: int,  # 当前处于第几个交易日
                                ):
        """
        获取当前股票持有时间超过{days}的股票数量占比。

        例如：我在第3、10, 30个交易日分别买入了100, 200, 700股。
        若当前为第35个交易日(curr_trade_days=35)，
        则持有超过10个交易日(days=10)的股票占比为0.3（(100+200)/(100+200+700)）
        """

        # 校验holding_days中的股票总数和n_shares中的是否一致
        if sum(self.holding_days.values()) != self.n_shares:
            raise RuntimeError("holding_days股票总数和n_shares中的不一致，代码存在bug！")

        sell_num = 0
        for buy_trading_day, number in self.holding_days.items():
            if curr_trade_days - buy_trading_day >= days:
                # 如果持有时间到了，就卖出
                sell_num += number

        return sell_num / self.n_shares

    def get_min_holding_trade_day(self):
        """
        获取最早的持有该股票的买入日期。

        若当前未持有该股票，则返回-1
        """
        day_keys = list(self.holding_days.keys())
        day_keys.sort()

        for trade_day in day_keys:
            if self.holding_days[trade_day] > 0:
                return trade_day

        return -1


class BacktestRecord(object):
    """
    记录回测过程中的持仓，盈利等情况
    """

    def __init__(self, init_money: float):
        self.ready_money = init_money  # 现金。即目前可以用于交易的金额。

        self.positions: Dict[str, BacktestDto] = dict()  # 持仓情况。key为stock_code, value为BacktestDto对象

        self.bug_frequency = 0  # 购买次数

        self.success_times = 0  # 成功（盈利）次数

        self.fail_times = 0  # 失败（赔钱）次数

    def buy_stock(self,
                  stock_code: str,
                  curr_cost: float,  # 预计本次买入金额。例如：curr_cost=1w表示本次预计花费1w块买入
                  price: float,  # 当前股价
                  trade_days: int,  # 开始回测到现在是第几个交易日
                  ) -> int:  # 返回买入的股数
        """
        买入股票
        """
        if curr_cost >= self.ready_money:
            curr_cost = self.ready_money

        number = int((curr_cost // price // 100) * 100)  # 可以买的股数

        if number <= 0:  # 目前的仓位钱连一手都买不了
            return 0

        if stock_code not in self.positions.keys():
            self.positions[stock_code] = BacktestDto(stock_code)

        dto = self.positions[stock_code]
        real_cost = dto.buy(price, number, trade_days)

        self.ready_money -= real_cost  # 更新余额

        self.bug_frequency += 1

        return number

    def sell_stock(self, stock_code: str, sell_ratio, price: float) -> int:  # 返回卖出的股数
        """
        卖出股票
        :param sell_ratio: 卖出比例。例如，0.5表示卖出该只股票的一半股票。1是全部卖出，0是不卖出
        """
        if stock_code not in self.positions.keys():
            return 0

        dto = self.positions[stock_code]
        number = int(math.ceil(dto.n_shares * sell_ratio / 100) * 100)  # 要卖出的股数

        if number <= 0:
            return number

        earning, profit_rate = dto.sell(price, number)

        self.ready_money += earning

        if profit_rate > 0:
            self.success_times += 1
        else:
            self.fail_times += 1

        return number

    def remove_stock(self, stock_code: str) -> int:  # 返回股票数量
        """
        当回测过程中发现某只股票数据有问题时（例如发现某天该股票突然退市了），则移除该股票的购买记录，假装没有买过。
        """

        if stock_code not in self.positions.keys():
            return 0

        dto = self.positions[stock_code]

        if dto.n_shares <= 0:
            return 0

        # 按平均成本价回退，不考虑其涨跌
        fprint(f"股票{stock_code}有问题，按成本价{dto.cost_price}卖出{dto.n_shares}股")

        sell_number = dto.n_shares

        earning, _ = dto.sell(dto.cost_price, dto.n_shares)
        self.ready_money += earning

        return sell_number

    def get_total_money(self):
        """
        获取现在总金额（包括持有的股票价格，但不考虑当前股票的涨跌情况，只考虑买入时的成本）
        """

        total_money = 0.
        for stock_code, dto in self.positions.items():
            total_money += dto.n_shares * dto.cost_price

        total_money += self.ready_money

        return total_money

    def get_stock_dto(self, stock_code) -> BacktestDto:
        """
        获取某只股票的持仓情况
        """
        if stock_code not in self.positions.keys():
            self.positions[stock_code] = BacktestDto(stock_code)

        return self.positions[stock_code]


def _read_data(require_data: tuple,  # 需要读取哪些数据
               stock_code: str,
               curr_date: str,
               start_date: str,
               end_date: str,
               record: BacktestRecord,
               abnormal_stock_code_list: set,
               buy_action_list) -> Tuple[TuShareDataReader, DataFrame, DataFrame]:
    """
    读取数据。读取到的数据会被放在dict中。由于内存问题，该方法假设所有读取的数据都必须是trade_date作为index
    例如：{
        "daily": daily_data,
        "daily_extra": daily_extra_data,
        "cyq_extra": cyq_extra,
    }

    # todo 要在该函数发现所有有问题的股票，然后将其全部都排除
    """
    if 'daily' not in require_data:
        require_data = list(require_data)
        require_data.append('daily')

    reader = TuShareDataReader(stock_code).set_date_range(start_date, end_date)

    daily_data = reader.get_daily()

    data = daily_data

    if len(require_data) > 1:
        data = reader.get_combine(require_data)

    if curr_date not in data.index:  # 股票数据异常，后续不参与回测计算
        sell_number = record.remove_stock(stock_code)  # 去掉之前的购买记录
        abnormal_stock_code_list.add(stock_code)
        buy_action_list.append([stock_code, curr_date, -sell_number])
        return None, None, None

    return reader, daily_data, data


def _can_buy(
        reader: TuShareDataReader,
        curr_date: str,  # 当天日期
        curr_data: Series,  # 当天数据
        buy_timing: str,  # 买入方式
) -> bool:
    """
    判断是否可以买入
    """
    limit_data = reader.get_stk_limit()
    up_limit = limit_data.loc[curr_date]['up_limit']

    # 以收盘价买入时，若收盘价=涨停价，即当天涨停，则买入失败
    if buy_timing == 'close' and abs(curr_data['close'] - up_limit) <= 0.05:
        return True

    # 开盘价=最低价，收盘价=最高价，即当天一字涨停，则买入失败
    if curr_data['open'] == curr_data['low'] and curr_data['close'] == curr_data['high']:
        return True

    return False


def _is_limit_down(
        reader: TuShareDataReader,
        curr_date: str,
        curr_data: Series,
) -> bool:
    """
    判断是否是跌停板
    """
    limit_data = reader.get_stk_limit()
    down_limit = limit_data.loc[curr_date]['down_limit']

    # 收盘价=跌停价，当天跌停
    if abs(curr_data['close'] - down_limit) <= 0.05:
        return True

    # 开盘价=最高价，收盘价=最低价：当天一字跌停
    if curr_data['open'] == curr_data['high'] and curr_data['close'] == curr_data['low']:
        return True

    return False


def sell_strategy_template(
        data: Dict[str, DataFrame],  # 所有历史数据，例如 data['daily']为日线数据，具体参考_read_data(..)
        curr_data: Series,  # 当天的数据
        dto: BacktestDto,
        trade_days: int,  # 当前是第几个交易日，用来标记当前时间
        *args,
        **kwargs,
) -> float:
    """
    卖出策略

    返回卖出的比例。
    """

    # 收益率
    profit_rate = (curr_data['close'] - dto.cost_price) / dto.cost_price

    if profit_rate >= 0.1 or profit_rate <= -0.1:  # 赚10%或亏10%时，都卖出股票
        return random.uniform(0.5, 1)  # 随机卖出50%~100%的仓位

    return 0


def buy_strategy_template(
        data: Dict[str, DataFrame],  # 所有历史数据，例如 data['daily']为日线数据，具体参考_read_data(..)
        curr_date: str,  # 当天日期
        prev_data: Series,  # 昨日的日线数据
        curr_data: Series,  # 当天的日线数据
        dto: BacktestDto,  # 当前的持仓情况
        *args,
        **kwargs,
) -> float:
    """
    回测策略，判断是否要买入。

    返回要买入的仓位比例。例如 0.1 表示要买入当前总金额的10%
    """

    if dto.n_shares > 0:  # 如果该股票已经持仓，则不再买入
        return 0.

    # 涨幅>=6%
    if not prev_data['pct_chg'] >= 6.:
        return 0.

    # 开盘价=最低价 and 收盘价=最高价
    if not (prev_data['open'] == prev_data['low'] and prev_data['close'] == prev_data['high']):
        # 不是光头光脚大阳线
        return 0.

    return random.uniform(0.1, 0.3)  # 随机买入 0%~30%的仓位


def backtest(
        init_money: float,  # 初始资金
        start_date: str,  # 开始时间，即从哪天开始模拟。format: %Y-%m-%d
        days: int,  # 持续多少天，时间到之后会立即卖出所有股票。非交易日也被算在内，因此可以用365天表示一年
        buy_strategy,  # 买入策略
        sell_strategy,  # 卖出策略
        require_data=('daily',),  # 本次回测的购买决策都需要用到哪些数据。
        buy_timing: str = 'open',  # 买入时机；开盘买入：open，收盘买入：close
        random_stock_list=False,  # 随机选取股票买入，而不是按照现有的股票顺序遍历。
        debug_limit=-1,  # debug模式，-1表示不开启，10表示只看前10个股票
):
    """
    回测，即根据用户指定的策略进行模拟交易，看看最后自己的盈利情况。
    """

    utils.data_reader.use_pin_memory = True  # 设置缓存读取到的数据，要不太慢了

    stock_code_list = get_all_china_stock_code_list()
    if random_stock_list:
        random.shuffle(stock_code_list)
    if debug_limit > 0:
        stock_code_list = stock_code_list[:debug_limit]

    abnormal_stock_code_list = set()  # 异常股票列表，其不参与回测计算。

    record = BacktestRecord(init_money)
    trade_days = 0  # 交易日的数量
    idle_money_list = []  # 记录每个交易日后的闲置资金
    # 记录购买行为，例如：[('000001', '2023-01-01', +100), ('000001', '2023-01-02', +100), ...]
    buy_action_list = []

    date_list = get_future_date_list(start_date, days)
    last_date = date_list[-1]
    last_date = TuShareDataReader.get_last_trade_date(last_date)  # 最后一个交易日

    # 先将所有的数据读取到内存
    for stock_code in tqdm(stock_code_list, desc="Read Data"):
        reader, _, _ = _read_data(require_data,
                                  stock_code,
                                  last_date,
                                  start_date,
                                  last_date,
                                  record,
                                  abnormal_stock_code_list,
                                  buy_action_list)

        if reader is None:
            continue

        _ = reader.get_stk_limit()

    print()
    print("开始进行回测计算")
    for curr_date in tqdm(date_list, desc="Backtest"):
        fprint(curr_date, ":")

        if not is_trading_day(curr_date):
            fprint("非交易日")
            fprint('-' * 20)
            continue

        trade_days += 1

        # 逐个遍历已经持仓的股票，进行卖出操作
        _backtest_sell_stock(abnormal_stock_code_list,
                             curr_date,
                             last_date,
                             record,
                             require_data,
                             sell_strategy,
                             start_date,
                             buy_action_list,
                             trade_days,
                             )

        # 逐个遍历股票，进行买入操作
        _backtest_buy_stock(abnormal_stock_code_list,
                            buy_strategy,
                            buy_timing,
                            curr_date,
                            last_date,
                            record,
                            require_data,
                            start_date,
                            stock_code_list,
                            buy_action_list,
                            trade_days,
                            random_stock_list,
                            )

        idle_money_list.append(record.ready_money)

        fprint('-' * 20)

        if curr_date == last_date:
            break

    # 最后结束的时候卖出所有股票。
    fprint("回测结束，结果如下：")
    profit_rate = (record.ready_money - init_money) / init_money * 100
    fprint(f"初始资金：{init_money}, 最终资金: {record.ready_money}, 收益率: {profit_rate}")

    # 计算同期沪深300指数
    _, _, market_profit_rate = analysis_index('000300.SH', start_date, last_date)
    fprint("同期沪深300：", market_profit_rate)

    success_rate = round(100 * record.success_times / (record.success_times + record.fail_times + 0.00001), 2)
    fprint(
        f"出手次数: {record.bug_frequency}，成功次数: {record.success_times}，失败次数: {record.fail_times}，成功率: {success_rate}%", )

    idle_money = sum(idle_money_list) / len(idle_money_list)
    fprint(f"平均每日闲置资金: {idle_money}")

    check_buy_action(buy_action_list)
    avg_hold_days = compute_hold_days(buy_action_list)

    fprint(f"平均持仓时间: {avg_hold_days}天")

    # 释放缓存
    utils.data_reader.remove_pin_memory()

    check_log()

    result = {
        "初始资金": "%.2fw" % (init_money / 10000),
        "结束资金": "%.2fw" % round(record.ready_money / 10000, 2),
        "收益率": "%.2f%%" % round(profit_rate, 2),
        "同期沪深300": "%.2f%%" % round(market_profit_rate, 2),
        "出手次数": record.bug_frequency,
        "成功次数": record.success_times,
        "失败次数": record.fail_times,
        "成功率": "%.2f%%" % success_rate,
        "平均每日闲置资金": round(idle_money, 2),
        "平均持仓时间": f"{int(avg_hold_days)}天",
    }

    print(result)

    return result


def _backtest_buy_stock(abnormal_stock_code_list,
                        buy_strategy,
                        buy_timing,
                        curr_date,
                        last_date,
                        record,
                        require_data,
                        start_date,
                        stock_code_list,
                        buy_action_list,
                        trade_days,
                        random_stock_list,
                        ):
    if random_stock_list:
        random.shuffle(stock_code_list)

    for stock_code in stock_code_list:
        if curr_date == last_date:  # 最后一天不再买入股票
            continue

        if stock_code in abnormal_stock_code_list:  # 异常股票，不参与回测计算
            continue

        reader, daily_data, data = _read_data(require_data,
                                              stock_code,
                                              curr_date,
                                              start_date,
                                              last_date,
                                              record,
                                              abnormal_stock_code_list,
                                              buy_action_list)
        if reader is None:
            continue

        if daily_data.loc[curr_date]['index'] <= 2:  # 从第三天正式开始，因为买入要参考前一天的数据
            continue

        buy_price: float  # 买入价格
        prev_data: DataFrame  # 昨日数据
        curr_data: DataFrame  # 今日数据
        if buy_timing == 'open':  # 以开盘价买入，即符合条件后，第二天才买入。
            buy_price = daily_data.loc[curr_date]['open']
            prev_data = data[data['index'] == (data.loc[curr_date]['index'] - 2)]
            prev_data = prev_data.iloc[0]
            curr_data = data[data['index'] == (data.loc[curr_date]['index'] - 1)]
            curr_data = curr_data.iloc[0]
        elif buy_timing == 'close':  # 以收盘价买入，即符合条件后，当天就买入。
            buy_price = daily_data.loc[curr_date]['close']
            prev_data = data[data['index'] == (data.loc[curr_date]['index'] - 1)]
            prev_data = prev_data.iloc[0]
            curr_data = data.loc[curr_date]
        else:
            raise RuntimeError("不支持的buy_timing:%s" % buy_timing)

        # 求买入的仓位比例
        buy_ratio = buy_strategy(data,
                                 curr_date,
                                 prev_data=prev_data,
                                 curr_data=curr_data,
                                 dto=record.get_stock_dto(stock_code),
                                 stock_code=stock_code)

        if buy_ratio <= 0:
            # 不满足买入条件
            continue

        # 若当天涨停，则买入失败
        if _can_buy(reader, curr_date, daily_data.loc[curr_date], buy_timing):  # 当天涨停，买入失败
            continue

        # 计算当前总金额
        total_money = record.get_total_money()

        # 本次预计花费
        curr_cost = buy_ratio * total_money
        # 买入股票
        buy_number = record.buy_stock(stock_code, curr_cost, buy_price, trade_days)
        buy_action_list.append([stock_code, curr_date, buy_number])


def _backtest_sell_stock(abnormal_stock_code_list,
                         curr_date,
                         last_date,
                         record,
                         require_data,
                         sell_strategy,
                         start_date,
                         buy_action_list,
                         trade_days,
                         ):
    for stock_code, dto in record.positions.items():
        if dto.n_shares <= 0:  # 该股票已经不持仓了
            continue

        if stock_code in abnormal_stock_code_list:  # 异常股票，不参与回测计算
            continue

        reader, daily_data, data = _read_data(require_data,
                                              stock_code,
                                              curr_date,
                                              start_date,
                                              last_date,
                                              record,
                                              abnormal_stock_code_list,
                                              buy_action_list)
        if reader is None:
            continue

        if curr_date != last_date:
            sell_ratio = sell_strategy(data,
                                       data.loc[curr_date],
                                       dto,
                                       trade_days,
                                       )
        else:  # 最后一天卖出所有股票
            sell_ratio = 1.

        if sell_ratio <= 0:
            # 不满足卖出条件
            continue

        close_price = daily_data.loc[curr_date]['close']  # 当天收盘价。注：以收盘价作为卖出价

        # 若当天跌停，则卖出失败
        if _is_limit_down(reader, curr_date, daily_data.loc[curr_date]):
            continue

        # 卖出股票
        sell_number = record.sell_stock(stock_code, sell_ratio, close_price)

        buy_action_list.append([stock_code, curr_date, -sell_number])


def compute_hold_days(_buy_action_list):
    """
    根据购买行为分析平局持仓时间
    """
    hold_days_list = []

    buy_action_list = copy.deepcopy(_buy_action_list)
    for i in range(len(buy_action_list)):
        stock_code, trade_date, number = buy_action_list[i]
        if number >= 0:
            # 买入行为，跳过。继续往后找卖出行为
            continue

        sell_number = number
        # 发现卖出行为，从前往后找，找到该股票最早的买入记录，然后记录持股时间
        for j in range(0, i):
            _stock_code, _trade_date, buy_number = buy_action_list[j]
            if stock_code != _stock_code:
                continue

            if buy_number < 0:
                raise RuntimeError("有bug")

            if buy_number == 0:  # 该买入记录已被平仓
                continue

            diff = abs(sell_number)  # 计划平仓份数：卖出的份数
            if buy_number < diff:  # 但本次数量不够平仓
                diff = buy_number  # 本次平仓的数量

            # 发现之前的交易记录，平仓
            sell_number += diff
            buy_number -= diff
            buy_action_list[i][2] = sell_number
            buy_action_list[j][2] = buy_number

            hold_days_list.append(date_diff(buy_action_list[i][1], buy_action_list[j][1]))  # 记录持股时间

            if sell_number > 0:
                raise RuntimeError("有bug")

            if buy_number < 0:
                raise RuntimeError("有bug")

            if sell_number == 0:  # 已平仓，不需要再往后找了
                break

        # 如果平仓结束后，卖出数量不为0，则说明有bug
        if buy_action_list[i][2] != 0:
            raise RuntimeError("有bug")

    for action in buy_action_list:
        if action[2] != 0:
            raise RuntimeError("有bug")

    return sum(hold_days_list) / (len(hold_days_list) + 0.00001)  # 平均持股时间


def check_buy_action(buy_action_list):
    """
    根据购买行为判断代码是否存在bug
    """
    stock_number_map = {}
    for stock_code, trade_date, number in buy_action_list:
        if stock_code not in stock_number_map:
            stock_number_map[stock_code] = 0

        stock_number_map[stock_code] += number

        if stock_number_map[stock_code] < 0:
            raise RuntimeError("代码存在bug，股票数量为负")

    for stock_code, number in stock_number_map.items():
        if number != 0:
            raise RuntimeError("代码存在bug，最终股票数量不为0")


def check_log():
    """
    分析log日志，看看程序有没有bug。

    日志应符合以下等式：
    最终资金 - 初始资金 = sum(收入明细) - sum(花费明细)
    """

    total_cost = 0.
    total_earn = 0.
    init_money = 0.
    final_money = 0.
    total_sell = 0.
    total_buy = 0.

    with open(ROOT / 'log' / f"{log_filename}.log", encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if '收入' in line:
            total_earn += float(re.findall('收入[0-9.]+元', line)[0].replace('收入', '').replace('元', ''))
            total_sell += int(re.findall('股票[0-9]+股', line)[0].replace('股票', "").replace("股", ""))

        if '花费' in line:
            total_cost += float(re.findall('花费[0-9.]+元', line)[0].replace('花费', '').replace('元', ''))
            total_buy += int(re.findall('股票[0-9]+股', line)[0].replace('股票', "").replace("股", ""))

        if '初始资金' in line:
            init_money = float(re.findall('初始资金：[0-9.]+,', line)[0].replace('初始资金：', '').replace(',', ''))
            final_money = float(re.findall('最终资金: [0-9.]+,', line)[0].replace('最终资金: ', '').replace(',', ''))

    if not abs((final_money - init_money) - (total_earn - total_cost)) <= 0.1:
        raise RuntimeError("最终资金 - 初始资金 = sum(收入明细) - sum(花费明细) 不成立，可能是代码出bug了，请检查！")

    if total_buy != total_sell:
        raise RuntimeError("总卖出股数 != 总买入股数，可能是代码出bug了，请检查！")


def multi_backtest(
        years,  # 对哪些年进行预测，可以是list，或str(格式为2018-2022)
        init_money: float,  # 初始资金
        buy_strategy,  # 买入策略
        sell_strategy,  # 卖出策略
        require_data=('daily',),  # 本次回测的购买决策都需要用到哪些数据。
        buy_timing: str = 'open',  # 买入时机；开盘买入：open，收盘买入：close
        random_stock_list=False,  # 是否随机选取股票，而非按顺序遍历
        debug_limit=-1,  # debug模式，-1表示不开启，10表示只看前10个股票
):
    """
    连续对若干年的数据进行回测，最后输出markdown格式结果
    """
    if type(years) == str:
        start, end = years.split("-")
        years = list(range(int(start), int(end) + 1))

    if type(years) != list:
        raise RuntimeError("years必须是list")

    years.sort(reverse=True)

    results = []
    for year in years:
        global log_filename  # 每年写一个日志
        global fprint
        log_filename = "analysis_" + get_now(format='%Y%m%d_%H%M%S')
        fprint = get_fprint(log_filename)

        print(year, "回测，日志文件：", log_filename)
        result = backtest(
            init_money=init_money,
            start_date='%d-01-01' % year,
            days=365,
            buy_strategy=buy_strategy,
            sell_strategy=sell_strategy,
            require_data=require_data,
            buy_timing=buy_timing,
            random_stock_list=random_stock_list,
            debug_limit=debug_limit,
        )

        result_dict = dict({'年份': str(year)})
        result_dict.update(result)
        results.append(result_dict)

    print()
    table_print(results, format='md')


def backtest_demo():
    backtest(
        init_money=100_000,
        start_date='2022-01-01',
        days=365,
        buy_strategy=buy_strategy_template,
        sell_strategy=sell_strategy_template,
        require_data=('daily',),
        buy_timing='open',
        debug_limit=50,
    )
