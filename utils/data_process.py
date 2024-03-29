# -*- coding: UTF-8 -*-
import sys
from pathlib import Path
from typing import Dict, Tuple, List

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import pandas as pd
from pandas import DataFrame

import utils.data_reader as data_reader
from utils.utils import time_func

# 统一的数据分析起止时间。若没有指定，那么分析的数据就是这个区间的
global_start_date = '2010-01-01'
global_end_date = '2022-12-31'


def get_all_china_stock():
    """
    获取所有的大陆股票
    """
    filename = ROOT / 'data' / ('stock_list.csv')
    # todo
    data = pd.read_csv(filename, dtype=str)
    return data


def get_all_china_ts_code_list():
    all_stock = get_all_china_stock()
    stock_code_list = list(all_stock['ts_code'])

    return stock_code_list


def get_all_china_stock_code_list() -> List[str]:
    data = data_reader.TuShareDataReader.get_stock_list()

    return list(data.index)


def format_tushare(data: DataFrame):
    data['trade_date'] = pd.to_datetime(data['trade_date'], format="%Y%m%d")
    data = data.sort_values(by="trade_date", ascending=True)
    data = data.set_index('trade_date')
    data['volume'] = data['vol']
    data['index'] = list(range(1, len(data) + 1))
    del data['vol']

    # 去掉未知Unnamed开头的列
    for column_name in data.columns:
        if column_name.startswith("Unnamed"):
            del data[column_name]

    return data


def read_data(stock_code: str, start_date: str = "2010-01-01", end_date="2022-12-31"):
    reader = data_reader.TuShareDataReader(stock_code).set_date_range(start_date, end_date)
    daily_data = reader.get_daily()
    return reader, daily_data


def add_MA(data: DataFrame, day_list=(5, 10, 20, 60),
           target_data: DataFrame = None,  # 如果传了target_data，则5MA这些列放到这个target_data里
           format='%dMA',  # 默认格式是'%dMA'，但后续逐步改成'MA%d'
           ):
    """
    计算MA均线
    day_list: 例如，如果你想计算5日均线和10日均线，则传 [5, 10]

    return：在原数据上增加 "5MA", "10MA"等列
    """
    if target_data is None:
        target_data = data

    for n in day_list:
        key = format % n
        if key in data:
            continue

        target_data[key] = data['close'].rolling(window=n).mean().round(2)

    return target_data


def add_amplitude(data, day_list=(5, 10, 20, 60)):
    """
    计算前一段时间的振幅（波动率）。
    振幅公式为：(前n天的最高价-前n天的最低价)/前n天的最低价 * 100%

    return: 在原数据上增加 "5AMP", "10AMP"等列
    """
    for n in day_list:
        key = "%dAMP" % n
        if key in data:
            continue

        highs = data['high'].rolling(n).max()
        lows = data['low'].rolling(n).min()

        data[key] = (highs - lows) / lows * 100

    return data


def add_high(data, day_list=(60,)):
    """
    增加近n天的最高价
    例如："60high"表示近60天的最高价
    """
    for n in day_list:
        key = "%dhigh" % n
        if key in data:
            continue

        data[key] = data['high'].rolling(n).max()

    return data


def add_low(data, day_list=(60,)):
    """
    增加近n天的最低价
    例如："60low"表示近60天的最低价
    """
    for n in day_list:
        key = "%dlow" % n
        if key in data:
            continue

        data[key] = data['low'].rolling(n).min()

    return data


def add_QRR(daily_data: DataFrame, day_list=(5,), target_data: DataFrame = None):
    """
    计算每日的量比。
    量比：当日交易量 / 前n天的平均交易量

    该方法会在data上增加列，例如：5QRR，代表5日的量比
    """
    if target_data is None:
        target_data = daily_data

    for n in day_list:
        value = daily_data['vol'] / daily_data['vol'].rolling(n, closed='left').mean()
        value = round(value, 2)
        target_data[f'QRR{n}'] = value

    return target_data


def add_high_vol(data, day_list=(40,)):
    """
    计算近n日（不包含当天）的最大交易量。

    该方法会在data上增加列，例如：40_high_vol，代表近40天的最大交易量
    """

    for n in day_list:
        data[f'{n}_high_vol'] = data['volume'].rolling(n, closed='left').max()

    return data


def add_prev_n(data, n, include_columns=None, exclude_columns=('ts_code',)):
    """
    求前n天的数据。
    例如：data有"high, low, close, open"四列
    求前1天的数据后，data会增加"prev_1_high, prev_1_low, prev_1_close, prev_1_open"四列

    最后返回值会去掉前n行
    """
    prev_data = data.iloc[0:-n].copy()
    curr_data = data.iloc[n:].copy()

    columns = list(curr_data.columns)
    for column in columns:
        if column in exclude_columns:
            continue

        if include_columns is not None and column not in include_columns:
            continue

        if column.startswith(f"prev_{n}_"):
            continue

        curr_data[f'prev_{n}_{column}'] = list(prev_data[column])

    return curr_data


def add_after_n(data, n, include_columns=None, exclude_columns=('ts_code',)):
    """
    求后n天的数据。
    例如：data有"high, low, close, open"四列
    求前1天的数据后，data会增加"after_1_high, after_1_low, after_1_close, after_1_open"四列

    最后返回值会去掉前n行
    """
    curr_data = data.iloc[0:-n].copy()
    after_data = data.iloc[n:].copy()

    columns = list(curr_data.columns)
    for column in columns:
        if column in exclude_columns:
            continue

        if include_columns is not None and column not in include_columns:
            continue

        if column.startswith(f"after_{n}_"):
            continue

        curr_data[f'after_{n}_{column}'] = list(after_data[column])

    return curr_data


def add_RPY(data: DataFrame,
            year=1,
            target_data: DataFrame = None,
            ):
    """
    年相对价位：当前股价在近n年中的相对位置
    RPY_n = (当前收盘价 - 近n年最低价) / (近n年最高价 - 近n年最低价)

    这里一年按250个交易日计算
    :param year: n
    :return: 在data中增加RPYN，例如：RPY1、RPY2等
    """
    if target_data is None:
        target_data = data

    high = data['high'].rolling(250 * year).max()
    low = data['low'].rolling(250 * year).min()

    value = (data['close'] - low) / (high - low) * 100
    value = round(value, 2)

    target_data['RPY%d' % year] = value

    return target_data


def add_RPM(data: DataFrame,
            month=1,
            target_data: DataFrame = None,
            ):
    """
    月相对价位：当前股价在近n月中的相对位置。类似RPY，但维度变成了月（20天/月）
    """
    if target_data is None:
        target_data = data

    high = data['high'].rolling(20 * month).max()
    low = data['low'].rolling(20 * month).min()

    value = (data['close'] - low) / (high - low) * 100
    value = round(value, 2)

    target_data['RPM%d' % month] = value

    return target_data


def add_gap(daily_data: DataFrame,
            target_data: DataFrame = None,
            ):
    """
    跳空幅度。即 今日的开盘价相比昨天的收盘价发生的位移。
    gap = (今日开盘价 - 昨日收盘价) / 昨日收盘价 * 100%

    若gap>0：跳空高开
    若gap<0：跳空低开
    """
    if target_data is None:
        target_data = daily_data

    open = daily_data['open']
    prev_close = daily_data['close'].shift(1)

    value = (open - prev_close) / prev_close * 100
    value = round(value, 2)

    target_data['gap'] = value

    return target_data


def add_wick(daily_data: DataFrame,
             target_data: DataFrame = None,
             ):
    """
    上、下影线长度（百分比）。
    upper_wick = (今日最高价 - 箱体上沿价格) / 箱体上沿价格 * 100%
    lower_wick = (今日最低价 - 箱体下沿价格) / 箱体下沿价格 * 100%
    """
    if target_data is None:
        target_data = daily_data

    high = daily_data['high']
    low = daily_data['low']

    body_high = pd.Series([max(open, close) for open, close in zip(daily_data['open'], daily_data['close'])],
                          index=daily_data.index)
    body_low = pd.Series([min(open, close) for open, close in zip(daily_data['open'], daily_data['close'])],
                         index=daily_data.index)

    upper_wick = (high - body_high) / body_high * 100
    lower_wick = (low - body_low) / body_low * 100

    upper_wick = round(upper_wick, 2)
    lower_wick = round(lower_wick, 2)

    target_data['upper_wick'] = upper_wick
    target_data['lower_wick'] = lower_wick

    return target_data


def add_amp(daily_data: DataFrame,
            target_data: DataFrame = None,
            ):
    """
    当日振幅。
    amp = (今日最高价 - 今日最低价) / 今日最低价 * 100%
    """
    if target_data is None:
        target_data = daily_data

    value = (daily_data['high'] - daily_data['low']) / daily_data['low'] * 100
    value = round(value, 2)

    target_data['amp'] = value

    return target_data


def add_FAR(daily_data: DataFrame,
            target_data: DataFrame = None,
            ):
    """
    冲高后回落幅度（fall after rise）：当日的收盘价相比最高价下跌了多少%
    下跌后上涨幅度（rise after fall）：当日的收盘价相比最低价上涨了多少%

    FAR = (收盘价 - 最高价) / 最高价 * 100%
    RAF = (收盘价 - 最低价) / 最低价 * 100%
    """
    if target_data is None:
        target_data = daily_data

    FAR = (daily_data['close'] - daily_data['high']) / daily_data['high'] * 100
    RAF = (daily_data['close'] - daily_data['low']) / daily_data['low'] * 100

    FAR = round(FAR, 2)
    RAF = round(RAF, 2)

    target_data['FAR'] = FAR
    target_data['RAF'] = RAF

    return target_data


def add_FPY(data: DataFrame,
            year=1,
            target_data: DataFrame = None,
            ):
    """
    近n年回撤，即当前股价相比近n年的最高点下跌了多少

    FPY_n = (当前收盘价 - 近n年最高价) / 近n年最高价

    这里一年按250个交易日计算
    """
    if target_data is None:
        target_data = data

    high = data['high'].rolling(250 * year).max()

    value = (data['close'] - high) / high * 100
    value = round(value, 2)

    target_data['FPY%d' % year] = value

    return target_data


def add_FPM(data: DataFrame,
            month=1,
            target_data: DataFrame = None,
            ):
    """
    近n个月回撤。类似FPY，但维度变成了月（20天/月）
    """
    if target_data is None:
        target_data = data

    high = data['high'].rolling(20 * month).max()

    value = (data['close'] - high) / high * 100
    value = round(value, 2)

    target_data['FPM%d' % month] = value

    return target_data


def add_CYQP(data: DataFrame,
             cyq_data: DataFrame,
             down: int,
             up: int,
             ):
    """
    筹码占比（自己发明的指标）：当前价位上下的位置总共包含了多少筹码。

    例如：CYQP_15_10表示：当前股价向下15%向上10%的空间内，一共有百分之多少的筹码

    :param data: 要求的数据，最终会在该指标中增加`CYQP_{down}_{up}`，值为百分比（10表示10%）
    :param cyq_data: 筹码分布数据
    """

    cyqp_list = []
    for trade_date, row in data.iterrows():
        cyq = cyq_data[cyq_data['trade_date'] == trade_date]

        if len(cyq) <= 0:
            cyqp_list.append(None)
            continue

        price = row['close']
        down_price = price * (1 - down / 100)
        up_price = price * (1 + up / 100)

        cyqp = cyq[(down_price <= cyq['price']) & (cyq['price'] <= up_price)]['percent'].sum()
        cyqp_list.append(cyqp)

    data[f'CYQP_{down}_{up}'] = cyqp_list

    return data


class CyqSpeed:
    """
    筹码分布数据加速器。
    由于筹码分布数据量太大，因此执行`cyq_data[cyq_data['trade_date'] == trade_date]`时比较慢，大约需要5毫秒
    虽然5ms看起来不多，但执行3年的数据可就要好几秒。因此还是需要优化的
    """

    def __init__(self, cyq_data: DataFrame):
        self.cyq_data = cyq_data

        # 记录当前cyq_data的每个trade_date对应的索引下表的起始和终止
        self.date_map: Dict[str, Tuple[int, int]] = self._init_date_map()

    def _init_date_map(self) -> Dict[str, Tuple[int, int]]:
        date_map: Dict[str, Tuple[int, int]] = dict()

        cyq_count = self.cyq_data['trade_date'].value_counts()
        cyq_count = cyq_count.sort_index()

        i = 0
        for trade_date, count in cyq_count.items():
            date_map[trade_date] = (i, i + count)
            i += count

        return date_map

    def get_cyq_by_trade_date(self, trade_date: str):
        l, r = self.date_map[trade_date]
        return self.cyq_data.iloc[l:r]


def add_CYQK(data: DataFrame,
             daily_data: DataFrame,
             cyq_data: DataFrame,
             ):
    """
    博弈K线，分别有：
            open_winner：开盘时的获利盘占比
            close_winner：收盘时的获利盘占比
            high_winner：最高价时的获利盘占比
            low_winner：最低价时的获利盘占比
    """
    cyq_data = CyqSpeed(cyq_data)

    open_winner_list = []
    close_winner_list = []
    high_winner_list = []
    low_winner_list = []
    for trade_date in data.index:
        daily_sr = daily_data.loc[trade_date]
        daily_cyq_data = cyq_data.get_cyq_by_trade_date(trade_date)

        open = round(daily_cyq_data['percent'][daily_cyq_data['price'] <= daily_sr['open']].sum(), 2)
        close = round(daily_cyq_data['percent'][daily_cyq_data['price'] <= daily_sr['close']].sum(), 2)
        high = round(daily_cyq_data['percent'][daily_cyq_data['price'] <= daily_sr['high']].sum(), 2)
        low = round(daily_cyq_data['percent'][daily_cyq_data['price'] <= daily_sr['low']].sum(), 2)

        open_winner_list.append(open)
        close_winner_list.append(close)
        high_winner_list.append(high)
        low_winner_list.append(low)

    data['open_winner'] = open_winner_list
    data['close_winner'] = close_winner_list
    data['high_winner'] = high_winner_list
    data['low_winner'] = low_winner_list

    return data


def compute_ASR(price: float,
                cyq_data: DataFrame
                ):
    """
    计算某个价格的活动筹码：统计在该价位上下10%的区间内的筹码总量（该区间散户最容易割肉）

    cyq_data: 当天的筹码分布数据
    """
    up_price = price * 1.1
    down_price = price * 0.9

    ASR = cyq_data['percent'][(down_price <= cyq_data['price']) & (cyq_data['price'] <= up_price)].sum()
    ASR = round(ASR, 2)

    return ASR


def add_ASR(data: DataFrame,
            daily_data: DataFrame,
            cyq_data: DataFrame,
            ):
    """
    活动筹码: 统计在该价位上下10%的区间内的筹码总量（该区间散户最容易割肉）

    以收盘价为准。
    """
    cyq_data = CyqSpeed(cyq_data)

    ASR_list = []
    for trade_date in data.index:
        daily_sr = daily_data.loc[trade_date]
        daily_cyq_data = cyq_data.get_cyq_by_trade_date(trade_date)

        price = daily_sr['close']

        ASR_list.append(compute_ASR(price, daily_cyq_data))

    data['ASR'] = ASR_list

    return data


def add_ASRC(data: DataFrame,
             daily_data: DataFrame,
             cyq_data: DataFrame,
             ):
    """
    活动筹码变化量 ASR Change（自己设计的指标）

    昨日的活动筹码变化量（自定义指标）：昨日收盘价的位置的活动筹码，在今天收盘后，减少或增加了多少。
    """
    cyq_data = CyqSpeed(cyq_data)

    result_list = [None]
    for i in range(1, len(data.index)):
        last_date = data.index[i - 1]  # 昨天
        trade_date = data.index[i]

        last_close_price = daily_data.loc[last_date]['close']  # 昨日收盘价

        # 计算昨日收盘价位置的“昨天的”活动筹码
        last_ASR = compute_ASR(last_close_price,
                               cyq_data.get_cyq_by_trade_date(last_date)  # 昨日筹码分布
                               )
        # 计算昨日收盘价位置的“今天的”活动筹码
        today_ASR = compute_ASR(last_close_price,
                                cyq_data.get_cyq_by_trade_date(trade_date)  # 今日筹码分布
                                )
        ASRC = today_ASR - last_ASR
        ASRC = round(ASRC, 2)

        result_list.append(ASRC)

    data['ASRC'] = result_list

    return data


def add_CKDP(data: DataFrame,
             daily_data: DataFrame,
             cyq_data: DataFrame,
             ):
    """
    筹码分布相对价位, CKDP = (当天收盘价 - 最低成本价) / (最高成本价 - 最低成本价) × 100%

    注意：忽略掉了占比小于0.2%的筹码
    """
    cyq_data = CyqSpeed(cyq_data)

    result_list = []
    for trade_date in data.index:
        daily_sr = daily_data.loc[trade_date]
        daily_cyq_data = cyq_data.get_cyq_by_trade_date(trade_date)

        price = daily_sr['close']
        high_price = daily_cyq_data['price'][daily_cyq_data['percent'] > 0.2].max()
        low_price = daily_cyq_data['price'][daily_cyq_data['percent'] > 0.2].min()

        ckdp = (price - low_price) / (high_price - low_price) * 100
        ckdp = round(ckdp, 2)

        result_list.append(ckdp)

    data['CKDP'] = result_list

    return data


def add_CKDW(data: DataFrame,
             daily_data: DataFrame,
             cyq_data: DataFrame,
             ):
    """
    成本重心，CKDW = (平均成本价 - 最低成本价) / (最高成本价 - 最低成本价) × 100%

    若CKDW较高，说明平均成本价较高，筹码处于高位密集状态。
    若CKDW较低，说明平均成本价较低，筹码处于低位密集状态。

    注意：忽略掉了占比小于0.2%的筹码。但计算平均成本时，不忽略。
    """
    cyq_data = CyqSpeed(cyq_data)

    avg_price_list = []  # 平均成本
    result_list = []
    for trade_date in data.index:
        daily_cyq_data = cyq_data.get_cyq_by_trade_date(trade_date)

        avg_price = (daily_cyq_data['price'] * daily_cyq_data['percent']).sum() / 100
        # tushare数据有问题，daily_cyq_data['percent'].sum() 不一定为100，因此需要对avg_price进行调节
        avg_price = avg_price * 100 / daily_cyq_data['percent'].sum()
        avg_price = round(avg_price, 2)
        avg_price_list.append(avg_price)

        high_price = daily_cyq_data['price'][daily_cyq_data['percent'] > 0.2].max()
        low_price = daily_cyq_data['price'][daily_cyq_data['percent'] > 0.2].min()

        result = (avg_price - low_price) / (high_price - low_price) * 100
        result = round(result, 2)
        result_list.append(result)

    data['avg_price'] = avg_price_list
    data['CKDW'] = result_list

    return data


def add_CBW(data: DataFrame,
            daily_data: DataFrame,
            cyq_data: DataFrame,
            ):
    """
    成本带宽，(最高成本价 - 最低成本价) / 最低成本价 × 100%

    注意：忽略掉了占比小于0.2%的筹码
    """
    cyq_data = CyqSpeed(cyq_data)

    high_price_list = []
    low_price_list = []
    result_list = []
    for trade_date in data.index:
        daily_cyq_data = cyq_data.get_cyq_by_trade_date(trade_date)

        high_price = daily_cyq_data['price'][daily_cyq_data['percent'] > 0.2].max()
        low_price = daily_cyq_data['price'][daily_cyq_data['percent'] > 0.2].min()

        cbw = (high_price - low_price) / (low_price) * 100
        cbw = round(cbw, 2)

        high_price = round(high_price, 2)
        low_price = round(low_price, 2)

        high_price_list.append(high_price)
        low_price_list.append(low_price)
        result_list.append(cbw)

    data['high_price'] = high_price_list
    data['low_price'] = low_price_list
    data['CBW'] = result_list

    return data


def compute_CYC_n():
    pass


def add_CYC(data: DataFrame,
            daily_data: DataFrame,
            cyq_data: DataFrame,
            ):
    """
    todo

    成本均线，包含4个：
    CYC5：5日成本均线
    CYC13：13日成本均线
    CYC34：34日成本均线
    CYC∞（用CYC+表示）：无穷成本均线

    n日成本均线计算公式：
    无穷成本均线计算公式：
    """
    CYC5_list = []
    CYC13_list = []
    CYC34_list = []
    CYC_INF_list = []
    for trade_date in data.index:
        daily_cyq_data = cyq_data[cyq_data['trade_date'] == trade_date]

        high_price = daily_cyq_data[daily_cyq_data['percent'] > 0.2]['price'].max()
        low_price = daily_cyq_data[daily_cyq_data['percent'] > 0.2]['price'].min()

        cbw = (high_price - low_price) / (low_price) * 100
        cbw = round(cbw, 2)

        high_price = round(high_price, 2)
        low_price = round(low_price, 2)

    data['CYC5'] = CYC5_list
    data['CYC13'] = CYC13_list
    data['CYC34'] = CYC34_list
    data['CYC+'] = CYC_INF_list

    return data


def remove_continual_data(data: DataFrame,
                          tolerant_day=1,  # 连续1天认为是连续。假设为2，则间隔1天也认为是连续。
                          ):
    """
    去重连续值。仅保留第一天。

    应用场景：在执行分析过程中，会存在连续几天都符合买入条件的情况。
            此时就需要去除连续值，仅保留第一天。毕竟实操时通常也
            不会因为连续两天符合买入条件就连续两天买入。
    """
    continual_data = (data['index'] - data['index'].shift(1)) <= tolerant_day
    return data[~continual_data]


def drop_duplicated_columns(data: DataFrame):
    """
    去除重复的列
    """
    is_series = False
    if type(data) == pd.Series:
        data = data.to_frame().T
        is_series = True

    data = data.loc[:, ~data.columns.duplicated()]

    if is_series:
        data = data.iloc[0]

    return data


if __name__ == '__main__':
    data1 = {'A': [1, 2, 3],
             'B': [4, 5, 6],
             'C': [7, 8, 9], }
    data1 = pd.DataFrame(data1)

    data2 = {'B': [1, 5, 6],
             'C': [2, 8, 9], }
    data2 = pd.DataFrame(data2)

    data = pd.concat([data1, data2], axis=1)

    print()
