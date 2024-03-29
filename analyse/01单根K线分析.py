# -*- coding: UTF-8 -*-


import sys
from pathlib import Path

from pandas import DataFrame

from src.analysis import get_results, plot_results, analysis_A_share, AnalysisOne

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

# 根据历史K线数据分析某种K线组合对后续股价的影响

from utils.data_process import read_data, get_all_china_ts_code_list, add_MA, add_amplitude, add_QRR
from utils.data_reader import TuShareDataReader


def DaYangXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分析
        pct_chg=6.,  # 当日涨幅超过6%认为是大阳线
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的大阳线
    """
    reader, data = read_data(stock_code)
    # 获取该股票历史上的大阳线数据
    curr_data = data[data['pct_chg'] > pct_chg]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def LowDaYangXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分析
        pct_chg=6.,  # 当日涨幅超过6%认为是大阳线
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的处在低位时的大阳线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data[data['pct_chg'] > pct_chg]
    # 若大阳线日的“60日均线>最高价”，则认为是低位大阳线
    curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def HighDaYangXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分析
        pct_chg=6.,  # 当日涨幅超过6%认为是大阳线
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的处在高位时的大阳线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data[data['pct_chg'] > pct_chg]
    # 若大阳线日的“60日均线<最低价”，则认为是低位大阳线
    curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def BareDaYangXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分析
        pct_chg=6.,  # 当日涨幅超过6%认为是大阳线
        print_result=False,
        plot_examples=False,
):
    """
    光头光脚大阳线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data[data['pct_chg'] > pct_chg]
    # 筛选“开盘价==最低价”
    curr_data = curr_data[curr_data['open'] == curr_data['low']]
    # 筛选“收盘价==最高价”
    curr_data = curr_data[curr_data['close'] == curr_data['high']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def LowBareDaYangXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分析
        pct_chg=6.,  # 当日涨幅超过6%认为是大阳线
        print_result=False,
        plot_examples=False,
):
    """
    低位光头光脚大阳线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data[data['pct_chg'] > pct_chg]
    # 筛选“开盘价==最低价”
    curr_data = curr_data[curr_data['open'] == curr_data['low']]
    # 筛选“收盘价==最高价”
    curr_data = curr_data[curr_data['close'] == curr_data['high']]

    # 筛选“60日均线>最高价”
    curr_data = curr_data[curr_data['RPY2'] < 30]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def HighBareDaYangXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分析
        pct_chg=6.,  # 当日涨幅超过6%认为是大阳线
        print_result=False,
        plot_examples=False,
):
    """
    高位光头光脚大阳线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data[data['pct_chg'] > pct_chg]
    # 筛选“开盘价==最低价”
    curr_data = curr_data[curr_data['open'] == curr_data['low']]
    # 筛选“收盘价==最高价”
    curr_data = curr_data[curr_data['close'] == curr_data['high']]
    # 筛选“60日均线>最高价”
    curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def DaYinXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        pct_chg=-6.,  # 当日涨幅低于6%认为是大阴线
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的大阴线
    """
    reader, data = read_data(stock_code)
    # 筛选“涨跌幅<=-6%”
    curr_data = data[data['pct_chg'] <= pct_chg]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(5, 10, 20, 60,), )

    return results


def LowDaYinXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        pct_chg=-6.,  # 当日跌幅大于-6%认为是大阴线
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的处在低位时的大阴线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取“涨跌幅<=-6%”
    curr_data = data[data['pct_chg'] <= pct_chg]
    # 若大阳线日的“60日均线>最高价”，则认为是低位大阴线
    curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def HighDaYinXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        pct_chg=-6.,  # 当日涨幅超过-6%认为是大阴线
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的处在高位时的大阴线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data[data['pct_chg'] <= pct_chg]
    # 60日均线<最低价
    curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def BareDaYinXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分析
        pct_chg=-6.,  # 当日跌幅超过-6%认为是大阴线
        print_result=False,
        plot_examples=False,
):
    """
    光头光脚大阴线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data[data['pct_chg'] <= pct_chg]
    # 筛选“开盘价==最高价”
    curr_data = curr_data[curr_data['open'] == curr_data['high']]
    # 筛选“收盘价==最低价”
    curr_data = curr_data[curr_data['close'] == curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def LowBareDaYinXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        pct_chg=-6.,  # 当日涨幅低于-6%认为是大阳线
        print_result=False,
        plot_examples=False,
):
    """
    低位光头光脚大阳线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data[data['pct_chg'] <= pct_chg]
    # 筛选“开盘价==最低价”
    curr_data = curr_data[curr_data['open'] == curr_data['high']]
    # 筛选“收盘价==最高价”
    curr_data = curr_data[curr_data['close'] == curr_data['low']]
    # 筛选“60日均线>最高价”
    curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def HighBareDaYinXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        pct_chg=-6.,  # 当日跌幅超过6%认为是大阴线
        print_result=False,
        plot_examples=False,
):
    """
    高位光头光脚大阳线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data[data['pct_chg'] <= pct_chg]
    # 筛选“开盘价==最高价”
    curr_data = curr_data[curr_data['open'] == curr_data['high']]
    # 筛选“收盘价==最低价”
    curr_data = curr_data[curr_data['close'] == curr_data['low']]
    # 筛选“60日均线<最低价”
    curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def ChangShangYingXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的“长上影线”
    """
    reader, data = read_data(stock_code)
    curr_data = data.copy()
    # body_high：实体的最高位
    curr_data['body_high'] = [max(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]
    # body_high：实体的最低位
    curr_data['body_low'] = [min(open, close) for
                             open, close in zip(curr_data['open'], curr_data['close'])]

    # 上影线涨幅 > 3%
    curr_data = curr_data[(curr_data['high'] - curr_data['body_high']) / curr_data['body_high'] >= 0.03]
    # 下影线跌幅 > -1%
    curr_data = curr_data[(curr_data['low'] - curr_data['body_low']) / curr_data['body_low'] > -0.01]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(5, 10, 20, 60,), )

    return results


def LowChangShangYingXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    低位光头光脚大阳线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    curr_data = data.copy()
    # body_high：实体的最高位
    curr_data['body_high'] = [max(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]
    # body_high：实体的最低位
    curr_data['body_low'] = [min(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]

    # 上影线涨幅 > 3%
    curr_data = curr_data[(curr_data['high'] - curr_data['body_high']) / curr_data['body_high'] >= 0.03]
    # 下影线跌幅 > -1%
    curr_data = curr_data[(curr_data['low'] - curr_data['body_low']) / curr_data['body_low'] > -0.01]

    # 筛选“60日均线>最高价”
    curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def HighChangShangYingXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    高位长上影线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    curr_data = data.copy()
    # body_high：实体的最高位
    curr_data['body_high'] = [max(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]
    # body_high：实体的最低位
    curr_data['body_low'] = [min(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]

    # 上影线涨幅 > 3%
    curr_data = curr_data[(curr_data['high'] - curr_data['body_high']) / curr_data['body_high'] >= 0.03]
    # 下影线跌幅 > -1%
    curr_data = curr_data[(curr_data['low'] - curr_data['body_low']) / curr_data['body_low'] > -0.01]

    # 筛选“60日均线<最低价”
    curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def XianRenZhiLu(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    长上影线-仙人指路
    """
    reader, data = read_data(stock_code)
    curr_data = data.copy()

    n_amp = 40  # 看近{n_amp}日是否横盘震荡
    curr_data = add_MA(curr_data, day_list=(60,))
    # 增添近60日的振幅
    curr_data = add_amplitude(curr_data, day_list=(n_amp,))
    # body_high：实体的最高位
    curr_data['body_high'] = [max(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]
    # body_high：实体的最低位
    curr_data['body_low'] = [min(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]

    # 近60日振幅 < 25% (横盘震荡)
    curr_data = curr_data[curr_data['%dAMP' % n_amp] <= 20]

    # 筛选“60日均线>最高价”
    curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    # 上影线涨幅 > 3%
    curr_data = curr_data[(curr_data['high'] - curr_data['body_high']) / curr_data['body_high'] >= 0.03]
    # 下影线跌幅 > -1%
    curr_data = curr_data[(curr_data['low'] - curr_data['body_low']) / curr_data['body_low'] > -0.01]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def ChangXiaYingXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的“长下影线”
    """
    reader, data = read_data(stock_code)
    curr_data = data.copy()
    # body_high：实体的最高位
    curr_data['body_high'] = [max(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]
    # body_high：实体的最低位
    curr_data['body_low'] = [min(open, close) for
                             open, close in zip(curr_data['open'], curr_data['close'])]

    # 上影线涨幅 < 1%
    curr_data = curr_data[(curr_data['high'] - curr_data['body_high']) / curr_data['body_high'] <= 0.01]
    # 下影线跌幅 > -3%
    curr_data = curr_data[(curr_data['low'] - curr_data['body_low']) / curr_data['body_low'] <= -0.03]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(5, 10, 20, 60,), )

    return results

# Deprecated
def LowChangXiaYingXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    低位长下影线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    curr_data = data.copy()
    # body_high：实体的最高位
    curr_data['body_high'] = [max(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]
    # body_high：实体的最低位
    curr_data['body_low'] = [min(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]

    # 上影线涨幅 < 1%
    curr_data = curr_data[(curr_data['high'] - curr_data['body_high']) / curr_data['body_high'] <= 0.01]
    # 下影线跌幅 > -3%
    curr_data = curr_data[(curr_data['low'] - curr_data['body_low']) / curr_data['body_low'] <= -0.03]

    # 筛选“60日均线>最高价”
    curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


class LowChangXiaYingXian_AnalysisOne(AnalysisOne):
    """
    低位长下影线
    """

    def filter_data(self, data: DataFrame, *args, **kwargs):
        data = data[-6 <= (data['pct_chg']) & (data['pct_chg'] <= -3)]

        data = data[(4 <= data['turnover_rate']) & (data['turnover_rate'] < 8)]

        data = data[(500000 <= data['circ_mv']) & (data['circ_mv'] < 5000000)]

        data = data[(-7 <= data['lower_wick']) & (data['lower_wick'] < -3)]

        data = data[(-7 <= data['gap']) & (data['gap'] < -3)]

        # 股价处于低位
        data = data[data['RPY1'] <= 10]
        data = data[data['RPM1'] <= 10]

        return data


def HighChangXiaYingXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    高位长下影线
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    curr_data = data.copy()
    # body_high：实体的最高位
    curr_data['body_high'] = [max(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]
    # body_high：实体的最低位
    curr_data['body_low'] = [min(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]

    # 上影线涨幅 < 1%
    curr_data = curr_data[(curr_data['high'] - curr_data['body_high']) / curr_data['body_high'] <= 0.01]
    # 下影线跌幅 > -3%
    curr_data = curr_data[(curr_data['low'] - curr_data['body_low']) / curr_data['body_low'] <= -0.03]

    # 筛选“60日均线<最低价”
    curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def ShiZiXing(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    十字星
    """
    reader, data = read_data(stock_code)
    curr_data = data.copy()
    # body_high：实体的最高位
    curr_data['body_high'] = [max(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]
    # body_high：实体的最低位
    curr_data['body_low'] = [min(open, close) for open, close in zip(curr_data['open'], curr_data['close'])]

    # 实体振幅 < 0.15%
    curr_data = curr_data[(curr_data['body_high'] - curr_data['body_low']) / curr_data['body_low'] * 100 < 0.15]
    # 上影线振幅 >= 2%
    curr_data = curr_data[(curr_data['high'] - curr_data['body_high']) / curr_data['body_high'] * 100 >= 2]
    # 下影线振幅 >= 2%
    curr_data = curr_data[(curr_data['body_low'] - curr_data['low']) / curr_data['low'] * 100 >= 2]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def Anyone(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
        new_data=False,
):
    """
    通用
    """
    if new_data:
        data = TuShareDataReader(stock_code.split(".")[0]).read_data()
    else:
        reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(5, 10, 20, 60,))  # 增加均线
    data = add_QRR(data)  # 增加量比

    n_amp = 40
    data = add_amplitude(data, day_list=(n_amp,))
    curr_data = data

    # 获取该股票历史上的大阳线数据
    curr_data = curr_data[curr_data['pct_chg'] >= 6]
    # 筛选“开盘价==最低价”
    curr_data = curr_data[curr_data['open'] == curr_data['low']]
    # 筛选“收盘价==最高价”
    curr_data = curr_data[curr_data['close'] == curr_data['high']]

    # 放量：量比>=2
    curr_data = curr_data[curr_data['5QRR'] > 2.]

    # 筛选“120日均线>最高价”
    # curr_data = curr_data[curr_data['80MA'] > curr_data['high']]

    # 经过一段时间的横盘整理：近40天振幅<平均40天振幅
    curr_data = curr_data[curr_data[f'{n_amp}AMP'] < data[f'{n_amp}AMP'].mean()]

    # 突破所有均线
    # 最低点<60MA<最高点
    curr_data = curr_data[(curr_data['low'] < curr_data['60MA']) & (curr_data['60MA'] < curr_data['high'])]
    # 最低点<20MA<最高点
    curr_data = curr_data[(curr_data['low'] < curr_data['20MA']) & (curr_data['20MA'] < curr_data['high'])]
    # 最低点<10MA<最高点
    curr_data = curr_data[(curr_data['low'] < curr_data['10MA']) & (curr_data['10MA'] < curr_data['high'])]
    # 最低点<5MA<最高点
    curr_data = curr_data[(curr_data['low'] < curr_data['5MA']) & (curr_data['5MA'] < curr_data['high'])]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(5, 10, 20, 60), )

    return results


if __name__ == '__main__':
    # DaYangXian("000001.SZ", print_result=True)  # 分析某只股票的大阳线
    # analysis_A_share()  # 分析整个大陆股市的大阳线
    # LowDaYangXian("000006.SZ", print_result=True)  # 分析某只股票的高位大阳线
    # analysis_A_share(line_type='low_DaYangXian')  # 分析整个A股的低位大阳线
    # HighDaYangXian("000007.SZ", print_result=True, plot_examples=True)  # 分析某只股票的高位大阳线
    # analysis_A_share(line_type='high_DaYangXian')  # 分析整个A股的高位大阳线
    # BareDaYangXian("000010", print_result=True, plot_examples=False)  # 分析某只股票的光头光脚大阳线
    # analysis_A_share(BareDaYangXian)  # 分析整个A股的光头光脚大阳线
    LowBareDaYangXian("000014.SZ", print_result=True, plot_examples=True)  # 分析某只股票的低位光头光脚大阳线
    # analysis_A_share(line_type='low_bare_DaYangXian')  # 分析整个A股的低位光头光脚大阳线
    # HighBareDaYangXian("000011.SZ", print_result=True, plot_examples=True)  # 分析某只股票的高位光头光脚大阳线
    # analysis_A_share(line_type='high_bare_DaYangXian')  # 分析整个A股的高位光头光脚大阳线
    # DaYinXian("000001.SZ", print_result=True, plot_examples=True)  # 分析某只股票的大阴线
    # analysis_A_share(line_type='normal_DaYinXian')  # 分析整个A股的高位光头光脚大阳线
    # LowDaYinXian("000001.SZ", print_result=True, plot_examples=True)  # 分析某只股票的低位大阴线
    # analysis_A_share(line_type='low_DaYinXian')  # 分析整个A股的低位大阴线
    # HighDaYinXian("000002.SZ", print_result=True, plot_examples=True)  # 分析某只股票的高位大阴线
    # analysis_A_share(line_type='high_DaYinXian')  # 分析整个A股的高位大阴线
    # BareDaYinXian("000004.SZ", print_result=True, plot_examples=True)  # 分析某只股票的光头光脚大阴线
    # analysis_A_share(line_type='bare_DaYinXian')  # 分析整个A股的光头光脚大阴线
    # LowBareDaYinXian("000014.SZ", print_result=True, plot_examples=True)  # 分析某只股票的低位光头光脚大阴线
    # analysis_A_share(line_type='low_bare_DaYinXian')  # 分析整个A股的低位光头光脚大阴线
    # HighBareDaYinXian("000004.SZ", print_result=True, plot_examples=True)  # 分析某只股票的低位光头光脚大阴线
    # analysis_A_share(line_type='high_bare_DaYinXian')  # 分析整个A股的高位光头光脚大阴线
    # ChangShangYingXian("000001.SZ", print_result=True, plot_examples=True)  # 分析某只股票的长上影线
    # analysis_A_share(line_type='normal_ChangShangYingXian')  # 分析整个A股的长上影线
    # LowChangShangYingXian("000001.SZ", print_result=True, plot_examples=True)  # 分析某只股票的低位长上影线
    # analysis_A_share(line_type='low_ChangShangYingXian')  # 分析整个A股的低位长上影线
    # HighChangShangYingXian("000001.SZ", print_result=True, plot_examples=True)  # 分析某只股票的高位长上影线
    # analysis_A_share(line_type='high_ChangShangYingXian')  # 分析整个A股的低位长上影线
    # XianRenZhiLu("000005.SZ", print_result=True, plot_examples=True)  # 分析某只股票的仙人指路长上影线
    # analysis_A_share(line_type='XianRenZhiLu')  # 分析整个A股的仙人指路长上影线
    # ChangXiaYingXian("000001.SZ", print_result=True, plot_examples=True)  # 分析某只股票的长下影线
    # analysis_A_share(line_type='normal_ChangXiaYingXian')  # 分析整个A股的长下影线
    # LowChangXiaYingXian("000001.SZ", print_result=True, plot_examples=True)  # 分析某只股票的低位长下影线
    # analysis_A_share(line_type='low_ChangXiaYingXian')  # 分析整个A股的低位长下影线
    # HighChangXiaYingXian("000001.SZ", print_result=True, plot_examples=True)  # 分析某只股票的高位长下影线
    # analysis_A_share(line_type='high_ChangXiaYingXian')  # 分析整个A股的高位长下影线
    # ShiZiXing("000001.SZ", print_result=True, plot_examples=True)  # 分析某只股票的十字星
    # analysis_A_share(ShiZiXing)  # 分析整个A股的十字星

    # Anyone("600476.SH", print_result=True, plot_examples=True)
    # analysis_A_share(Anyone)

    # Anyone("301183", print_result=True, plot_examples=True, new_data=True)
