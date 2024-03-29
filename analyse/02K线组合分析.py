# -*- coding: UTF-8 -*-

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

# 根据历史K线数据分析某种K线组合对后续股价的影响

import pandas as pd
from pandas import DataFrame

from utils.data_reader import TuShareDataReader
from utils.data_process import read_data, add_MA, add_amplitude, add_QRR, add_prev_n
from src.analysis import get_results, plot_results, analysis_A_share, common_analysis, AnalysisOne


def SanYangKaiTai(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的三阳开泰
    """
    reader, data = read_data(stock_code)
    # 获取该股票历史上的大阳线数据
    curr_data = data
    curr_data = curr_data[curr_data['pct_chg'].rolling(3).min() >= 3.]

    # 去掉相邻的大阳线。例如连续4个大阳线会被认为是2个三阳开泰，因此去掉第二个
    curr_data = curr_data[(curr_data['index'] - curr_data['index'].rolling(2).min() > 3) \
                          | (curr_data['index'].rolling(2).min().isna())]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def LowSanYangKaiTai(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的低位三阳开泰
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data
    curr_data = curr_data[curr_data['pct_chg'].rolling(3).min() >= 3.]

    # 去掉相邻的大阳线。例如连续4个大阳线会被认为是2个三阳开泰，因此去掉第二个
    curr_data = curr_data[(curr_data['index'] - curr_data['index'].rolling(2).min() > 3) \
                          | (curr_data['index'].rolling(2).min().isna())]

    # 筛选“60日均线>最高价”
    curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def HighSanYangKaiTai(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的低位三阳开泰
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    # 获取该股票历史上的大阳线数据
    curr_data = data
    curr_data = curr_data[curr_data['pct_chg'].rolling(3).min() >= 3.]

    # 去掉相邻的大阳线。例如连续4个大阳线会被认为是2个三阳开泰，因此去掉第二个
    curr_data = curr_data[(curr_data['index'] - curr_data['index'].rolling(2).min() > 3) \
                          | (curr_data['index'].rolling(2).min().isna())]

    # 筛选“60日均线<最低价”
    curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def SanZhiWuYa(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    分析某只股票的三只乌鸦
    """
    reader, data = read_data(stock_code)
    curr_data = data
    curr_data = curr_data[curr_data['pct_chg'].rolling(3).max() <= -3.]

    curr_data = curr_data[(curr_data['index'] - curr_data['index'].rolling(2).min() > 3) \
                          | (curr_data['index'].rolling(2).min().isna())]

    # 筛选“60日均线>最高价”
    curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def LowSanZhiWuYa(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    低位三只乌鸦
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    curr_data = data
    curr_data = curr_data[curr_data['pct_chg'].rolling(3).max() <= -3.]

    curr_data = curr_data[(curr_data['index'] - curr_data['index'].rolling(2).min() > 3) \
                          | (curr_data['index'].rolling(2).min().isna())]

    # 筛选“60日均线>最高价”
    curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def HighSanZhiWuYa(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    高位三只乌鸦
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    curr_data = data
    curr_data = curr_data[curr_data['pct_chg'].rolling(3).max() <= -3.]

    curr_data = curr_data[(curr_data['index'] - curr_data['index'].rolling(2).min() > 3) \
                          | (curr_data['index'].rolling(2).min().isna())]

    # 筛选“60日均线<最低价”
    curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


def ChuShuiFuRong(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    出水芙蓉
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(5, 10, 20, 60, 120))  # 增加均线
    data = add_QRR(data)  # 增加量比
    n_amp = 60
    data = add_amplitude(data, day_list=(n_amp,))
    curr_data = data

    # 大阳线：涨幅>=6%
    curr_data = curr_data[curr_data['pct_chg'] >= 6.]

    # 放量：量比>=2
    curr_data = curr_data[curr_data['5QRR'] > 2.]

    # 筛选“120日均线>最高价”
    curr_data = curr_data[curr_data['120MA'] > curr_data['high']]

    # 经过一段时间的横盘整理：近40天振幅<平均40天振幅*80%
    curr_data = curr_data[curr_data[f'{n_amp}AMP'] < data[f'{n_amp}AMP'].mean() * 0.8]

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
            plot_results(index, data, l_offset=120, r_offset=60, ma_list=(5, 10, 20, 60, 120), )

    return results


def DuanTouZhaDao(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    断头铡刀
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(5, 10, 20, 60, 120))  # 增加均线
    data = add_QRR(data)  # 增加量比
    n_amp = 60
    curr_data = data

    # 大阴线：涨跌幅<=-6%
    curr_data = curr_data[curr_data['pct_chg'] <= -6.]

    # 放量：量比>=2
    curr_data = curr_data[curr_data['5QRR'] > 2]

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
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(5, 10, 20, 60, 120), )

    return results


class 旭日东升_AnalysisOne(AnalysisOne):

    def filter_data(self, data: DataFrame, *args, **kwargs):
        data = add_prev_n(data, 1, include_columns=['pct_chg', 'high'])

        data = data[data['pct_chg'] >= 3.]
        data = data[data['prev_1_pct_chg'] <= -3.]
        # 当天的阳线收盘价比前一天阴线最高点还高：当日最高价>前一天最高价
        data = data[data['close'] > data['prev_1_high']]

        # 处在低位
        data = data[data['RPY1'] <= 25]

        return data

    def _require_data(self):
        return ('daily', 'daily_extra',)


def WuYunGaiDing(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    乌云盖顶
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(5, 10, 20, 60))  # 增加均线

    # 求连续的两天，data_1为1,2,3,4,...， data_2为2,3,4,5,...
    data_1 = data.iloc[:-1].copy()
    data_2 = data.iloc[1:].copy()

    # 求前一天的涨跌
    data_2['prev_pct_chg'] = list(data_1['pct_chg'])
    # 求前一天的最高价
    data_2['prev_low'] = list(data_1['low'])

    curr_data = data_2

    # 当天为中阴线或大阴线：涨跌幅<=-3.%
    curr_data = curr_data[curr_data['pct_chg'] <= -3.]
    # 前一天为中阳线或大阳线：涨跌幅>=3%
    curr_data = curr_data[curr_data['prev_pct_chg'] >= 3.]

    # 当天的阴线收盘价比前一天阳线最低点还低：当日收盘价<前一天最低价
    curr_data = curr_data[curr_data['close'] < curr_data['prev_low']]

    # 股票在上涨行情中：“最低价>60日均线”
    curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(5, 10, 20, 60), )

    return results


def ZaoChenZhiXing(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    早晨之星
    """
    reader, data = read_data(stock_code)
    data = add_prev_n(data, n=1)
    data = add_prev_n(data, n=2)
    data = add_MA(data, day_list=(5, 10, 20, 60))  # 增加均线

    curr_data = data
    # 第一天为大阴线：第一天涨跌幅<=-6%
    curr_data = curr_data[curr_data['prev_2_pct_chg'] <= -6.]

    # 第二天是小阴线或小阳线：-3%<第二天涨跌幅<3%
    curr_data = curr_data[(-3. < curr_data['prev_1_pct_chg']) & (curr_data['prev_1_pct_chg'] < 3.)]

    # 第二天跳空：第二天开盘价<第一天的收盘价
    curr_data = curr_data[curr_data['prev_1_open'] < curr_data['prev_2_close']]

    # 第三天大阳线：第三天涨跌幅>=6%
    curr_data = curr_data[curr_data['pct_chg'] >= 6.]

    # 第三天收复第一天的大部分失地：第三天收盘价 > 第一天的开盘价
    curr_data = curr_data[curr_data['close'] > curr_data['prev_2_open']]

    # 股票经过长期下跌：“60日均线>最高价”
    curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(5, 10, 20, 60), )

    return results


def HuangHunZhiXing(
        stock_code,
        future_days=(5, 10, 20, 40, 60),  # 对多少个交易日后的数据进行分析
        print_result=False,
        plot_examples=False,
):
    """
    黄昏之星
    """
    reader, data = read_data(stock_code)
    data = add_prev_n(data, n=1)
    data = add_prev_n(data, n=2)
    data = add_MA(data, day_list=(5, 10, 20, 60))  # 增加均线

    curr_data = data
    # 第一天为大阳线：第一天涨跌幅>=6%
    curr_data = curr_data[curr_data['prev_2_pct_chg'] >= 6.]

    # 第二天是小阴线或小阳线：-3%<第二天涨跌幅<3%
    curr_data = curr_data[(-3. < curr_data['prev_1_pct_chg']) & (curr_data['prev_1_pct_chg'] < 3.)]

    # 第二天跳空：第二天开盘价>第一天的收盘价
    curr_data = curr_data[curr_data['prev_1_open'] > curr_data['prev_2_close']]

    # 第三天大阴线：第三天涨跌幅<=-6%
    curr_data = curr_data[curr_data['pct_chg'] <= -6.]

    # 第三天跌破第一天的开盘价：第三天收盘价 < 第一天的开盘价
    curr_data = curr_data[curr_data['close'] < curr_data['prev_2_open']]

    # 股票经过长期上涨：“60日均线<最低价”
    curr_data = curr_data[curr_data['60MA'] < curr_data['close']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(5, 10, 20, 60), )

    return results


if __name__ == '__main__':
    # SanYangKaiTai("000002.SZ", print_result=True, plot_examples=True)  # 分析某只股票的三阳开泰
    # analysis_A_share(SanYangKaiTai)  # 分析整个A股的三阳开泰
    # LowSanYangKaiTai("000004.SZ", print_result=True, plot_examples=True)  # 分析某只股票的低位三阳开泰
    # analysis_A_share(LowSanYangKaiTai)  # 分析整个A股的低位三阳开泰
    # HighSanYangKaiTai("000004.SZ", print_result=True, plot_examples=True)  # 分析某只股票的高位三阳开泰
    # analysis_A_share(HighSanYangKaiTai)  # 分析整个A股的高位三阳开泰
    # SanZhiWuYa("000004.SZ", print_result=True, plot_examples=True)  # 分析某只股票的三只乌鸦
    # analysis_A_share(SanZhiWuYa)  # 分析整个A股的三只乌鸦
    # LowSanZhiWuYa("000004.SZ", print_result=True, plot_examples=True)  # 分析某只股票的低位三只乌鸦
    # analysis_A_share(LowSanZhiWuYa)  # 分析整个A股的低位三只乌鸦
    # HighSanZhiWuYa("000005.SZ", print_result=True, plot_examples=True)  # 分析某只股票的低位三只乌鸦
    # analysis_A_share(HighSanZhiWuYa)  # 分析整个A股的高位三只乌鸦
    # ChuShuiFuRong("000005.SZ", print_result=True, plot_examples=True)  # 分析某只股票的出水芙蓉
    # analysis_A_share(ChuShuiFuRong)  # 分析整个A股的出水芙蓉
    # DuanTouZhaDao("000007.SZ", print_result=True, plot_examples=True)  # 分析某只股票的断头铡刀
    # analysis_A_share(DuanTouZhaDao)  # 分析整个A股的断头铡刀
    # XuRiDongSheng("300947", print_result=True, plot_examples=False)  # 分析某只股票的旭日东升
    # analysis_A_share(XuRiDongSheng, limit=-1)  # 分析整个A股的旭日东升
    analysis_A_share(旭日东升_AnalysisOne, limit=-1)
    # WuYunGaiDing("000001.SZ", print_result=True, plot_examples=True)
    # analysis_A_share(WuYunGaiDing)
    # ZaoChenZhiXing("000159.SZ", print_result=True, plot_examples=True)
    # analysis_A_share(ZaoChenZhiXing)
    # HuangHunZhiXing("000006.SZ", print_result=True, plot_examples=True)
    # analysis_A_share(HuangHunZhiXing)
