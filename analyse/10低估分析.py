# -*- coding: UTF-8 -*-

import sys
from pathlib import Path

from pandas import DataFrame

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

# 根据历史K线数据分析某种K线组合对后续股价的影响
import pandas as pd
from src.analysis import get_results, plot_results, analysis_A_share, analysis_factor, AnalysisOne, common_analysis
from utils.data_process import read_data, get_all_china_ts_code_list, add_MA, add_amplitude, add_QRR

cache = {}


class LowPeTTM_AnalysisOne(AnalysisOne):
    """
    每天筛选近1年平均市盈率最低的50个股票，然后看它们后续的涨跌情况
    """

    def low_avg_pe_ttm(self, start_date, end_date, n: int):
        f"""
        统计从{start_date}到{end_date}期间，平均市盈率最低的n个公司。
        注意：该期间，不能出现亏损，即pe_ttm必须大于0
        """
        cache_key = f"{start_date}_{end_date}"
        if cache_key in cache:
            return cache[cache_key]

        data = self.db.select_union('daily_basic',
                                    f"select avg(case when pe_ttm>0 then pe_ttm else 99999999 end) as pe_ttm "
                                    f"from {{table_name}} where trade_date>='{start_date}' and trade_date<='{end_date}'",
                                    add_stock_code=True)

        data = data[data['pe_ttm'] > 0]
        data = data.sort_values(by='pe_ttm').iloc[:n]

        cache[cache_key] = data
        return data

    def filter_data(self, data: DataFrame, *args, **kwargs):
        # 获取该股票历史上的大阳线数据
        data['day_chg'] = (data['close'] - data['open']) / data['open'] * 100

        # 当天涨幅>=6%
        data = data[data['day_chg'] >= 6.]

        # 换手率较低
        min_turnover, max_turnover = turnover_rate
        data = data[(min_turnover <= data['turnover_rate']) & (data['turnover_rate'] < max_turnover)]

        ######### 新增探索指标 ###############
        # 股价处于低位
        # curr_data = curr_data[curr_data['RPY1'] <= 25]

        return data


def DaYinXian(
        stock_code,
        future_days=(5, 10, 20, 40, 60),
        print_result=False,
        plot_examples=False,
):
    """
    """
    reader, daily_data = read_data(stock_code)
    # 获取该股票历史上的大阳线数据

    daily_data['day_chg'] = (daily_data['close'] - daily_data['open']) / daily_data['open'] * 100

    curr_data = daily_data

    # 当天涨幅<=-6%
    curr_data = curr_data[curr_data['day_chg'] <= -6.]

    # 换手率较低
    # curr_data = curr_data[curr_data['turnover_rate'] <= 0.5]

    results = get_results(stock_code, curr_data, daily_data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, daily_data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def DaYinXian_LowTurnover(
        stock_code,
        future_days=(5, 10, 20, 40, 60),
        print_result=False,
        plot_examples=False,
):
    """
    当出现大阴线时，通常伴随高换手率。

    实验结果：若`当日涨幅<=-6%`，则`平均换手率<7%`。
    """
    reader, daily_data = read_data(stock_code)
    # 获取该股票历史上的大阳线数据

    daily_data['day_chg'] = (daily_data['close'] - daily_data['open']) / daily_data['open'] * 100

    curr_data = daily_data

    # 当天涨幅<=-6%
    curr_data = curr_data[curr_data['day_chg'] <= -6.]

    # 换手率较低
    curr_data = curr_data[curr_data['turnover_rate'] <= 1]

    results = get_results(stock_code, curr_data, daily_data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, daily_data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def DaYinXian_HighTurnover(
        stock_code,
        future_days=(5, 10, 20, 40, 60),
        print_result=False,
        plot_examples=False,
):
    """
    大阴线，高换手率
    """
    reader, daily_data = read_data(stock_code)

    daily_data['day_chg'] = (daily_data['close'] - daily_data['open']) / daily_data['open'] * 100

    curr_data = daily_data

    # 当天涨幅<=-6%
    curr_data = curr_data[curr_data['day_chg'] <= -6.]

    # 换手率较高
    curr_data = curr_data[curr_data['turnover_rate'] >= 35.]

    results = get_results(stock_code, curr_data, daily_data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, daily_data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def 大盘跌个股涨():
    """
          | n个交易日后 | 样本总数 | 上涨样本数 | 上涨概率 | 平均涨跌幅 | 上涨时平均涨幅 | 下跌时平均跌幅 | 样本集中度 | 胜率标准差 |
    |:-----------:|:--------:|:----------:|:--------:|:----------:|:--------------:|:--------------:|:----------:|:----------:|
    |      5      |   9357   |    4267    |   45.6   |   1.06%    |     11.93%     |     -8.05%     |   22.68%   |   7.52%    |
    |      10     |   9357   |    4116    |  43.99   |   1.37%    |     15.61%     |     -9.81%     |   22.68%   |   7.11%    |
    |      20     |   9357   |    4117    |   44.0   |   1.87%    |     19.51%     |    -11.98%     |   22.68%   |   7.50%    |
    |      40     |   9357   |    3607    |  38.55   |   -0.13%   |     24.08%     |    -15.32%     |   22.67%   |   10.81%   |
    |      60     |   9357   |    3620    |  38.69   |   1.63%    |     28.79%     |     -15.5%     |   19.11%   |   15.67%   |
    """
    common_analysis(
        factors_range={
            'pct_chg': (3, 999),
            'index_pct_chg': (-999, -2),
        },
        limit=-1
    )


def 大盘涨个股跌():
    """
          | n个交易日后 | 样本总数 | 上涨样本数 | 上涨概率 | 平均涨跌幅 | 上涨时平均涨幅 | 下跌时平均跌幅 | 样本集中度 | 胜率标准差 |
    |:-----------:|:--------:|:----------:|:--------:|:----------:|:--------------:|:--------------:|:----------:|:----------:|
    |      5      |   5236   |    2593    |  49.52   |   0.39%    |     7.92%      |     -7.0%      |   19.06%   |   18.98%   |
    |      10     |   5236   |    2699    |  51.55   |   1.19%    |     10.65%     |     -8.88%     |   19.06%   |   21.08%   |
    |      20     |   5236   |    2578    |  49.24   |   1.21%    |     14.69%     |    -11.87%     |   19.06%   |   20.07%   |
    |      40     |   5236   |    2108    |  40.26   |   -1.95%   |     21.06%     |    -17.46%     |   17.23%   |   19.23%   |
    |      60     |   5236   |    2094    |  39.99   |   0.65%    |     28.32%     |    -17.79%     |   15.04%   |   20.76%   |
    """
    common_analysis(
        factors_range={
            'pct_chg': (-999, -3),
            'index_pct_chg': (2, 999),
        },
        limit=-1
    )


if __name__ == '__main__':
    # LowTurnoverRate("000001", print_result=True, plot_examples=False)
    # analysis_A_share(DaYangXianLowTurnover_AnalysisOne, line_func_kwargs={
    #     'turnover_rate': (0, 10)
    # }, limit=-1)  # 有效
    # DaYangXian_LowTurnoverRate("000001", turnover_rate=(1, 10), print_result=True)

    大盘涨个股跌()
