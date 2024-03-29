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


class LowChangXiaYingXian_AnalysisOne(AnalysisOne):
    """
    低位长下影线
    """

    def filter_data(self, data: DataFrame, *args, **kwargs):
        data = data[(-6 <= data['pct_chg']) & (data['pct_chg'] <= -3)]

        data = data[(4 <= data['turnover_rate']) & (data['turnover_rate'] <= 8)]

        # data = data[(500000 <= data['circ_mv']) & (data['circ_mv'] <= 5000000)]
        #
        # data = data[(-7 <= data['lower_wick']) & (data['lower_wick'] <= -3)]
        #
        # data = data[(-7 <= data['gap']) & (data['gap'] <= -3)]
        #
        # # 股价处于低位
        # data = data[data['RPY1'] <= 10]
        # data = data[data['RPM1'] <= 10]

        return data

    def _require_data(self):
        return ('daily', 'daily_basic', 'daily_extra',)
    

def demo_test_1():
    common_analysis(
        factors_range={
            'pct_chg': (-8, -2),
            'turnover_rate': (2, 8),
            'circ_mv': (500000, 5000000),
            'lower_wick': (-8, -2),
            'gap': (-8, -2),
            'RPY1': (0, 10),
            'RPM1': (0, 10),
        },
        limit=-1
    )


def 近一年超跌():
    common_analysis(factors_range={
        'FPY1': (-100, -60),
        'FAR': (-100, -6.25),
        'turnover_rate_f': (1, 7),
        'upper_wick': (0, 2.5),
        'pe_ttm': (3, 75),
    }, remove_continual=True, limit=10)

if __name__ == '__main__':
    近一年超跌()