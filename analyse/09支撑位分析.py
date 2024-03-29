# -*- coding: UTF-8 -*-

import sys
import time
from pathlib import Path

from pandas import DataFrame

from utils.data_reader import TuShareDataReader
from utils.utils import time_func, is_null

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

# 根据历史K线数据分析某种K线组合对后续股价的影响
import pandas as pd
from src.analysis import get_results, plot_results, analysis_A_share, analysis_factor, AnalysisOne
from utils.data_process import read_data, get_all_china_ts_code_list, add_MA, add_amplitude, add_QRR, \
    remove_continual_data
from utils.date_utils import date_add


class Support_AnalysisOne(AnalysisOne):
    """
    通过支撑位选股，当股价到达支撑位附近时买入股票

    结果看一般，也就50%左右。
    """

    def filter_data(self, data: DataFrame,
                    window=60,
                    offset=10,
                    interval_days=30,  # 间隔天数。即今天买入后，{interval_days}天之内不再买入相同股票。
                    *args, **kwargs):
        # 获取支撑位的数据
        support_data = self.reader.find_support(window=window, offset=offset)
        support_data = support_data[['low']]

        result_data_list = []

        i = -1
        while i < len(data) - 1:
            i += 1

            data_item = data.iloc[i:i + 1]
            trade_date = data_item.iloc[0].name

            # 找出近n天的支撑位。结束日期减去窗口大小，要不然会出现昨天是支撑位，今天就买入的情况。那肯定就不对了
            support_items = support_data[date_add(trade_date, -365):date_add(trade_date, -window)]
            if len(support_items) <= 0:
                continue

            # FIXME，该股价虽然在近一年内是支撑位，但现在有可能已经变成阻力位了。
            # 如果当天最低价是近一年内的某个支撑位，则符合买入条件。
            if (((data_item.iloc[0]['low'] - support_items['low']) / support_items['low'] * 100).abs() <= 5).any():
                result_data_list.append(data_item)
                i += interval_days  # 买入后直接往后跳n天。

            continue

        if is_null(result_data_list):
            return None

        data = pd.concat(result_data_list)
        data = remove_continual_data(data)

        return data


if __name__ == '__main__':
    # Support_AnalysisOne("000002", print_result=True).analysis(window=40, offset=20, interval_days=0)

    # time_func(Support_AnalysisOne("000002", print_result=False), times=10, window=200, offset=10, interval_days=40)

    analysis_A_share(Support_AnalysisOne, line_func_kwargs={
        "window": 300,
        "offset": 30,
        "interval_days": 30,
    }, limit=-1)  # 有效
