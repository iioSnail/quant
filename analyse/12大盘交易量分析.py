# -*- coding: UTF-8 -*-

import sys
from pathlib import Path

from utils.log_utils import HiddenPrints

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import numpy as np

from data_reader.index.index_daily import IndexDailyDataReader
from src.analysis import get_results

"""
分析大盘K线图与交易量的关系。

分析背景：
1. 个股容易受主力、庄家、游资等影响，导致交易量和上涨关系混乱。
   但大盘不一样，参与人数众多，一个人或组织很难影响。
2. A股中，大盘对个股的走势影响较大。一般同涨同跌。

网格分析（不同影响因素组合）：
1. 成交量：与近一年的平均值相比，成家量上涨或下跌的百分比
   分为：极度缩量，缩量，正常量，放量，极度放量
2. K线：当天的收盘价与开盘价相比的涨幅。
   分为：大阴线、中阴线、小阴线、小阳线、中阳线、大阳线
3. 当天涨幅：当天收盘价和前一天收盘价相比
   分为：大跌、中跌、小跌、小涨、中涨、大涨
4. 位置：看上证指数的当前位置
"""


def index_volume_analyse(
        vol_range,
        k_range,
        pct_range,
        pos_range,
):
    index_code = "上证指数"

    data = IndexDailyDataReader(index_code, index_prefix=False).get_data()
    data['index'] = list(range(1, len(data) + 1))

    data_bak = data.copy()

    # 当天成交量与近一年成交量的百分比
    amount_mean = data['amount'].rolling(250).mean()
    data['vol_pos'] = (data['amount'] - amount_mean) / amount_mean * 100
    # 当天的K线情况
    data['k_line'] = (data['close'] - data['open']) / data['open'] * 100
    # 当日上涨情况
    # data['pct_chg']

    data = data[(pos_range[0] < data['close']) & (data['close'] < pos_range[1])]
    data = data[(vol_range[0] < data['vol_pos']) & (data['vol_pos'] < vol_range[1])]
    data = data[(k_range[0] < data['k_line']) & (data['k_line'] < k_range[1])]
    data = data[(pct_range[0] < data['pct_chg']) & (data['pct_chg'] < pct_range[1])]

    print("vol_range:", vol_range, "; k_range:", k_range, "; pct_range:", pct_range, "; pos_range:", pos_range)
    results = get_results(index_code,
                          data,
                          data_bak,
                          future_days=[1, 2, 3, 5, 10, 20, 40, 60],
                          print_result=True,
                          remove_continual=True,
                          remove_2015=False,
                          )

    return results


def main():
    vol_map = {
        "常量": (-20, 30),
        "温和放量": (30, 120),
        "放量": (120, 200),
        "巨量": (200, 9999),
        "温和缩量": (-40, -20),
        "缩量": (-60, -40),
        "严重缩量": (-100, -60),
    }

    k_map = {
        "小阳线": (0, 1),
        "中阳线": (1, 2.5),
        "大阳线": (2.5, 4),
        "巨阳线": (4, 999),
        "小阴线": (-1, 0),
        "中阴线": (-2.5, -1),
        "大阴线": (-4, -2.5),
        "巨阴线": (-999, -4),
    }

    pct_map = {
        "小涨": (0, 1),
        "中涨": (1, 2.5),
        "大涨": (2.5, 4),
        "巨涨": (4, 999),
        "小跌": (-1, 0),
        "中跌": (-2.5, -1),
        "大跌": (-4, -2.5),
        "巨跌": (-999, -4),
    }

    pos_map = {
        "严重低位": (0, 2000),
        "低位": (2000, 2500),
        "较低位": (2500, 2850),
        "正常位": (2850, 3150),
        "较高位": (3150, 3300),
        "高位": (3300, 3700),
        "严重高位": (3700, 9999),
    }

    for vol in vol_map.keys():
        for k in k_map.keys():
            for pct in pct_map.keys():
                for pos in pos_map.keys():
                    # result = index_volume_analyse(
                    #     vol_range=vol_map[vol],
                    #     k_range=k_map[k],
                    #     pct_range=pct_map[pct],
                    #     pos_range=pos_map[pos],
                    # )

                    with HiddenPrints():
                        result = index_volume_analyse(
                            vol_range=vol_map[vol],
                            k_range=k_map[k],
                            pct_range=pct_map[pct],
                            pos_range=pos_map[pos],
                        )

                    if len(result) > 0 and result[0]['total_num'] >= 5:
                        print(vol, k, pct, pos)
                        index_volume_analyse(
                            vol_range=vol_map[vol],
                            k_range=k_map[k],
                            pct_range=pct_map[pct],
                            pos_range=pos_map[pos],
                        )
                        print("-------------------------------")


if __name__ == '__main__':
    main()
