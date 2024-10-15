# -*- coding: UTF-8 -*-

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

from data_reader.index.index_daily import IndexDailyDataReader
from src.analysis import get_results

"""
分析大盘K线图与交易量的关系。

分析背景：
1. 个股容易受主力、庄家、游资等影响，导致交易量和上涨关系混乱。
   但大盘不一样，参与人数众多，一个人或组织很难影响。
2. A股中，大盘对个股的走势影响较大。一般同涨同跌。

网格分析（不同影响因素组合）：
1. 成交量：与近一年的平均”成交额“相比，所处的百分位。
   分为：极度缩量(0~0.2)，缩量(0.2~0.4)，正常量(0.4~0.6)，放量(0.6~0.8)，极度放量(0.8~1)
2. K线：当天的收盘价与开盘价相比的涨幅。
   分为：大阴线(<-6%)、中阴线(-6%< <-3%)、小阴线(-3%< <0%)、小阳线、中阳线、大阳线
3. 当天涨幅：当天收盘价和前一天收盘价相比
   分为：大跌、中跌、小跌、小涨、中涨、大涨
"""


def index_volume_analyse(
        vol_range,
        k_range,
        pct_range,
):
    index_code = "上证指数"

    data = IndexDailyDataReader(index_code, index_prefix=False).get_data()
    data['index'] = list(range(1, len(data) + 1))

    data_bak = data.copy()

    # 当天成交量与近一年成交量的平均的比值
    data['vol_pos'] = data['amount'] / data['amount'].rolling(250).mean()
    # 当天的K线情况
    data['k_line'] = (data['close'] - data['open']) / data['open'] * 100
    # 当日上涨情况
    # data['pct_chg']

    data = data[(vol_range[0] < data['vol_pos']) & (data['vol_pos'] < vol_range[1])]
    data = data[(k_range[0] < data['k_line']) & (data['k_line'] < k_range[1])]
    data = data[(pct_range[0] < data['pct_chg']) & (data['pct_chg'] < pct_range[1])]

    results = get_results(index_code,
                          data,
                          data_bak,
                          future_days=[5, 10, 20, 40, 60],
                          print_result=True,
                          remove_continual=True
                          )

    return results


if __name__ == '__main__':
    index_volume_analyse(
        vol_range=(0.8, 1.2),
        k_range=(0, 1),
        pct_range=(0, 1)
    )
