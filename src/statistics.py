# -*- coding: UTF-8 -*-

"""
统计分析
"""

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import pandas as pd
from pandas import DataFrame
from utils.db import SqliteDB


class CommonStatistics(object):

    def __init__(self, cache=False):
        self.cache = cache

        self.db = SqliteDB()

        self.cache_map = {}


    def low_avg_pe_ttm(self, start_date, end_date, n: int):
        f"""
        统计从{start_date}到{end_date}期间，平均市盈率最低的n个公司。
        注意：该期间，不能出现亏损，即pe_ttm必须大于0
        """
        cache_key = f'low_avg_pe_ttm_{start_date}_{end_date}'
        if self.cache and cache_key in self.cache_map:
            return self.cache_map[cache_key]

        data = self.db.select_union('daily_basic',
                                    f"select avg(case when pe_ttm>0 then pe_ttm else 99999999 end) as pe_ttm "
                                    f"from {{table_name}} where trade_date>='{start_date}' and trade_date<='{end_date}'",
                                    add_stock_code=True)

        data = data[data['pe_ttm'] > 0]
        data = data.sort_values(by='pe_ttm').iloc[:n]

        if self.cache:
            self.cache_map[cache_key] = data

        return data
