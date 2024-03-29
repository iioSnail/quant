# -*- coding: UTF-8 -*-

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

from src.find_stock import PickStock
from utils.data_reader import TuShareDataReader


class DayangXianLowTurnover_PickStock(PickStock):
    """
    大阳线低换手率
    """

    def buy_or_not(self, reader: TuShareDataReader) -> bool:
        # 以下是样例，子类重写这段代码即可。
        last_daily_data = self.get_last_daily_data(reader)
        last_daily_basic_data = self.get_last_daily_basic_data(reader)

        # 当天涨幅 >= 6%
        is_DaYangXian = (last_daily_data['close'] - last_daily_data['open']) / last_daily_data['open'] * 100 >= 6
        # 换手率低
        low_turnover = last_daily_basic_data['turnover_rate'] <= 0.5

        return is_DaYangXian and low_turnover


if __name__ == '__main__':
    pick_stock = DayangXianLowTurnover_PickStock()
    pick_stock.find_stock().print_stocks()
