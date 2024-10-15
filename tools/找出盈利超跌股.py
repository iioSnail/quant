# -*- coding: UTF-8 -*-

import sys
from pathlib import Path

from data_reader.base import DataReader

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

from src.find_stock import PickStock


class OverfallPickStock(PickStock):

    def buy_or_not(self, reader: DataReader) -> bool:
        daily_basic = self.get_last_data(reader, 'daily_basic')
        has_profit = daily_basic['pe'] > 0 and daily_basic['pe_ttm'] > 0

        if not has_profit:
            return False

        daily = self.get_last_daily_data(reader)
        his_daily = self.get_his_data(365 * 7, reader, 'daily')

        high_price = his_daily['high'].max()
        curr_price = daily['close']

        change = (curr_price - high_price) / high_price

        return change <= -0.7


if __name__ == '__main__':
    OverfallPickStock().find_stock().print_stocks()