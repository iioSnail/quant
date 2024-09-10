"""
This experiment is aim to analyze the A-share market, such as its average PE, overall market value, etc.
"""

import sys
from pathlib import Path
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

from data_reader.index.index_daily import IndexDailyDataReader
from utils.plot_utils import plot_multiple_lines

def main():
    data = IndexDailyDataReader("上证指数").get_data()

    dates = list(data.index)
    close_data = list(data['index_close'])
    labels = ['close']

    plot_multiple_lines(dates, [close_data], labels, title="ShangZheng Index", ylabel="Close", xlabel="Date")



if __name__ == '__main__':
    main()
