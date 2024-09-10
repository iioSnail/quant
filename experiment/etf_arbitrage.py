"""
This experiment aims to study whether we can make a profit by taking
advantage of ETF discounts.

My idea is as follows:
The NASDAQ trading hours are completely different from the A-share.
In the A-share market, there are some NASDAQ ETFs that track the
NASDAQ index. However, due to fluctuations in market sentiment,
NASDAQ ETFs are often traded at a discount compared to the NASDAQ
index. That means I can buy the NASDAQ ETF when it is slightly discount,
and sell it the next day when the discount disappears.

I'm not sure if my idea will be profitable, so I'm conducting this
experiment to verify.
"""

import sys
from pathlib import Path
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

from data_reader.index.index_global import IndexGlobalDataReader
from data_reader.fund.fund_daily import FundDailyDataReader
from utils.plot_utils import plot_multiple_lines

def main(start_date, end_date):
    index_data = IndexGlobalDataReader("纳指").get_data()
    etf_data = FundDailyDataReader("513300").get_data()

    index_data = index_data[(index_data.index >= start_date) & (index_data.index <= end_date)]
    etf_data = etf_data[(etf_data.index >= start_date) & (etf_data.index <= end_date)]

    start_date = etf_data.iloc[0].name
    index_data = index_data[index_data.index >= start_date]

    # The first day's closing price is as the initial price.
    etf_init_price = etf_data.iloc[0]['close']
    etf_trend = (1 / etf_init_price) * etf_data['close']

    index_init_price = index_data.iloc[0]['close']
    index_trend = (1 / index_init_price) * index_data['close']

    # Computing our profits. The strategy is that buying the stock
    # if the change of ETF less than the change of index by 0.5% percent.
    curr_profit = 1
    my_trend = []
    for trade_date, etf_row in etf_data.iterrows():
        # Get the NASDAQ stock data in the last trading day.
        index_data_part = index_data[index_data.index < trade_date]
        if len(index_data_part) <= 0:
            my_trend.append([trade_date, curr_profit])
            continue
        index_row = index_data_part.iloc[-1]

        # Get the ETF data in the next trading day.
        etf_data_part = etf_data[etf_data.index > trade_date]
        if len(etf_data_part) <= 0:
            my_trend.append([trade_date, curr_profit])
            continue
        etf_next_row = etf_data_part.iloc[0]

        """
        etf_row index_row
        0.2     0.9         -0.6    √
        0.5     0.5         0       ×
        0.2     0.0         0.2     ×
        0.2     -0.5        0.7     ×
        """
        if etf_row['pct_chg'] - index_row['pct_chg'] <= -0.5:
            curr_profit = curr_profit * (1 + (etf_next_row['close'] - etf_row['close']))

        my_trend.append([trade_date, curr_profit])

    my_trend = pd.DataFrame(my_trend, columns=['trade_date', 'profit'])
    my_trend = my_trend.set_index('trade_date')['profit']

    dates = list(set(etf_trend.index).intersection(set(index_trend.index)).intersection(set(my_trend.index)))
    dates = sorted(dates)
    etf_trend = etf_trend[dates]
    index_trend = index_trend[dates]
    my_trend = my_trend[dates]

    # Plot the result.
    etf_trend_data = list(etf_trend)
    index_trend_data = list(index_trend)
    my_trend_data = list(my_trend)
    data_sets = [etf_trend_data, index_trend_data, my_trend_data]
    labels = ['etf', 'nasdaq', 'ours']

    plot_multiple_lines(dates, data_sets, labels, title="Profit Chart", ylabel="Change", xlabel="Date")
if __name__ == '__main__':
    main(start_date='2021-11-08', end_date='2023-01-01')
