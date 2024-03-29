# -*- coding: UTF-8 -*-
from math import ceil
from pathlib import Path

import mplfinance as mpf
import numpy as np
import pandas as pd
from pandas import DataFrame

from utils.data_process import format_tushare, add_MA

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class CandlePlot(object):

    def __init__(self,
                 stock_code: str = None,
                 data: DataFrame = None,
                 dir_path='data',
                 ):
        self.stock_code = stock_code
        self.dir_path = ROOT / dir_path

        if data is not None:
            self.data = data.copy()
        else:
            self.filename = self.dir_path / ('%s.csv' % stock_code)

            self.data = pd.read_csv(self.filename)
            self.data = format_tushare(self.data)

        self.plot_type = 'candle'  # 绘制类型，默认为K线
        self.show_volume = False  # 是否绘制交易量

        self.ma_list = ()  # 绘制均线

        self.extra_plots = []

        # 设置样式
        color_style = mpf.make_marketcolors(
            up="red",  # 上涨K线的颜色
            down="green",  # 下跌K线的颜色
            edge='inherit',  # K线箱体不描边
            volume="inherit",  # 成交量与上涨下跌保持一致
        )

        self.style = mpf.make_mpf_style(
            marketcolors=color_style,
        )

    def type(self, plot_type):
        self.plot_type = plot_type
        return self

    def volume(self):
        self.show_volume = True
        return self

    def vol(self):
        self.show_volume = True
        return self

    def add_MA(self, ma_list=(5, 10, 20, 60)):
        self.ma_list = ma_list

        return self

    def add_plot(self, plot):
        self.extra_plots.append(plot)

        return self

    def add_marks(self, indices, date_indices=False):
        """
        增加标记点
        :param indices: 位置列表，例如：[1, 20, 50]表示在1,20,50这三个位置增加标记点
                        或['2012-12-10', '2012-12-15']表示在这两天进行标记点，如果使用日期，则需要指定 date_indices=True
        """
        if not '_mark_point' in self.data:
            self.data['_mark_point'] = np.NAN

        if date_indices:
            self.data.loc[indices, '_mark_point'] = self.data.loc[indices, 'low'] - 0.2
        else:
            self.data.loc[self.data.iloc[indices].index, '_mark_point'] = self.data.iloc[indices]['low'] - 0.2
        return self

    def color_windows(self, win_num: int):
        """
        将data分成n个窗口，然后对不同的窗口使用不同的背景色
        """
        win_len = ceil(len(self.data) / win_num)

        y1 = np.array([self.data['high'].max()] * len(self.data))
        y2 = np.array([self.data['low'].min()] * len(self.data))

        offset = 0
        while offset < len(self.data):
            where_values = np.array([False] * len(self.data))
            where_values[offset:offset + win_len] = True

            apt = mpf.make_addplot(self.data['close'],
                                   fill_between=dict(y1=y1, y2=y2,
                                                     where=where_values,
                                                     color='%.2f' % (offset / len(self.data)),
                                                     alpha=0.2))
            self.add_plot(apt)

            offset = offset + win_len

        return self

    def plot(self, start: str = None, end: str = None):
        if len(self.ma_list) > 0:
            self.data = add_MA(self.data, self.ma_list)

        data = self.data

        if start is not None:
            data = data.loc[start:, :]

        if end is not None:
            data = data.loc[start:, :]

        extra_plots = self.extra_plots.copy()
        for n in self.ma_list:
            ma_color = None
            if n == 5:
                ma_color = 'blue'
            if n == 10:
                ma_color = 'orange'
            if n == 20:
                ma_color = 'purple'
            if n == 60:
                ma_color = 'green'

            apt = mpf.make_addplot(data['%dMA' % n], color=ma_color)
            extra_plots.append(apt)

        if '_mark_point' in data and data['_mark_point'].count() > 0:
            apt = mpf.make_addplot(data['_mark_point'], type='scatter',
                                   markersize=50, marker='^', color='b')
            extra_plots.append(apt)

        plot_args = {}
        if self.stock_code is not None:
            plot_args['title'] = self.stock_code

        # todo fixme
        mpf.plot(data,
                 datetime_format='%Y-%m-%d',
                 type=self.plot_type,
                 volume=self.show_volume,
                 style=self.style,
                 figratio=(6, 3),
                 addplot=extra_plots,
                 **plot_args)


if __name__ == '__main__':
    plot = CandlePlot('002364.SZ') \
        .volume() \
        .type("line") \
        .add_marks([5, 20, 100], date_indices=False) \
        .color_windows(5) \
        .plot()
