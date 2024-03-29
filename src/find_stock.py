# -*- coding: UTF-8 -*-

"""
选股模块：当找出好的买入卖出时机后，并且经过回测验证可行后。就需要用到选股模块。
        该模块是按照设置好的条件，筛选当天（即最后一个交易日）符合条件的股票。
"""

from pandas import Series
from tqdm import tqdm

from utils.data_process import get_all_china_stock_code_list
from utils.data_reader import TuShareDataReader
from utils.date_utils import get_today, date_add
from utils.utils import is_null


class PickStock(object):
    """
    选股函数类。需要选股的时候就继承该类，并实现`buy_or_not`方法。

    使用样例：
    """

    def __init__(self):
        self.stock_code_list = TuShareDataReader.get_stock_list(list_status='L', only_stock_code=True)
        self.today = TuShareDataReader.get_last_trade_date()  # 最近一个交易日

        self.buy_stocks = []  # 记录符合条件的股票代码

        self.error_msgs = []  # 错误信息（某些股票可能有问题）

    def buy_or_not(self, reader: TuShareDataReader) -> bool:
        """
        子类需实现该方法。
        通过reader可以获取当前股票的所有数据，然后判断今天是否买入条件。
        当然，当你执行选股的时候，当天已经收盘了，因此你需要在第二天开盘买入。
        """

        # 以下是样例，子类重写这段代码即可。
        last_daily_data = self.get_last_daily_data(reader)

        # 如果当天大涨6个点，则符合买入条件。
        if last_daily_data['pct_chg'] >= 6.:
            return True

        return False

    def get_prev_data(self, n, reader: TuShareDataReader, data_type) -> Series:
        """
        获取往前推第n天的数据
        """
        data = reader.get_data_method(data_type)()
        if is_null(data):
            return None

        if len(data) < n:
            return None

        return data.iloc[-n]

    def get_last_data(self, reader: TuShareDataReader, data_type) -> Series:
        """
        获取某只股票最后一个交易日的数据
        """
        last_data = self.get_prev_data(1, reader, data_type)
        if last_data is None:
            return last_data

        if last_data.name != self.today:
            return None

        return last_data


    def get_last_daily_data(self, reader: TuShareDataReader) -> Series:
        """
        获取最后一个交易日的日线数据
        """
        return self.get_last_data(reader, 'daily')

    def get_last_daily_basic_data(self, reader: TuShareDataReader) -> Series:
        """
        获取最后一个交易日的日线数据
        """
        return self.get_last_data(reader, 'daily_basic')

    def find_stock(self):
        """
        根据最后一个交易日的数据，挑选出符合条件的股票。
        """
        for stock_code in tqdm(self.stock_code_list, desc="Find Stock(%s)" % self.__class__.__name__):
            try:
                # 一年的数据应该够用了
                reader = TuShareDataReader(stock_code).set_date_range(start_date=date_add(get_today(), n_day=-365),
                                                                      end_date=get_today())

                if self.get_last_daily_data(reader) is None:
                    # 最后一天根本没数据
                    continue

                if self.buy_or_not(reader):
                    self.buy_stocks.append(stock_code)
            except Exception as e:
                error_msg = f"股票”{stock_code}“异常，异常信息：" + str(e)
                self.error_msgs.append(error_msg)
                print(error_msg)

        return self

    def print_stocks(self):
        """
        打印符合条件的股票信息 TODO
        """
        print("符合条件的股票：", self.buy_stocks)


def CommonPickStock(PickStock):
    """
    通用选股子类

    todo
    """

    def buy_or_not(self, reader: TuShareDataReader) -> bool:
        pass


def PickStock_demo():
    pick_stock = PickStock()
    pick_stock.find_stock().print_stocks()


if __name__ == '__main__':
    PickStock_demo()