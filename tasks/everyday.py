# -*- coding: UTF-8 -*-

import sys
import time
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from utils.data_reader import TuShareDataReader
from src.find_stock import PickStock
from utils.log_utils import set_log_level
from utils.date_utils import get_today
from utils.email_utils import send_mail
from utils.utils import is_null

"""
每日要执行的定时任务
"""


class DayangXianLowTurnover_PickStock(PickStock):
    """
    大阳线低换手率
    """

    def buy_or_not(self, reader: TuShareDataReader) -> bool:
        last_daily_data = self.get_last_daily_data(reader)
        last_daily_basic_data = self.get_last_daily_basic_data(reader)

        # 当天涨幅 >= 6%
        is_DaYangXian = (last_daily_data['close'] - last_daily_data['open']) / last_daily_data['open'] * 100 >= 6
        # 换手率低
        low_turnover = last_daily_basic_data['turnover_rate'] <= 0.5

        return is_DaYangXian and low_turnover


class 旭日东升_PickStock(PickStock):

    def buy_or_not(self, reader: TuShareDataReader) -> bool:
        last_daily_data = self.get_last_daily_data(reader)
        last_daily_extra_data = self.get_last_data(reader, 'daily_extra')
        prev_daily_data = self.get_prev_data(2, reader, 'daily')
        if prev_daily_data is None:
            return False

        con1 = last_daily_data['pct_chg'] >= 3.
        con2 = prev_daily_data['pct_chg'] <= -3.
        con3 = last_daily_data['close'] > prev_daily_data['high']
        con4 = last_daily_extra_data['RPY1'] < 25.

        return con1 and con2 and con3 and con4


def obtain_daily_data(trade_date):
    """
    获取每日数据
    """
    set_log_level("debug")

    start_time = time.time()

    stock_list = TuShareDataReader.get_stock_list(update=True, list_status='L')
    # 整体获取的数据
    daily_data = TuShareDataReader._get_all_by_trade_date(trade_date, data_type='daily')
    daily_basic_data = TuShareDataReader._get_all_by_trade_date(trade_date, data_type='daily_basic')
    index_daily_data = TuShareDataReader('上证指数', is_index=True).set_date_range(start_date=trade_date,
                                                                               end_date=trade_date).get_index_daily()
    overall_extra_data = TuShareDataReader.get_overall_extra(trade_date, trade_date)

    # 下面的是需要逐条获取的
    def one_by_one(data_type):
        obtained_stock_list = TuShareDataReader.get_union(data_type, trade_date, trade_date)['stock_code']
        obtained_stock_list = set(obtained_stock_list)
        for stock_code in tqdm(stock_list.index, desc="Get %s Data" % data_type):
            if stock_code in obtained_stock_list:
                # 该股票已经获取过了数据了
                continue

            reader = TuShareDataReader(stock_code).set_date_range(trade_date, trade_date)
            _ = reader.get_data_method(data_type)()

    one_by_one('daily_extra')

    return "获取数据完毕，耗时：%d秒\n" % int(time.time() - start_time)


def find_stock():
    """
    选股
    """
    log_content = ""

    pick_stock = DayangXianLowTurnover_PickStock()
    log_content += "大阳线低换手率（当天涨幅>=6%, 换手率<=0.5%）:\n"
    log_content += str(pick_stock.find_stock().buy_stocks) + "\n"

    if not is_null(pick_stock.error_msgs):
        log_content += "“大阳线低换手率”包含以下异常股票：\n"
        log_content += "\n".join(pick_stock.error_msgs)
        log_content += "\n\n"

    pick_stock = 旭日东升_PickStock()
    log_content += "旭日东升:\n"
    log_content += str(pick_stock.find_stock().buy_stocks) + "\n"

    if not is_null(pick_stock.error_msgs):
        log_content += "“旭日东升”包含以下异常股票：\n"
        log_content += "\n".join(pick_stock.error_msgs)
        log_content += "\n\n"

    return log_content


def main(trade_date=None):
    if trade_date is None:
        trade_date = TuShareDataReader.get_last_trade_date()
    title = trade_date + "股票情况"
    try:
        log_content = ""
        log_content += obtain_daily_data(trade_date)
        log_content += find_stock()
    except Exception as e:
        send_mail(title, "执行出错，错误信息如下：" + str(e))
        raise e

    print()
    print(log_content)

    send_mail(title, log_content)


if __name__ == '__main__':
    main()
