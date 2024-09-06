import sys
import time
from pathlib import Path
from typing import Callable

from pandas import DataFrame, Series

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import pandas as pd
import tushare as ts

from utils import date_utils
from utils.date_utils import get_yesterday, get_today, date_add
from utils.db import SqliteDB
from utils.log_utils import print_verbose
from data_reader.utils import convert_alias_to_ts_code, set_trade_date_as_index, pin_memory, drop_duplicates_by_index
from utils.utils import is_null

pin_memory_cache = dict()
use_pin_memory = False  # 是否使用pin_memory，全局配置
from config import tushare_token


class DataReader(object):
    token = tushare_token
    ts.set_token(token)

    pro = ts.pro_api()

    root_dir = ROOT / 'data' / 'tushare'

    db = SqliteDB()

    stock_list = None

    def __init__(self, stock_code: str, data_type='stock'):
        super(DataReader, self).__init__()

        self.root_dir = DataReader.root_dir
        self.pro = DataReader.pro

        self.stock_code = stock_code
        self.data_type = data_type

        if data_type == 'stock':
            self.stock_properties = self.get_stock_properties()
        elif data_type == 'fund':
            self.stock_properties = self.get_fund_properties()
        elif data_type == 'index':
            self.stock_properties = {
                'stock_code': stock_code,
                'ts_code': convert_alias_to_ts_code(stock_code),
            }
        elif data_type == "basic":
            self.stock_properties = {
                'stock_code': "",
                'ts_code': ""
            }
        else:
            raise RuntimeError(f"未知的数据类型: {data_type}")

        self.ts_code = self.stock_properties['ts_code']
        self.start_date = '2010-01-01'
        self.end_date = get_yesterday()

        self.db = DataReader.db

    def set_date_range(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        # todo 判断end_date是否大于今天

        return self

    @staticmethod
    def get_trade_cal(start_date: str, end_date: str):
        """
        获取交易日历
        """
        dtype = {
            "trade_date": str,
            "exchange": str,
            "is_open": str,
            "pretrade_date": str,
        }

        data = DataReader.db.read_data('trade_cal', start_date, end_date, dtype=dtype)

        if data is None or end_date not in data.index:
            # 重新全量获取
            data = DataReader.pro.trade_cal()

            data['trade_date'] = pd.to_datetime(data['cal_date'], format='%Y%m%d')
            data = set_trade_date_as_index(data)
            del data['cal_date']

            print_verbose("获取交易日历trade_cal")

            DataReader.db.to_sql(data, 'trade_cal', dtype=dtype, if_exists='replace')

        if start_date not in data.index:
            raise RuntimeError("start_date异常: %s" % start_date)

        if end_date not in data.index:
            raise RuntimeError("end_date异常: %s" % end_date)

        return data.loc[start_date:end_date]

    @staticmethod
    def get_last_trade_date(trade_date: str = None, close=True, yesterday_limit=True) -> str:
        """
        获取上一个交易日

        :param close: trade_date本身是否包含在内。
        :param yesterday_limit: 限制到昨天。例如：trade_date传的比昨天大，那么trade_date就取昨天
                                但如果当前时间已经超过了18:00点，则就最后一个交易日就是今天。

        例如：
        （1）当close=True时， 2023-11-13的上一个交易日为2023-11-13
        （1）当close=False时， 2023-11-13的上一个交易日为2023-11-10

        :return: 例如：'2023-11-10'
        """
        if trade_date is None:
            trade_date = get_today()

        if yesterday_limit and date_utils.compare_to(trade_date, get_today()) >= 0:
            if date_utils.compare_to(date_utils.get_now(), get_today() + ' 18:00:00', format='%Y-%m-%d %H:%M:%S') >= 0:
                # 如果当前时间已经大于18点，今天的数据已经有了，可以取今天。
                trade_date = get_today()
            else:
                trade_date = get_yesterday()

        data = DataReader.get_trade_cal(trade_date, trade_date)
        if data is None or len(data) != 1:
            raise RuntimeError("trade_date异常：%s" % trade_date)

        if close and (data.iloc[0]['is_open'] == 1 or data.iloc[0]['is_open'] == '1'):
            # 如果trade_date就是交易日，那么返回其自身
            return trade_date

        pretrade_date = data.iloc[0]['pretrade_date']
        return date_utils.convert_format(pretrade_date, '%Y%m%d', '%Y-%m-%d')

    @staticmethod
    def get_next_trade_date(trade_date: str, close=True) -> str:
        """
        获取下一个交易日

        :param close: trade_date本身是否包含在内。

        例如：
        （1）当close=True时， 2023-11-10的下一个交易日为2023-11-10
        （1）当close=False时， 2023-11-10的下一个交易日为2023-11-13
        :return: 例如：'2023-11-10'
        """
        data = DataReader.get_trade_cal(trade_date, date_add(trade_date, 20))
        if data is None and len(data[data['is_open'] == '1']) <= 0:
            raise RuntimeError("trade_date异常：%s" % trade_date)

        if not close:
            data = data.iloc[1:]

        next_trade_date = data[data['is_open'] == '1'].index[0]
        return next_trade_date

    def get_stock_properties(self) -> Series:
        data = DataReader.get_stock_list(list_status='all')

        if self.stock_code not in data.index:
            raise NameError(f'未找到股票代码"{self.stock_code}"，请确认是否填写正确！')

        return data.loc[self.stock_code]

    def get_fund_properties(self) -> Series:
        from data_reader.fund.fund_basic import FundBasicDataReader
        data = FundBasicDataReader().get_data()

        if self.stock_code not in data.index:
            raise NameError(f'未找到基金代码"{self.stock_code}"，请确认是否填写正确！')

        return data.loc[self.stock_code]

    @staticmethod
    @pin_memory()
    def get_stock_list(market='all',  # 市场类别 （主板/创业板/科创板/CDR/北交所）
                       list_status='all',  # 上市状态 L上市 D退市 P暂停上市，默认是L
                       exchange='all',  # 交易所 SSE上交所 SZSE深交所 BSE北交所
                       update=False,  # 为True时，删除原表重新获取
                       only_stock_code=False,  # 是否只返回stock_code. 若为True，则返回list
                       ) -> DataFrame:

        dtype = {
            "ts_code": str,
            "symbol": str,
            "name": str,
            "area": str,
            "industry": str,
            "fullname": str,
            "enname": str,
            "cnspell": str,
            "market": str,
            "exchange": str,
            "curr_type": str,
            "list_status": str,
            "list_date": str,
            "delist_date": str,
            "is_hs": str,
            "act_name": str,
            "act_ent_type": str,
        }
        table_name = 'stock_basic'

        if update:
            DataReader.db.del_table(table_name)

        data = DataReader.db.select(table_name,
                                    conditions=["(delist_date is null or delist_date>='2010-01-01')"],
                                    index_col='symbol',
                                    dtype=dtype,
                                    )

        if data is None:
            fields = ','.join(dtype.keys())
            data = DataReader.pro.stock_basic(exchange='', list_status='L', fields=fields)
            time.sleep(0.1)
            data2 = DataReader.pro.stock_basic(exchange='', list_status='D', fields=fields)
            time.sleep(0.1)
            data3 = DataReader.pro.stock_basic(exchange='', list_status='P', fields=fields)

            print_verbose("获取股票列表，API:stock_basic")

            data = pd.concat([data, data2, data3])

            data = data.set_index('symbol')

            DataReader.db.to_sql(data, table_name, dtype=dtype)

        if market != 'all':
            data = data[data['market'] == market]

        if list_status != 'all':
            data = data[data['list_status'] == list_status]

        if exchange != 'all':
            data = data[data['exchange'] == exchange]

        DataReader.stock_list = data

        if only_stock_code:
            return list(data.index)

        return data

    def get_something(self,
                      table_name,  # 表名，通常为`{数据接口名}_{stock_code}`，例如：`daily_000001`
                      dtype: dict,  # 接口的返回字段（用于和数据库类型进行映射）
                      req_func: Callable,  # 调用tushare请求数据的函数。可参考`get_daily_basic(...)`
                      start_date: str = None,  # 开始日期。若不传，则使用self.start_date
                      end_date: str = None,  # 结束日期。若不传，则使用self.end_date
                      request=True,  # 是否请求tushare服务器获取最新数据。在某些场景下，不读取最新数据
                      call_method=None,  # 哪个方法调用
                      sort_by=None,  # 依据哪一列对返回结果进行排序。Sort the return data by some columns.
                      ascending=True,  # sort_by的是否正序字段。The parameter of `sort_by`.
                      index_col=None,
                      *args, **kwargs):
        """
        对`get_daily`、`basic_daily`等每日数据的共用代码抽象
        """
        if start_date is None:
            start_date = self.start_date

        # end_date取上一个交易日
        if end_date is None:
            end_date = self.end_date
        end_date = DataReader.get_last_trade_date(end_date)

        # 先从表中读取数据
        if call_method == 'cyq':
            data = self.db.read_data(table_name, start_date, end_date=None, dtype=dtype, index_col='index')
            last_data_date = '2010-01-01' if is_null(data) else data['trade_date'].max()
        elif self.data_type == 'basic':
            return self.db.read_data(table_name, start_date=None, end_date=None, dtype=dtype, index_col=index_col)
        else:
            data = self.db.read_data(table_name, start_date, end_date=None, dtype=dtype)
            last_data_date = '2010-01-01' if is_null(data) else data.index[-1]

        def req_data(data):
            # 数据不为空的话，查一下数据的开盘情况，看看有没有必要拉远程接口
            if not is_null(data) and call_method != 'daily_open':
                daily_open_data = self.get_daily_open(last_data_date, end_date, *args, **kwargs)
                if daily_open_data['is_open'].sum() <= 1:
                    # 无需请求远程接口，因为该日期后面本身也没有数据
                    return

            req_start_date = None
            if data is not None and len(data) > 0:
                # 若data不为空，则从data的最后一个日期的下个交易日开始请求，避免请求重复数据
                req_start_date = str(last_data_date)
                req_start_date = DataReader.get_next_trade_date(req_start_date, close=False)

            if data is not None and len(data) <= 0:
                # 若data不为None，但没请求到数据，则从数据库中的最后一个交易日的下一个交易日开始请求
                db_data = self.db.get_last_data(table_name)
                if db_data is not None and len(db_data) > 0:
                    req_start_date = str(db_data.iloc[0]['trade_date'])
                    req_start_date = DataReader.get_next_trade_date(req_start_date, close=False)

            if req_start_date is not None:
                req_start_date = req_start_date.replace("-", "")

            # 调用tushare接口获取数据或通过计算获得
            resp_data = req_func(req_start_date, None)  # 每次请求接口end_date都获取到最新的数据

            # 去除重复值，避免频繁报主键重复
            resp_data = drop_duplicates_by_index(data, resp_data)

            if not is_null(resp_data):
                # 存到数据库中
                self.db.to_sql(resp_data, table_name, dtype=dtype)

            if data is not None:
                data = pd.concat([data, resp_data], axis=0)
            else:
                data = resp_data

            return data

        if request and (is_null(data) or date_utils.compare_to(last_data_date, end_date) < 0):
            # 获取数据的开盘情况
            data = req_data(data)

        # 表中没有读到数据，则从接口获取数据
        if data is None:
            return DataFrame(columns=dtype.keys())

        if call_method == 'cyq':
            data = data[(start_date <= data['trade_date']) & (data['trade_date'] <= end_date)]
        else:
            data = data.loc[start_date:end_date]

        if sort_by is not None:
            data = data.sort_values(by=sort_by, ascending=ascending)

        return data

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_daily_open(self, start_date: str = None, end_date: str = None, request=True, *args,
                       **kwargs):
        """
        获取股票每日是否开盘的情况。由于许多股票在某天就退市了，然后之后又重新上市，中间就会有一部分数据缺失。
        这个表就是用来记录每只股票每天的开盘情况。
        """
        dtype = {
            "trade_date": str,
            "is_open": bool,
        }

        table_name = f"daily_open_{self.stock_code}"

        def req_func(req_start_date, req_end_date):
            if is_null(req_start_date):
                req_start_date = '20100101'

            resp_data = self.pro.daily(ts_code=self.ts_code,
                                       start_date=req_start_date,
                                       fields='trade_date',
                                       )

            print(f"获取daily_open数据, ts_code: {self.ts_code}")

            resp_data['trade_date'] = pd.to_datetime(resp_data['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

            first_trade_date = resp_data['trade_date'].min()
            if is_null(first_trade_date):
                first_trade_date = date_utils.convert_format(req_start_date, '%Y%m%d', '%Y-%m-%d')
            last_trade_date = self.get_last_trade_date()

            date_list = date_utils.get_date_list(first_trade_date, last_trade_date)
            resp_date_list = set(resp_data['trade_date'])

            data_items = []
            for trade_date in date_list:
                if trade_date in resp_date_list:
                    data_items.append([trade_date, True])
                else:
                    data_items.append([trade_date, False])

            resp_data = DataFrame(data_items, columns=['trade_date', 'is_open'])
            resp_data = set_trade_date_as_index(resp_data, format='%Y-%m-%d')

            return resp_data

        return self.get_something(table_name=table_name,
                                  dtype=dtype,
                                  req_func=req_func,
                                  start_date=start_date,
                                  end_date=end_date,
                                  request=request,
                                  call_method='daily_open',
                                  *args,
                                  **kwargs
                                  )

    @staticmethod
    def is_trading_day(date: str):
        """
        判断date是否为交易日。
        例如：
        2023-11-19为False（因为是周日）
        2023-11-20为Ture
        """
        data = DataReader.get_trade_cal(date, date)

        return data.iloc[0]['is_open'] == 1 or data.iloc[0]['is_open'] == "1"
