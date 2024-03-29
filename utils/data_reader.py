import os
import sys
from pathlib import Path
from typing import Callable

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import time
import pandas as pd
import numpy as np
import tushare as ts
from pandas import DataFrame, Series
from tqdm import tqdm

from utils import date_utils
import utils.data_process as data_process
from utils.date_utils import get_yesterday, get_today, date_add, split_month, get_month_list
from utils.db import SqliteDB
from utils.log_utils import print_verbose
from utils.utils import time_func, is_null

pin_memory_cache = dict()
use_pin_memory = False  # 是否使用pin_memory，全局配置


def get_require_data_list(factor_names,
                          require_daily=True,  # True时，不管factor_names中有没有daily的数据，最终结果都会包含daily
                          ):
    """
    根据指标获取都需要哪些参数。例如：factor_names为['pct_chg', 'turnover_rate', 'ASR']
                            则，需要'daily', 'daily_basic', 'cyq_extra' 三种数据，
                            因此会返回 ['daily', 'daily_basic', 'cyq_extra']
    """
    daily_dtype = TuShareDataReader.daily_dtype()
    daily_basic_dtype = TuShareDataReader.daily_basic_dtype()
    daily_extra_dtype = TuShareDataReader.daily_extra_dtype()
    index_daily_dtype = TuShareDataReader.index_daily_dtype(index_prefix=True)

    # 判断一下都需要读取哪些数据
    require_data = set()
    for factor in factor_names:
        if factor in daily_dtype:
            require_data.add('daily')
        elif factor in daily_basic_dtype:
            require_data.add('daily_basic')
        elif factor in daily_extra_dtype:
            require_data.add('daily_extra')
        elif factor in index_daily_dtype:
            require_data.add('index_daily')
        else:
            raise RuntimeError(f"不支持{factor}参数！")

    if require_daily and 'daily' not in require_data:
        require_data.add('daily')

    return require_data


def set_trade_date_as_index(data: DataFrame, format='%Y%m%d'):
    if data is None:
        return data

    if data.index.name == 'trade_date':
        return data

    data['trade_date'] = pd.to_datetime(data['trade_date'], format=format).dt.strftime('%Y-%m-%d')
    data = data.sort_values(by='trade_date', ascending=True)

    # 把trade_date放到最前面
    columns = list(data.columns)
    columns.remove('trade_date')
    columns.insert(0, 'trade_date')
    data = data[columns]

    data = data.set_index('trade_date')

    return data


def is_trading_day(date: str):
    """
    判断date是否为交易日。
    例如：
    2023-11-19为False（因为是周日）
    2023-11-20为Ture
    """
    data = TuShareDataReader.get_trade_cal(date, date)

    return data.iloc[0]['is_open'] == 1 or data.iloc[0]['is_open'] == "1"


def drop_duplicates_by_index(data: DataFrame, resp_data: DataFrame):
    """
    去除resp_data的重复数据。重复数据是指：已经在data中存在的数据。

    该函数的应用：在从tushare中获取数据后，有部分数据在库里（data里）已经有了，因此不需要再插入了。
    """
    if data is None:
        return resp_data

    return resp_data[~resp_data.index.isin(data.index)]


def convert_alias_to_ts_code(ts_code: str):
    """
    将常用中文名转换成对应的ts_code
    """
    # 常用指数转换
    if ts_code in ['上证指数', '上证']:
        ts_code = '000001.SH'
    elif ts_code in ['沪深300']:
        ts_code = '000300.SH'

    return ts_code


def pin_memory(is_obj=False, obj_fields=()):
    """
    装饰器，将func的请求参数和返回结果进行缓存。这样下一次再调用的时候就可以直接从内存中读取了

    :param is_obj: 是否是类对象
    :param obj_fields: 哪些类对象属性要作为key的一部分
    """

    def wrapper_out(func):

        def wrapper(*args, **kwargs):
            func_name = func.__name__

            use_cache = use_pin_memory  # 是否使用缓存

            # 如果参数中明确包含了'pin_memory'参数，则根据参数决定是否使用缓存
            if 'pin_memory' in kwargs:
                use_cache = kwargs.get('pin_memory')

            obj_args_str = ""
            args_str = ""
            if is_obj:  # 类上的方法
                self = args[0]
                for field in obj_fields:
                    if not hasattr(self, field):
                        raise RuntimeError(f"{func_name}类对象不包含{field}属性")

                    field_value = getattr(self, field)
                    obj_args_str += f"_{field_value}"

                args_str += str(args[1:])
            else:
                args_str += str(args)

            key = func_name + obj_args_str + args_str + '_' + str(kwargs)

            if key in pin_memory_cache and use_cache:  # 如果在缓存中，直接返回
                return pin_memory_cache[key]

            result = func(*args, **kwargs)
            pin_memory_cache[key] = result
            return result

        return wrapper

    return wrapper_out


def remove_pin_memory():
    """
    释放pin_memory中的数据
    """
    global pin_memory_cache

    pin_memory_cache = dict()


class TuShareDataReader(object):
    token = ""  # FIXME
    ts.set_token(token)

    pro = ts.pro_api()

    root_dir = ROOT / 'data' / 'tushare'

    db = SqliteDB()

    stock_list = None

    def __init__(self, stock_code: str, is_index=False):
        super(TuShareDataReader, self).__init__()

        self.root_dir = TuShareDataReader.root_dir
        self.pro = TuShareDataReader.pro

        self.stock_code = stock_code
        self.is_index = is_index

        if not is_index:
            self.stock_properties = self.get_stock_properties()
        else:
            self.stock_properties = {
                'stock_code': stock_code,
                'ts_code': convert_alias_to_ts_code(stock_code),
            }

        self.ts_code = self.stock_properties['ts_code']
        self.start_date = '2010-01-01'
        self.end_date = get_yesterday()

        self.db = TuShareDataReader.db

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

        data = TuShareDataReader.db.read_data('trade_cal', start_date, end_date, dtype=dtype)

        if data is None or end_date not in data.index:
            # 重新全量获取
            data = TuShareDataReader.pro.trade_cal()

            data['trade_date'] = pd.to_datetime(data['cal_date'], format='%Y%m%d')
            data = set_trade_date_as_index(data)
            del data['cal_date']

            print_verbose("获取交易日历trade_cal")

            TuShareDataReader.db.to_sql(data, 'trade_cal', dtype=dtype, if_exists='replace')

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

        data = TuShareDataReader.get_trade_cal(trade_date, trade_date)
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
        data = TuShareDataReader.get_trade_cal(trade_date, date_add(trade_date, 20))
        if data is None and len(data[data['is_open'] == '1']) <= 0:
            raise RuntimeError("trade_date异常：%s" % trade_date)

        if not close:
            data = data.iloc[1:]

        next_trade_date = data[data['is_open'] == '1'].index[0]
        return next_trade_date

    @staticmethod
    def get_index_basic(market='SSE'):
        """
        获取指数的基本信息。
        """
        dtype = {
            "ts_code": str,
            "name": str,
            "fullname": str,
            "market": str,
            "publisher": str,
            "index_type": str,
            "category": str,
            "base_date": str,
            "base_point": float,
            "list_date": str,
            "weight_rule": str,
            "desc": str,
            "exp_date": str,
        }

        market_list = [
            "MSCI",  # MSCI指数
            "CSI",  # 中证指数
            "SSE",  # 上交所指数
            "SZSE",  # 深交所指数
            "CICC",  # 中金指数
            "SW",  # 申万指数
            "OTH",  # 其他指数
        ]

        if market not in market_list:
            raise RuntimeError("market不在范围内，可用范围: " + str(market_list))

        data = TuShareDataReader.db.select("index_basic",
                                           conditions=[f"market='{market}'"],
                                           index_col='ts_code',
                                           dtype=dtype,
                                           )

        if data is None:
            data = TuShareDataReader.pro.index_basic(market=market, fields=','.join(dtype.keys()))
            print_verbose(f"获取指数基本信息，market: {market}")
            data = data.set_index('ts_code')

            TuShareDataReader.db.to_sql(data, 'index_basic', dtype=dtype)

        return data

    @staticmethod
    def index_daily_dtype(
            index_prefix=False,  # 是否给每个列名都加上index前缀
    ):
        dtype = {
            "ts_code": str,
            "trade_date": str,
            "close": float,
            "open": float,
            "high": float,
            "low": float,
            "pre_close": float,
            "change": float,
            "pct_chg": float,
            "vol": float,
            "amount": float,
        }

        if index_prefix:
            dtype = {
                "index_ts_code": str,
                "index_trade_date": str,
                "index_close": float,
                "index_open": float,
                "index_high": float,
                "index_low": float,
                "index_pre_close": float,
                "index_change": float,
                "index_pct_chg": float,
                "index_vol": float,
                "index_amount": float,
            }

        return dtype

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_index_daily(self,
                        ts_code=None,
                        index_prefix=True,  # 是否给每个列名都加上index前缀
                        start_date: str = None,
                        end_date: str = None,
                        request=True,
                        *args,
                        **kwargs,
                        ) -> DataFrame:
        """
        获取指数行情
        """
        dtype = TuShareDataReader.index_daily_dtype()

        if not self.is_index:
            ts_code = convert_alias_to_ts_code('上证')

        if ts_code is None:
            ts_code = self.ts_code

        table_name = f"index_daily_{ts_code}"

        def req_func(req_start_date, req_end_date):
            resp_data = TuShareDataReader.pro.index_daily(
                ts_code=ts_code,
                start_date=req_start_date,
                end_date=req_end_date,
            )

            print_verbose(f"获取指数日线行情，ts_code: {ts_code}")

            resp_data = set_trade_date_as_index(resp_data)

            return resp_data

        data = self._get_something(table_name=table_name,
                                   dtype=dtype,
                                   req_func=req_func,
                                   start_date=start_date,
                                   end_date=end_date,
                                   request=request,
                                   call_method='index_daily',
                                   *args,
                                   **kwargs,
                                   )

        if index_prefix:
            columns = []
            for col in data.columns:
                if not col.startswith("index_"):
                    col = 'index_' + col

                columns.append(col)

            data.columns = columns

        return data

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
            TuShareDataReader.db.del_table(table_name)

        data = TuShareDataReader.db.select(table_name,
                                           conditions=["(delist_date is null or delist_date>='2010-01-01')"],
                                           index_col='symbol',
                                           dtype=dtype,
                                           )

        if data is None:
            fields = ','.join(dtype.keys())
            data = TuShareDataReader.pro.stock_basic(exchange='', list_status='L', fields=fields)
            time.sleep(0.1)
            data2 = TuShareDataReader.pro.stock_basic(exchange='', list_status='D', fields=fields)
            time.sleep(0.1)
            data3 = TuShareDataReader.pro.stock_basic(exchange='', list_status='P', fields=fields)

            print_verbose("获取股票列表，API:stock_basic")

            data = pd.concat([data, data2, data3])

            data = data.set_index('symbol')

            TuShareDataReader.db.to_sql(data, table_name, dtype=dtype)

        if market != 'all':
            data = data[data['market'] == market]

        if list_status != 'all':
            data = data[data['list_status'] == list_status]

        if exchange != 'all':
            data = data[data['exchange'] == exchange]

        TuShareDataReader.stock_list = data

        if only_stock_code:
            return list(data.index)

        return data

    def get_stock_properties(self) -> Series:
        data = TuShareDataReader.get_stock_list(list_status='all')

        if self.stock_code not in data.index:
            raise NameError(f'未找到股票代码"{self.stock_code}"，请确认是否填写正确！')

        return data.loc[self.stock_code]

    def get_data_method(self, data_type: str) -> Callable:
        """
        根据data_type返回对应的获取数据方法。

        例如：daily返回get_daily
        """
        if data_type == 'daily':
            return self.get_daily
        elif data_type == 'daily_extra':
            return self.get_daily_extra
        elif data_type == 'daily_basic':
            return self.get_daily_basic
        elif data_type == 'cyq_extra':
            return self.get_cyq_extra
        elif data_type == 'index_daily':
            return self.get_index_daily
        else:
            raise RuntimeError("不支持的数据类型：" % data_type)

    def get_combine(self,
                    require_data: tuple = ('daily',),  # 需要哪些数据。例如：('daily', 'daily_basic', ...)
                    rule='error',  # 当这几种数据量不一致后，采用哪种处理规则。① error报错
                    ) -> DataFrame:
        """
        同时获取多种数据，然后合并到一起。
        """
        concat_data = []

        for data_name in require_data:
            concat_data.append(self.get_data_method(data_name)())

        # todo rule

        if len(concat_data) <= 0:
            return DataFrame(columns=['stock_code'])

        data = pd.concat(concat_data, axis=1, join='inner')
        data = data_process.drop_duplicated_columns(data)

        return data

    def _get_something(self,
                       table_name,  # 表名，通常为`{数据接口名}_{stock_code}`，例如：`daily_000001`
                       dtype: dict,  # 接口的返回字段（用于和数据库类型进行映射）
                       req_func: Callable,  # 调用tushare请求数据的函数。可参考`get_daily_basic(...)`
                       start_date: str = None,  # 开始日期。若不传，则使用self.start_date
                       end_date: str = None,  # 结束日期。若不传，则使用self.end_date
                       request=True,  # 是否请求tushare服务器获取最新数据。在某些场景下，不读取最新数据
                       call_method=None,  # 哪个方法调用
                       *args, **kwargs):
        """
        对`get_daily`、`basic_daily`等每日数据的共用代码抽象
        """
        if start_date is None:
            start_date = self.start_date

        # end_date取上一个交易日
        if end_date is None:
            end_date = self.end_date
        end_date = TuShareDataReader.get_last_trade_date(end_date)

        # 先从表中读取数据
        if call_method == 'cyq':
            data = self.db.read_data(table_name, start_date, end_date=None, dtype=dtype, index_col='index')
            last_data_date = '2010-01-01' if is_null(data) else data['trade_date'].max()
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
                req_start_date = TuShareDataReader.get_next_trade_date(req_start_date, close=False)

            if data is not None and len(data) <= 0:
                # 若data不为None，但没请求到数据，则从数据库中的最后一个交易日的下一个交易日开始请求
                db_data = self.db.get_last_data(table_name)
                if db_data is not None and len(db_data) > 0:
                    req_start_date = str(db_data.iloc[0]['trade_date'])
                    req_start_date = TuShareDataReader.get_next_trade_date(req_start_date, close=False)

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

        return data

    @staticmethod
    def _get_all_by_trade_date(trade_date: str = None, data_type='daily'):
        """
        按日期获取数据所有股票的数据。主要用于每日全量获取当天的数据，要不一次一个接口拉太慢了。

        trade_date若为None，则视为上一个交易日。
        """
        if trade_date is None:
            trade_date = TuShareDataReader.get_last_trade_date(get_today())

        if not is_trading_day(trade_date):
            raise RuntimeError(f"{trade_date}不是交易日！")

        stock_list = TuShareDataReader.get_stock_list()
        last_trade_date = TuShareDataReader.get_last_trade_date(trade_date, close=False, yesterday_limit=False)

        dtype: dict
        resp_data: DataFrame
        table_name_template: str
        if data_type == 'daily':
            table_name_template = 'daily_{stock_code}'
            dtype = TuShareDataReader.daily_dtype()
            resp_data = TuShareDataReader.pro.daily(trade_date=trade_date.replace("-", ""))
            print(f"获取多个股票{trade_date}日线行情(daily)")
            time.sleep(0.1)
        elif data_type == 'daily_basic':
            table_name_template = 'daily_basic_{stock_code}'
            dtype = TuShareDataReader.daily_basic_dtype()
            resp_data = TuShareDataReader.pro.daily_basic(trade_date=trade_date.replace("-", ""),
                                                          fields=','.join(dtype.keys()))
            print(f"获取多个股票{trade_date}基础日线行情(daily_basic)")
            time.sleep(0.1)
        else:
            raise RuntimeError(f"暂不支持{data_type}")

        # 已经获取过的股票列表，避免一条一条再次检索，加快速度。
        obtained_stock_list = TuShareDataReader.get_union(data_type, trade_date, trade_date)['stock_code']
        obtained_stock_list = set(obtained_stock_list)

        for i in tqdm(range(len(resp_data)), desc='Saving %s Data' % data_type):
            resp_data_item = resp_data.iloc[i:i + 1].copy()
            ts_code = resp_data_item['ts_code'].item()
            stock_data = stock_list[stock_list['ts_code'] == ts_code]

            if len(stock_data) <= 0:
                print_verbose(f"[WARN] 股票列表中不存在“{ts_code}”，可能是新上市公司！")
                continue

            stock_code = stock_data.iloc[0].name

            if stock_code in obtained_stock_list:
                # 该股票已经获取过了数据了
                continue

            table_name = table_name_template.format(stock_code=stock_code)

            resp_data_item = set_trade_date_as_index(resp_data_item)

            if 'ts_code' in resp_data_item.columns:
                del resp_data_item['ts_code']

            # 连续性判断，不连续则不存。
            last_data = TuShareDataReader.db.get_last_data(table_name, dtype=dtype)

            if last_data is None or len(last_data) <= 0:
                print_verbose(f"[WARN] {table_name}表不存在，重新请求全量数据!")
                TuShareDataReader(stock_code).get_daily()
                TuShareDataReader(stock_code).get_daily_basic()
                continue

            if date_utils.compare_to(last_data.iloc[0]['trade_date'], last_trade_date) > 0:
                # 已经保存了，无需再次保存
                continue

            if date_utils.compare_to(last_data.iloc[0]['trade_date'], last_trade_date) < 0:
                print_verbose(f"[WARN] {table_name}表数据不连续，不保存!")
                continue

            TuShareDataReader.db.to_sql(resp_data_item, table_name, dtype=dtype)

        return resp_data

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

        return self._get_something(table_name=table_name,
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
    def daily_basic_dtype():
        return {
            "ts_code": str,
            "trade_date": str,  # 交易日期
            "close": float,  # 当日收盘价
            "turnover_rate": float,  # 换手率（%）
            "turnover_rate_f": float,  # 换手率（自由流通股）
            "volume_ratio": float,  # 量比
            "pe": float,  # 市盈率（总市值/净利润， 亏损的PE为空）
            "pe_ttm": float,  # 市盈率（TTM，亏损的PE为空）
            "pb": float,  # 市净率（总市值/净资产）
            "ps": float,  # 市销率
            "ps_ttm": float,  # 市销率（TTM）
            "dv_ratio": float,  # 股息率 （%）
            "dv_ttm": float,  # 股息率（TTM）（%）
            "total_share": float,  # 总股本 （万股）
            "float_share": float,  # 流通股本 （万股）
            "free_share": float,  # 自由流通股本 （万）
            "total_mv": float,  # 总市值 （万元）
            "circ_mv": float,  # 流通市值（万元）
        }

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_daily_basic(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs):
        """
        获取每日的基础数据，例如：总市值，流通市值等。
        """
        dtype = TuShareDataReader.daily_basic_dtype()

        table_name = f"daily_basic_{self.stock_code}"

        def req_func(req_start_date, req_end_date):
            resp_data = self.pro.daily_basic(ts_code=self.ts_code,
                                             start_date=req_start_date,
                                             end_date=req_end_date,
                                             fields=','.join(dtype.keys()),
                                             )

            print(f"获取日线基础数据, ts_code: {self.ts_code}")

            resp_data = set_trade_date_as_index(resp_data)

            if 'ts_code' in resp_data.columns:
                del resp_data['ts_code']

            return resp_data

        return self._get_something(table_name=table_name,
                                   dtype=dtype,
                                   req_func=req_func,
                                   start_date=start_date,
                                   end_date=end_date,
                                   request=request,
                                   call_method='daily_basic',
                                   *args,
                                   **kwargs
                                   )

    @staticmethod
    def daily_dtype():
        return {
            "trade_date": str,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "pre_close": float,
            "change": float,
            "pct_chg": float,
            "vol": float,
            "amount": float,
            "turnover_rate": float,  # deprecated。使用daily_basic中的
            "volume_ratio": float,  # deprecated。使用daily_basic中的
        }

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_daily(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs) -> DataFrame:
        """
        股票日线行情（默认前复权）
        """
        dtype = TuShareDataReader.daily_dtype()

        table_name = f"daily_{self.stock_code}"

        def req_func(req_start_date, req_end_date):
            resp_data = ts.pro_bar(ts_code=self.ts_code,
                                   start_date=req_start_date,
                                   adj="qfq",
                                   factors=['tor', "vr"])

            print(f"获取日线行情，ts_code: {self.ts_code}")

            resp_data = set_trade_date_as_index(resp_data)

            if resp_data is None:
                print(f"[ERROR]获取日线行情为None，ts_code: {self.ts_code}")
                return DataFrame()

            if 'ts_code' in resp_data.columns:
                del resp_data['ts_code']

            return resp_data

        data = self._get_something(table_name=table_name,
                                   dtype=dtype,
                                   req_func=req_func,
                                   start_date=start_date,
                                   end_date=end_date,
                                   request=request,
                                   call_method='daily',
                                   *args,
                                   **kwargs
                                   )

        # 增加index（序号）列
        if 'index' in data.columns:
            del data['index']

        data['index'] = list(range(1, len(data) + 1))

        if 'turnover_rate' in data.columns:  # turnover_rate去daily_basic中取
            del data['turnover_rate']

        return data

    @staticmethod
    def daily_extra_dtype():
        return {
            "trade_date": str,
            "MA5": float,  # 5日均线
            "MA10": float,  # 10日均线
            "MA20": float,  # 20日均线
            "MA60": float,  # 60日均线,
            "RPY1": float,  # 1年相对价位（250个交易日算一年）
            "RPY2": float,  # 2年相对价位
            "RPM1": float,  # 1月相对价位（20个交易日算一年）
            "RPM2": float,  # 2月相对价位
            "RPM3": float,  # 3月相对价位
            "RPM6": float,  # 6月相对价位
            "gap": float,  # 跳空幅度,
            "upper_wick": float,  # 上影线长度（百分比）
            "lower_wick": float,  # 下影线长度（百分比）
            "amp": float,  # 当日振幅,
            "FAR": float,  # 冲高后回落幅度（fall after rise）
            "RAF": float,  # 下跌后上涨幅度（rise after fall）
            "FPY1": float,  # 近1年回撤（change per year）
            "FPY2": float,  # 近2年回撤（change per year）
            "FPM1": float,  # 近1月回撤
            "FPM2": float,  # 近2月回撤
            "FPM3": float,  # 近3月回撤
            "FPM6": float,  # 近6月回撤
            "QRR5": float,  # 5日量比
            "QRR10": float,  # 10日量比
            "QRR20": float,  # 20日量比
            "QRR60": float,  # 60日量比
            "QRR250": float,  # 250日量比
        }

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_daily_extra(self, start_date: str = None, end_date: str = None, *args, **kwargs):
        """
        获取日线行情的额外指标
        """
        dtype = TuShareDataReader.daily_extra_dtype()

        table_name = f"daily_extra_{self.stock_code}"

        def compute_func(req_start_date, req_end_date):
            if req_start_date is None:
                req_start_date = '20100101'

            req_start_date = date_utils.convert_format(req_start_date, '%Y%m%d', '%Y-%m-%d')
            daily_data = self.get_daily('1980-01-01', get_today(), request=False)

            if daily_data is None or len(daily_data) <= 0:
                return DataFrame(columns=dtype.keys())

            resp_data = DataFrame(daily_data[req_start_date:].index)
            resp_data = resp_data.set_index('trade_date')

            if is_null(resp_data):
                return DataFrame(columns=dtype.keys())

            if len(set(dtype.keys()) - set(resp_data.columns)) > 0:  # 存在新加的指标，该指标要全部重新计算
                if 'MA5' not in resp_data.columns:
                    # 均线
                    resp_data = data_process.add_MA(daily_data, day_list=(5, 10, 20, 60), target_data=resp_data,
                                                    format='MA%d')

                if "RPY1" not in resp_data.columns:
                    # 年相对价位
                    resp_data = data_process.add_RPY(daily_data, year=1, target_data=resp_data)
                    resp_data = data_process.add_RPY(daily_data, year=2, target_data=resp_data)

                if "RPM1" not in resp_data.columns:
                    # 月相对价位
                    resp_data = data_process.add_RPM(daily_data, month=1, target_data=resp_data)
                    resp_data = data_process.add_RPM(daily_data, month=2, target_data=resp_data)
                    resp_data = data_process.add_RPM(daily_data, month=3, target_data=resp_data)
                    resp_data = data_process.add_RPM(daily_data, month=6, target_data=resp_data)

                if "gap" not in resp_data.columns:
                    # 跳空幅度
                    resp_data = data_process.add_gap(daily_data, target_data=resp_data)

                if "upper_wick" not in resp_data.columns:
                    # 上下影线长度
                    resp_data = data_process.add_wick(daily_data, target_data=resp_data)

                if "amp" not in resp_data.columns:
                    # 当日振幅
                    resp_data = data_process.add_amp(daily_data, target_data=resp_data)

                if "FAR" not in resp_data.columns:
                    # 当日振幅
                    resp_data = data_process.add_FAR(daily_data, target_data=resp_data)

                if "FPY1" not in resp_data.columns:
                    # 近n年回撤
                    resp_data = data_process.add_FPY(daily_data, year=1, target_data=resp_data)
                    resp_data = data_process.add_FPY(daily_data, year=2, target_data=resp_data)

                if "FPM1" not in resp_data.columns:
                    # 近n个月回撤
                    resp_data = data_process.add_FPM(daily_data, month=1, target_data=resp_data)
                    resp_data = data_process.add_FPM(daily_data, month=2, target_data=resp_data)
                    resp_data = data_process.add_FPM(daily_data, month=3, target_data=resp_data)
                    resp_data = data_process.add_FPM(daily_data, month=6, target_data=resp_data)

                if "QRR5" not in resp_data.columns:
                    # n日量比
                    resp_data = data_process.add_QRR(daily_data, day_list=(5, 10, 20, 60, 250), target_data=resp_data)

            return resp_data

        return self._get_something(table_name=table_name,
                                   dtype=dtype,
                                   req_func=compute_func,
                                   start_date=start_date,
                                   end_date=end_date,
                                   request=True,
                                   call_method='daily_extra',
                                   *args,
                                   **kwargs,
                                   )

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_stk_limit(self) -> DataFrame:
        """
        获取涨跌停价格
        """

        os.makedirs(TuShareDataReader.root_dir / 'stk_limit', exist_ok=True)
        cache_file = TuShareDataReader.root_dir / 'stk_limit' / ('%s.csv' % self.stock_code)

        # end_date取上一个交易日
        end_date = TuShareDataReader.get_last_trade_date(self.end_date)

        data = None
        if os.path.exists(cache_file):
            # todo
            data = pd.read_csv(cache_file, index_col='trade_date', parse_dates=['trade_date'])

        if data is None or date_utils.compare_to(data.index[-1].strftime('%Y-%m-%d'), end_date) < 0:
            # 获取全部数据；该接口一次仅能返回5800条数据，这里也只取这么多，够用了。
            data = self.pro.stk_limit(ts_code=self.ts_code)
            print(f"获取涨跌停价格，ts_code: {self.ts_code}")
            data = set_trade_date_as_index(data)

            data.to_csv(cache_file)

        return data.loc[self.start_date:end_date]

    def _get_cyq_by_month(self, month: str):
        """
        从tushare获取一个月的筹码分布
        :param month: %Y-%m  例如：2023-10
        """

        if date_utils.compare_to(month + "-01", get_today()) > 0:
            raise RuntimeError("month异常：" + month)

        # 筹码分布一次只能返回2000条数据，一个月的数据分多次提取
        data_list = []
        for start_date, end_date in split_month(month):
            if date_utils.compare_to(start_date, get_today()) >= 0:
                break

            data_list.append(self.pro.cyq_chips(ts_code=self.ts_code, start_date=start_date, end_date=end_date))
            print_verbose(f"获取筹码分布, {self.ts_code} [{start_date}, {end_date}]")
            time.sleep(0.35)  # 该接口每分钟至多200次

        if is_null(data_list):
            return None

        data = pd.concat(data_list)

        data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
        data['trade_date'] = data['trade_date'].dt.strftime('%Y-%m-%d')
        data = data.sort_values(by=['trade_date', 'price'], ascending=[True, False])

        return data

    @staticmethod
    def cyq_dtype():
        return {
            "index": str,
            "trade_date": str,
            "price": float,
            "percent": float,
        }

    def get_cyq(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs) -> DataFrame:
        """
        获取筹码分布
        """
        dtype = TuShareDataReader.cyq_dtype()

        table_name = f'cyq_{self.stock_code}'

        def req_func(req_start_date, req_end_date):
            if is_null(req_start_date):
                req_start_date = '20100101'

            if date_utils.compare_to(req_start_date, '20100101', format='%Y%m%d') < 0:
                # Tushare的筹码分布从2010年开始的，但实际上很多数据从2018年才开始有
                req_start_date = '20100101'

            req_start_date = date_utils.convert_format(req_start_date, '%Y%m%d', '%Y-%m-%d')
            last_trade_date = TuShareDataReader.get_last_trade_date()

            if date_utils.date_diff(last_trade_date, req_start_date) <= 15:
                # 数据量较少，直接调用接口
                resp_data = self.pro.cyq_chips(ts_code=self.ts_code,
                                               start_date=req_start_date.replace("-", ""),
                                               end_date=last_trade_date.replace("-", ""))

                print_verbose(f"获取筹码分布, {self.ts_code} [{req_start_date}, {last_trade_date}]")
                time.sleep(0.35)  # 该接口每分钟至多200次

                resp_data['trade_date'] = pd.to_datetime(resp_data['trade_date'], format='%Y%m%d')
                resp_data['trade_date'] = resp_data['trade_date'].dt.strftime('%Y-%m-%d')
                resp_data = resp_data.sort_values(by=['trade_date', 'price'], ascending=[True, False])
            else:
                # 数据量较大，逐月获取
                trade_month_list = get_month_list(req_start_date, last_trade_date)
                trade_month_list.reverse()

                data_list = []

                for month in trade_month_list:
                    cyq_data = self._get_cyq_by_month(month)
                    if cyq_data is None or len(cyq_data) <= 0:
                        break

                    data_list.insert(0, cyq_data)

                if len(data_list) <= 0:
                    print(f"{self.stock_code}没有筹码分布数据")
                    return DataFrame(columns=dtype.keys())

                resp_data = pd.concat(data_list)

            resp_data.index = resp_data['trade_date'] + '_' + resp_data['price'].apply(str)
            resp_data.index.name = 'index'

            if 'ts_code' in resp_data.columns:
                del resp_data['ts_code']

            return resp_data

        return self._get_something(table_name=table_name,
                                   dtype=dtype,
                                   req_func=req_func,
                                   start_date=start_date,
                                   end_date=end_date,
                                   request=request,
                                   call_method='cyq',
                                   *args,
                                   **kwargs,
                                   )

    def get_cyq_by_trade_date(self, trade_date: str):
        """
        获取某一天的筹码分布
        """
        return self.get_cyq(trade_date, trade_date)

    @staticmethod
    def cyq_extra_dtype():
        return {
            "trade_date": str,
            "open_winner": float,  # 开盘价时的获利盘
            "close_winner": float,  # 收盘价时的获利盘
            "high_winner": float,  # 最高价时的获利盘
            "low_winner": float,  # 最低价时的获利盘。 上面这四个可以绘制成博弈K线
            "ASR": float,  # 活动筹码，统计在该价位上下10%的区间内的筹码总量（该区间散户最容易割肉）
            "ASRC": float,  # 活动筹码变化量 ASR Change。这个看下来和换手率其实差不多
            "CKDP": float,  # 筹码分布相对价位, CKDP = (当天收盘价 - 最低成本价) / (最高成本价 - 最低成本价) × 100%
            "CKDW": float,  # 成本重心，CKDW = (平均成本价 - 最低成本价) / (最高成本价 - 最低成本价) × 100%，而平均成本价是50%百分位的筹码价格
            "CBW": float,  # 成本带宽，(最高成本价 - 最低成本价) / 最低成本价 × 100%
        }

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_cyq_extra(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs):
        """
        获取筹码分布衍生指标
        """
        dtype = TuShareDataReader.cyq_extra_dtype()

        table_name = f"cyq_extra_{self.stock_code}"

        def req_func(req_start_date, req_end_date):
            if req_start_date is None:
                req_start_date = '20100101'

            req_start_date = date_utils.convert_format(req_start_date, '%Y%m%d', '%Y-%m-%d')
            cyq_data = self.get_cyq(req_start_date, None)

            if cyq_data is None or len(cyq_data) <= 0:
                return DataFrame(columns=dtype.keys())

            data = DataFrame(cyq_data['trade_date'].drop_duplicates())
            data = data.set_index('trade_date')

            req_start_date = data.index[0]
            req_end_date = data.index[-1]

            if len(set(dtype.keys()) - set(data.columns)) > 0:  # 存在新加的指标，该指标要全部重新计算
                daily_data = self.get_daily(req_start_date, req_end_date)
                cyq_data = self.get_cyq(req_start_date, req_end_date)

                if 'open_winner' not in data.columns:
                    # 博弈K线，open_winner，close_winner，high_winner，low_winner
                    data = data_process.add_CYQK(data, daily_data, cyq_data)

                if 'ASR' not in data.columns:
                    data = data_process.add_ASR(data, daily_data, cyq_data)

                if 'ASRC' not in data.columns:
                    data = data_process.add_ASRC(data, daily_data, cyq_data)

                if 'CKDP' not in data.columns:
                    data = data_process.add_CKDP(data, daily_data, cyq_data)

                if 'CKDW' not in data.columns:
                    data = data_process.add_CKDW(data, daily_data, cyq_data)

                if 'CBW' not in data.columns:
                    data = data_process.add_CBW(data, daily_data, cyq_data)

            return data

        return self._get_something(table_name=table_name,
                                   dtype=dtype,
                                   req_func=req_func,
                                   start_date=start_date,
                                   end_date=end_date,
                                   request=request,
                                   call_method='cyq_extra',
                                   *args,
                                   **kwargs,
                                   )

    @staticmethod
    def overall_extra_dtype():
        return {
            'trade_date': str,
            'company_number': int,  # 上市公司数量
            'profitable_company_number': int,  # 近一年盈利的上市公司数量（即PE>0）
            'avg_pe_ttm': float,  # 平均动态市盈率。市场上所有市盈率大于0的
            'weight_avg_pe_ttm': float,  # 加权平均动态市盈率（按市值进行加权）。市场上所有市盈率大于0的
            'rise_company_number': int,  # 当天上涨的公司数
            'fall_company_number': int,  # 当天下跌公司数（包含涨幅为0的）
            'overall_total_mv': float,  # 全体公司的总市值（万亿元）
            'overall_circ_mv': float,  # 全体公司的流通市值（万亿元）
        }

    @staticmethod
    def get_overall_extra(start_date, end_date, request=True, *args, **kwargs):
        # 000001是随便写的，什么都可以
        return TuShareDataReader("000001").set_date_range(start_date, end_date)._get_overall_extra(request=request,
                                                                                                   *args, **kwargs)

    @pin_memory(is_obj=True, obj_fields=('start_date', 'end_date',))
    def _get_overall_extra(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs):
        """
        获取全局指标，即整个大盘的指标情况
        """
        dtype = TuShareDataReader.overall_extra_dtype()

        table_name = 'overall_extra'

        def compute_func(req_start_date, req_end_date):
            if req_start_date is None:
                req_start_date = '20100101'

            if req_end_date is None:
                req_end_date = self.get_last_trade_date(get_today())

            req_start_date = date_utils.convert_format(req_start_date, '%Y%m%d', '%Y-%m-%d')

            data_item_list = []
            # 一天一天处理，要不内存扛不住
            for trade_date in tqdm(date_utils.get_date_list(req_start_date, req_end_date), desc="overall_extra"):
                if not is_trading_day(trade_date):
                    continue

                sql_template = "select * from {table_name} where trade_date='%s'" % trade_date
                overall_daily_data = self.db.select_union('daily', sql_template)
                overall_daily_basic_data = self.db.select_union('daily_basic', sql_template)

                if len(overall_daily_data) != len(overall_daily_basic_data):
                    print("[WARN]daily和daily_basic数量不相等！trade_date:" + trade_date)

                profitable_daily_basic_data = overall_daily_basic_data[
                    overall_daily_basic_data['pe_ttm'] > 0]  # 盈利的公司basic数据

                # 逐个计算每个指标
                company_number = len(overall_daily_data)  # 上市公司数量
                rise_company_number = (overall_daily_data['pct_chg'] > 0).sum()  # 当天上涨
                fall_company_number = (overall_daily_data['pct_chg'] <= 0).sum()  # 当天下跌

                profitable_company_number = len(profitable_daily_basic_data)  # 盈利公司数量
                if profitable_company_number <= 0:
                    print("[WARN]盈利公司数量为0，数据有问题！trade_date:" + trade_date)
                    data_item_list.append({
                        'trade_date': trade_date,
                        'company_number': company_number,
                        'profitable_company_number': profitable_company_number,
                        'rise_company_number': rise_company_number,
                        'fall_company_number': fall_company_number,
                    })
                    continue

                # 计算与市盈率相关的指标
                avg_pe_ttm = round(profitable_daily_basic_data['pe_ttm'].mean(), 2)  # 平均动态市盈率
                weight_avg_pe_ttm = np.average(profitable_daily_basic_data['pe_ttm'],
                                               weights=profitable_daily_basic_data['total_mv'])  # 加权平均动态市盈率

                overall_total_mv = round(overall_daily_basic_data['total_mv'].sum() / 1_0000_0000, 2)  # 全体公司的总市值（万亿元）
                overall_circ_mv = round(overall_daily_basic_data['circ_mv'].sum() / 1_0000_0000, 2)  # 全体公司的流通市值（万亿元）

                data_item_list.append({
                    'trade_date': trade_date,
                    'company_number': company_number,
                    'profitable_company_number': profitable_company_number,
                    'avg_pe_ttm': avg_pe_ttm,
                    'weight_avg_pe_ttm': weight_avg_pe_ttm,
                    'rise_company_number': rise_company_number,
                    'fall_company_number': fall_company_number,
                    'overall_total_mv': overall_total_mv,
                    'overall_circ_mv': overall_circ_mv,
                })

            resp_data = DataFrame(data_item_list)
            resp_data = set_trade_date_as_index(resp_data, format='%Y-%m-%d')
            return resp_data

        return self._get_something(table_name=table_name,
                                   dtype=dtype,
                                   req_func=compute_func,
                                   start_date=start_date,
                                   end_date=end_date,
                                   request=request,
                                   call_method='overall_extra',
                                   *args,
                                   **kwargs,
                                   )

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def find_support(
            self,
            window=40,
            offset=10,
            resistance=False,  # 若想寻找压力位，则指定该值为True
            ascending=True,  # 是否升序。默认为False
            *args,
            **kwargs,
    ) -> DataFrame:
        """
        寻找股票的支撑位。(该方法经过测试，速度很快。因此当速度慢时，不需要考虑这里)

        寻找思路：使用滑动窗口进行遍历，若窗口内最低点处在窗口的相对中间位置，则该位置就是压力位。越靠近当前时间的压力位越有效。
        具体流程如下
        1. 定义一个窗口，大小为w天。
        2. 定义一个offset，大小为o天。o<=n
        3. 从股票上市日期，开始使用滑动窗口遍历。
        4. 若当前窗口的最低点的处在中间局域，则为支撑位。
           中间区域按滑动窗口大小进行定义。即：若 (w-o)/2 <=最低点<= (w+o)/2
        5. 接着将窗口向后滑动o天，然后继续执行4动作
        6. 不断重复4,5过程，直到窗口大小不满足为止。

        该思路不同取值性质：
        1. 窗口大小w越大，支撑位数量越少，但支撑位支撑越强。反之亦然
        2. offset越小，查找速度越慢，但查找出来的支撑位更准确。

        （压力位同理）

        建议：
        1. 寻找支撑位时，window尽量“大”一点，这样可以保证支撑位更有效
        2. 寻找阻力位时，window尽量“小”一点，这样可以防止股票没涨到阻力位就回落的情况。
        """
        daily_data = self.get_daily()

        trade_date_set = set()  # 记录一下trade_date，避免重复
        result_daily_list = []
        for i in range(0, len(daily_data), offset):
            daily_sharding = daily_data.iloc[i: i + window]

            if len(daily_sharding) < window:  # 忽略最后一个大小不足的窗口
                continue

            index: int
            if not resistance:
                # 支撑位
                index = daily_sharding['low'].argmin()
            else:
                # 阻力位
                index = daily_sharding['low'].argmax()

            trade_date = daily_sharding.iloc[index].name
            if (window - offset) / 2 <= index <= (window + offset) / 2 and trade_date not in trade_date_set:
                trade_date_set.add(trade_date)
                result_daily_list.append(daily_sharding.iloc[index:index + 1])

            continue

        if is_null(result_daily_list):
            return daily_data.iloc[0:1].drop(index=daily_data.iloc[0:1].index)

        result_daily_data = pd.concat(result_daily_list)
        result_daily_data = result_daily_data.sort_index(ascending=ascending)

        return result_daily_data

    @staticmethod
    def get_union(data_type: str, start_date: str, end_date: str):
        """
        获取某种数据某一段时间内的所有数据。主要用于解决一张张表读取速度太慢的问题。

        例如：data_type='daily'
        那么就是获取所有股票某一段时间内的daily数据。
        """
        table_name = data_type
        sql_template = "select * from {table_name} where trade_date>='%s' and trade_date<='%s'" % (start_date, end_date)

        return TuShareDataReader.db.select_union(table_name, sql_template, add_stock_code=True)


if __name__ == '__main__':
    # _data = TuShareDataReader.get_trade_cal("2024-01-01", "2024-02-02")
    # _data = TuShareDataReader.get_index_basic()
    # _data = TuShareDataReader.get_index_daily("上证指数", "2023-01-01", "2023-02-02")
    # _data = TuShareDataReader.get_stock_list(update=True)
    _data = TuShareDataReader("000001").set_date_range("2023-01-01", "2024-02-02").get_daily()
    # _data = TuShareDataReader("000046").set_date_range("2024-02-02", "2024-02-02").get_cyq()
    # _data = TuShareDataReader("000001").set_date_range("2023-01-01", "2024-02-02").get_cyq_extra()

    # _data = TuShareDataReader("832317").set_date_range("2024-02-01", "2024-02-01").get_daily_extra()
    #

    # _data = TuShareDataReader("上证指数", is_index=True).set_date_range("2023-01-01", "2024-01-02").get_index_daily(index_prefix=True)

    # stock_list = TuShareDataReader.get_stock_list()
    #
    # for stock_code in tqdm(list(stock_list.index)):
    #     TuShareDataReader(stock_code).set_date_range("2010-01-01", "2024-02-01").get_daily_extra()

    # _data = TuShareDataReader("000001").set_date_range(start_date='2010-01-01',
    #                                                    end_date='2024-02-01').get_daily_extra()

    # _data = TuShareDataReader("000005").set_date_range(start_date=date_add(get_today(), n_day=-365),
    #                                                    end_date=get_today()).get_daily_basic()

    # 获取今日最新的数据
    # _data = TuShareDataReader._get_all_by_trade_date(data_type='daily')
    # _data = TuShareDataReader._get_all_by_trade_date(data_type='daily_basic')

    # _data = TuShareDataReader("000001").set_date_range("2010-01-01", "2024-02-01").get_overall_extra()

    # _data = TuShareDataReader("000001").set_date_range(start_date='2012-01-01',
    #                                                    end_date='2024-02-01').get_daily_open()

    # overall_extra_data = TuShareDataReader.get_overall_extra('2024-01-31', '2024-01-31')

    # _data = TuShareDataReader.get_union("daily", "2024-02-01", "2024-02-01")

    _data = TuShareDataReader.get_stock_list(list_status='L', only_stock_code=True)

    print()
    pass


