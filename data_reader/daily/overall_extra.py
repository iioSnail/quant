"""
This data is not from any data API. It's computed by other data.

Note that this data is based on the daily data and daily_basic data.
"""
import numpy as np

from pandas import DataFrame
from tqdm import tqdm

from data_reader.base import DataReader
from data_reader.daily import db_name
from data_reader.daily.daily import DailyDataReader
from data_reader.daily.daily_basic import DailyBasicDataReader
from data_reader.utils import pin_memory, set_trade_date_as_index
from utils import date_utils
from utils.date_utils import get_today


class OverallExtraDataReader(DataReader):
    data_name = "overall_extra"

    def __init__(self):
        super().__init__("", data_type='extra', db_name=db_name)

    def _makeup_data(self, end_date):
        """
        The overall extra data is based on "daily" and "daily_basic" data. Therefore,
        it has to ensure both "daily" data and "daily_basic" data is not absent.
        Otherwise, the method will makeup these data first.
        """
        data = self.db.get_last_data(self.data_name, order_by='trade_date')
        start_date = data.iloc[0].trade_date
        if date_utils.date_diff(end_date, start_date) > 20:
            print("[WARN] 数据缺口较大，建议使用全量获取方式！")

        if date_utils.compare_to(end_date, self.get_last_trade_date()) > 0:
            end_date = self.get_last_trade_date()

        date_list = date_utils.get_date_list(start_date, end_date)[1:]

        for trade_date in date_list:
            print(f"[INFO]补足{trade_date}数据...")
            DailyDataReader.refresh_data(trade_date)
            DailyBasicDataReader.refresh_data(trade_date)

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_data(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs):
        table_name = f"{self.data_name}"

        self._makeup_data(end_date)

        def compute_func(req_start_date, req_end_date):
            if req_start_date is None:
                req_start_date = '20100101'

            if req_end_date is None:
                req_end_date = self.get_last_trade_date(get_today())

            req_start_date = date_utils.convert_format(req_start_date, '%Y%m%d', '%Y-%m-%d')

            data_item_list = []
            # 一天一天处理，要不内存扛不住
            for trade_date in tqdm(date_utils.get_date_list(req_start_date, req_end_date), desc="overall_extra"):
                if not DataReader.is_trading_day(trade_date):
                    continue

                sql_template = "select * from {table_name} where trade_date='%s'" % trade_date
                overall_daily_data = self.db.select_union('daily', sql_template)
                overall_daily_basic_data = self.db.select_union('daily_basic', sql_template)

                if len(overall_daily_data) != len(overall_daily_basic_data):
                    # set(overall_daily_data['ts_code']) - set(overall_daily_basic_data['ts_code'])
                    if len(overall_daily_data) < 100 or len(overall_daily_basic_data) < 100:
                        print("[WARN]daily或daily_basic数据有问题，停止计算！trade_date:" + trade_date)
                        break
                    else:
                        print("[WARN]daily和daily_basic数量不相等！trade_date:" + trade_date)

                # 未盈利的公司，将市盈率设置为0
                overall_daily_basic_data['pe_ttm'] = overall_daily_basic_data['pe_ttm'].fillna(0)
                profitable_daily_basic_data = overall_daily_basic_data[
                    overall_daily_basic_data['pe_ttm'] > 0]  # 盈利的公司basic数据

                # 逐个计算每个指标
                company_number = len(overall_daily_data)  # 上市公司数量
                rise_company_number = (overall_daily_data['pct_chg'] > 0).sum()  # 当天上涨
                fall_company_number = (overall_daily_data['pct_chg'] <= 0).sum()  # 当天下跌

                profitable_company_number = len(profitable_daily_basic_data)  # 盈利公司数量
                if profitable_company_number <= 0:
                    print("[ERROR]盈利公司数量为0，数据有问题！trade_date:" + trade_date)
                    break

                # 计算与市盈率相关的指标
                avg_pe_ttm = round(overall_daily_basic_data['pe_ttm'].mean(), 2)  # 平均动态市盈率
                weight_avg_pe_ttm = np.average(overall_daily_basic_data['pe_ttm'],
                                               weights=overall_daily_basic_data['total_mv'])  # 加权平均动态市盈率
                weight_avg_pe_ttm = round(weight_avg_pe_ttm, 2)

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

        return self.get_something(table_name=table_name,
                                  dtype=self.dtype(),
                                  req_func=compute_func,
                                  start_date=start_date,
                                  end_date=end_date,
                                  request=request,
                                  call_method=self.data_name,
                                  *args,
                                  **kwargs
                                  )

    def dtype(self):
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
