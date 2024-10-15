import time

from pandas import DataFrame
from tqdm import tqdm

from data_reader.base import DataReader
from data_reader.daily import db_name
from data_reader.utils import pin_memory, set_trade_date_as_index
from utils import date_utils
from utils.log_utils import print_verbose


class DailyDataReader(DataReader):
    data_name = "daily"

    def __init__(self, stock_code: str):
        super().__init__(stock_code, data_type='stock', db_name=db_name)
        self.stock_code = stock_code

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_data(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs) -> DataFrame:
        table_name = f"{self.data_name}_{self.stock_code}"

        def req_func(req_start_date, req_end_date):
            if req_start_date is None:
                req_start_date = '20100101'

            print_verbose(f"获取{self.data_name}数据, ts_code：{self.ts_code}")
            resp_data = self.pro.daily(ts_code=self.ts_code,
                                       start_date=req_start_date)
            resp_data = set_trade_date_as_index(resp_data)
            time.sleep(60 / 500 * 2)

            return resp_data

        data = self.get_something(table_name=table_name,
                                  dtype=self.dtype(),
                                  req_func=req_func,
                                  start_date=start_date,
                                  end_date=end_date,
                                  request=request,
                                  call_method=self.data_name,
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
    def dtype():
        return {
            "trade_date": str,
            "ts_code": str,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "pre_close": float,
            "change": float,
            "pct_chg": float,
            "vol": float,
            "amount": float,
        }

    @staticmethod
    def refresh_all_data():
        """
        Request new data from API and store them into database.
        Note that the method is used for refreshing data when the data has large lack.
        """
        stock_list = DataReader.get_stock_list(update=False)
        for i, stock in tqdm(stock_list.iterrows(),
                             total=len(stock_list),
                             desc="Refresh %s" % DailyDataReader.data_name):
            end_date = None
            if stock['delist_date'] != 'None' and stock['delist_date'] is not None:
                end_date = date_utils.convert_format(stock['delist_date'], "%Y%m%d", "%Y-%m-%d")

            DailyDataReader(stock_code=stock.name).get_data(end_date=end_date)

        print(f"Finish Refresh {DailyDataReader.data_name} data!")

    @staticmethod
    def refresh_data(trade_date: str = None):
        def get_all_data_func(trade_date):
            resp_data = DataReader.pro.daily(trade_date=trade_date.replace("-", ""))
            print(f"获取多个股票{trade_date}日线行情(daily)")
            return resp_data

        def refresh_stock_data_func(stock_code):
            DailyDataReader(stock_code).get_data()

        DataReader._refresh_data(
            table_name_template='daily_{stock_code}',
            dtype=DailyDataReader.dtype(),
            get_all_data_func=get_all_data_func,
            refresh_stock_data_func=refresh_stock_data_func,
            data_type='daily',
            db_name=db_name,
            trade_date=trade_date,
        )
