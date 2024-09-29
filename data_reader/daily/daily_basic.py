import time

from pandas import DataFrame
from tqdm import tqdm

from data_reader.base import DataReader
from data_reader.daily import db_name
from data_reader.utils import pin_memory, set_trade_date_as_index
from utils import date_utils
from utils.log_utils import print_verbose


class DailyBasicDataReader(DataReader):
    data_name = "daily_basic"

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
            resp_data = self.pro.daily_basic(ts_code=self.ts_code,
                                             start_date=req_start_date,
                                             fields=','.join(self.dtype().keys()),
                                             )

            resp_data = set_trade_date_as_index(resp_data)

            return resp_data

        return self.get_something(table_name=table_name,
                                  dtype=self.dtype(),
                                  req_func=req_func,
                                  start_date=start_date,
                                  end_date=end_date,
                                  request=request,
                                  call_method=self.data_name,
                                  *args,
                                  **kwargs
                                  )

    def dtype(self):
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

    @staticmethod
    def refresh_data():
        """
        Request new data from API and store them into database.
        Note that the method is used for refreshing data when the data has large lack.
        """
        stock_list = DataReader.get_stock_list(update=False)
        for i, stock in tqdm(stock_list.iterrows(),
                             total=len(stock_list),
                             desc="Refresh %s" % DailyBasicDataReader.data_name):
            end_date = None
            if stock['delist_date'] != 'None' and stock['delist_date'] is not None:
                end_date = date_utils.convert_format(stock['delist_date'], "%Y%m%d", "%Y-%m-%d")

            DailyBasicDataReader(stock_code=stock.name).get_data(end_date=end_date)
            time.sleep(60 / 500 * 2)

        print(f"Finish Refresh {DailyBasicDataReader.data_name} data!")
