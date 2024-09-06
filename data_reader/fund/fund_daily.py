import pandas as pd

from data_reader.base import DataReader
from data_reader.utils import pin_memory, set_trade_date_as_index
from utils.log_utils import print_verbose
from utils.utils import is_null


class FundDailyDataReader(DataReader):
    data_name = "fund_daily"

    def __init__(self, stock_code: str):
        super().__init__(stock_code, data_type='fund')
        self.stock_code = stock_code

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_data(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs):
        table_name = f"{self.data_name}_{self.stock_code}"

        def req_func(req_start_date, req_end_date):
            if req_start_date is None:
                req_start_date = '20100101'

            print_verbose(f"获取{self.data_name}数据, ts_code：{self.ts_code}")
            resp_data = self.pro.fund_daily(ts_code=self.ts_code, start_date=req_start_date)
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
            "ts_code": str,  # TS代码
            "trade_date": str,  # 交易日期
            "open": float,  # 开盘价(元)
            "high": float,  # 最高价(元)
            "low": float,  # 最低价(元)
            "close": float,  # 收盘价(元)
            "pre_close": float,  # 昨收盘价(元)
            "change": float,  # 涨跌额(元)
            "pct_chg": float,  # 涨跌幅(%)
            "vol": float,  # 成交量(手)
            "amount": float,  # 成交额(千元)
        }
