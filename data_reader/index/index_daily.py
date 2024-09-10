from data_reader.base import DataReader
from data_reader.utils import pin_memory, set_trade_date_as_index
from utils.log_utils import print_verbose


class IndexDailyDataReader(DataReader):
    data_name = "index_daily"

    def __init__(self, stock_code,
                 index_prefix=True,  # 是否给每个列名都加上index前缀
                 ):
        super().__init__(stock_code, data_type='index')
        self.stock_code = stock_code
        self.index_prefix = index_prefix

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_data(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs):
        table_name = f"{self.data_name}_{self.ts_code}"

        def req_func(req_start_date, req_end_date):
            if req_start_date is None:
                req_start_date = '20100101'

            resp_data = self.pro.index_daily(
                ts_code=self.ts_code,
                start_date=req_start_date,
            )

            print_verbose(f"获取指数日线行情，ts_code: {self.ts_code}")
            resp_data = set_trade_date_as_index(resp_data)

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

        if self.index_prefix:
            columns = []
            for col in data.columns:
                if not col.startswith("index_"):
                    col = 'index_' + col

                columns.append(col)

            data.columns = columns

        return data

    def dtype(self):
        return {
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
