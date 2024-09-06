"""
Global index. For example: NASDAQ index.
"""

import os
import json

import pandas as pd

from data_reader.base import DataReader
from data_reader.utils import pin_memory, set_trade_date_as_index
from utils.log_utils import print_verbose
from utils.utils import is_null

"""
Because the TuShare API requires 600 yuan for accessing, I decided to use another free API to obtain the data.   
In this code snippet, the Baidu Stock API will be used, namely "https://gushitong.baidu.com/stock/us-NDAQ".
The first time you use it, you need to find the "getquotation" request, copy its response, and paste it into the `./baidu_json.json` file.
Then, you can run the main method, and the data will be stored in the database. 
The next time you run it, the program will retrieve the data from the database.
"""

data_json = ""


class IndexGlobalDataReader(DataReader):

    data_name = "index_global"

    def __init__(self, stock_code: str, json_file="./baidu_json.json"):
        """
        Common Index:
        纳斯达克指数:
        """
        super().__init__(stock_code, data_type='index')
        self.stock_code = stock_code
        self.stock_code = self._extract_index()

        self.json_file = json_file

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_data(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs):
        table_name = f"{self.data_name}_{self.stock_code}"

        return self.get_something(table_name=table_name,
                                  dtype=self.dtype(),
                                  req_func=self.resolve_data_json,
                                  start_date=start_date,
                                  end_date=end_date,
                                  request=request,
                                  call_method=self.data_name,
                                  *args,
                                  **kwargs
                                  )

    def resolve_data_json(self, req_start_date=None, req_end_date=None):
        """
        This method is used for resolve the Baidu Stock API's response.
        """
        print_verbose(f"解析百度API数据, ts_code：{self.ts_code}")

        assert os.path.exists(self.json_file), f"文件{self.json_file}不存在！"

        with open(self.json_file, encoding='utf-8') as f:
            data_json = json.load(f)

        data_json = data_json['Result']['newMarketData']

        headers = data_json['keys']
        data = data_json['marketData']

        data_list = []
        for row in data.split(";"):
            row = row.replace("--", "").replace("+", "")
            items = row.split(",")
            data_list.append(items)

        resp_data = pd.DataFrame(data=data_list, columns=headers)
        resp_data['ts_code'] = self.stock_code
        resp_data['trade_date'] = resp_data['time']
        resp_data['open'] = resp_data['open']
        resp_data['close'] = resp_data['close']
        resp_data['high'] = resp_data['high']
        resp_data['low'] = resp_data['low']
        resp_data['pre_close'] = resp_data['preClose']
        resp_data['change'] = resp_data['range']
        resp_data['pct_chg'] = resp_data['ratio']
        resp_data['vol'] = resp_data['volume']
        resp_data['amount'] = resp_data['amount']

        resp_data = resp_data[list(self.dtype().keys())]
        resp_data = set_trade_date_as_index(resp_data, format='%Y-%m-%d')

        return resp_data

    def dtype(self):
        return {
            "ts_code": str,  # TS指数代码
            "trade_date": str,  # 交易日
            "open": float,  # 开盘点位
            "close": float,  # 收盘点位
            "high": float,  # 最高点位
            "low": float,  # 最低点位
            "pre_close": float,  # 昨日收盘点
            "change": float,  # 涨跌点位
            "pct_chg": float,  # 涨跌幅
            # "swing": float,  # 振幅
            "vol": float,  # 成交量 （大部分无此项数据）
            "amount": float,  # 成交额 （大部分无此项数据）
        }

    def _extract_index(self):
        common_index = {
            "纳斯达克指数": "IXIC",
            "纳指": "IXIC"
        }

        if self.stock_code in common_index:
            return common_index[self.stock_code]

        return self.stock_code

