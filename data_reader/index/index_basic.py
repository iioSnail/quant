from data_reader.base import DataReader
from data_reader.utils import pin_memory
from utils.log_utils import print_verbose


class IndexBasicDataReader(DataReader):
    data_name = "index_basic"

    def __init__(self, market='SSE'):
        super().__init__("", data_type='basic')
        self.market = market

    @pin_memory(is_obj=True, obj_fields=())
    def get_data(self, *args, **kwargs):
        market_list = [
            "MSCI",  # MSCI指数
            "CSI",  # 中证指数
            "SSE",  # 上交所指数
            "SZSE",  # 深交所指数
            "CICC",  # 中金指数
            "SW",  # 申万指数
            "OTH",  # 其他指数
        ]

        if self.market not in market_list:
            raise RuntimeError("market不在范围内，可用范围: " + str(market_list))

        data = DataReader.db.select("index_basic",
                                    conditions=[f"market='{self.market}'"],
                                    index_col='ts_code',
                                    dtype=self.dtype(),
                                    )

        if data is None:
            data = DataReader.pro.index_basic(market=self.market, fields=','.join(self.dtype().keys()))
            print_verbose(f"获取指数基本信息，market: {self.market}")
            data = data.set_index('ts_code')

            DataReader.db.to_sql(data, 'index_basic', dtype=self.dtype())

        return data

    def dtype(self):
        return {
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
