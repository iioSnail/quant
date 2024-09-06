import pandas as pd

from data_reader.base import DataReader
from data_reader.utils import pin_memory, set_trade_date_as_index
from utils.log_utils import print_verbose
from utils.utils import is_null


class FundBasicDataReader(DataReader):
    data_name = "fund_basic"

    def __init__(self):
        super().__init__("", data_type='basic')

    @pin_memory(is_obj=True, obj_fields=())
    def get_data(self, request=True, *args, **kwargs):
        table_name = f"{self.data_name}"

        def req_func(req_start_date, req_end_date):
            print_verbose(f"获取{self.data_name}数据")
            resp_data = self.pro.fund_basic(market='E', status='L')
            resp_data['stock_code'] = resp_data.ts_code.str[:6]
            resp_data = resp_data.set_index('stock_code')

            return resp_data

        return self.get_something(table_name=table_name,
                                  dtype=self.dtype(),
                                  req_func=req_func,
                                  request=request,
                                  call_method=self.data_name,
                                  index_col='stock_code',
                                  *args,
                                  **kwargs
                                  )

    def dtype(self):
        return {
            "ts_code": str,  # 基金代码
            "name": str,  # 简称
            "management": str,  # 管理人
            "custodian": str,  # 托管人
            "fund_type": str,  # 投资类型
            "found_date": str,  # 成立日期
            "due_date": str,  # 到期日期
            "list_date": str,  # 上市时间
            "issue_date": str,  # 发行日期
            "delist_date": str,  # 退市日期
            "issue_amount": float,  # 发行份额(亿)
            "m_fee": float,  # 管理费
            "c_fee": float,  # 托管费
            "duration_year": float,  # 存续期
            "p_value": float,  # 面值
            "min_amount": float,  # 起点金额(万元)
            "exp_return": float,  # 预期收益率
            "benchmark": str,  # 业绩比较基准
            "status": str,  # 存续状态D摘牌 I发行 L已上市
            "invest_type": str,  # 投资风格
            "type": str,  # 基金类型
            "trustee": str,  # 受托人
            "purc_startdate": str,  # 日常申购起始日
            "redm_startdate": str,  # 日常赎回起始日
            "market": str,  # E场内O场外
        }
