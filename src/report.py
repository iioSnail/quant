"""
This file is to generate a company's health report that tries to tell which indicator is risky.
"""
from utils.data_reader import TuShareDataReader
from utils.log_utils import table_print
from utils.utils import null_to_0


class ReportTable:

    def __init__(self,
                 stock_code,
                 report_period,
                 ):
        self.stock_code = stock_code
        self.report_period = report_period

        self.rows = []

    def add_row(self,
                factor_name,
                factor_value,
                normal_range
                ):
        self.rows.append(
            {
                "指标名称": factor_name,
                "指标值": factor_value,
                "参考范围": normal_range,
            }
        )

    def print_table(self):
        table_print(self.rows)


class CompanyReport(object):

    def __init__(self, stock_code,
                 request=False,  # whether to request Tushare server to get new data.
                 ):
        self.stock_code = stock_code
        self.reader = TuShareDataReader(stock_code)

        # 资产负债表。使用最后一次报告的数据
        # Using last report.
        self.balance_sheet_data = self.reader.get_balance_sheet(request=request)
        self.balance_sheet = self.balance_sheet_data.iloc[0]

        self.income_data = self.reader.get_income(request=request)
        # 最近一年的利润表
        # The last year of income report.
        self.year_income = None

        self.report_table = ReportTable(self.stock_code, self.balance_sheet.name)

    def factor_debt_to_asset_ratio(self):
        """
        debt to asset ratio
        资产负债率 = 总负债 / 总资产
        一般越小越好
        """
        value = self.balance_sheet['total_liab'] / self.balance_sheet['total_assets']
        value = round(value, 3)

        return self.report_table.add_row(
            "资产负债率",
            value,
            "0-1"
        )

    def factor_current_ratio(self):
        """
        流动比率 = 流动资产 / 流动负债
        一般越大越好
        """
        value = self.balance_sheet['total_cur_assets'] / self.balance_sheet['total_cur_liab']
        value = round(value, 3)

        return self.report_table.add_row(
            "流动比率",
            value,
            "1-5"
        )

    def factor_quick_ratio(self):
        """
        速动比率 = 速动资产 / 流动负债
        速动资产 = 流动资产 - 存货 - 预付账款 - 待摊费用
        """
        deno = self.balance_sheet['total_cur_assets'] - self.balance_sheet['inventories']
        deno = deno - null_to_0(self.balance_sheet['prepayment']) - null_to_0(self.balance_sheet['amor_exp'])
        value = deno / self.balance_sheet['total_cur_liab']
        value = round(value, 3)

        return self.report_table.add_row(
            "速动比率",
            value,
            "1-5"
        )

    def factor_interest_coverage_ratio(self):
        """
        利息保障倍数 = 息税前利润(EBIT) / 利息
        越大越好。若小于1，表示挣的钱练利息都不够还
        """



    def generate(self):
        for method in dir(self):
            if not method.startswith("factor_"):
                continue

            try:
                getattr(self, method)()
            except:
                print("[ERROR]%s计算出错" % method)

        return self.report_table



if __name__ == '__main__':
    report_table = CompanyReport("600000", request=True).generate()

    report_table.print_table()
