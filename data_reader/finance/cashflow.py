"""
Cashflow
现金流量表
"""

import pandas as pd

from data_reader.base import DataReader
from data_reader.utils import pin_memory, set_trade_date_as_index
from utils.log_utils import print_verbose
from utils.utils import is_null


class CashflowDataReader(DataReader):

    def __init__(self, stock_code: str):
        super().__init__(stock_code)
        self.stock_code = stock_code

    @pin_memory(is_obj=True, obj_fields=('stock_code', 'start_date', 'end_date',))
    def get_data(self, start_date: str = None, end_date: str = None, request=True, *args, **kwargs):
        table_name = f"cashflow_{self.stock_code}"

        def req_func(req_start_date, req_end_date):
            if req_start_date is None:
                req_start_date = '19980101'

            print_verbose(f"获取cashflow数据, ts_code：{self.ts_code}")
            resp_data = self.pro.cashflow(ts_code=self.ts_code, start_date=req_start_date)

            data_list = []
            for end_date in list(resp_data['end_date'].drop_duplicates()):
                item_data = resp_data[resp_data['end_date'] == end_date]
                if len(item_data) > 1:  # 如果同一个报告期有多个，则取最新的
                    item_data = item_data[item_data['update_flag'] == '1']

                if is_null(item_data):
                    print_verbose(self.stock_code, "股票", end_date, "报告期数据有问题")
                    continue

                data_list.append(item_data.iloc[0])

            resp_data = pd.DataFrame(data_list)
            resp_data['trade_date'] = resp_data['end_date']
            resp_data = set_trade_date_as_index(resp_data)

            return resp_data

        return self.get_something(table_name=table_name,
                                  dtype=self.dtype(),
                                  req_func=req_func,
                                  start_date=start_date,
                                  end_date=end_date,
                                  request=request,
                                  call_method='cashflow',
                                  *args,
                                  **kwargs
                                  )

    def dtype(self):
        return {
            "ts_code": str,  # TS股票代码
            "ann_date": str,  # 公告日期
            "f_ann_date": str,  # 实际公告日期
            "end_date": str,  # 报告期
            "comp_type": str,  # 公司类型(1一般工商业2银行3保险4证券)
            "report_type": str,  # 报表类型
            "end_type": str,  # 报告期类型
            "net_profit": float,  # 净利润
            "finan_exp": float,  # 财务费用
            "c_fr_sale_sg": float,  # 销售商品、提供劳务收到的现金
            "recp_tax_rends": float,  # 收到的税费返还
            "n_depos_incr_fi": float,  # 客户存款和同业存放款项净增加额
            "n_incr_loans_cb": float,  # 向中央银行借款净增加额
            "n_inc_borr_oth_fi": float,  # 向其他金融机构拆入资金净增加额
            "prem_fr_orig_contr": float,  # 收到原保险合同保费取得的现金
            "n_incr_insured_dep": float,  # 保户储金净增加额
            "n_reinsur_prem": float,  # 收到再保业务现金净额
            "n_incr_disp_tfa": float,  # 处置交易性金融资产净增加额
            "ifc_cash_incr": float,  # 收取利息和手续费净增加额
            "n_incr_disp_faas": float,  # 处置可供出售金融资产净增加额
            "n_incr_loans_oth_bank": float,  # 拆入资金净增加额
            "n_cap_incr_repur": float,  # 回购业务资金净增加额
            "c_fr_oth_operate_a": float,  # 收到其他与经营活动有关的现金
            "c_inf_fr_operate_a": float,  # 经营活动现金流入小计
            "c_paid_goods_s": float,  # 购买商品、接受劳务支付的现金
            "c_paid_to_for_empl": float,  # 支付给职工以及为职工支付的现金
            "c_paid_for_taxes": float,  # 支付的各项税费
            "n_incr_clt_loan_adv": float,  # 客户贷款及垫款净增加额
            "n_incr_dep_cbob": float,  # 存放央行和同业款项净增加额
            "c_pay_claims_orig_inco": float,  # 支付原保险合同赔付款项的现金
            "pay_handling_chrg": float,  # 支付手续费的现金
            "pay_comm_insur_plcy": float,  # 支付保单红利的现金
            "oth_cash_pay_oper_act": float,  # 支付其他与经营活动有关的现金
            "st_cash_out_act": float,  # 经营活动现金流出小计
            "n_cashflow_act": float,  # 经营活动产生的现金流量净额
            "oth_recp_ral_inv_act": float,  # 收到其他与投资活动有关的现金
            "c_disp_withdrwl_invest": float,  # 收回投资收到的现金
            "c_recp_return_invest": float,  # 取得投资收益收到的现金
            "n_recp_disp_fiolta": float,  # 处置固定资产、无形资产和其他长期资产收回的现金净额
            "n_recp_disp_sobu": float,  # 处置子公司及其他营业单位收到的现金净额
            "stot_inflows_inv_act": float,  # 投资活动现金流入小计
            "c_pay_acq_const_fiolta": float,  # 购建固定资产、无形资产和其他长期资产支付的现金
            "c_paid_invest": float,  # 投资支付的现金
            "n_disp_subs_oth_biz": float,  # 取得子公司及其他营业单位支付的现金净额
            "oth_pay_ral_inv_act": float,  # 支付其他与投资活动有关的现金
            "n_incr_pledge_loan": float,  # 质押贷款净增加额
            "stot_out_inv_act": float,  # 投资活动现金流出小计
            "n_cashflow_inv_act": float,  # 投资活动产生的现金流量净额
            "c_recp_borrow": float,  # 取得借款收到的现金
            "proc_issue_bonds": float,  # 发行债券收到的现金
            "oth_cash_recp_ral_fnc_act": float,  # 收到其他与筹资活动有关的现金
            "stot_cash_in_fnc_act": float,  # 筹资活动现金流入小计
            "free_cashflow": float,  # 企业自由现金流量
            "c_prepay_amt_borr": float,  # 偿还债务支付的现金
            "c_pay_dist_dpcp_int_exp": float,  # 分配股利、利润或偿付利息支付的现金
            "incl_dvd_profit_paid_sc_ms": float,  # 其中:子公司支付给少数股东的股利、利润
            "oth_cashpay_ral_fnc_act": float,  # 支付其他与筹资活动有关的现金
            "stot_cashout_fnc_act": float,  # 筹资活动现金流出小计
            "n_cash_flows_fnc_act": float,  # 筹资活动产生的现金流量净额
            "eff_fx_flu_cash": float,  # 汇率变动对现金的影响
            "n_incr_cash_cash_equ": float,  # 现金及现金等价物净增加额
            "c_cash_equ_beg_period": float,  # 期初现金及现金等价物余额
            "c_cash_equ_end_period": float,  # 期末现金及现金等价物余额
            "c_recp_cap_contrib": float,  # 吸收投资收到的现金
            "incl_cash_rec_saims": float,  # 其中:子公司吸收少数股东投资收到的现金
            "uncon_invest_loss": float,  # 未确认投资损失
            "prov_depr_assets": float,  # 加:资产减值准备
            "depr_fa_coga_dpba": float,  # 固定资产折旧、油气资产折耗、生产性生物资产折旧
            "amort_intang_assets": float,  # 无形资产摊销
            "lt_amort_deferred_exp": float,  # 长期待摊费用摊销
            "decr_deferred_exp": float,  # 待摊费用减少
            "incr_acc_exp": float,  # 预提费用增加
            "loss_disp_fiolta": float,  # 处置固定、无形资产和其他长期资产的损失
            "loss_scr_fa": float,  # 固定资产报废损失
            "loss_fv_chg": float,  # 公允价值变动损失
            "invest_loss": float,  # 投资损失
            "decr_def_inc_tax_assets": float,  # 递延所得税资产减少
            "incr_def_inc_tax_liab": float,  # 递延所得税负债增加
            "decr_inventories": float,  # 存货的减少
            "decr_oper_payable": float,  # 经营性应收项目的减少
            "incr_oper_payable": float,  # 经营性应付项目的增加
            "others": float,  # 其他
            "im_net_cashflow_oper_act": float,  # 经营活动产生的现金流量净额(间接法)
            "conv_debt_into_cap": float,  # 债务转为资本
            "conv_copbonds_due_within_1y": float,  # 一年内到期的可转换公司债券
            "fa_fnc_leases": float,  # 融资租入固定资产
            "im_n_incr_cash_equ": float,  # 现金及现金等价物净增加额(间接法)
            "net_dism_capital_add": float,  # 拆出资金净增加额
            "net_cash_rece_sec": float,  # 代理买卖证券收到的现金净额(元)
            "credit_impa_loss": float,  # 信用减值损失
            "use_right_asset_dep": float,  # 使用权资产折旧
            "oth_loss_asset": float,  # 其他资产减值损失
            "end_bal_cash": float,  # 现金的期末余额
            "beg_bal_cash": float,  # 减:现金的期初余额
            "end_bal_cash_equ": float,  # 加:现金等价物的期末余额
            "beg_bal_cash_equ": float,  # 减:现金等价物的期初余额
            "update_flag": str,  # 更新标志(1最新）
        }
