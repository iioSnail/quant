# -*- coding: UTF-8 -*-
from src.analysis import get_results, plot_results, analysis_A_share
from utils.data_process import read_data, add_QRR, add_high_vol, add_MA, add_RPY
from utils.data_reader import TuShareDataReader

# 分析区间
start_date = '2018-01-01'
end_date = '2023-10-31'


def DiWeiMiJi(
        stock_code,
        future_days=(5, 10, 20, 40, 60),
        print_result=False,
        plot_examples=False,
):
    """
    筹码低位密集

    定义：
    1. 主力高度控盘：当前股价下30%聚集了40%的筹码。
    2. 股价在低位：PRY2 <= 30%
    3. 有利好：例如大盘行情。TODO
    """
    reader = TuShareDataReader(stock_code).set_date_range(start_date, end_date)

    data = reader.get_daily(extras=("RPY2", "CYQP_30_0"))
    data['index'] = list(range(1, len(data) + 1))

    curr_data = data

    # 当前处于低位
    curr_data = curr_data[curr_data['RPY2'] <= 30]

    # 筹码密集
    curr_data = curr_data[curr_data['CYQP_30_0'] >= 40]

    results = get_results(curr_data, curr_data, future_days, print_result)

    if print_result:
        print("trade_date:", list(curr_data.index))

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=180, r_offset=60, volume=False)

    return results


if __name__ == '__main__':
    # DiWeiMiJi("000002", print_result=True, plot_examples=True)
    analysis_A_share(DiWeiMiJi)
