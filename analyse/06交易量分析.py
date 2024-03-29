# -*- coding: UTF-8 -*-
from src.analysis import get_results, plot_results, analysis_A_share
from utils.data_process import read_data, add_QRR, add_high_vol, add_MA


def DanRiFangLiangShangZhang(
        stock_code,
        future_days=(5, 10, 20, 40, 60),
        qrr=2,
        print_result=False,
        plot_examples=False,
):
    """
    当日突然放量上涨
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    data = add_QRR(data, day_list=(5,))
    data = add_high_vol(data, day_list=(40, ))

    curr_data = data

    # 量比>=2
    curr_data = curr_data[(curr_data['5QRR'] >= qrr) & (curr_data['5QRR'] <= qrr+1)]

    # 当日涨幅>=3%
    curr_data = curr_data[curr_data['pct_chg'] >= 3]

    # 当日交易量 > 近40日最大交易量的1.5倍
    curr_data = curr_data[curr_data['volume'] > curr_data['40_high_vol'] * 1.5]

    # 当日收盘价 > 当日开盘价
    curr_data = curr_data[curr_data['close'] > curr_data['open']]

    # 股价处于低位
    # curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    # 股价处在高位
    # curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=45, r_offset=60, ma_list=(5, 20, 60,), )

    return results


def DanRiFangLiangXiaDie(
        stock_code,
        future_days=(5, 10, 20, 40, 60),
        qrr=2,
        print_result=False,
        plot_examples=False,
):
    """
    当日突然放量下跌
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(60,))
    data = add_QRR(data, day_list=(5,))
    data = add_high_vol(data, day_list=(40, ))

    curr_data = data

    # 量比>=2
    curr_data = curr_data[(curr_data['5QRR'] >= qrr) & (curr_data['5QRR'] <= qrr+1)]

    # 当日涨跌幅<=-3%
    curr_data = curr_data[curr_data['pct_chg'] <= -3]

    # 当日交易量 > 近40日最大交易量的1.5倍
    curr_data = curr_data[curr_data['volume'] > curr_data['40_high_vol'] * 1.5]

    # 当日收盘价 < 当日开盘价
    curr_data = curr_data[curr_data['close'] < curr_data['open']]

    # 股价处于低位
    # curr_data = curr_data[curr_data['60MA'] > curr_data['high']]

    # 股价处在高位
    # curr_data = curr_data[curr_data['60MA'] < curr_data['low']]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=45, r_offset=60, ma_list=(5, 20, 60,), )

    return results


if __name__ == '__main__':
    # DanRiFangLiangShangZhang("000001.SZ", qrr=20, print_result=True, plot_examples=True)

    for i in range(2, 20):
        # analysis_A_share(DanRiFangLiangShangZhang, line_func_kwargs={"qrr": i})
        analysis_A_share(DanRiFangLiangXiaDie, line_func_kwargs={"qrr": i})

    pass
