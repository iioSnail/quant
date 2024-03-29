from src.analysis import get_results, plot_results, analysis_A_share
from utils.data_process import read_data, add_prev_n, add_after_n, add_MA, add_amplitude, add_high


def ShangZhangQueKou(
        stock_code,
        open_pct_chg=0.03,  # 开盘涨幅
        future_days=(5, 10, 20, 40, 60),
        print_result=False,
        plot_examples=False,
):
    """
    上涨缺口
    """
    reader, data = read_data(stock_code)
    data = add_prev_n(data, n=1)
    curr_data = data

    # 收盘价 > 开盘价
    curr_data = curr_data[curr_data['close'] >= curr_data['open']]

    # 开盘价相较于昨日收盘价上涨>=3%
    curr_data = curr_data[(curr_data['open'] - curr_data['prev_1_close']) / curr_data['prev_1_close'] >= open_pct_chg]

    # 开盘价相较于昨日收盘价上涨<4%
    curr_data = curr_data[
        (curr_data['open'] - curr_data['prev_1_close']) / curr_data['prev_1_close'] < open_pct_chg + 0.01]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def YiZiZhangTing(
        stock_code,
        open_pct_chg=0.03,  # 开盘涨幅
        future_days=(5, 10, 20, 40, 60),
        print_result=False,
        plot_examples=False,
):
    """
    一字涨停
    """
    reader, data = read_data(stock_code)
    data = add_prev_n(data, n=1)
    data = add_after_n(data, n=1)
    curr_data = data

    # 开盘价 = 最低价
    curr_data = curr_data[curr_data['open'] == curr_data['low']]

    # 收盘价 = 最高价
    curr_data = curr_data[curr_data['close'] == curr_data['high']]

    # 开盘价 = 收盘价
    curr_data = curr_data[curr_data['open'] == curr_data['close']]

    # 开盘价相较于昨日收盘价上涨>=10%
    curr_data = curr_data[(curr_data['open'] - curr_data['prev_1_close']) / curr_data['prev_1_close'] >= 0.1]

    # 后一天不是一字涨停，即可以买入
    curr_data = curr_data[curr_data['after_1_high'] - curr_data['after_1_low'] > 0]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=10, r_offset=60, ma_list=(60,), )

    return results


def ShangZhangTuPoQueKou(
        stock_code,
        future_days=(5, 10, 20, 40, 60),
        retracement=0.3,  # 近60日回撤
        print_result=False,
        plot_examples=False,
):
    """
    上涨的突破缺口
    """
    reader, data = read_data(stock_code)
    data = add_prev_n(data, n=1)
    data = add_after_n(data, n=1)
    data = add_MA(data, day_list=(60,))  # 增加均线
    data = add_high(data, day_list=(60,))
    n_amp = 40
    data = add_amplitude(data, day_list=(n_amp,))

    curr_data = data

    # 开盘价相较于昨日收盘价上涨>=3%
    curr_data = curr_data[(curr_data['open'] - curr_data['prev_1_close']) / curr_data['prev_1_close'] >= 0.03]

    # 当天上涨，且不是一字涨停（收盘价 > 开盘价）
    curr_data = curr_data[curr_data['close'] > curr_data['open']]

    # 缺口没有被回补（当日最低价>昨日收盘价）
    curr_data = curr_data[curr_data['low'] > curr_data['prev_1_close']]

    # 当前位置相比近60天的最高价下跌了30%
    curr_data = curr_data[(curr_data['close'] - curr_data['60high']) / curr_data['60high'] <= -retracement]

    # > -40%
    curr_data = curr_data[(curr_data['close'] - curr_data['60high']) / curr_data['60high'] > -(retracement + 0.1)]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(60,), )

    return results


if __name__ == '__main__':
    # ShangZhangQueKou("000001.SZ", print_result=True, plot_examples=True)
    # analysis_A_share(ShangZhangQueKou)

    # YiZiZhangTing("000004.SZ", print_result=True, plot_examples=True)
    # analysis_A_share(YiZiZhangTing)
    # ShangZhangTuPoQueKou("000004.SZ", print_result=True, plot_examples=True)
    analysis_A_share(ShangZhangTuPoQueKou, line_func_kwargs={"retracement": 0.2})
    analysis_A_share(ShangZhangTuPoQueKou, line_func_kwargs={"retracement": 0.3})
    analysis_A_share(ShangZhangTuPoQueKou, line_func_kwargs={"retracement": 0.4})
    analysis_A_share(ShangZhangTuPoQueKou, line_func_kwargs={"retracement": 0.5})
    analysis_A_share(ShangZhangTuPoQueKou, line_func_kwargs={"retracement": 0.6})
    analysis_A_share(ShangZhangTuPoQueKou, line_func_kwargs={"retracement": 0.7})
