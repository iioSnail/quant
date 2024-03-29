from src.analysis import get_results, plot_results, analysis_A_share
from utils.data_process import read_data, add_prev_n, add_after_n, add_MA, add_amplitude, add_high, add_low


def HuangJinJiaoCha(
        stock_code,
        future_days=(5, 10, 20, 40, 60),
        print_result=False,
        plot_examples=False,
):
    """
    黄金交叉
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(5, 20, 60, ))  # 增加均线
    data = add_prev_n(data, n=1)
    data = add_high(data, day_list=(60,))

    curr_data = data

    # 当日5MA>当日20MA
    curr_data = curr_data[curr_data['5MA'] > curr_data['20MA']]

    # 昨日5MA < 昨日20MA
    curr_data = curr_data[curr_data['prev_1_5MA'] < curr_data['prev_1_20MA']]

    # 当前位置相比近60天的最高价下跌了30%
    curr_data = curr_data[(curr_data['close'] - curr_data['60high']) / curr_data['60high'] <= -0.3]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(5, 20, 60,), )

    return results


def SiWangJiaoCha(
        stock_code,
        future_days=(5, 10, 20, 40, 60),
        print_result=False,
        plot_examples=False,
):
    """
    死亡交叉
    """
    reader, data = read_data(stock_code)
    data = add_MA(data, day_list=(5, 20, 60, ))  # 增加均线
    data = add_prev_n(data, n=1)
    data = add_low(data, day_list=(60,))

    curr_data = data

    # 当日5MA<当日20MA
    curr_data = curr_data[curr_data['5MA'] < curr_data['20MA']]

    # 昨日5MA > 昨日20MA
    curr_data = curr_data[curr_data['prev_1_5MA'] > curr_data['prev_1_20MA']]

    # 当前位置相比近60天的最低价上涨了30%
    curr_data = curr_data[(curr_data['close'] - curr_data['60low']) / curr_data['60low'] >= 0.3]

    results = get_results(stock_code, curr_data, data, future_days, print_result)

    if plot_examples:
        for index in curr_data.index:
            plot_results(index, data, l_offset=60, r_offset=60, ma_list=(5, 20, 60,), )

    return results


if __name__ == '__main__':
    # HuangJinJiaoCha("000004.SZ", print_result=True, plot_examples=True)
    # analysis_A_share(HuangJinJiaoCha)
    # SiWangJiaoCha("000001.SZ", print_result=True, plot_examples=True)
    analysis_A_share(SiWangJiaoCha)

    pass
