import numpy as np
import mplfinance as mpf
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from utils.data_process import read_data
from utils.math_utils import seek_extremum
from utils.plot import CandlePlot


def TouJianDing(
        stock_code,
        plot_examples=False,
):
    """
    头肩顶
    """
    reader, data = read_data(stock_code)

    # 窗口大小，一个头肩顶形态可以分为5个窗口，分别是左键左侧，左肩，头部，右肩，右肩，右肩右侧
    win_length = 10
    # 拟合时特殊点位的权重大小
    special_weights = 10.

    for offset in tqdm(range(0, len(data), win_length)):
        L = offset
        R = L + 5 * win_length

        curr_data = data[L:R]

        model = linear_model.LinearRegression()
        X = np.arange(len(curr_data)).reshape(-1, 1)
        # 头肩顶一共7个关键点位
        X = PolynomialFeatures(degree=7, include_bias=False).fit_transform(X)
        Y = np.array(curr_data['close'])

        # 权重，关键点权重大
        sample_weight = np.ones(len(curr_data))
        # 左肩左侧最低点（窗口1）
        l = 0
        r = l + 1 * win_length
        win1_high = curr_data[l:r]['close'].argmin()
        sample_weight[l + win1_high] = special_weights + 1

        # 左肩最高点
        l = l + 1 * win_length
        r = l + 1 * win_length
        win2_high = l + curr_data[l:r]['close'].argmax()
        sample_weight[win2_high] = special_weights + 2
        # 左肩最低点（从左肩最高点到窗口2结束）
        if win2_high < r:
            win2_low = l + curr_data[win2_high:r]['close'].argmin()
            sample_weight[win2_low] = special_weights + 3

        # 头部最高点
        l = l + 1 * win_length
        r = l + 1 * win_length
        sample_weight[l + curr_data[l:r]['close'].argmax()] = special_weights + 4
        # 右肩最高点
        l = l + 1 * win_length
        r = l + 1 * win_length
        win4_high = l + curr_data[l:r]['close'].argmax()
        sample_weight[win4_high] = special_weights + 5
        # 右肩最低点（从窗口4到右肩最高点）
        if win4_high > l:
            win4_low = l + curr_data[l:win4_high]['close'].argmin()
            sample_weight[win4_low] = special_weights + 6

        # 右肩右侧最低点
        l = l + 1 * win_length
        r = l + 1 * win_length
        sample_weight[l + curr_data[l:r]['close'].argmin()] = special_weights + 7

        model.fit(X, Y, sample_weight)

        # 求极大值点和极小值点，如果是极大值则，用1表示，否则为-1，非极值点则为0
        extremum_indices, extremum_values = seek_extremum(X[:, 0], model)

        if (extremum_indices != 0).sum() < 5:
            # 极值点数量小于5
            continue

        extre_indices = extremum_indices[extremum_indices != 0][:5]
        if (extre_indices == [1, -1, 1, -1, 1]).sum() < 5:
            # 极值点不符合头肩顶形态
            continue

        extre_values = extremum_values[extremum_indices != 0][:5]

        if not (extre_values[0] < extre_values[2] and extre_values[2] > extre_values[4]):
            # 头部必须高于左右两肩
            continue

        if not (extre_values[0] > extre_values[1] and extre_values[4] > extre_values[3]):
            # 左肩头必须大于左颈线，右肩头必须大于由肩头
            continue

        if abs((extre_values[3] - extre_values[1]) / extre_values[1]) > 0.03:
            # 颈线斜率不能太高
            continue

        if plot_examples:
            # np.argwhere(sample_weight >= special_weights).reshape(-1)
            CandlePlot(data=curr_data) \
                .volume() \
                .type("line") \
                .add_plot(mpf.make_addplot(model.predict(X))) \
                .add_marks(np.argwhere(extremum_indices != 0).reshape(-1)) \
                .color_windows(5) \
                .plot()

            print()


if __name__ == '__main__':
    TouJianDing("000001.SZ", plot_examples=True)
