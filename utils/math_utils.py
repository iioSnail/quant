import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def seek_extremum(X, model) -> (np.ndarray, np.ndarray):
    """
    寻找极大值点
    :param X: 例如：[1, 2, 3, 4, ...]
    :param model: sklearn的model，可以调用predict方法
    :return: 与X同大小的结果。例如：[0, 0, 1, 0, -1]。
    其中0表示该位置非极值，1表示极大值，-1表示极小值
    """

    extremum = np.zeros(len(X), dtype=int)

    X = X.reshape(-1, 1)
    LX = X - 1
    RX = X + 1

    processor = PolynomialFeatures(degree=7, include_bias=False)
    X = processor.fit_transform(X)
    LX = processor.fit_transform(LX)
    RX = processor.fit_transform(RX)

    Y = model.predict(X)
    LY = model.predict(LX)
    RY = model.predict(RX)

    extremum[(LY < Y) & (Y > RY)] = 1  # 极大值
    extremum[(LY > Y) & (Y < RY)] = -1  # 极小值

    return extremum, Y


def is_number(obj):
    try:
        t = 0 + obj
        return True
    except Exception:
        return False
