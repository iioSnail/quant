"""
This python file is used to value the given stock.
"""


def valuation_DCF(
        next_decade_fcf: list,
        next_decade_dr: float,
        future_gr: float,
        future_dr: float,
        n_shares: float,
) -> float:
    """
    Value the given stock by the DCF model. The return value is a estimated
    stock price of the stock.

    Parameters:
    ----------
    next_decade_fcf: The estimated free cash flow in the next decade. The unit is "亿元".
    next_decade_dr: The estimated discount rate for the next decade. For example, 0.05.
    future_gr: The estimated growth rate in the future. Usually, the GDP growth speed rate will be as the value.
    future_dr: The estimated discount rate after the next decade.
    n_shares: Number of shares outstanding. The unit is "万股".
    """
    assert len(next_decade_fcf) == 10, "The length of the parameter 'next_decade fcf' is not equals 10."

    pv = 0
    # Compute the present value with the future values of the next decade.
    for i, fv in enumerate(next_decade_fcf):
        pv += fv / (1 + next_decade_dr) ** (i + 1)

    # Compute the discounted present value of perpetual annuity.
    pv += next_decade_fcf[-1] * (1. + future_gr) / (future_dr - future_gr) / (1 + future_dr) ** 10

    return round((pv * 100_000_000) / (n_shares * 10000), 2)


if __name__ == '__main__':
    # 茅台
    price = valuation_DCF(
        next_decade_fcf=[650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100],
        next_decade_dr=0.05,
        future_gr=0.04,
        future_dr=0.07,
        n_shares=252485.11,
    )

    print("贵州茅台: ", price)

    # 药明康德
    price = valuation_DCF(
        next_decade_fcf=[70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
        next_decade_dr=0.05,
        future_gr=0.04,
        future_dr=0.07,
        n_shares=252485.11,
    )

    print("药明康德：", price)

    price = valuation_DCF(
        next_decade_fcf=[70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
        next_decade_dr=0.05,
        future_gr=0.04,
        future_dr=0.07,
        n_shares=757803.72,
    )
    print("隆基绿能：", price)

    price = valuation_DCF(
        next_decade_fcf=[60, 80, 90, 100, 110, 120, 130, 140, 150, 150],
        next_decade_dr=0.05,
        future_gr=0.04,
        future_dr=0.07,
        n_shares=150270.68,
    )
    print("洋河股份：", price)

    price = valuation_DCF(
        next_decade_fcf=[400, 420, 440, 460, 480, 500, 520, 540, 560, 580],
        next_decade_dr=0.05,
        future_gr=0.04,
        future_dr=0.07,
        n_shares=388152.59,
    )
    print("五粮液：", price)

    price = valuation_DCF(
        next_decade_fcf=[80, 90, 100, 110, 120, 130, 140, 150, 160, 170],
        next_decade_dr=0.05,
        future_gr=0.04,
        future_dr=0.07,
        n_shares=910599.88,
    )
    print("海康威视：", price)
    
    price = valuation_DCF(
        next_decade_fcf=[19, 21, 23, 25, 27, 29, 31, 33, 35, 37],
        next_decade_dr=0.05,
        future_gr=0.04,
        future_dr=0.07,
        n_shares=46252.56,
    )
    print("金山办公：", price)

    price = valuation_DCF(
        next_decade_fcf=[40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
        next_decade_dr=0.05,
        future_gr=0.04,
        future_dr=0.07,
        n_shares=121_442.70,
    )
    print("韦尔股份：", price)



