# 量化选股

# 项目简介

本项目用于帮助股民对技术分析进行验证，找出有效的技术分析手段。

本项目包含以下几个部分：

- 分析：用户指定交易策略，分析历史数据，计算按该交易策略买入的话，后续n天上涨的概率
- 回测：用户指定买入策略、卖出策略，通过历史数据分析按该策略进行交易的收益情况
- 选股：用户指定交易策略，然后拉取当天数据，查询符合交易条件的股票

# 项目配置

本项目数据来源于[Tushare](https://tushare.pro/) ，因此需要花200元/年买会员，如果需要用到筹码分布，则500/年。

项目配置：

- 填写Tushare的Token，在 `./utils/data_reader`下
```
class TuShareDataReader(object):
    token = ""  # 这里填写token TODO，后续改成配置文件
```

# 分析模块

## 简单分析

如果你要做的分析比较简单，那么可以调用`./src/analysis/common_analysis`函数。

示例：

```python
from src.analysis import common_analysis

common_analysis(
        factors_range={
            'pct_chg': (-8, -2),  # 当天跌幅-8%~-2%
            'turnover_rate': (2, 8),  # 当天换手率 2%~8%
            'circ_mv': (500000, 5000000),  # 流通市值 50w(万元) ~ 500w(万元) 
            'RPY1': (0, 10),  # 一年相对价位在 0%~10%
        }
    )
```

运行该函数后，系统会开始用历史数据进行分析，最终会输出如下表格（该表格仅为示例，不是上述示例的分析结果）：

| n个交易日后 | 样本总数 | 上涨样本数 | 上涨概率 | 平均涨跌幅 | 上涨时平均涨幅 | 下跌时平均跌幅 | 样本集中度 | 胜率标准差 |
|:-----------:|:--------:|:----------:|:--------:|:----------:|:--------------:|:--------------:|:----------:|:----------:|
|      5      |   1825   |    1320    |  72.33   |   26.76%   |     38.77%     |     -4.62%     |   17.40%   |   18.42%   |
|      10     |   1825   |    1314    |   72.0   |   53.09%   |     76.06%     |     -5.97%     |   17.40%   |   16.63%   |
|      20     |   1825   |    1371    |  75.12   |   75.72%   |    103.39%     |     -7.84%     |   17.41%   |   16.10%   |
|      40     |   1825   |    1345    |   73.7   |   73.08%   |    102.91%     |    -10.49%     |   17.46%   |   17.28%   |
|      60     |   1825   |    1310    |  71.78   |   71.55%   |    103.98%     |    -10.95%     |   17.59%   |   17.27%   |

## 复杂分析

若你的技术指标比较复杂，简单分析无法满足，则可以采用继承抽象类`src.analysis.AnalysisOne`的方式完成。

示例：

```python
from pandas import DataFrame
from src.analysis import AnalysisOne

class LowChangXiaYingXian_AnalysisOne(AnalysisOne):
    """
    低位长下影线
    """

    def filter_data(self, data: DataFrame, *args, **kwargs):
        data = data[(-6 <= data['pct_chg']) & (data['pct_chg'] <= -3)]
        data = data[(4 <= data['turnover_rate']) & (data['turnover_rate'] <= 8)]

        return data
```

data包含了一只股票的所有历史信息（按天），你需要筛选出符合你的交易条件的数据，然后进行返回。

若你只想分析某一直股票的情况，则使用如下代码：

```python
LowChangXiaYingXian_AnalysisOne(stock_code="000001").analysis()
```

若你想分析整个A股的情况，则需要使用`src.analysis.analysis_A_share`，例如：

```
analysis_A_share(LowChangXiaYingXian_AnalysisOne)
```

# 回测模块

## 简单策略回测

与简单分析同理，若你的交易策略不复杂，则可以通用的回测函数`src.backtest.multi_backtest`。

示例：

```python
factors_range = {
    'FPY1': (-100, -60),
    'FAR': (-100, -6.25),
    'turnover_rate_f': (1, 7),
    'upper_wick': (0, 2.5),
    'pe_ttm': (3, 75),
}

multi_backtest(
    years='2018-2023',  # 要回测的年份
    init_money=100000,  # 初始资金
    buy_strategy=common_buy_strategy(factors_range, buy_ratio=0.1),  # 买入策略
    sell_strategy=sell_by_day(20),  # 卖出策略
    require_data=get_require_data_list(factors_range.keys()),  # 需要哪些数据（不读取不必要的数据可以加速回测）
    buy_timing='open',  # 什么节点买入（open：第二天的开盘价，close：当天的收盘价）
)
```

当回测函数执行完毕后，会输出以下表格：

| 年份 | 初始资金 | 结束资金 |  收益率 | 同期沪深300 | 出手次数 | 成功次数 | 失败次数 | 成功率 | 平均每日闲置资金 | 平均持仓时间 |
|:----:|:--------:|:--------:|:-------:|:-----------:|:--------:|:--------:|:--------:|:------:|:----------------:|:------------:|
| 2022 |  10.00w  |  20.48w  | 104.78% |   -21.27%   |    38    |    28    |    9     | 75.68% |     22348.55     |     32天     |
| 2021 |  10.00w  |  17.72w  |  77.17% |    -6.21%   |    39    |    26    |    13    | 66.67% |     45811.09     |     25天     |
| 2020 |  10.00w  |  26.78w  | 167.77% |    23.16%   |    26    |    17    |    9     | 65.38% |     87301.02     |     29天     |
| 2019 |  10.00w  |  20.53w  | 105.26% |    37.95%   |    27    |    18    |    8     | 69.23% |     20529.44     |     37天     |
| 2018 |  10.00w  |  9.88w   |  -1.19% |   -26.34%   |    38    |    16    |    19    | 45.71% |     16746.15     |     25天     |

## 复杂策略回测

若你的交易策略比较复杂，则可以自己编写买入策略。

示例：

```
def buy_strategy(
        data: Dict[str, DataFrame],  # 截止到今天的所以历史数据
        curr_date: str,  # 当天日期
        prev_data: Series,  # 昨日的数据
        curr_data: Series,  # 当天的数据
        dto: BacktestDto,  # 当前的持仓情况
        *args,
        **kwargs,
) -> float:
    if dto.n_shares > 0:  # 如果该股票已经持仓，则不再买入
        return 0.

    # 当天涨幅 >= 6%
    is_DaYangXian = (curr_data['close'] - curr_data['open']) / curr_data['open'] * 100 >= 6
    # 换手率低
    low_turnover = curr_data['turnover_rate'] <= 0.5

    if is_DaYangXian and low_turnover:
        # return 0.5 # 半仓
        return kelly_buy_ratio_strategy(0.65, 0.2, 0.1)  # 凯利公式
    else:
        return 0.  # 不买入

multi_backtest(
    years='2018-2023',  # 要回测的年份
    init_money=100000,  # 初始资金
    buy_strategy=buy_strategy,  # 买入策略
    sell_strategy=sell_by_day(20),  # 卖出策略
    require_data=('daily',)  # 需要哪些数据（不读取不必要的数据可以加速回测）
    buy_timing='open',  # 什么节点买入（open：第二天的开盘价，close：当天的收盘价）
)
```

## 买入策略

买入策略回调函数会在回测时被调用，传入当前股票响应的数据，用户可以根据自己的策略来判断是否要买入当前股票，返回值为买入的仓位比例。

```python
def buy_strategy(
            data: Dict[str, DataFrame],
            curr_date: str,  # 当天日期
            prev_data: Series,  # 昨日的数据
            curr_data: Series,  # 当天的数据
            dto: BacktestDto,  # 当前的持仓情况
            *args,
            **kwargs,
    ) -> float:
        return 0. # 返回买入的仓位比例，0代表不买入
```

## 卖出策略

本项目提供了部分卖出策略，包括：

- `src.strategy.sell_by_day`: 按天卖出，持有n天后不管涨跌都卖掉
- `src.sell_by_drawdown`: 回撤卖出，从买入后的最高股价回撤超过n%时，卖出

可以仿照上面的策略自定义策略


# TODO

- [ ] 文档、代码待整理：由于本项目之前是自己一个人用，所以文档和代码都不完善
- [ ] UI界面
- [ ] 注释
- [ ] 提供初始数据

