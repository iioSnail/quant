# -*- coding: UTF-8 -*-

"""
分析模块：用于分析在某种条件下买入后，n天之后上涨的概率。以及上涨和下跌情况下，各个指标的情况。
        主要的作用是找出好的买入和卖出时机。
"""
import copy
import sys
import types
from pathlib import Path
from typing import Dict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from prettytable import PrettyTable
from tqdm import tqdm

from utils import date_utils
from utils.plot_utils import plot_analysis_factor, plot_analysis_sample_distribution
from utils.data_process import get_all_china_stock_code_list, remove_continual_data, drop_duplicated_columns
from utils.data_reader import TuShareDataReader, get_require_data_list
from utils.date_utils import get_now, date_add, get_today
from utils.log_utils import get_fprint, table_print, FilePrint
from utils.plot import CandlePlot
from utils.utils import load_obj, dict_change_place, is_null, equals, save_obj

log_dir = ROOT / 'log' / ("analysis_" + get_now(format='%Y%m%d_%H%M%S'))
fprint = FilePrint('analysis', console=True, log_dir=log_dir)

print(f"日志文件目录:{str(log_dir)}")


def analysis_candle(
        daily_data: DataFrame,  # 历史所有数据
        daily_basic_data: DataFrame,
        daily_extra_data: DataFrame,
        cyq_extra_data: DataFrame,
        index_daily_data: DataFrame,
        # 根据K线状态筛选后的数据。为curr_data为data的子集。
        # 例如curr_data为data中当日涨幅>6%的数据
        curr_data: DataFrame,
        # 想要计算curr_data中的数据在n个交易日后的涨跌幅
        # 例如，我想看curr_data中（某天涨幅超过6%）数据，在20个交易日后，它的涨幅情况
        future_day: int,
) -> dict:
    results = {
        'total_num': 0,  # 总样本数
        'rise_num': 0,  # N个交易日后股价上涨的数量
        'rise_pct': 0,  # N个交易日后股价总涨幅
        'avg_pct_chg': 0,  # 平均涨幅
        'up_rise_pct_list': [],  # 上涨情况下N个交易日后的涨幅
        'down_rise_pct_list': [],  # 下跌情况下N个交易日后的跌幅
        'up_year_dist': dict(),  # 上涨时的年份分布（例如：{"2015": 3, "2016: 4} 表示2015年有3个上涨样本...）
        'down_year_dist': dict(),  # 下跌时的年份分布
    }

    """
    在结果中增加其他的指标，用于帮助对上涨和下跌进行分析。
    例如：
    daily_factors中的pct_chg就表示：统计n日上涨和下跌时的平均pct_chg情况，在最后的结果中增加"up_pct_chg"和"down_pct_chg"，
                                  这些都代表发生事件当天的情况。
    假设最终：up_pct_chg=6% 就表示：对于n日后上涨的那批数据中，当日出现特定K线时的平均涨幅为6%,
            down_pct_chg=3% 就表示：对于n日后下跌的那批数据中，当日出现特定K线时的平均涨幅为3%,
            因此，我们就可以根据上述指标得出一个结论，当日上涨要超过6%，这样胜率会更大。从而调整我们的K线组合，进行重新测试。
            
    注意：统计结果会放在up_XXX_list和down_XXXX_list里。
    """

    if daily_data is None or len(daily_data) <= 0:
        return results

    """
    在这里增加指标后，输出结果会默认加上“%”，若你增加的指标不需要带“%”，则在后面可以排除。
    搜“counter[factor] = str(counter[factor]) + "%"”，就在这段代码那。
    """
    daily_factors = ('pct_chg',)
    daily_basic_factors = ('turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe_ttm', 'ps_ttm', 'pb', 'circ_mv',)
    daily_extra_factors = ("RPY1", "RPY2", "RPM1", "RPM2", "gap", "upper_wick", "lower_wick",
                           "amp", "FAR", "RAF", "FPY1", "FPM1", "FPM2")
    cyq_extra_factors = ("open_winner", "close_winner", "high_winner", "low_winner", "ASR", "ASRC", "CKDP",
                         "CKDW", "CBW",)
    index_daily_factors = ('index_close', 'index_pct_chg',)

    for factor in daily_factors + daily_basic_factors + daily_extra_factors + cyq_extra_factors + index_daily_factors:
        results[f"up_{factor}_list"] = []
        results[f"down_{factor}_list"] = []

    # 找出${future_days}个交易日后的当天行情数据
    future_data = daily_data[daily_data['index'].isin((curr_data['index'] + future_day))]

    for i in range(min(len(curr_data), len(future_data))):
        curr_sr = curr_data.iloc[i]
        future_sr = future_data.iloc[i]
        close = curr_sr['close']
        future_close = future_sr['close']
        year = curr_sr.name[:4]

        pct = (future_close - close) / close * 100
        if future_close > close:  # 上涨
            results['rise_num'] += 1
            results['up_rise_pct_list'].append(pct)

            # 统计年份分布
            results['up_year_dist'][year] = results['up_year_dist'].get(year, 0) + 1

            # 统计额外指标
            _stat_factors(curr_sr, daily_data, daily_factors, results, 'up_')
            _stat_factors(curr_sr, daily_basic_data, daily_basic_factors, results, 'up_')
            _stat_factors(curr_sr, daily_extra_data, daily_extra_factors, results, 'up_')
            # FIXME 这里有bug，tushare的cyq数据是从2018年开始的，最后的除数如果使用total_num的话，值会变小。
            _stat_factors(curr_sr, cyq_extra_data, cyq_extra_factors, results, 'up_')
            _stat_factors(curr_sr, index_daily_data, index_daily_factors, results, 'up_')

        else:  # 下跌
            results['down_rise_pct_list'].append(pct)

            # 统计年份分布
            results['down_year_dist'][year] = results['down_year_dist'].get(year, 0) + 1

            _stat_factors(curr_sr, daily_data, daily_factors, results, 'down_')
            _stat_factors(curr_sr, daily_basic_data, daily_basic_factors, results, 'down_')
            _stat_factors(curr_sr, daily_extra_data, daily_extra_factors, results, 'down_')
            _stat_factors(curr_sr, cyq_extra_data, cyq_extra_factors, results, 'down_')
            _stat_factors(curr_sr, index_daily_data, index_daily_factors, results, 'down_')

        results['rise_pct'] += pct

    results['total_num'] = len(curr_data)

    # 平均涨跌幅
    if results['total_num'] != 0:
        results['avg_pct_chg'] = results['rise_pct'] / results['total_num']

    return results


def _stat_factors(curr_sr, data, factors, results, prefix):
    for factor in factors:
        if curr_sr.name not in data.index:
            value = 0.
        else:
            value = data.loc[curr_sr.name][factor]

        if pd.isna(value):
            value = 0.

        results[f'{prefix}{factor}_list'].append(value)


def get_results(stock_code,
                curr_data,
                daily_data,
                future_days,
                print_result,
                daily_basic_data=None,
                daily_extra_data=None,
                cyq_extra_data=None,
                index_daily_data=None,  # 上证指数数据
                remove_2015=True,  # 是否去除2015年数据（2015-01-01~2015-10-01）
                remove_continual=False,  # 是否去除连续数据。（例如：2023-01-01,2023-01-02,2023-01-03，仅保留2023-01-01）
                ):
    pt = None

    results = []

    factor_props = None
    stat_factor_props = False  # 是否统计了factor_props
    if type(curr_data) == dict:
        curr_data, factor_props = curr_data['data'], curr_data['factor_props']
        stat_factor_props = True

    if is_null(curr_data) or is_null(daily_data):
        if print_result:
            print("无符合条件的记录！")

        if stat_factor_props:
            return {
                'results': results,
                'factor_props': factor_props,
            }
        else:
            return results

    if len(curr_data) > 0:
        fprint(f"股票{stock_code}, 样本: {list(curr_data.index)}")

    if remove_2015:
        curr_data = curr_data.loc[(curr_data.index < '2015-01-01') | (curr_data.index > '2015-10-01')]

    if remove_continual:
        # 去除连续数据会导致样本数“看起来”异常。例如：一开始我只有条件A，共1000个样本。之后我在这基础上增加了条件B，
        # 结果，增加了条件B后，总样本数反倒变为了2000。这是因为，条件A下，大多数样本都连续，因此都被去除掉了。而增加
        # 条件B后，很多样本A的连续性被打破了，因此总样本数反而变大了。
        # 因此，建议当探索期间，指定remove_continual=False，而最终要确定真正的胜率时，使用True。
        curr_data = remove_continual_data(curr_data)

    if date_utils.compare_to(daily_data.index[0], daily_data.index[-1]) > 0:
        raise RuntimeError(f"{stock_code}数据有问题，请检查！")

    # 读取额外统计数据
    reader = TuShareDataReader(stock_code).set_date_range(daily_data.index[0], daily_data.index[-1])

    if daily_extra_data is None:
        daily_extra_data = reader.get_daily_extra()

    if cyq_extra_data is None:
        cyq_extra_data = reader.get_cyq_extra()

    if daily_basic_data is None:
        daily_basic_data = reader.get_daily_basic()

    index_reader = TuShareDataReader('上证', is_index=True).set_date_range(daily_data.index[0], daily_data.index[-1])
    if index_daily_data is None:
        index_daily_data = index_reader.get_index_daily(index_prefix=True, pin_memory=True)

    for i, day in enumerate(future_days):
        # 分析K线，获取上涨下跌结果
        result = analysis_candle(daily_data,
                                 daily_basic_data,
                                 daily_extra_data,
                                 cyq_extra_data,
                                 index_daily_data,
                                 curr_data, future_day=day)
        results.append(result)

        if print_result:
            if i == 0:  # 初始化表头
                pt = PrettyTable()

                field_names = ['future_day',  # 未来多少天的数据
                               'total_num',  # 数据总量
                               'rise_num',  # 上涨数量
                               'rise_ratio',  # 上涨概率
                               'avg_pct_chg',  # 平均上涨幅度
                               ]

                # 个股暂不打印额外指标信息
                # extra_factors = list_remove(list(result.keys()), field_names)  # 除上述提到的外，其他的指标
                # field_names.extend(extra_factors)
                pt.field_names = field_names

            row_items = [day,
                         result['total_num'],
                         result['rise_num'],
                         "%.2f %%" % (result['rise_num'] / max(result['total_num'], 1) * 100),
                         "%.2f %%" % result['avg_pct_chg'],
                         ]

            pt.add_row(row_items, )
    if print_result:
        print(pt)

    if stat_factor_props:
        return {
            'results': results,
            'factor_props': factor_props,
        }
    else:
        return results


def plot_results(date_index, data, l_offset=10, r_offset=60, ma_list=(5, 10, 20, 60), volume=True):
    """
    FIXME 后来不知道改了什么，这里有问题了。后续再说吧
    """
    index = data.loc[date_index]['index']
    left = int(max(0, index - l_offset - 1))
    right = int(min(len(data), index + r_offset))

    data = data.iloc[left:right]

    plot = CandlePlot(data=data)
    if volume:
        plot = plot.volume()
    plot.add_marks(date_index, date_indices=True).add_MA(ma_list).plot()


def _factor_props_print(factor_props: dict, print_func=print):
    """
    打印_factor_props的结果
    """
    if is_null(factor_props):
        return

    row_dict = {
        '': 'filter ratio',
    }
    for factor, (total_num, after_num) in factor_props.items():
        filter_ratio = (1 - after_num / total_num) * 100
        row_dict[factor] = "%.4f%%" % round(filter_ratio, 2)

    table_print([row_dict], print_func=print_func)


def analysis_A_share(
        line_func,  # 分析函数。推荐使用继承AnalysisOne的方式
        line_func_kwargs=None,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分
        remove_continual=False,  # 是否去掉连续数据。仅用于AnalysisOne的方式
        limit=-1,  # 限制股票数量，用于debug
        _print=None,  # 日志输出函数
        require_data=None,  # 本次分析需要用到哪些数据。若不传，则根据AnalysisOne._require_data方法进行判断
):
    """
    分析整个A股的不同K线情况
    """
    if limit > 0:  # 如果是debug模式，则不生成日志
        fprint.disable()

    if _print is None:
        _print = fprint

    _print("line_func: ", line_func.__name__)
    if line_func_kwargs is not None:
        _print("line_func_kwargs", line_func_kwargs)
    stock_code_list = get_all_china_stock_code_list()

    # 初始化一个列表，负责记录每个future_day的情况
    counters = []
    year_dist_list = []  # 负责记录年份分布
    for day in future_days:
        counters.append({
            "future_day": day,
            "stock_num": 0,  # 符合条件的股票数量
        })

        year_dist_list.append({})

    factor_props = {}  # 记录每个指标使样本数减少的量。Dict[factor_name, [total_num, after_num]]
    special_stock_codes = set()
    for num, stock_code in tqdm(enumerate(stock_code_list), total=len(stock_code_list)):
        if line_func_kwargs is None:
            line_func_kwargs = {}

        if isinstance(line_func, types.FunctionType):
            # 最早的方式，后面会逐渐淘汰
            # 这个results是从 get_results(..) 函数返回的
            results = line_func(stock_code, future_days=future_days, **line_func_kwargs)
        elif issubclass(line_func, AnalysisOne):
            # 推荐使用该方式
            analysis_one = line_func(stock_code,
                                     future_days=future_days,
                                     # todo require_data
                                     # todo data_start_date
                                     # todo data_end_date
                                     print_result=False,
                                     require_data=require_data,
                                     remove_continual=remove_continual,
                                     )
            results = analysis_one(**line_func_kwargs)
        else:
            raise RuntimeError("line_func不合法！")

        one_factor_props: Dict = None
        stat_factor_props = False
        if type(results) == dict:
            results, one_factor_props = results['results'], results['factor_props']
            stat_factor_props = True

        if stat_factor_props and one_factor_props is not None:
            # 统计各个factor对数据量的影响
            for factor, (total_num, after_num) in one_factor_props.items():
                prop_item = factor_props.get(factor, [0, 0])
                prop_item[0] += total_num
                prop_item[1] += after_num
                factor_props[factor] = prop_item

        # 统计结果
        for i, result in enumerate(results):
            for key, value in result.items():
                if key in ('up_year_dist', 'down_year_dist'):
                    # 年份分布
                    prefix = key.split('_')[0] + "_"
                    for year in value:
                        year_dist_list[i][prefix + str(year)] = year_dist_list[i].get(prefix + str(year), 0) \
                                                                + value[year]
                    continue

                if type(value) == list:
                    counters[i][key] = counters[i].get(key, []) + value
                else:
                    counters[i][key] = counters[i].get(key, 0) + value

            if result['total_num'] >= 1:
                counters[i]['stock_num'] += 1

                special_stock_codes.add(stock_code)

        if limit >= 0 and num >= limit:
            break

    counters_bak = copy.deepcopy(counters)

    # 整理分析结果
    columns = list(counters[0].keys())
    for i, counter in enumerate(counters):
        if counter['stock_num'] <= 0:
            continue

        counter['rise_ratio'] = '%.2f%%' % round((counter['rise_num'] / (counter['total_num'] + 0.000001) * 100), 2)
        counter['rise_pct'] = round((counter['rise_pct'] / (counter['total_num'] + 0.000001)), 2)

        # 对指标求平均等操作
        for factor in columns:
            if factor.endswith('_list'):
                counter[factor[:-5]] = round(sum(counter[factor]) / (len(counter[factor]) + 0.000001), 2)
                del counter[factor]
                factor = factor[:-5]
            else:
                # 其他指标保持原状
                counter[factor] = counter[factor]

            # 为指标增加“%”，但排除以下字段
            if factor in ['future_day', 'total_num', 'rise_num', 'up_index_close', 'down_index_close', 'stock_num',
                          'up_pe_ttm', 'down_pe_ttm', 'up_ps_ttm', 'down_ps_ttm']:
                # 不用增加任何单位
                pass
            elif factor in ['up_circ_mv', 'down_circ_mv']:
                counter[factor] = str(round(counter[factor] / 100)) + "m"
            else:
                # 大部分都是百分比，所以输出后面增加个%
                counter[factor] = str(counter[factor]) + "%"

        counters[i] = dict_change_place(counter, key='rise_ratio', after_key='rise_num')

    # 打印结果
    table_print(counters, exclude_columns=('stock_num', 'avg_pct_chg'), print_func=_print)

    _factor_props_print(factor_props, print_func=_print)

    # 计算并打印每年的样本分布
    ratio_con_list, rise_std_list = print_year_dist(year_dist_list, future_days, _print)
    for i, counter in enumerate(counters):  # 将年份样本分布计算结果写入
        if ratio_con_list is None:
            continue
        counter['dist_con'] = "%.2f%%" % ratio_con_list[i]  # 样本集中度
        counter['rise_std'] = "%.2f%%" % rise_std_list[i]  # 胜率标准差

    _print()
    _print("markdown format:")

    # 打印md格式结果
    table_print(counters, format='md',
                columns=('future_day', 'total_num', 'rise_num', 'rise_ratio', 'rise_pct', 'up_rise_pct',
                         'down_rise_pct', 'dist_con', 'rise_std'),
                translate={
                    'future_day': 'n个交易日后',
                    'total_num': '样本总数',
                    'rise_num': '上涨样本数',
                    'rise_ratio': '上涨概率',
                    'rise_pct': '平均涨跌幅',
                    'up_rise_pct': '上涨时平均涨幅',
                    'down_rise_pct': '下跌时平均跌幅',
                    'dist_con': '样本集中度',
                    'rise_std': '胜率标准差',
                }, print_func=_print)

    # print(special_stock_codes)

    if not fprint.disabled:
        plot_analysis_sample_distribution(counters_bak, root_dir=log_dir, line_func_kwargs=line_func_kwargs, limit=limit)

    return counters, ratio_con_list, rise_std_list


def analysis_factor(
        line_func,  # 要分析函数
        factor_name: str,  # 要分析的指标的参数名称
        factor_value_list: list,  # 要分析的指标的哪些值
        line_func_kwargs=None,
        future_days=(5, 10, 20, 40, 60),  # 对大阳线后多少个交易日后的数据进行分
        remove_continual=False,  # 是否去掉连续数据。仅用于AnalysisOne的方式
        limit=-1,  # 限制股票数量，用于debug
):
    """
    这个方法就是循环调用 analysis_A_share函数，然后汇总它们的结果。

    举例：假设想测试不同换手率对光头大脚大阳线的胜率。那么就可以这么调用：
    analysis_factor(
        line_func=BareDaYangXian,  # 光头光脚大阳线的函数
        factor_name='turnover',  # turnover必须是BareDaYangXian的参数之一
        factor_value_list=[0.5, 1.0, 1.5, 2.0],  # 该list中的元素必须是turnover参数的合法参数值
    )
    """

    # 日志怎么说 todo


    results = []

    if line_func_kwargs is None:
        line_func_kwargs = {}

    for factor_value in factor_value_list:
        line_func_kwargs[factor_name] = factor_value
        # 分析整个A股
        result = analysis_A_share(line_func,
                                  line_func_kwargs,
                                  future_days,
                                  remove_continual=remove_continual,
                                  limit=limit)

        fprint()
        fprint("-" * 20)

        results.append(result)

    # 整理数据结构
    plot_analysis_factor(factor_name, factor_value_list, results, root_dir=log_dir)


def print_year_dist(year_dist_list, future_days, print_func=None):
    # 计算年份边界
    years = set()
    for year_dist in year_dist_list:
        _ = [years.add(key.split("_")[1]) for key in year_dist.keys()]

    years = list(years)
    years.sort()

    if len(years) <= 0:
        print_func("无数据，不打印年份分布！")
        return None, None

    start_year = years[0]
    end_year = years[-1]

    years = [str(year) for year in range(int(start_year), int(end_year) + 1)]

    pt = PrettyTable()
    field_names = ['future_day', ]
    for year in years:
        field_names += ['up_' + year, 'down_' + year, 'ratio_' + year, 'rise_' + year]

    pt.field_names = field_names

    ratio_con_list = []  # 样本数量分布标准差
    rise_std_list = []  # 样本胜率分布标准差

    for i, day in enumerate(future_days):
        year_dist = year_dist_list[i]
        total_num = sum(year_dist.values())

        row_items = [day, ]

        ratio_list = []
        rise_list = []
        for year in years:
            up = year_dist.get('up_' + year, 0)
            down = year_dist.get('down_' + year, 0)
            ratio = round((up + down) / (total_num + 0.0001) * 100, 2)
            rise = round(up / (up + down + 0.0001) * 100, 2)
            row_items += [up,
                          down,
                          "%.2f%%" % ratio,
                          "%.2f%%" % rise
                          ]

            ratio_list.append(ratio)
            rise_list.append(rise)

        ratio_con_list.append(round(max(ratio_list) - min(ratio_list), 2))
        rise_std_list.append(round(np.std(rise_list), 2))

        pt.add_row(row_items)

    print_func()
    print_func('年份分布如下：')
    print_func(pt)

    return ratio_con_list, rise_std_list


def analysis_index(ts_code: str,  # 指数的ts_code
                   start_date: str,  # 开始日期 %Y-%m-%d，以当天的收盘价为准
                   end_date: str,  # 结束日期 %Y-%m-%d，以当天的收盘价为准
                   ):
    """
    分析指数收益。

    :return: (
                start_price,  # 买入价
                end_price,  # 卖出价
                profit_rate,  # 收益率
             )
    """
    data = TuShareDataReader(ts_code, is_index=True).set_date_range(start_date, end_date).get_index_daily(
        index_prefix=False)

    start_price = data.iloc[0]['close']
    end_price = data.iloc[-1]['close']

    profit_rate = (end_price - start_price) / start_price * 100

    return start_price, end_price, profit_rate


class AnalysisOne:
    """
    对一只股票进行分析。

    后续analysis_A_share要改用这个
    """

    def __init__(self,
                 stock_code,
                 future_days=(5, 10, 20, 40, 60),
                 data_start_date='2010-01-01',
                 data_end_date='2022-12-31',
                 print_result=True,
                 require_data=None,  # 分析需要用到哪些数据，若不填，则根据_require_data函数决定。
                 remove_continual=False,
                 ):
        self.stock_code = stock_code
        self.future_days = future_days
        self.print_result = print_result
        self.remove_continual = remove_continual

        self.reader = TuShareDataReader(stock_code).set_date_range(data_start_date, data_end_date)
        self.index_reader = TuShareDataReader("上证指数", is_index=True).set_date_range(data_start_date, data_end_date)
        self.db = TuShareDataReader.db

        # {stock_code}的股票每日各指标情况
        self.daily_data = self.reader.get_daily()

        if require_data is None:
            require_data = self._require_data()  # 分析需要用到哪些额外数据, 'daily'是必须的，填不填都会有

        self.daily_basic_data = None
        if 'daily_basic' in require_data:
            self.daily_basic_data = self.reader.get_daily_basic()

        self.daily_extra_data = None
        if 'daily_extra' in require_data:
            self.daily_extra_data = self.reader.get_daily_extra()

        self.cyq_extra_data = None
        if 'cyq_extra' in require_data:
            self.cyq_extra_data = self.reader.get_cyq_extra()

        # 上证指数每日情况
        self.index_daily_data = None
        if 'index_daily' in require_data:
            self.index_daily_data = self.index_reader.get_index_daily(index_prefix=True, pin_memory=True)

    def filter_data(self, data: DataFrame, *args, **kwargs):
        """
        子类需要实现该方法。
        按照你的技术指标对data进行过滤，最后剩下的就是符合技术指标条件的数据。
        """
        return data

    def _require_data(self):
        """
        子类需要实现，告知分析过程中需要用到哪些数据。

        例如：你的买入策略中需要用到流通市值(circ_mv)，那么该函数的返回值中就要包含'daily_basic'（因为流通市值是在这个表中的）。
        """
        return ('daily',)

    def __call__(self, *args, **kwargs):
        concat_data = [self.daily_data]

        if self.daily_basic_data is not None:
            concat_data.append(self.daily_basic_data)
        if self.daily_extra_data is not None:
            concat_data.append(self.daily_extra_data)
        if self.cyq_extra_data is not None:
            concat_data.append(self.cyq_extra_data)
        if self.index_daily_data is not None:
            concat_data.append(self.index_daily_data)

        data = pd.concat(concat_data, axis=1, join='inner')
        data = drop_duplicated_columns(data)

        if len(self.daily_data) != len(data):
            raise RuntimeError("合并前后数据量不一致，请查看数据或代码是否有问题！")

        try:
            data = self.filter_data(data, *args, **kwargs)
        except KeyError as err:
            raise RuntimeError(f"KeyError({str(err)})！请确认Key是否填错，且子类实现了`_require_data`方法并填入了需要的数据！") from err

        results = get_results(self.stock_code, data, self.daily_data, self.future_days,
                              print_result=self.print_result,
                              daily_extra_data=self.daily_extra_data,
                              cyq_extra_data=self.cyq_extra_data,
                              index_daily_data=self.index_daily_data,
                              remove_continual=self.remove_continual
                              )

        return results

    def analysis(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)


class CommonAnalysisOne(AnalysisOne):
    """
    通用的AnalysisOne子类
    """

    def filter_data(self,
                    data: DataFrame,
                    factors_range: Dict[str, tuple] = None,
                    stat_factor_props=True,  # 是否统计factor_props，不统计的话会更快一点。（大概快几分钟）
                    *args, **kwargs):
        """
        factors_range: 描述你的指标。例如：{
            "pct_chg": (2, 6), # 即当天涨幅在2%~6%之间,
            "turnover_rate": (0, 10), # 换手率在0~10%之间
        }
        """

        # 统计每个factor会使data减少的量。这样就知道哪个指标设置的过于激进了。
        # key为factor_name, value为tuple, value[0]为总量，value[1]为调整后的数量
        factor_props = {}
        data_bak = data  # 用于统计factor_prop

        for factor, (min_factor, max_factor) in factors_range.items():
            if min_factor > max_factor:
                raise RuntimeError(f"{factor}指标的范围({min_factor}, {max_factor})异常")

            data = data[(min_factor <= data[factor]) & (data[factor] <= max_factor)]

            if stat_factor_props:
                total_num = len(data_bak)
                after_num = ((min_factor <= data_bak[factor]) & (data_bak[factor] <= max_factor)).sum()
                factor_props[factor] = (total_num, after_num)

            if is_null(data) and not stat_factor_props:
                break

        return {
            'data': data,
            'factor_props': factor_props,
        }

    def _require_data(self):
        return ('daily',)


def common_analysis(
        # 要分析的指标。key为指标名称，value为指标的范围，目前仅支持tuple，即一个最小值一个最大值。
        # 详见CommonAnalysisOne.filter_data说明
        factors_range: dict,
        remove_continual=False,
        limit=-1,
):
    """
    通用的分析函数
    """
    require_data = get_require_data_list(factors_range.keys())

    analysis_A_share(CommonAnalysisOne,
                     line_func_kwargs={'factors_range': factors_range},
                     remove_continual=remove_continual,
                     limit=limit,
                     require_data=require_data)


def analysis_same_condition(
        stock_code: str,
        trade_date: str = None,  # 若不填，则取最近一个交易日
        tolerant=0.2,  # 上下波动度
        limit=-1,
):
    """
    当通过find_stock筛选出一直股票后，可以使用该函数进行进一步验证。

    该函数会将该股票的各种情况找出来，然后分析历史上与该股票出现相似情况时，后续上涨的概率。
    """
    if trade_date is None:
        trade_date = get_today()

    print("trade_date:", trade_date)

    reader = TuShareDataReader(stock_code).set_date_range(start_date=date_add(trade_date, n_day=-30),
                                                          end_date=trade_date)

    # 先获取stock_code最后一个交易日的数据情况
    last_daily: Series = reader.get_daily().iloc[-1]
    last_daily_basic: Series = reader.get_daily_basic().iloc[-1]
    # last_daily_extra: Series = reader.get_daily_extra().iloc[-1]
    # last_cyq_extra: Series = reader.get_cyq_extra().iloc[-1]

    if not equals(last_daily.name, last_daily_basic.name):
        raise RuntimeError("存在trade_date不一致，请确认数据是否正确！")

    last_data = pd.concat([last_daily, last_daily_basic], axis=0)
    last_data = drop_duplicated_columns(last_data)

    # 需要考虑的指标
    factors = ('pct_chg', 'turnover_rate', 'circ_mv',)
    # 每个指标都会有一个范围(min, max)，波动范围为20%。
    # 例如，某股票当天涨幅5%，那与之相似范围为： 4%~6%。
    factors_range = {}
    for factor in factors:
        down = last_data[factor] - tolerant * last_data[factor]
        up = last_data[factor] + tolerant * last_data[factor]
        factors_range[factor] = (min(down, up), max(down, up))

    analysis_A_share(CommonAnalysisOne, line_func_kwargs={
        'factors_range': factors_range
    }, limit=limit)


if __name__ == '__main__':
    # backtest_demo()
    # check_log()
    # multi_backtest(
    #     years='2020-2022',
    #     init_money=100_000,
    #     buy_strategy=buy_strategy_template,
    #     sell_strategy=sell_strategy_template,
    #     require_data=('daily',),
    #     buy_timing='open',
    #     debug_limit=20,
    # )
    # temp = load_obj(ROOT / 'tools' / 'test.pkl')
    # print_year_dist(temp, (5, 10, 20, 40, 60))
    # analysis_same_condition('601166', trade_date='2023-12-29', limit=-1)

    print()
