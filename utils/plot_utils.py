import math
import sys
import os

from pathlib import Path

from pandas import DataFrame
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import time
import pandas as pd
import altair as alt
import altair_viewer
import vl_convert as vlc

from utils.utils import save_obj, load_obj, remove_pd_percent, write_txt
from utils.date_utils import get_now


def altair_save(chart, filepath):
    # Create PNG image data and then write to a file
    png_data = vlc.vegalite_to_png(chart.to_json(), scale=2)
    with open(filepath, "wb") as f:
        f.write(png_data)


def plot_analysis_factor(factor_name,  # 指标名称
                         factor_value_list,  # 指标的不同取值
                         results: list,  # 指标计算结果
                         root_dir: Path,  # 放在哪个目录下面
                         debug_file=None,
                         ):
    """
    绘制analysis中analysis_factor方法的结果
    """

    # 将结果存一下，防止报错没打印图像
    if debug_file is None:
        os.makedirs(root_dir, exist_ok=True)
        save_obj({
            'factor_name': factor_name,
            'factor_value_list': factor_value_list,
            'results': results,
        }, root_dir / f'{factor_name}_data.pkl')

        print("pkl结果文件：", f'{factor_name}_data.pkl')
    else:
        pkl_obj = load_obj(root_dir / debug_file)
        factor_name = pkl_obj['factor_name']
        factor_value_list = pkl_obj['factor_value_list']
        results = pkl_obj['results']

    width = len(factor_value_list) * 50

    # 将factor增加到results，并将results合并成一个list
    data = []
    dist_data = []  # 样本年份分布数据  # TODO
    for i, result_list in enumerate(results):
        factor_value = str(factor_value_list[i]).replace(' ', "")
        if type(factor_value) == tuple:
            factor_start, factor_end = factor_value
        else:
            factor_start, factor_end = factor_value, factor_value

        # data
        for result_one in result_list[0]:
            result_one['factor'] = factor_value
            result_one['factor_start'] = factor_start
            result_one['factor_end'] = factor_end

            data.append(result_one)

        # dist_data

    data = pd.DataFrame(data)
    data = remove_pd_percent(data)
    data['tmp_0'] = [0] * len(data)

    # year_data = pd.DataFrame(
    #     columns=('')
    # )

    # 绘制胜率折线图
    rise_ratio_line = alt.Chart(data).mark_line().encode(
        x=alt.X('factor:N', title=factor_name),
        y=alt.Y('rise_ratio:Q',
                scale=alt.Scale(domain=[data['rise_ratio'].min() - 5, data['rise_ratio'].max() + 5]),
                title='胜率(%)'),
        color=alt.Color('future_day:N', title='n个交易日后')  # N是nominal，可理解为字符串
    ).properties(
        title="各参数胜率",
        width=width,
    )
    rise_ratio_line = rise_ratio_line + rise_ratio_line.mark_point(color="#333")

    # 绘制样本数柱状图
    sample_num_data = data[['total_num', 'rise_num', 'factor']].drop_duplicates()
    sample_num_bar = alt.Chart(sample_num_data).mark_bar(color='green').encode(
        x=alt.X('factor:N', title=factor_name),
        y=alt.Y('total_num:Q', title="上涨样本数/总样本数"),
        text='total_num'
    )
    sample_num_bar += sample_num_bar.mark_text(align='center', dy=-5, color='green')
    up_sample_num_bar = alt.Chart(sample_num_data).mark_bar(color='red').encode(
        x=alt.X('factor:N', title=factor_name),
        y=alt.Y('mean(rise_num):Q', title="上涨样本数/总样本数"),
    )
    sample_num_bar = (sample_num_bar + up_sample_num_bar).properties(
        title="各参数样本数",
        width=width,
    )

    # 绘制平均涨跌幅
    rise_pct_line = alt.Chart(data).mark_line().encode(
        x=alt.X('factor:N', title=factor_name),
        y=alt.Y('rise_pct:Q',
                title='平均涨跌幅(%)'),
        color=alt.Color('future_day:N', title='n个交易日后')
    )
    rise_pct_line += rise_pct_line.mark_point(color="#333")
    rise_pct_line = rise_pct_line.properties(
        title="平均涨跌幅(%)（收益率数学期望）",
        width=width
    )

    # 绘制上涨时平均涨幅与下跌时平均跌幅
    up_rise_pct_line = alt.Chart(data).mark_line().encode(
        x=alt.X('factor:N', title=factor_name),
        y=alt.Y('up_rise_pct:Q',
                title='平均涨幅(%)'),
        color=alt.Color('future_day:N', title='n个交易日后')
    )
    up_rise_pct_line += up_rise_pct_line.mark_point(color="#333")
    down_rise_pct_line = alt.Chart(data).mark_line().encode(
        x=alt.X('factor:N', title=factor_name),
        y=alt.Y('down_rise_pct:Q',
                title='平均涨幅(%)'),
        color=alt.Color('future_day:N', title='n个交易日后')
    )
    down_rise_pct_line += down_rise_pct_line.mark_point(color="#333")

    up_down_rise_pct_line = up_rise_pct_line + down_rise_pct_line
    # 在0处增加一条线
    up_down_rise_pct_line += alt.Chart().mark_rule(color='black').encode(
        y='tmp_0:Q',
        size=alt.SizeValue(3)
    )

    up_down_rise_pct_line = up_down_rise_pct_line.properties(
        title="上涨与下跌时的平均涨跌幅",
        width=width
    )

    # 绘制样本分布图
    # year_dist_chart = alt

    chart_dir = root_dir / 'chart'
    os.makedirs(chart_dir, exist_ok=True)
    print('chart目录:', chart_dir)

    altair_save(rise_ratio_line, chart_dir / f"rise_ratio.png")
    altair_save(sample_num_bar, chart_dir / f"sample_num_bar.png")
    altair_save(rise_pct_line, chart_dir / f"rise_pct.png")
    altair_save(up_down_rise_pct_line, chart_dir / f"up_down_rise_pct.png")

    print(altair_viewer.display((rise_ratio_line & sample_num_bar) | (rise_pct_line & up_down_rise_pct_line)))

    time.sleep(10)


def plot_analysis_sample_distribution(counters, root_dir: Path, *args, **kwargs):
    """
    打印分析出来的样本的分布情况。用来分析那些涨了和跌了的股票的各个指标都分布在哪些位置。
    """
    alt.data_transformers.disable_max_rows()

    save_obj(counters, root_dir / 'counters.pkl')
    # counters = load_obj(ROOT / 'tmp' / 'counters.pkl')

    chart_dir = root_dir / f'chart'
    os.makedirs(chart_dir, exist_ok=True)
    write_txt(str(kwargs), chart_dir / 'args.txt')
    print('sample_distribution chart目录:', chart_dir)

    for counter in tqdm(counters, desc='Gene Chart'):
        # 将counter结果转为DataFrame
        data = []
        for key in counter.keys():
            if not key.endswith("_list"):
                continue

            if key.startswith("up_"):
                factor = key[3:-5]
                up_down = 'up'
            elif key.startswith("down_"):
                factor = key[5:-5]
                up_down = 'down'
            else:
                continue

            factor_value_list = counter[key]
            factor_value_list.sort()
            factor_value_list = factor_value_list[int(len(factor_value_list) * 0.05):int(len(factor_value_list) * 0.95)]

            for factor_value in factor_value_list:
                data.append([factor, factor_value, up_down])

        data = DataFrame(data=data, columns=['factor', 'factor_value', 'up_down'])

        factors = list(data['factor'].drop_duplicates())

        chart = None
        for factor in factors:
            data_fragment = data[data['factor'] == factor]

            while len(data_fragment) > 20000:
                # 如果data_fragment太大，那么就去掉一些样本
                data_fragment = data_fragment.sort_values('factor_value')
                data_fragment = data_fragment.iloc[range(0, len(data_fragment), 2)]
                data_fragment = data_fragment.reset_index()
                if 'level_0' in data_fragment.columns:
                    del data_fragment['level_0']

            sub_chart = alt.Chart(data_fragment).mark_tick().encode(
                x=alt.X('factor_value:Q', title='value',
                        scale=alt.Scale(domain=[data_fragment['factor_value'].min(),
                                                data_fragment['factor_value'].max()])),
                y=alt.Y('up_down:N', title=factor),
            )

            if chart is None:
                chart = sub_chart
            else:
                chart = chart & sub_chart

        chart = chart.configure_mark(
            opacity=1 / (math.log(500, 2) + 1),
            color='green'
        )

        altair_save(chart, chart_dir / f"sample_factor_distribution_{counter['future_day']}.png")

    time.sleep(10)


if __name__ == '__main__':
    # plot_analysis_factor(None, None, None, 'turnover_rate_data_20231229_154323.pkl')
    plot_analysis_sample_distribution(None)
