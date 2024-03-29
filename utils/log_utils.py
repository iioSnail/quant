import os
import sys
from pathlib import Path
from typing import List, Dict

import prettytable
from prettytable import PrettyTable

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

level = 'verbose'


def set_log_level(log_level):
    global level
    level = log_level


def print_verbose(*args, **kwargs):
    if level == 'verbose':
        print(*args, **kwargs)


class FilePrint(object):

    def __init__(self, filename, console=False, log_dir=None):
        self.filename = filename
        self.console = console
        self.log_dir = log_dir

        if self.log_dir is None:
            self.log_dir = ROOT / 'log'

        self.file = None
        self.disabled = False  # 不打印到文件

    def disable(self):
        self.disabled = True

    def reset_filename(self, filename):
        self.filename = filename
        self.__del__()
        self.file = None

    def __call__(self, *args, **kwargs):
        console = kwargs.get('console', None)
        if console is None:
            console = self.console
        else:
            del kwargs['console']

        if console:
            print(*args, **kwargs)

        if self.disabled:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        if self.file is None:
            self.file = open(self.log_dir / f'{self.filename}.log', mode='a', encoding='utf-8')

        print(*args, **kwargs, file=self.file, flush=True)


    def __del__(self):
        if self.file is not None:
            self.file.close()


def get_fprint(filename, console=False, log_dir=None):
    """
    获取一个打印到日志文件的print函数。返回的函数功能与print一样，但不输出到控制台，而是文件
    :param filename: 文件名。文件会被放在 ~/log/{filename}.log下
    :param console: 是否在控制台打印
    :return: print_func, 一个类似print的函数
    """
    if log_dir is None:
        log_dir = ROOT / 'log'

    f_list = []

    def print_func(*args, **kwargs):
        if len(f_list) <= 0:  # 第一写入时再生成文件
            os.makedirs(log_dir, exist_ok=True)
            f_list.append(open(log_dir / f'{filename}.log', mode='a', encoding='utf-8'))

        print(*args, **kwargs, file=f_list[0], flush=True)
        if console:
            print(*args, **kwargs)

    return print_func


def table_print(rows: List[Dict],
                format='normal',
                columns=None,  # 仅打印的列
                exclude_columns=None,  # 不打印的列
                translate: Dict[str, str] = None,
                print_func=print,  # 打印函数，默认用原生print
                ):
    """
    将rows的内容以表格形式进行输出
    rows示例：
    [{'初始资金': '10.00w',
      '结束资金': '9.40w',
      '收益率': '-5.99%',
      '同期沪深300': '-21.27%',
      '出手次数': 3,
      '平均每日闲置资金': 94025.99,
      '平均持仓时间': '30天'},
     {'初始资金': '10.00w',
      '结束资金': '10.74w',
      '收益率': '7.41%',
      '同期沪深300': '-6.21%',
      '出手次数': 13,
      '平均每日闲置资金': 86746.98,
      '平均持仓时间': '29天'}]

    translate: 转换列名。例如：{"future_day": "n个交易日后"}，即将“future_day”列名转为"n个交易日后"
    """
    if len(rows) <= 0:
        print_func("未找到表格数据！")
        return

    pt = PrettyTable()

    if columns is None:
        columns = list(rows[0].keys())

    columns = list(columns)

    if exclude_columns is not None:
        for col in exclude_columns:
            if col not in columns:
                continue

            columns.remove(col)

    field_names = columns.copy()
    for i in range(len(field_names)):
        if translate is None:
            continue
        field_names[i] = translate.get(field_names[i], field_names[i])
    pt.field_names = field_names

    for row in rows:
        pt_row = []
        for key in columns:
            pt_row.append(row.get(key, ''))

        pt.add_row(pt_row)

    if format in ['md', 'markdown']:
        pt.set_style(prettytable.MARKDOWN)

    print_func(pt)


if __name__ == '__main__':
    # fprint = get_fprint('test')
    # fprint("hello", "world")
    # fprint("hello", "world", 2)
    table_print([{'初始资金': '10.00w',
                  '结束资金': '9.40w',
                  '收益率': '-5.99%',
                  '同期沪深300': '-21.27%',
                  '出手次数': 3,
                  '平均每日闲置资金': 94025.99,
                  '平均持仓时间': '30天'},
                 {'初始资金': '10.00w',
                  '结束资金': '10.74w',
                  # '收益率': '7.41%',
                  '同期沪深300': '-6.21%',
                  '出手次数': 13,
                  '平均每日闲置资金': 86746.98,
                  '平均持仓时间': '29天'}], format='md',
                # columns=['初始资金', '出手次数'],
                exclude_columns=['结束资金'],
                translate={"初始资金": "init_money"}
                )
