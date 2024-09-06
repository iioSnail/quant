import sys
from pathlib import Path

import pandas as pd
from pandas import DataFrame

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

pin_memory_cache = dict()
use_pin_memory = False  # 是否使用pin_memory，全局配置


def set_trade_date_as_index(data: DataFrame, format='%Y%m%d'):
    if data is None:
        return data

    if data.index.name == 'trade_date':
        return data

    data['trade_date'] = pd.to_datetime(data['trade_date'], format=format).dt.strftime('%Y-%m-%d')
    data = data.sort_values(by='trade_date', ascending=True)

    # 把trade_date放到最前面
    columns = list(data.columns)
    columns.remove('trade_date')
    columns.insert(0, 'trade_date')
    data = data[columns]

    data = data.set_index('trade_date')

    return data


def drop_duplicates_by_index(data: DataFrame, resp_data: DataFrame):
    """
    去除resp_data的重复数据。重复数据是指：已经在data中存在的数据。

    该函数的应用：在从tushare中获取数据后，有部分数据在库里（data里）已经有了，因此不需要再插入了。
    """
    if data is None:
        return resp_data

    return resp_data[~resp_data.index.isin(data.index)]


def convert_alias_to_ts_code(ts_code: str):
    """
    将常用中文名转换成对应的ts_code
    """
    # 常用指数转换
    if ts_code in ['上证指数', '上证']:
        ts_code = '000001.SH'
    elif ts_code in ['沪深300']:
        ts_code = '000300.SH'

    return ts_code

def fill_zero_to_empty_str(data: DataFrame, dtype: dict, fill_value=0.0):
    """
    If the real column has a empty string, the method will fill it with zero.
    """
    for key in dtype.keys():
        if key not in data.columns:
            continue

        if dtype[key] not in (float, int):
            continue

        data[key] = data[key].apply(pd.to_numeric, errors='coerce').fillna(fill_value)

    return data


def pin_memory(is_obj=False, obj_fields=()):
    """
    装饰器，将func的请求参数和返回结果进行缓存。这样下一次再调用的时候就可以直接从内存中读取了

    :param is_obj: 是否是类对象
    :param obj_fields: 哪些类对象属性要作为key的一部分
    """

    def wrapper_out(func):

        def wrapper(*args, **kwargs):
            func_name = func.__name__

            use_cache = use_pin_memory  # 是否使用缓存

            # 如果参数中明确包含了'pin_memory'参数，则根据参数决定是否使用缓存
            if 'pin_memory' in kwargs:
                use_cache = kwargs.get('pin_memory')

            obj_args_str = ""
            args_str = ""
            if is_obj:  # 类上的方法
                self = args[0]
                for field in obj_fields:
                    if not hasattr(self, field):
                        raise RuntimeError(f"{func_name}类对象不包含{field}属性")

                    field_value = getattr(self, field)
                    obj_args_str += f"_{field_value}"

                args_str += str(args[1:])
            else:
                args_str += str(args)

            key = func_name + obj_args_str + args_str + '_' + str(kwargs)

            if key in pin_memory_cache and use_cache:  # 如果在缓存中，直接返回
                return pin_memory_cache[key]

            result = func(*args, **kwargs)
            pin_memory_cache[key] = result
            return result

        return wrapper

    return wrapper_out


def remove_pin_memory():
    """
    释放pin_memory中的数据
    """
    global pin_memory_cache

    pin_memory_cache = dict()
