import pickle
import time

import pandas as pd
from pandas import DataFrame


def save_obj(obj, filepath):
    with open(filepath, "bw") as f:
        pickle.dump(obj, f)


def load_obj(filepath):
    with open(filepath, "br") as f:
        return pickle.load(f)


def write_txt(txt, filepath):
    with open(filepath, "w") as f:
        f.write(txt)


def time_func(func, times=1, *args, **kwargs):
    """
    计算一个函数的耗时（精确到毫秒）
    """
    start_time = int(time.time() * 1000)
    for _ in range(times):
        func(*args, **kwargs)  # todo 暂不支持传参

    end_time = int(time.time() * 1000)

    def print_result(total_time):
        minutes = total_time // 60000
        seconds = total_time // 1000 % 60
        ms = total_time % 1000
        if minutes > 0:
            print(f"{minutes}分", end='')

        if seconds > 0:
            print(f"{seconds}秒", end='')

        if ms > 0:
            print(f"{ms}毫秒", end='')

        print()

    print('total time: ', end='')
    print_result(end_time - start_time)
    print('avg time: ', end='')
    print_result(int((end_time - start_time) / times))


def list_remove(src_list: list, remove_list: list):
    for item in remove_list:
        if item not in src_list:
            continue

        src_list.remove(item)

    return src_list


def dict_change_place(old_dict: dict, key: str, after_key: str):
    """
    修改dict中key的位置，将`key`放到`after_key`后面
    """
    if key not in old_dict:
        raise RuntimeError("Key“%s”不在dict中" % key)

    if after_key not in old_dict:
        raise RuntimeError("Key“%s”不在dict中" % after_key)

    new_dict = {}
    for k in old_dict.keys():
        if k == after_key:
            new_dict[k] = old_dict[k]
            new_dict[key] = old_dict[key]
            continue

        if k == key:
            continue

        new_dict[k] = old_dict[k]

    return new_dict


def str_endwiths(string: str, suffix_list):
    """
    如果string以suffix_list中的任意一个结尾，则为True
    :param string:
    :param suffix_list:
    :return:
    """

    for suffix in suffix_list:
        if string.endswith(suffix):
            return True

    return False


def remove_pd_percent(data: DataFrame):
    """
    将data中以%结尾的数据都转为float
    """
    for col in data.columns:
        try:
            if data[col].str.endswith('%')[0] == True:
                data[col] = data[col].str.replace("%", "")
                data[col] = data[col].astype('float')
        except:
            pass

    return data


def is_null(obj):
    """
    判断一个对象是否为空，不同的对象有不同的判定方法
    """
    if obj is None:
        return True

    if hasattr(obj, '__len__'):
        return len(obj) <= 0

    if pd.isnull(obj):
        return True

    raise RuntimeError("不支持的obj类型：" + str(type(obj)))


def equals(*args):
    """
    判断args中的结果是否全都相等
    """
    if is_null(args):
        return True

    reference = args[0]
    for arg in args[1:]:
        if reference != arg:
            return False

    return True


if __name__ == '__main__':
    print(equals(1, 1, 1))
