import calendar
import datetime
import time
from typing import List


def get_today():
    today = datetime.date.today()

    return str(today)


def get_yesterday():
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    return str(yesterday)


def get_date_list(start_date: str, end_date: str, date_format="%Y-%m-%d", tgt_format="%Y-%m-%d") -> list:
    """
    获取一段时间内的所有日期。

    例如：start_date="2023-01-01", end_date="2023-10-01"
    则，return=["2023-01-01", "2023-01-02", ..., "2023-09-30", "2023-10-01"]
    """

    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)

    date_list = []
    current_date = start_date

    while current_date <= end_date:
        date_list.append(current_date.strftime(date_format))
        current_date += datetime.timedelta(days=1)

    return date_list


def get_future_date_list(start_date: str, n_day: int, date_format="%Y-%m-%d") -> List[str]:
    """
    获取未来{n_day}天的日期。

    例如：start_date='2023-02-27', n_day=4
    则，return=['2023-02-27', '2023-02-28', '2023-03-01', '2023-03-02']
    """
    start_date = datetime.datetime.strptime(start_date, date_format)

    date_list = []
    current_date = start_date

    for _ in range(n_day):
        date_list.append(current_date.strftime(date_format))
        current_date += datetime.timedelta(days=1)

    return date_list


def get_month_list(start_date: str, end_date: str, date_format="%Y-%m-%d", tgt_format="%Y-%m"):
    """
    获取一段时间内的所有月份。

    例如：start_date="2023-01-01", end_date="2023-10-01"
    则，return=["2023-01", "2023-02", ..., "2023-09", "2023-10"]
    """

    start_date = datetime.datetime.strptime(start_date, date_format)
    end_date = datetime.datetime.strptime(end_date, date_format)
    start_date = start_date.replace(day=1)
    end_date = end_date.replace(day=1)

    result = []

    # 循环生成月份列表
    current_datetime = start_date
    while current_datetime <= end_date:
        result.append(current_datetime.strftime(tgt_format))
        # 增加一个月
        if current_datetime.month == 12:
            current_datetime = current_datetime.replace(year=current_datetime.year + 1, month=1)
        else:
            current_datetime = current_datetime.replace(month=current_datetime.month + 1)

    return result


def compare_to(date1: str, date2: str, format='%Y-%m-%d') -> int:
    """
    比较两个日期的大小，若date1>date2，返回1，否则返回-1，若相等，返回0
    """
    timestamp1 = int(time.mktime(time.strptime(date1, format)))
    timestamp2 = int(time.mktime(time.strptime(date2, format)))

    if timestamp1 > timestamp2:
        return 1
    elif timestamp1 < timestamp2:
        return -1
    else:
        return 0


def convert_format(date: str, src_format: str, tgt_format: str):
    """
    日期格式转换，例如：20231101 到 2023-11-01
    """
    src_date = datetime.datetime.strptime(date, src_format)

    return src_date.strftime(tgt_format)


def split_month(month: str):
    """
    将一个月分成两份。

    例如：month='2023-10'
    则返回值为：[
        ('2023-10-01', '2023-10-15'),
        ('2023-10-16', '2023-10-31'),
    ]
    """

    results = []

    results.append((month + "-01", month + "-15"))

    date = datetime.datetime.strptime(month, '%Y-%m')
    last_day = calendar.monthrange(date.year, date.month)[1]

    results.append((month + "-16", month + "-" + str(last_day)))

    return results


def date_add(date: str, n_day: int):
    """
    求未来n天的日期
    例如，date='2023-02-28'，n_day=3，则return '2023-03-03'
    """
    date = datetime.datetime.strptime(date, '%Y-%m-%d')

    date += datetime.timedelta(days=n_day)

    return date.strftime('%Y-%m-%d')


def next_day(date: str):
    """
    求下一天。
    例如，date='2023-02-28'，则return '2023-03-01'
    """
    return date_add(date, 1)


def get_now(format='%Y-%m-%d %H:%M:%S') -> str:
    now = datetime.datetime.now()
    return now.strftime(format)


def to_date(date: str, format='%Y-%m-%d') -> datetime.date:
    return datetime.datetime.strptime(date, format).date()


def date_diff(date2: str, date1: str, format='%Y-%m-%d') -> int:
    """
    计算两个日期的天数差, date2-date1
    """
    diff = datetime.datetime.strptime(date2, format) - datetime.datetime.strptime(date1, format)
    return diff.days


if __name__ == '__main__':
    # print(get_yesterday())
    # print(get_date_list("2023-01-01", "2023-10-01"))
    # print(get_today())
    # print(compare_to('2023-01-01', '2023-01-01'))
    # print(convert_format('20231101', '%Y%m%d', '%Y-%m-%d'))
    # print(get_month_list('2022-09-23', '2023-04-02'))
    # print(split_month("2023-10"))
    # print(next_day('2020-02-28'))
    # print(get_now())

    # print(get_now(format='%Y%m%d_%H%M%S'))
    # print(get_future_date_list('2023-02-27', n_day=4))
    # print(date_add('2020-02-28', 3))
    # print(to_date('2020-02-28'))
    print(date_diff('2023-01-02', '2022-12-29'))
