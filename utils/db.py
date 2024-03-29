"""
用sqlite去存拉下来的tushare数据
"""
import copy
import datetime
import sys
from pathlib import Path
from typing import List, Dict

from pandas import DataFrame

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.insert(0, str(ROOT))

import sqlite3
import pandas as pd
from utils.utils import is_null


class SqliteDB(object):
    db_dir = ROOT / 'data'

    # db_dir = Path(ROOT / 'data' / 'test')
    # db_dir = Path(r"E:/data")

    def __init__(self, db_name='tushare.sqlite'):
        self.con = sqlite3.connect(str(SqliteDB.db_dir / db_name))

        self.exist_tables = set()  # 记录哪些表已经存在
        self.table_columns: Dict[str, List[str]] = dict()
        self.table_structures: Dict[str, DataFrame] = dict()  # 记录表结构
        self.index_structures: Dict[str, DataFrame] = dict()  # 记录表的索引结构

    def _table_exist(self, table_name):
        if table_name in self.exist_tables:
            return True

        cursor = self.con.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        result = cursor.fetchone()
        cursor.close()
        if result:
            self.exist_tables.add(table_name)
            return True
        else:
            return False

    def _get_table_structure(self, table_name: str) -> DataFrame:
        """
        获取表结构
        """

        structure_columns = ['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk']

        if not self._table_exist(table_name):
            return DataFrame(data=[], columns=structure_columns, dtype=str)

        if table_name in self.table_structures.keys():  # 表结构缓存
            return self.table_structures[table_name]

        table_columns = []

        cursor = self.con.cursor()
        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        result = cursor.fetchall()
        for row in result:
            table_columns.append(list(row))
        cursor.close()

        table_structure = DataFrame(data=table_columns, columns=structure_columns, dtype=str)

        self.table_structures[table_name] = table_structure

        return table_structure

    def _get_column_name_by_index_name(self, index_name: str) -> str:
        """
        根据index_name获取对应的列名。（sqlite3上，不同的表索引名也不能重复，因此只用index_name即可）
        """
        cursor = self.con.cursor()
        cursor.execute(f"PRAGMA index_info(`{index_name}`)")
        result = cursor.fetchall()
        if len(result) != 1:
            raise RuntimeError(f"未找到{index_name}索引")

        return result[0][2]

    def _get_index_structure(self, table_name: str) -> DataFrame:
        """
        获取表的索引结构
        """

        structure_columns = ['seq', 'name', 'unique', 'origin', 'partial', 'column_name']

        if not self._table_exist(table_name):
            return DataFrame(data=[], columns=structure_columns, dtype=str)

        if table_name in self.index_structures.keys():  # 表结构缓存
            return self.index_structures[table_name]

        index_columns = []

        cursor = self.con.cursor()
        cursor.execute(f"PRAGMA index_list(`{table_name}`)")
        result = cursor.fetchall()
        for row in result:
            row = list(row) + [self._get_column_name_by_index_name(row[1])]  # index中不包含column_name，还得再查一遍
            index_columns.append(row)
        cursor.close()

        index_structure = DataFrame(data=index_columns, columns=structure_columns, dtype=str)

        self.index_structures[table_name] = index_structure

        return index_structure

    def _get_table_columns(self, table_name: str) -> List[str]:
        """
        根据表名获取表中的列名
        """
        if not self._table_exist(table_name):
            return []

        if table_name in self.table_columns.keys():
            return self.table_columns[table_name]

        table_structure = self._get_table_structure(table_name)
        table_columns = list(table_structure['name'])

        self.table_columns[table_name] = table_columns

        return table_columns

    def _clean_dtype(self, table_name: str, dtype: dict):
        """
        pandas读取数据时，dtype里不能有不存在的列，所以这里处理一下
        """
        if dtype is None:
            return None

        columns = self._get_table_columns(table_name)
        new_dtype = {}

        for k, v in dtype.items():
            if k in columns:
                new_dtype[k] = v

        return new_dtype

    def execute(self, sql):
        """
        执行一个非查询SQL
        """
        try:
            cursor = self.con.cursor()
            cursor.execute(sql)
            self.con.commit()
        except Exception as e:
            print("执行SQL失败，SQL:", sql)
            raise e

    def _unique_index_check(self, data: DataFrame, table_name: str, index_label=None):
        """
        检测表是否增添了唯一索引。
        因为使用sqlite3+pandas自动创建表时并不会增添主键，只会增添一个普通索引，
        因此需要在第二次插入的时候，将普通索引改成唯一索引。
        """
        if not self._table_exist(table_name):
            return

        index_structure = self._get_index_structure(table_name)
        key_column = data.index.name
        if index_label is not None:
            key_column = index_label

        index_data = index_structure[index_structure['column_name'] == key_column]

        if len(index_data) == 1 and index_data['unique'].iloc[0] == '1':  # 已经是唯一索引了
            return

        if sum(index_data['unique'] == '1') >= 1:  # 虽然有多个索引或约束，但是已经有唯一索引了
            return

        if len(index_data) > 1:
            raise RuntimeError(f"{table_name}的{key_column}列索引怎么会大于1呢？请检查表结构，是否增加了主键或约束")

        index_name = f'ix_{table_name}_{key_column}'
        if len(index_data) == 1:  # 删除原来的索引
            index_name = index_data.iloc[0]['name']
            self.execute(f"DROP INDEX `{index_name}`;")

        # 为该列增添新的唯一索引
        self.execute(f"CREATE UNIQUE INDEX `{index_name}` ON `{table_name}` (`{key_column}` ASC);")

    def read_data(self, table_name, start_date: str = None, end_date: str = None, dtype=None,
                  index_col='trade_date') -> DataFrame:
        """
        读取那些以trade_date为主键(index)的数据
        """
        if not self._table_exist(table_name):
            return None

        dtype = self._clean_dtype(table_name, dtype)

        sql = f"select * from `{table_name}` where 1=1"
        if start_date is not None and start_date != "":
            sql += f" and trade_date>='{start_date}'"

        if end_date is not None and end_date != "":
            sql += f" and trade_date<='{end_date}'"

        return pd.read_sql(sql, self.con, index_col=index_col, coerce_float=False, dtype=dtype)

    def select(self,
               table_name: str,
               sql: str = None,  # 要执行sql语句。
               conditions: List[str] = None,  # sql语句的条件
               index_col: str = None,  # 索引列
               dtype=None,  # 类型
               ):
        """
        通用查询接口
        """
        if not self._table_exist(table_name):
            return None

        dtype = self._clean_dtype(table_name, dtype)

        if sql is None:
            sql = f"select * from `{table_name}` where 1=1"

        if conditions is not None:
            for con in conditions:
                sql += f' and {con}'

        return pd.read_sql(sql, self.con, index_col=index_col, coerce_float=False, dtype=dtype)

    def select_by_sql(self, sql, index_col=None, dtype=None):
        return pd.read_sql(sql, self.con, index_col=index_col, coerce_float=False, dtype=dtype)

    def select_union(self,
                     table_name,  # 例如：`daily`, `daily_extra`等
                     sql_template,  # 例如: `select * from {table_name} where 1=1`
                     add_stock_code=False,  # 是否在最终结果中增加stock_code
                     ):
        table_names = self.select_by_sql(
            f"select name from sqlite_master where type='table' and name like '{table_name}_%';")

        # 生成逐条的sql
        sqls = []
        for item_name in table_names['name']:
            if '_' in item_name[len(table_name) + 1:]:
                continue

            stock_code = item_name.split('_')[-1]

            sql = sql_template.format(table_name=item_name)

            # 在select列表中增加stock_code
            if add_stock_code:
                if not sql.startswith("select"):
                    raise RuntimeError("要增添stock_code, sql必须以select开头。sql: %s" % sql)


                sql = f"select '{stock_code}' as stock_code," + sql[6:]

            sqls.append(sql)

        # 将其union起来
        data_list = []
        # sqlite3一次只能union500个
        for i in range(0, len(sqls), 450):
            sql = '\nunion\n'.join(sqls[i:i + 450])

            data_list.append(self.select_by_sql(sql))

        if is_null(data_list):
            return DataFrame()

        return pd.concat(data_list)

    def to_sql(self, data: DataFrame, table_name: str, dtype=None, index_label=None, if_exists=None):
        if is_null(data):
            return

        if dtype is None:
            dtype = {}

        dtype = copy.deepcopy(dtype)

        for k, v in dtype.items():
            if v in [datetime.date, str]:
                dtype[k] = 'text'

            if v in [float, int]:
                dtype[k] = 'real'

            if v in [bool]:
                dtype[k] = 'integer'

        self._unique_index_check(data, table_name, index_label)

        if not self._table_exist(table_name):
            # 表不存在，一次性插入。加快速度
            data.to_sql(table_name, self.con, if_exists='fail', dtype=dtype, index_label=index_label)
            return

        if if_exists == 'replace':
            # 有些情况下确实需要全部替换
            data.to_sql(table_name, self.con, if_exists='replace', dtype=dtype, index_label=index_label)
            return

        try:
            # 先尝试一次性插入，若存在冲突，则一条一条插入
            data.to_sql(table_name, self.con, if_exists='append', dtype=dtype, index_label=index_label)
            return
        except sqlite3.IntegrityError as e:
            # 主键冲突忽略  # 一条一条插入
            print(f'[WARN]{table_name}存在主键重复，改用逐条插入！')
            pass

        # 如果表存在，一条一条插入
        for i in range(len(data)):
            try:
                # 一条一条插入，若有主键冲突，则忽略
                data.iloc[i:i + 1].to_sql(table_name, self.con, if_exists='append', dtype=dtype,
                                          index_label=index_label)
            except sqlite3.IntegrityError as e:
                # 主键冲突忽略
                # print(f'[WARN]{table_name}的index数据{data.iloc[i:i + 1].index}存在主键重复！')
                pass

    def del_table(self, tabel_name: str):
        self.execute(f"DROP TABLE IF EXISTS `{tabel_name}`;")

        if tabel_name in self.exist_tables:
            self.exist_tables.remove(tabel_name)

    def get_last_data(self,
                      table_name: str,
                      order_by='trade_date',  # 按哪个字段进行降序
                      dtype=None,
                      ):
        """
        获取一张表的最后一条数据，
        """
        sql = f"select * from `{table_name}` order by `{order_by}` desc limit 1;"
        return self.select(table_name, sql=sql, dtype=dtype)

    def __del__(self):
        self.con.close()


if __name__ == '__main__':
    db = SqliteDB()
    # data = pd.read_csv(SqliteDB.db_dir / 'tushare' / 'index_daily' / '000001.SH.csv')
    # data.to_sql("test", db.con, if_exists="replace")

    # data = db.read_data('trade_cal2', start_date='2023-01-01', end_date='2023-01-08')
    # data = db._get_table_columns('trade_cal')

    # data = db._get_table_columns(table_name='daily_000008')
    # data = db._get_column_name_by_index_name('ix_daily_000002_trade_date')
    # data = db._get_index_structure(table_name='daily_000014')

    # db._unique_index_check(data=DataFrame([], columns=['trade_date']), table_name='daily_000011', index_label='trade_date')

    data = db.select_union('daily',
                           "select * from {table_name} where trade_date>='2023-01-01' and trade_date<='2024-01-01'")

    print()
