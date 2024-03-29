import datetime

import streamlit as st
import altair as alt

import utils.data_reader
from ui import plot
from ui.bean import UiDto
from ui.data import get_stock_list
from utils import data_process
from utils.date_utils import date_add, to_date

utils.data_reader.use_pin_memory = True

st.set_page_config(layout="wide")

dto = UiDto()

# 选择股票
dto.stock_code = st.sidebar.selectbox("股票代码", dto.stock_list)

# 选择时间范围, 选择展示trade_date左右多少天的数据
offset_L_st, trade_date_st, offset_R_st = st.sidebar.columns([1, 2, 1], gap='small')
dto.trade_date = trade_date_st.date_input("交易日期",
                                          value=to_date(dto.trade_date),
                                          format='YYYY-MM-DD'
                                          )
dto.offset_L = offset_L_st.number_input(label="<-", value=60, key="offset_L_st")
dto.offset_R = offset_R_st.number_input(label="->", value=60, key="offset_R_st")

candle_st, cyq_st = st.columns([4, 1])

candle_chart, x_scale, y_scale = plot.candle(dto)

# K线
candle_st.altair_chart(candle_chart, theme="streamlit", use_container_width=True)

# 筹码分布
cyq_st.altair_chart(plot.cyq(dto, y_scale), theme="streamlit", use_container_width=True)

# st.altair_chart(plot.test_plot(), theme="streamlit", use_container_width=True)
