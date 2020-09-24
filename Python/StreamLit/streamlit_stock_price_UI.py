
from alpha_vantage.timeseries import TimeSeries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def display_referenced_stock(ref_type, data, sym):
    fig, ax = plt.subplots()
    ax.plot(data[ref_type],'-*', color='#1fff10')
    st.title(f" Company: {sym} ")
    st.subheader(f"Looking at {ref_type}")
    st.pyplot(fig)


def display_stock_period(period_name, stock_sym):

    if period_name == 'daily':
        data, meta_data = ts.get_daily(symbol=stock_sym)

    elif period_name == 'weekly':
        data, meta_data = ts.get_weekly(symbol=stock_sym)

    else:
        # Monthly
        data, meta_data = ts.get_monthly(symbol=stock_symbol)

    return data, meta_data


list_of_stock_symbols = ["NVDA","SCHW","TSLA","NIO","MSTF","AMD","AAPL","BA","INTC","AMZN"]
ts = TimeSeries(key='4SPI2UTSXLEHEL14', output_format='pandas')

stock_symbol = st.sidebar.selectbox("Stock Symbols", list_of_stock_symbols)
stock_period = st.sidebar.selectbox("Plot Period", ['daily', 'weekly', 'monthly'])

data, meta_data = display_stock_period(stock_period, stock_symbol)

key_list = data.keys()
ref_type = st.sidebar.selectbox("Stock Reference", key_list)


display_referenced_stock(ref_type, data, stock_symbol)



