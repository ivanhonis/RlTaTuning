import sys
import time
import datetime
import math

import pandas as pd
from binance import AsyncClient, BinanceSocketManager, Client
from binance.exceptions import BinanceAPIException
from binance.enums import *
from threading import Thread
import asyncio

import  numpy as np
cimport numpy as np
import tensorflow as tf

from keras import Input, Model
from tensorflow.keras import optimizers
from tensorflow.keras import layers

# cimport tensorflow as tf
np.set_printoptions(precision=8)
np.import_array()

DTYPEAD = np.double
ctypedef np.double_t DTYPEAD_t

DTYPEAI = np.int64
ctypedef np.int64_t DTYPEAI_t


cdef class CythonTaTuning:

    cdef double last_trade_gross_profit, last_trade_fee
    cdef double fee_percent, initial_cash
    cdef double cash, cash_min, cash_max
    cdef double stop_loss_long_percent, stop_loss_long_price
    cdef double stop_loss_short_percent, stop_loss_short_price
    cdef double trailer_long_activation_percent, trailer_long_offset_percent
    cdef double trailer_long_activation_price, trailer_long_highest_price
    cdef double trailer_long_stop_price

    cdef double trailer_short_activation_percent, trailer_short_offset_percent,
    cdef double trailer_short_activation_price, trailer_short_lowest_price,
    cdef double trailer_short_stop_price

    cdef double market_price, position, position_enter_price, position_enter_qty

    cdef bint short_signal, long_signal
    cdef bint trailer_long_active, trailer_short_active

    cdef int memory_size, step_count, longest_ta, real_action, trade_steps, closed_trades_count

    cdef int asset_precision_decimal, data_precision_decimal

    cdef np.ndarray nopen, nclose
    cdef np.ndarray ema_fast_long_down_shift, ema_slow_long
    cdef np.ndarray ema_fast_short_up_shift, ema_slow_short

    cdef double total_gross_profit, total_fee, total_net_profit

    cdef int long_trades_count, long_close_trades_count
    cdef int short_trades_count, short_close_trades_count

    cdef int p1, p3, p4, p6
    cdef double p2, p5

    cdef bint is_long, is_short


    # cdef object
    #
    # cdef str

    def __init__(self, nopen, nclose, is_long=True, is_short=True):

        self.p1 = 0
        self.p2 = 0.0
        self.p3 = 0
        self.p4 = 0
        self.p5 = 0.0
        self.p6 = 0

        self.is_long = is_long
        self.is_short = is_short

        self.nopen = nopen
        self.nclose = nclose

        self.ema_fast_long_down_shift = np.zeros_like(nclose, dtype=DTYPEAD)
        self.ema_slow_long = np.zeros_like(nclose, dtype=DTYPEAD)

        self.ema_fast_short_up_shift = np.zeros_like(nclose, dtype=DTYPEAD)
        self.ema_slow_short = np.zeros_like(nclose, dtype=DTYPEAD)

        self.memory_size = 10
        self.step_count = 0
        self.longest_ta = 1000

        self.short_signal = False
        self.long_signal = False
        # self.cd = DataCollector(initial_size=self.memory_size, length=self.df.shape[0] + self.memory_size)

        self.last_trade_gross_profit = 0.0
        self.last_trade_fee = 0.0

        # for macktest
        self.fee_percent = 0.045
        self.initial_cash = 100000.0
        self.cash = self.initial_cash
        self.cash_min = self.initial_cash * 10
        self.cash_max = 0.0

        self.stop_loss_long_percent = 3.0  # %
        self.stop_loss_long_price = 0.0

        self.stop_loss_short_percent = 3  # %
        self.stop_loss_short_price = 0.0

        self.trailer_long_activation_percent = 1.0  # %
        self.trailer_long_offset_percent = 0.5  # %
        self.trailer_long_activation_price = 0.0
        self.trailer_long_highest_price = 0.0
        self.trailer_long_stop_price = 0.0
        self.trailer_long_active = False

        self.trailer_short_activation_percent = 1  # %
        self.trailer_short_offset_percent = 0.5  # %
        self.trailer_short_activation_price = 0.0
        self.trailer_short_lowest_price = 0.0
        self.trailer_short_stop_price = 0.0
        self.trailer_short_active = False

        self.market_price = 0.0
        self.position = 0.0
        self.real_action = 0

        self.position_enter_price = 0.0
        self.position_enter_qty = 0.0
        self.trade_steps = 0
        self.closed_trades_count = 0
        self.long_trades_count = 0
        self.long_close_trades_count = 0
        self.short_trades_count = 0
        self.short_close_trades_count = 0

        self.asset_precision_decimal = 4
        self.data_precision_decimal = 6

        self.total_gross_profit = 0.0
        self.total_fee = 0.0
        self.total_net_profit = 0.0

    def set_data(self, p1, p2, p3, p4, p5 ,p6):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6

        # 'ema_fast_long_down_shift_length': 65,
        # 'ema_fast_long_down_shift': 0.99,
        # 'ema_slow_long_length': 346,
        # 'ema_fast_short_up_shift_length': 35,
        # 'ema_fast_short_up_shift': 1.01,
        # 'ema_slow_short_length': 331,
        #
        # p1 = ta_config[symbol]['ema_fast_long_down_shift_length']
        # p2 = ta_config[symbol]['ema_fast_long_down_shift']
        # p3 = ta_config[symbol]['ema_slow_long_length']
        # p4 = ta_config[symbol]['ema_fast_short_up_shift_length']
        # p5 = ta_config[symbol]['ema_fast_short_up_shift']
        # p6 = ta_config[symbol]['ema_slow_short_length']

        self.ema_fast_long_down_shift = self.ema(self.nclose, p1) * p2
        self.ema_slow_long = self.ema(self.nclose, p3)

        self.ema_fast_short_up_shift = self.ema(self.nclose, p4) * p5
        self.ema_slow_short = self.ema(self.nclose, p6)

    def ema(self, data, window):
        cdef double alpha
        cdef np.ndarray ret_ema

        alpha = 2 / (window + 1)
        ret_ema = np.zeros_like(data, dtype=DTYPEAD)
        ret_ema[0] = data[0]

        for i in range(1, len(data)):
            ret_ema[i] = alpha * data[i] + (1 - alpha) * ret_ema[i - 1]

        return ret_ema

    def run_backtest(self):
        cdef double meter

        for i in range(len(self.nclose) - self.longest_ta):
            self.next(i + self.longest_ta)
        meter = self.total_net_profit
        return meter, [self.p1, self.p2 * 100, self.p3, self.p4, self.p5 * 100, self.p6]

    def dround(self, value):
        cdef double factor

        factor = 10 ** self.asset_precision_decimal
        return math.floor(value * factor) / factor

    def long(self, price):

        self.position_enter_price = price
        self.stop_loss_long_price = price * (1 - (self.stop_loss_long_percent / 100))
        self.trailer_long_activation_price = price * (1 + (self.trailer_long_activation_percent / 100))
        self.position_enter_qty = self.dround(self.cash / price)
        self.position = self.position_enter_qty
        self.trade_steps = 1
        self.real_action = 1

        self.trailer_long_active = False
        self.trailer_short_active = False
        self.trailer_long_stop_price = 0.0
        self.trailer_short_stop_price = 0.0
        self.trailer_long_highest_price = 0.0
        self.trailer_short_lowest_price = 100_000_000.0

        self.long_trades_count +=1

    def short(self, price):

        self.position_enter_price = price
        self.stop_loss_short_price = price * (1 + (self.stop_loss_short_percent / 100))
        self.trailer_short_activation_price = price * (1 - (self.trailer_short_activation_percent / 100))
        self.position_enter_qty = self.dround(self.cash / price)
        self.position = -self.position_enter_qty
        self.trade_steps = 1
        self.real_action = 3

        self.trailer_long_active = False
        self.trailer_short_active = False
        self.trailer_long_stop_price = 0.0
        self.trailer_short_stop_price = 0.0
        self.trailer_long_highest_price = 0.0
        self.trailer_short_lowest_price = 100_000_000.0

        self.short_trades_count += 1

    def close_all(self, price, source=""):
        cdef double fee, profit

        fee = 0
        profit = 0
        if self.position > 0:  # close long
            profit = (price - self.position_enter_price) * self.position_enter_qty
            fee = (self.position_enter_qty * price) * (self.fee_percent / 100)
            fee += (self.position_enter_qty * self.position_enter_price) * (self.fee_percent / 100)
            self.real_action = 2
            self.long_close_trades_count += 1

        elif self.position < 0:  # close short
            profit = (self.position_enter_price - price) * self.position_enter_qty
            fee = (self.position_enter_qty * price) * (self.fee_percent / 100)
            fee += (self.position_enter_qty * self.position_enter_price) * (self.fee_percent / 100)
            self.real_action = 4
            self.short_close_trades_count += 1

        else:
            print("BUG: position not 1 or -1.")

        self.last_trade_gross_profit = profit
        self.last_trade_fee = fee

        self.total_gross_profit += self.last_trade_gross_profit
        self.total_fee += self.last_trade_fee
        self.total_net_profit += (self.last_trade_gross_profit - self.last_trade_fee)

        # self.closed_trades_profit += self.last_trade_profit
        self.cash += (profit - fee)
        self.cash = round(self.cash, 2)

        self.cash_min = min(self.cash_min, self.cash)
        self.cash_max = max(self.cash_max, self.cash)

        self.position = 0.0
        self.closed_trades_count += 1
        self.position_enter_qty = 0.0
        self.position_enter_price = 0.0
        self.trade_steps = 0

        self.stop_loss_long_price = 0.0
        self.stop_loss_short_price = 0.0
        self.trailer_long_activation_price = 0.0
        self.trailer_short_activation_price = 0.0
        self.trailer_long_highest_price = 0.0
        self.trailer_short_lowest_price = 100_000_000.0

    # def get_symbol_position(self, symbol):
    #     if self.live:
    #         return self.broker.get_symbol_position(symbol)
    #     else:
    #         return self.position[symbol]

    def next(self, row):
        cdef int p0, p1
        cdef double open_price, close_price, s_pos

        self.step_count += 1

        # self.d = self.df.iloc[row, :]

        p0 = row  # utolsó, legújabb
        p1 = p0 - 1  # utolsó előtti, egyel előtte

        long_signal = (
                self.ema_fast_long_down_shift[p0] > self.ema_slow_long[p0] and
                self.ema_fast_long_down_shift[p1] <= self.ema_slow_long[p1])

        short_signal = (
                self.ema_fast_short_up_shift[p0] < self.ema_slow_short[p0] and
                self.ema_fast_short_up_shift[p1] >= self.ema_slow_short[p1])

        long_signal = long_signal and self.is_long
        short_signal = short_signal and self.is_short

        # if self.step_count == 2 and symbol == "ADAUSDT":
        #     long_signal = True
        #     short_signal = False
        # else:
        #     long_signal = False
        #     short_signal = False

        open_price = self.nopen[p0]
        close_price = self.nclose[p0]
        self.market_price = close_price

        # print("- " * 25)
        # print("trade loop", data['date'], self.step_count, step_count,
        #       f"long: {long_signal}, short: {short_signal} position: {self.position}")
        #
        # print(f"close_price {close_price}, open_price {open_price}")
        # print(f"trailer_short_active {self.trailer_short_active}, trailer_short_active {self.trailer_short_active}")
        # print(f"stop_loss_long_price {self.stop_loss_long_price}, stop_loss_short_price {self.stop_loss_short_price}")
        # print(f"trailer_long_activation_price {self.trailer_long_activation_price}, trailer_short_activation_price {self.trailer_short_activation_price}")
        # print(f"trailer_long_stop_price {self.trailer_long_stop_price}, trailer_short_stop_price {self.trailer_short_stop_price}")

        s_pos = self.position
        if long_signal and s_pos == 0:

            self.long(open_price)
            # self.position = 1
            long_signal = False
            short_signal = False
        elif s_pos > 0 and close_price < self.stop_loss_long_price:
            self.close_all(close_price, "stop_loss_long")
            # print(self.last_trade_gross_profit)
        elif s_pos > 0 and self.trailer_long_active and close_price < self.trailer_long_stop_price:
            self.close_all(close_price, "trailer_long_stop")
            # print(self.last_trade_gross_profit)

        if s_pos > 0 and close_price > self.trailer_long_activation_price and not self.trailer_long_active:
            self.trailer_long_active = True

        if s_pos > 0 and self.trailer_long_active:
            self.trailer_long_highest_price = max(self.trailer_long_highest_price, close_price)
            self.trailer_long_stop_price = self.trailer_long_highest_price * (
                    1 - (self.trailer_long_offset_percent / 100))

        if short_signal and s_pos == 0:
            self.short(open_price)
            long_signal = False
            short_signal = False
            # self.position = -1
        elif s_pos < 0 and close_price > self.stop_loss_short_price:
            self.close_all(close_price, "stop_loss_short")
            # print(self.last_trade_gross_profit)
        elif s_pos < 0 and self.trailer_short_active and close_price > self.trailer_short_stop_price:
            self.close_all(close_price, "trailer short stop")
            # print(self.last_trade_gross_profit)

        if s_pos < 0 and close_price < self.trailer_short_activation_price and not self.trailer_short_active:
            self.trailer_short_active = True

        if s_pos < 0 and self.trailer_short_active:
            self.trailer_short_lowest_price = min(self.trailer_short_lowest_price, close_price)
            self.trailer_short_stop_price = self.trailer_short_lowest_price * (
                    1 + (self.trailer_short_offset_percent / 100))

        # self.cd.trade_steps.add(self.trade_steps)
        # self.cd.position.add(s_pos)
        # # self.cd.current_date.add(self.data['date'])
        # self.cd.portfolio_value.add(self.get_portfolio_value())
        # self.cd.open_position.add(self.get_open_position_value())
        #
        # self.cd.real_action.add(self.real_action)
        # self.cd.action.add(self.real_action)
        #
        # self.cd.close_price.add(self.d['close'])
        # self.cd.open_price.add(self.d['open'])
        #
        # self.cd.last_trade_net_profit.add(self.last_trade_gross_profit - self.last_trade_fee)
        # self.cd.last_trade_gross_profit.add(self.last_trade_gross_profit)
        # self.cd.last_trade_fee.add(self.last_trade_fee)

        self.real_action = 0
        self.last_trade_gross_profit = 0.0
        self.last_trade_fee = 0.0


    def render(self):
        print("-" * 25)
        print(f"             total_gross_profit {self.total_gross_profit:,.2f}")
        print(f"                      total_fee {self.total_fee:,.2f}")
        print(f"               total_net_profit {self.total_net_profit:,.2f}")
        print(f"                           cash {self.cash:,.2f}")
        print(f"                       cash_min {self.cash_min:,.2f}")
        print(f"                       cash_max {self.cash_max:,.2f}")
        print("")
        print(f"            closed_trades_count {self.closed_trades_count}")
        print(f"              long_trades_count {self.long_trades_count}")
        print(f"        long_close_trades_count {self.long_close_trades_count}")
        print(f"             short_trades_count {self.short_trades_count}")
        print(f"       short_close_trades_count {self.short_close_trades_count}")
        print("")
        print(f"ema_fast_long_down_shift_length {self.p1}")
        print(f"       ema_fast_long_down_shift {self.p2}")
        print(f"           ema_slow_long_length {self.p3}")
        print("")
        print(f" ema_fast_short_up_shift_length {self.p4}")
        print(f"        ema_fast_short_up_shift {self.p5}")
        print(f"          ema_slow_short_length {self.p6}")
        print("")

    def reset(self):
        self.short_signal = False
        self.long_signal = False

        self.last_trade_gross_profit = 0.0
        self.last_trade_fee = 0.0

        # for macktest
        self.cash = self.initial_cash
        self.cash_min = self.initial_cash * 10
        self.cash_max = 0.0

        self.stop_loss_long_price = 0.0

        self.stop_loss_short_price = 0.0

        self.trailer_long_activation_price = 0.0
        self.trailer_long_highest_price = 0.0
        self.trailer_long_stop_price = 0.0
        self.trailer_long_active = False

        self.trailer_short_activation_price = 0.0
        self.trailer_short_lowest_price = 0.0
        self.trailer_short_stop_price = 0.0
        self.trailer_short_active = False

        self.market_price = 0.0
        self.position = 0.0
        self.real_action = 0

        self.position_enter_price = 0.0
        self.position_enter_qty = 0.0
        self.trade_steps = 0
        self.closed_trades_count = 0
        self.long_trades_count = 0
        self.long_close_trades_count = 0
        self.short_trades_count = 0
        self.short_close_trades_count = 0

        self.total_gross_profit = 0.0
        self.total_fee = 0.0
        self.total_net_profit = 0.0

    # def get_open_position_value(self):
    #     if self.position > 0:
    #         ret = (self.market_price - self.position_enter_price) * self.position_enter_qty
    #     elif self.position < 0:
    #         ret = (self.position_enter_price - self.market_price) * self.position_enter_qty
    #     else:
    #         ret = 0.0
    #     return round(ret, 2)
    #
    # def get_portfolio_value(self):
    #     ret = 0.0
    #     if self.position == 0:
    #         ret = self.cash
    #     elif self.position > 0:
    #         ret = self.cash + ((self.market_price - self.position_enter_price) * self.position_enter_qty)
    #     elif self.position < 0:
    #         ret = self.cash + ((self.position_enter_price - self.market_price) * self.position_enter_qty)
    #     return round(ret, 2)
    #
    # @staticmethod
    # def count_long_short(df):
    #     l_count = 0
    #     s_count = 0
    #     for step_count in range(len(df)):
    #
    #         p0 = step_count
    #         p1 = p0 - 1
    #         long_signal = (
    #                 round(df.ema_fast_long_down_shift[p0], 8) > round(df.ema_slow_long[p0], 8) and
    #                 round(df.ema_fast_long_down_shift[p1], 8) <= round(df.ema_slow_long[p1], 8))
    #
    #         short_signal = (
    #                 round(df.ema_fast_short_up_shift[p0], 8) < round(df.ema_slow_short[p0], 8) and
    #                 round(df.ema_fast_short_up_shift[p1], 8) >= round(df.ema_slow_short[p1], 8))
    #
    #         if long_signal:
    #             l_count += 1
    #
    #         if short_signal:
    #             s_count += 1
    #
    #     return l_count, s_count
    #
    # @staticmethod
    # def check_trade_sequence_quality(trade_array):
    #     current_position = 'none'
    #
    #     for action in trade_array:
    #         if current_position == 'none':
    #             if action == 0:
    #                 continue
    #             elif action == 1:
    #                 current_position = 'long'
    #             elif action == 3:
    #                 current_position = 'short'
    #             else:
    #                 return False
    #
    #         elif current_position == 'long':
    #             if action == 0:
    #                 continue
    #             elif action == 2:
    #                 current_position = 'none'
    #             else:
    #                 return False
    #
    #         elif current_position == 'short':
    #             if action == 0:
    #                 continue
    #             elif action == 4:
    #                 current_position = 'none'
    #             else:
    #                 return False
    #
    #     return True
