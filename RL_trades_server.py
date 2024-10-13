import datetime
import math
import sys
import time
from prettytable import PrettyTable


import pandas as pd
import pandas_ta as ta
import numpy as np
import json
# import pandas_ta as ta
# from RL_Tools import df_split, binance_download, check_trade_sequence_quality, get_dd_du
from RL_Tools import DataCollector
from tqdm import tqdm
from prettytable import PrettyTable
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from sb3_contrib import MaskablePPO
from RL_Futures_Env import SingleFuturesEnv
from sb3_contrib.common.wrappers import ActionMasker
from RL_Tools import get_dataset
from RL_Broker import RL_Broker
# import torch
from binance import Client, ThreadedWebsocketManager

rl_path = "RL_Futures/"


def make_env(df, rank, seed=0, predict_mode=False):
    def _init():
        env = SingleFuturesEnv(symbol=symbol, worker_id=rank, predict_mode=predict_mode)
        env = ActionMasker(env, 'action_mask')
        env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env

    #
    # def get_action_mask(env):
    #     return env.action_mask()

    return _init


class RL_trade_server:
    def __init__(self, data, ta_config, live=False, ai=False):

        self.live = live
        self.traded_symbols = []
        self.df = {}

        if not self.live:
            data = [data[0]]

        for d in data:
            self.traded_symbols.append(d['symbol'])
            self.df[d['symbol']] = d['df']
        data = None  # just clear memory

        self.ai = ai
        self.ta_config = ta_config

        self.vec_envs = {}
        self.models = {}
        self.excluded_features = ['date']

        self.memory_size = 10
        self.step_count = self.defa_dict(0)
        self.longest_ta = 1000

        self.short_signal = False
        self.long_signal = False
        self.cd = {}
        for s in self.traded_symbols:
            self.cd[s] = DataCollector(initial_size=self.memory_size, length=self.df[s].shape[0] + self.memory_size)

        self.last_trade_gross_profit = self.defa_dict(0.0)
        self.last_trade_fee = self.defa_dict(0.0)

        # for macktest
        self.fee_percent = 0.045
        self.initial_cash = 100_000.0
        self.cash = self.initial_cash
        if self.live:
            self.lot = 60
        else:
            self.lot = self.cash

        self.stop_loss_long_percent = 3  # %
        self.stop_loss_long_price = self.defa_dict(0.0)

        self.stop_loss_short_percent = 3  # %
        self.stop_loss_short_price = self.defa_dict(0.0)

        self.trailer_long_activation_percent = 1  # %
        self.trailer_long_offset_percent = 0.5  # %
        self.trailer_long_activation_price = self.defa_dict(0.0)
        self.trailer_long_highest_price = self.defa_dict(0.0)
        self.trailer_long_stop_price = self.defa_dict(0.0)
        self.trailer_long_active = self.defa_dict(False)

        self.trailer_short_activation_percent = 1  # %
        self.trailer_short_offset_percent = 0.5  # %
        self.trailer_short_activation_price = self.defa_dict(0.0)
        self.trailer_short_lowest_price = self.defa_dict(0.0)
        self.trailer_short_stop_price = self.defa_dict(0.0)
        self.trailer_short_active = self.defa_dict(False)

        self.market_price = self.defa_dict(0.0)
        self.position = self.defa_dict(0)
        self.real_action = self.defa_dict(0)
        self.d = {}  # current data

        self.position_enter_price = self.defa_dict(0.0)
        self.position_enter_qty = self.defa_dict(0.0)
        self.trade_steps = self.defa_dict(0)
        self.closed_trades_count = self.defa_dict(0)

        self.asset_precision_decimal = 4
        self.data_precision_decimal = 6

        if self.ai:
            self.create_models()

        if self.live:
            self.broker = RL_Broker(self.traded_symbols)
            self.broker.handle_kline_socket_message = self.handle_kline_socket_message
            self.broker.futures_account_position_callback = self.futures_account_position_callback
            self.broker.start_user_socket()
            self.broker.get_futures_account_balance()
            self.broker.get_futures_account_position()
            self.broker.start_market_socket()
            time.sleep(5)
            for s in traded_symbols:
                self.broker.close(s)
            time.sleep(5)
            for s in traded_symbols:
                print(f"{s} position: {self.get_symbol_position(s)}")

            self.start_date_time = datetime.datetime.now()
            self.start_potrfolio_value = self.broker.get_portfolio_value()
            print(f"Portfolio value: {self.start_potrfolio_value}")

        else:
            self.run_backtest()

    def futures_account_position_callback(self):
        self.render_live(full=True)

    def defa_dict(self, value):
        ret = {}
        for s in self.traded_symbols:
            ret[s] = value
        return ret

    def handle_kline_socket_message(self, symbol, df):
        s = symbol
        new_dt = df['date'].iloc[-1]
        last_dt = self.df[s]['date'].iloc[-1]
        if new_dt == last_dt:
            self.df[s] = self.df[s].drop(self.df[s].index[-1])

        self.df[s] = pd.concat([self.df[s], df], ignore_index=True)

        ta_fll = self.ta_config[s]['ema_fast_long_down_shift_length']
        ta_fls = self.ta_config[s]['ema_fast_long_down_shift']
        ta_sll = self.ta_config[s]['ema_slow_long_length']
        ta_fsl = self.ta_config[s]['ema_fast_short_up_shift_length']
        ta_fss = self.ta_config[s]['ema_fast_short_up_shift']
        ta_ssl = self.ta_config[s]['ema_slow_short_length']

        self.df[s]['ema_fast_long_down_shift'] = ta.ema(self.df[s]['close'], length=ta_fll) * ta_fls
        self.df[s]['ema_slow_long'] = ta.ema(self.df[s]['close'], length=ta_sll)

        self.df[s]['ema_fast_short_up_shift'] = ta.ema(self.df[s]['close'], length=ta_fsl) * ta_fss
        self.df[s]['ema_slow_short'] = ta.ema(self.df[s]['close'], length=ta_ssl)

        self.df[s]['RSI_14'] = ta.rsi(self.df[s]['close'], length=14)
        self.df[s][['MACD', 'MACD_S', 'MACD_H']] = ta.macd(self.df[s]['close'], fast=12, slow=26, signal=9)
        self.df[s].drop(['MACD', 'MACD_S'], axis=1,
                        inplace=True)  # ez nem kell mert a MACD_H kifejezi a másik kettőt is
        self.df[s]['MACD_H'] = self.df[s]['MACD_H'] * 100

        if self.df[s].shape[0] > self.longest_ta:
            self.df[s] = self.df[s].drop(self.df[s].index[0])
        last_row = self.df[s].shape[0] - 1
        self.next(symbol, last_row)

    def create_models(self):
        model_base = "RL_Futures_model_Ta"
        model_ver = "dev_01l"

        for s in self.traded_symbols:
            model_name = s + '_' + model_base + '_' + model_ver
            self.vec_envs[s] = [make_env(df=self.df[s], rank=0, predict_mode=True) for i in range(1)]
            self.vec_envs[s] = SubprocVecEnv(self.vec_envs[s])
            self.vec_envs[s] = VecNormalize.load(rl_path + 'Vecnormalize_' + model_name + '.pkl', self.vec_envs[s])
            self.vec_envs[s].training = False
            self.vec_envs[s].norm_reward = False
            self.models[s] = MaskablePPO.load(rl_path + model_name, env=self.vec_envs[s])

    def ai_validate(self, symbol, row, long_signal, short_signal):
        s = symbol

        if self.ai:
            state = self.create_state_2d(symbol, row)
            # self.action_space = Discrete(3)  # 0 pass, 1 go long, 2 go short

            if long_signal:
                action_mask = np.array([1, 1, 0], dtype=np.int32)
                ori = "long"
            elif short_signal:
                action_mask = np.array([1, 0, 1], dtype=np.int32)
                ori = "short"
            else:
                action_mask = np.array([1, 0, 0], dtype=np.int32)
                ori = "none"

            normalized_state = self.vec_envs[s].normalize_obs(state)
            # action_mask = torch.tensor(action_mask,dtype=torch.bool).unsqueeze(0)
            action, _ = self.models[s].predict(normalized_state, action_masks=action_mask, deterministic=True)

            # actions = ["pass", "long", "short"]
            # print("ai_walidate", ori, actions[action])

            if long_signal and action == 1:
                return True, False
            elif short_signal and action == 2:
                return False, True
            else:
                return False, False

        else:
            return long_signal, short_signal

    # def prepare_data(self, df):
    #     self.df = df

    # self.df = self.add_ta(self.df)
    # self.df = self.set_order(self.df)

    # def set_order(self, df):
    #     df = df.dropna()
    #     df = df.reset_index(drop=True)
    #     df = df.sort_values(by=["date"]).reset_index(drop=True)
    #     return df

    # def add_data(self, row_no):
    #     new_row = self.temp_df.iloc[row_no]
    #     self.df = pd.concat([self.df, new_row.T], ignore_index=True)

    # def add_ta(self, df):
    #     df['ema_fast_long_down_shift'] = ta.ema(df['close'], length=53) * 0.99
    #     df['ema_slow_long'] = ta.ema(df['close'], length=265)
    #
    #     df['ema_fast_short_up_shift'] = ta.ema(df['close'], length=39) * 1.01
    #     df['ema_slow_short'] = ta.ema(df['close'], length=397)
    #     return df

    def run_backtest(self):
        s = symbol = self.traded_symbols[0]
        for i in tqdm(range(len(self.df[s]) - self.longest_ta)):
            self.next(symbol, i + self.longest_ta)
        return self.render(self.traded_symbols[0])

    def dround(self, value):
        factor = 10 ** self.asset_precision_decimal
        return math.floor(value * factor) / factor

    def long(self, symbol, price):
        s = symbol
        # print(self.data['date'], "long")
        bqty = 0
        if self.live:
            bqty = self.broker.long(symbol, self.lot, price)

        self.position_enter_price[s] = price
        self.stop_loss_long_price[s] = price * (1 - (self.stop_loss_long_percent / 100))
        self.trailer_long_activation_price[s] = price * (1 + (self.trailer_long_activation_percent / 100))
        self.position_enter_qty[s] = self.dround(self.lot / price) if not self.live else bqty
        self.position[s] = self.position_enter_qty[s]
        self.trade_steps[s] = 1
        self.real_action[s] = 1

        self.trailer_long_active[s] = False
        self.trailer_short_active[s] = False
        self.trailer_long_stop_price[s] = 0.0
        self.trailer_short_stop_price[s] = 0.0
        self.trailer_long_highest_price[s] = 0.0
        self.trailer_short_lowest_price[s] = 100_000_000.0

    def short(self, symbol, price):
        s = symbol

        bqty = 0
        if self.live:
            bqty = self.broker.short(symbol, self.lot, price)

        # print(self.data['date'], "short")
        # self.position = -1
        self.position_enter_price[s] = price
        self.stop_loss_short_price[s] = price * (1 + (self.stop_loss_short_percent / 100))
        self.trailer_short_activation_price[s] = price * (1 - (self.trailer_short_activation_percent / 100))
        self.position_enter_qty[s] = self.dround(self.lot / price) if not self.live else bqty
        self.position[s] = -self.position_enter_qty[s]
        self.trade_steps[s] = 1
        self.real_action[s] = 3

        self.trailer_long_active[s] = False
        self.trailer_short_active[s] = False
        self.trailer_long_stop_price[s] = 0.0
        self.trailer_short_stop_price[s] = 0.0
        self.trailer_long_highest_price[s] = 0.0
        self.trailer_short_lowest_price[s] = 100_000_000.0

    def close_all(self, symbol, price, source=""):
        s = symbol
        fee = 0
        profit = 0
        if self.get_symbol_position(symbol) > 0:  # close long
            profit = (price - self.position_enter_price[s]) * self.position_enter_qty[s]
            fee = (self.position_enter_qty[s] * price) * (self.fee_percent / 100)
            fee += (self.position_enter_qty[s] * self.position_enter_price[s]) * (self.fee_percent / 100)
            self.real_action[s] = 2

        elif self.get_symbol_position(symbol) < 0:  # close short
            profit = (self.position_enter_price[s] - price) * self.position_enter_qty[s]
            fee = (self.position_enter_qty[s] * price) * (self.fee_percent / 100)
            fee += (self.position_enter_qty[s] * self.position_enter_price[s]) * (self.fee_percent / 100)
            self.real_action[s] = 4

        else:
            print("BUG: position not 1 or -1.")

        # self.fee_total += fee
        # self.fee_total = round(self.fee_total, 2)

        if self.live:
            self.broker.close(symbol)

        self.last_trade_gross_profit[s] = profit
        self.last_trade_fee[s] = fee
        # self.closed_trades_profit += self.last_trade_profit
        self.cash += (profit - fee)
        self.cash = round(self.cash, 2)
        if not self.live:
            self.lot = self.cash

        self.position[s] = 0.0
        self.closed_trades_count[s] += 1
        self.position_enter_qty[s] = 0.0
        self.position_enter_price[s] = 0.0
        self.trade_steps[s] = 0

        self.stop_loss_long_price[s] = 0.0
        self.stop_loss_short_price[s] = 0.0
        self.trailer_long_activation_price[s] = 0.0
        self.trailer_short_activation_price[s] = 0.0
        self.trailer_long_highest_price[s] = 0.0
        self.trailer_short_lowest_price[s] = 100_000_000.0

    def get_symbol_position(self, symbol):
        if self.live:
            return self.broker.get_symbol_position(symbol)
        else:
            return self.position[symbol]

    def next(self, symbol, row):
        s = symbol
        self.step_count[s] += 1

        self.d[s] = self.df[s].iloc[row, :]

        p0 = row  # utolsó, legújabb
        p1 = p0 - 1  # utolsó előtti, egyel előtte

        long_signal = (
                self.df[s].ema_fast_long_down_shift[p0] > self.df[s].ema_slow_long[p0] and
                self.df[s].ema_fast_long_down_shift[p1] <= self.df[s].ema_slow_long[p1])

        short_signal = (
                self.df[s].ema_fast_short_up_shift[p0] < self.df[s].ema_slow_short[p0] and
                self.df[s].ema_fast_short_up_shift[p1] >= self.df[s].ema_slow_short[p1])

        # if self.step_count[s] == 2 and symbol == "ADAUSDT":
        #     long_signal = True
        #     short_signal = False
        # else:
        #     long_signal = False
        #     short_signal = False

        if self.ai and self.get_symbol_position(s) == 0 and (long_signal or short_signal):
            long_signal, short_signal = self.ai_validate(symbol, row, long_signal, short_signal)

        open_price = self.df[s].loc[p0, 'open']
        close_price = self.df[s].loc[p0, 'close']
        self.market_price[s] = close_price

        # print("- " * 25)
        # print("trade loop", data['date'], self.step_count, step_count,
        #       f"long: {long_signal}, short: {short_signal} position: {self.position}")
        #
        # print(f"close_price {close_price}, open_price {open_price}")
        # print(f"trailer_short_active {self.trailer_short_active}, trailer_short_active {self.trailer_short_active}")
        # print(f"stop_loss_long_price {self.stop_loss_long_price}, stop_loss_short_price {self.stop_loss_short_price}")
        # print(f"trailer_long_activation_price {self.trailer_long_activation_price}, trailer_short_activation_price {self.trailer_short_activation_price}")
        # print(f"trailer_long_stop_price {self.trailer_long_stop_price}, trailer_short_stop_price {self.trailer_short_stop_price}")

        s_pos = self.get_symbol_position(s)
        if long_signal and s_pos == 0:

            self.long(symbol, open_price)
            # self.position[s] = 1
            long_signal = False
            short_signal = False
        elif s_pos > 0 and close_price < self.stop_loss_long_price[s]:
            self.close_all(symbol, close_price, "stop_loss_long")
            # print(self.last_trade_gross_profit)
        elif s_pos > 0 and self.trailer_long_active[s] and close_price < self.trailer_long_stop_price[s]:
            self.close_all(symbol, close_price, "trailer_long_stop")
            # print(self.last_trade_gross_profit)

        if s_pos > 0 and close_price > self.trailer_long_activation_price[s] and not self.trailer_long_active[s]:
            self.trailer_long_active[s] = True

        if s_pos > 0 and self.trailer_long_active[s]:
            self.trailer_long_highest_price[s] = max(self.trailer_long_highest_price[s], close_price)
            self.trailer_long_stop_price[s] = self.trailer_long_highest_price[s] * (
                    1 - (self.trailer_long_offset_percent / 100))

        if short_signal and s_pos == 0:
            self.short(symbol, open_price)
            long_signal = False
            short_signal = False
            # self.position = -1
        elif s_pos < 0 and close_price > self.stop_loss_short_price[s]:
            self.close_all(symbol, close_price, "stop_loss_short")
            # print(self.last_trade_gross_profit)
        elif s_pos < 0 and self.trailer_short_active[s] and close_price > self.trailer_short_stop_price[s]:
            self.close_all(symbol, close_price, "trailer short stop")
            # print(self.last_trade_gross_profit)

        if s_pos < 0 and close_price < self.trailer_short_activation_price[s] and not self.trailer_short_active[s]:
            self.trailer_short_active[s] = True

        if s_pos < 0 and self.trailer_short_active[s]:
            self.trailer_short_lowest_price[s] = min(self.trailer_short_lowest_price[s], close_price)
            self.trailer_short_stop_price[s] = self.trailer_short_lowest_price[s] * (
                    1 + (self.trailer_short_offset_percent / 100))

        if not self.live:
            self.cd[s].trade_steps.add(self.trade_steps[s])
            self.cd[s].position.add(s_pos)
            # self.cd.current_date.add(self.data['date'])
            self.cd[s].portfolio_value.add(self.get_portfolio_value(symbol))
            self.cd[s].open_position.add(self.get_open_position_value(symbol))

            self.cd[s].real_action.add(self.real_action[s])
            self.cd[s].action.add(self.real_action[s])

            self.cd[s].close_price.add(self.d[s]['close'])
            self.cd[s].open_price.add(self.d[s]['open'])

            self.cd[s].last_trade_net_profit.add(self.last_trade_gross_profit[s] - self.last_trade_fee[s])
            self.cd[s].last_trade_gross_profit.add(self.last_trade_gross_profit[s])
            self.cd[s].last_trade_fee.add(self.last_trade_fee[s])

        self.real_action[s] = 0
        self.last_trade_gross_profit[s] = 0.0
        self.last_trade_fee[s] = 0.0

    @staticmethod
    def qprint(*args, table_indent=25):
        table = PrettyTable()
        table.header = False
        table.align = "l"
        for arg in args:
            table.add_row([arg])
        table_str = table.get_string()
        indented_table = "\n".join((" " * table_indent) + line for line in table_str.splitlines())
        # if self.predict_mode:
        #     pass
        # else:
        #     # with self.lock:
        print(indented_table)

    def render(self, symbol):
        s = symbol
        # qc = self.check_trade_sequence_quality(self.cd.real_action.get())

        # rtickets = "\n".join([f"  {key}: {value['count']} / {value['amount']:,.2f}" for key, value in self.price_list.items()])
        gross_profit = self.cd[s].last_trade_gross_profit.sum()
        total_fee = self.cd[s].last_trade_fee.sum()
        net_profit = gross_profit - total_fee

        count_1 = np.count_nonzero(self.cd[s].real_action.get() == 1)
        count_2 = np.count_nonzero(self.cd[s].real_action.get() == 2)
        count_3 = np.count_nonzero(self.cd[s].real_action.get() == 3)
        count_4 = np.count_nonzero(self.cd[s].real_action.get() == 4)

        # print(self.collect.last_trade_net_profit.describe())
        # print('- ' * 50)

        self.qprint(
                    f"               Datetime: {datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')}",
                    f"     Total gross profit: {gross_profit:,.2f}",
                    f"              Total fee: {total_fee:,.2f}",
                    f"             Net profit: {net_profit:,.2f}",
                    f"          Closed trades: {self.closed_trades_count[s]}",
                    f"AVG net profit / trades: {net_profit / (self.closed_trades_count[s] + 0.01):,.2f}",
                    f"       Num of open long: {count_1}",
                    f"      Num of close long: {count_2}",
                    f"      Num of open short: {count_3}",
                    f"     Num of close short: {count_4}",
                    )
        return

    def render_live(self, full=False):

        positions = ''
        for s in self.traded_symbols:
            sy = s + "           "
            positions += (f"{sy[:10]} -> "
                          f"{self.broker.get_symbol_position(s):,.6f} "
                          f"Enter p: {self.position_enter_price[s]:,.6f} "
                          f"Market p: {self.market_price[s]:,.6f} "
                          # f"Stop lp: {self.stop_loss_long_price[s]:,.6f} "
                          # f"Stop sp: {self.stop_loss_short_price[s]:,.6f} "
                          # f"Activation lp: {self.trailer_long_activation_price[s]:,.6f} "
                          # f"Activation sp: {self.trailer_short_activation_price[s]:,.6f} "
                          # f"Trailer stop lp: {self.trailer_long_stop_price[s]:,.6f} "
                          # f"Trailer stop sp: {self.trailer_short_stop_price[s]:,.6f} "
                          f"\n")

        if full:
            total_closed_trades_clount = sum(self.closed_trades_count.values())
            portfolio_value = self.broker.get_portfolio_value()

            self.qprint(f"Render: ",
                        f"           Datetime now: {datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')}",
                        f"         Start datetime: {self.start_date_time.strftime('%Y.%m.%d %H:%M:%S')}",
                        f"  Start portfolio value: {self.start_potrfolio_value:,.2f}",
                        f"        Portoflio value: {portfolio_value:,.2f}",
                        f"                    P&L: {self.start_potrfolio_value - portfolio_value:,.2f}",
                        f"          Closed trades: {total_closed_trades_clount:,.0f}",
                        positions
                        )
        else:
            print(positions)
            print("")

    def create_state_2d(self, symbol, row):
        s = symbol
        idf = self.df[s][row - self.memory_size:row].copy()

        idf = idf.drop(columns=self.excluded_features)
        # idf = idf.drop(columns=['date'])
        state_dict = {
            "data": idf.round(self.data_precision_decimal),
            # "position": self.cd.position.get_last(numpy=True, precision=self.data_precision_decimal),
            # "trade_steps": self.cd.trade_steps.get_last(numpy=True, precision=self.data_precision_decimal),
            # "potfolio_value": np.round(self.cd.portfolio_value.get_last(numpy=True, precision=self.data_precision_decimal) / self.initial_cash, self.data_precision_decimal),
            # "open_position_value": np.round(self.cd.open_position.get_last(numpy=True, precision=self.data_precision_decimal) / self.initial_cash, self.data_precision_decimal),
        }

        # return {'2d_array': self.convert_2d(state_dict, self.data_precision_decimal)}
        return self.convert_2d(state_dict, self.data_precision_decimal)

    @staticmethod
    def convert_2d(data_dict, data_precision_decimal=6):

        result_df = pd.DataFrame()

        for key, value in data_dict.items():
            if isinstance(value, pd.DataFrame):
                result_df = pd.concat([result_df, value], axis=1)
            elif isinstance(value, list):
                result_df[key] = np.array(value)
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                result_df[key] = value
            else:
                # Hibakezelés: ha nem megfelelő típusú adatot kapunk
                raise ValueError(f"A dictionary {key} kulcshoz tartozó értéke nem DataFrame vagy 1D numpy array",
                                 type(value))

        # ret = result_df.to_numpy(dtype=np.float64).flatten().round(data_precision_decimal).tolist()
        # ret = np.reshape(ret,(self.features_space, self.memory_size))
        # ret = result_df.to_numpy(dtype=np.float64).T
        # ret = result_df.to_numpy(dtype=np.float64)
        # ret = np.expand_dims(ret.T, axis=0)
        ret = result_df.T.values.flatten().tolist()
        return ret

    def get_open_position_value(self, symbol):
        s = symbol
        if self.position[s] > 0:
            ret = (self.market_price[s] - self.position_enter_price[s]) * self.position_enter_qty[s]
        elif self.position[s] < 0:
            ret = (self.position_enter_price[s] - self.market_price[s]) * self.position_enter_qty[s]
        else:
            ret = 0.0
        return round(ret, 2)

    def get_portfolio_value(self, symbol):
        s = symbol
        ret = 0.0
        if self.position[s] == 0:
            ret = self.cash
        elif self.position[s] > 0:
            ret = self.cash + ((self.market_price[s] - self.position_enter_price[s]) * self.position_enter_qty[s])
        elif self.position[s] < 0:
            ret = self.cash + ((self.position_enter_price[s] - self.market_price[s]) * self.position_enter_qty[s])
        return round(ret, 2)

    @staticmethod
    def count_long_short(df):
        l_count = 0
        s_count = 0
        for step_count in range(len(df)):

            p0 = step_count
            p1 = p0 - 1
            long_signal = (
                    round(df.ema_fast_long_down_shift[p0], 8) > round(df.ema_slow_long[p0], 8) and
                    round(df.ema_fast_long_down_shift[p1], 8) <= round(df.ema_slow_long[p1], 8))

            short_signal = (
                    round(df.ema_fast_short_up_shift[p0], 8) < round(df.ema_slow_short[p0], 8) and
                    round(df.ema_fast_short_up_shift[p1], 8) >= round(df.ema_slow_short[p1], 8))

            if long_signal:
                l_count += 1

            if short_signal:
                s_count += 1

        return l_count, s_count

    @staticmethod
    def check_trade_sequence_quality(trade_array):
        current_position = 'none'

        for action in trade_array:
            if current_position == 'none':
                if action == 0:
                    continue
                elif action == 1:
                    current_position = 'long'
                elif action == 3:
                    current_position = 'short'
                else:
                    return False

            elif current_position == 'long':
                if action == 0:
                    continue
                elif action == 2:
                    current_position = 'none'
                else:
                    return False

            elif current_position == 'short':
                if action == 0:
                    continue
                elif action == 4:
                    current_position = 'none'
                else:
                    return False

        return True



if __name__ == "__main__":

    # live = False
    live = True

    traded_symbols = ['ADAUSDT', 'AVAXUSDT', 'WLDUSDT', 'TRXUSDT', 'TIAUSDT', 'SEIUSDT', 'FTMUSDT']
    # traded_symbols = ['FTMUSDT']

    if live:
        trans_data = False
        from_dt = datetime.datetime.now() - datetime.timedelta(minutes=1000 + (2 * 60))
        cutoff_dt = None
        refresh = True

    else:
        # trans_data = True
        trans_data = False
        from_dt = datetime.datetime(year=2024, month=8, day=1, hour=0, minute=0)
        cutoff_dt = None
        refresh = False

    ta_config = {
        'FTMUSDT': {
            'ema_fast_long_down_shift_length': 3,
            'ema_fast_long_down_shift': 0.99,
            'ema_slow_long_length': 210,
            'ema_fast_short_up_shift_length': 3,
            'ema_fast_short_up_shift': 1.01,
            'ema_slow_short_length': 162,
        },
        'SEIUSDT': {
            'ema_fast_long_down_shift_length': 3,
            'ema_fast_long_down_shift': 0.98,
            'ema_slow_long_length': 386,
            'ema_fast_short_up_shift_length': 3,
            'ema_fast_short_up_shift': 1.0,
            'ema_slow_short_length': 271,
        },

        'TIAUSDT': {
            'ema_fast_long_down_shift_length': 3,
            'ema_fast_long_down_shift': 0.98,
            'ema_slow_long_length': 246,
            'ema_fast_short_up_shift_length': 3,
            'ema_fast_short_up_shift': 1.01,
            'ema_slow_short_length': 126,
        },

        'WLDUSDT': {
            'ema_fast_long_down_shift_length': 3,
            'ema_fast_long_down_shift': 0.99,
            'ema_slow_long_length': 154,
            'ema_fast_short_up_shift_length': 3,
            'ema_fast_short_up_shift': 1.01,
            'ema_slow_short_length': 321,
        },
        'TRXUSDT': {
            'ema_fast_long_down_shift_length': 36,
            'ema_fast_long_down_shift': 0.99,
            'ema_slow_long_length': 215,
            'ema_fast_short_up_shift_length': 24,
            'ema_fast_short_up_shift': 1.01,
            'ema_slow_short_length': 205,
        },

        # 'ETHUSDT': {
        #     'ema_fast_long_down_shift_length': 25,
        #     'ema_fast_long_down_shift': 0.99,
        #     'ema_slow_long_length': 252,
        #     'ema_fast_short_up_shift_length': 5,
        #     'ema_fast_short_up_shift': 1.01,
        #     'ema_slow_short_length': 324,
        # },

        'AVAXUSDT': {
            'ema_fast_long_down_shift_length': 6,
            'ema_fast_long_down_shift': 0.99,
            'ema_slow_long_length': 129,
            'ema_fast_short_up_shift_length': 8,
            'ema_fast_short_up_shift': 1.01,
            'ema_slow_short_length': 430,
        },
        'ADAUSDT': {
            'ema_fast_long_down_shift_length': 18,
            'ema_fast_long_down_shift': 0.99,
            'ema_slow_long_length': 169,
            'ema_fast_short_up_shift_length': 5,
            'ema_fast_short_up_shift': 1.0,
            'ema_slow_short_length': 419,
        },

    }

    data = []
    for symbol in traded_symbols:
        data_symbol = {'symbol': symbol}
        if trans_data:
            print("Load dataset from transdata.")
            file_name = "binance_data/" + symbol + "_trans_data"
            data_symbol['df'] = pd.DataFrame(pd.read_hdf(file_name, "df"))
        else:
            data_symbol['df'] = get_dataset(symbol, from_dt, cutoff_dt, ta_config, refresh=refresh)
            file_name = "binance_data/" + symbol + "_trans_data"
            print("Create new trans_data.")
            data_symbol['df'].to_hdf(file_name, key='df', mode='w')

        data.append(data_symbol)

        print(symbol)
        print(data_symbol['df'].T)
        print("shape", data_symbol['df'].shape)

    # rlt2 = RL_trade_server(model_name)
    # rlt2.prepare_data(df)
    # algo_ai = rlt2.run_strategy(ai=True)

    rlt = RL_trade_server(data, ta_config, live=live, ai=False, )
    # algo = rlt.run_back_test(ai=False)

    if live:
        while True:
            time.sleep(1200)

    # print("algo / ai:", ((algo_ai - algo) / 100_000) * 100)
