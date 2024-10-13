import datetime
# import sys
import copy
# import sys
# import time
import random
import time

# import sys
import gymnasium as gym

# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.box import Box

# from gymnasium.utils import seeding
# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat
import math
import torch as th

from prettytable import PrettyTable
# from RL_Futures_Visualization import RLVisualizer
from RL_Tools import DataCollector


class SingleFuturesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, symbol, worker_id=0, start_step_count=0, predict_mode=False):

        time.sleep(worker_id / 10)
        self.worker_id = worker_id
        file_name = "binance_data/" + symbol + "_trans_data"
        self.df = pd.DataFrame(pd.read_hdf(file_name, "df"))
        if self.worker_id == 0:
            print(self.df.T)
            print("size (ori)", self.df.shape)
            print(self.df.dtypes)

        self.df_rows = self.df.shape[0]
        self.rolling_net_profit = {}
        self.start_step_count = start_step_count
        self.predict_mode = predict_mode

        self.nn_request = False
        self.long_signal = False
        self.short_signal = False

        # self.xinfo = ["", "", "", ""]

        # különböző workerek különböző abalkokkal dolgoznak
        # window_management_types = ["growing", "slideing", "slideing-overlay"]
        # self.window_type = window_management_types[worker_id % 3]
        self.window_type = "slideing-overlay"
        # self.window_type = "slideing"
        # self.window_type = "growing"
        # self.window_type = "full"
        # if self.worker_id == 0:
        #     self.window_type = "full"

        # if symbol == 'ETHUSDT':
        #     a = 10
        #     b = 20
        # else:
        #     a = 50
        #     b = 80

        if self.window_type == "growing":
            self.window_size = int(self.df_rows / 6)
        elif self.window_type == "slideing":
            self.window_size = int(self.df_rows / 5)
        elif self.window_type == "slideing-overlay":
            r_slice = random.randint(50, 80)
            # self.xinfo[0] = r_slice
            self.window_size = int(self.df_rows / r_slice)
            self.df = self.df.iloc[:r_slice * self.window_size]
            self.df_rows = r_slice * self.window_size
            # print(self.window_size, self.df_rows)

        elif self.window_type == "full":
            self.window_size = self.df_rows
        else:
            print("Error: window_type")
            self.window_size = int(self.df_rows / 1)

        # const
        self.rl_steps = 0  # a sb3 steps eltért mert vannak belső loopok step loop, trade loop
        self.reset_count = 0

        self.memory_size = 10
        self.fee_percent = 0.045
        self.initial_cash = 100_000.0

        self.stop_loss_long_percent = 3  # %
        self.stop_loss_long_price = 0

        self.stop_loss_short_percent = 3  # %
        self.stop_loss_short_price = 0

        self.trailer_long_activation_percent = 1  # %
        self.trailer_long_offset_percent = 0.5  # %
        self.trailer_long_activation_price = 0  # %
        self.trailer_long_highest_price = 0
        self.trailer_long_stop_price = 0
        self.trailer_long_active = False

        self.trailer_short_activation_percent = 1  # %
        self.trailer_short_offset_percent = 0.5  # %
        self.trailer_short_activation_price = 0  # %
        self.trailer_short_lowest_price = 0
        self.trailer_short_stop_price = 0
        self.trailer_short_active = False

        self.asset_precision_decimal = 4
        self.data_precision_decimal = 6

        self.action_space = Discrete(3)  # 0 pass, 1 go long, 2 go short

        extra_inline_features = 0
        self.excluded_features = [
            'date',
            # 'ema_fast_long_down_shift',
            # 'ema_slow_long',
            # 'ema_fast_short_up_shift',
            # 'ema_slow_short',
            # 'open',
            # 'high',
            # 'low',
            # 'close',
            # 'ETHUSDTtaker_buy_base_asset_volume',
            # 'ETHUSDTtaker_buy_quote_asset_volume',
        ]
        removed_features = len(self.excluded_features)  #date
        self.features_space = (self.df.shape[1] - removed_features + extra_inline_features) * self.memory_size
        self.observation_space = Box(low=-2_000_000, high=2_000_000, shape=(self.features_space, ))

        # self.features_space = (self.df.shape[1] - removed_features + extra_inline_features)
        # self.observation_space = Box(low=-2_000_000, high=2_000_000, shape=(self.features_space, self.memory_size ))
        # self.observation_space = Dict({
        #     '2d_array': Box(low=0, high=3_000_000, shape=(self.features_space, self.memory_size), dtype=np.float32),
        #     # 2D array (10x10)
        # })


        # end data for tensorboard
        self.portfolio_value_end = self.initial_cash
        self.closed_trade_end = 1000
        self.netprofit_per_closed_trades_end = -60
        self.reward_end = 0


        # reset adja a kezdő értéket
        # self.fee_total = None
        self.position_enter_qty = None
        self.cash = None
        self.market_price = None
        self.position_enter_price = None
        self.step_count = None
        self.data = None
        self.current_date = None
        self.terminal = None
        self.truncated = None
        self.closed_trades_count = None

        self.last_trade_gross_profit = None
        self.last_trade_fee = None
        self.last_trade_steps = None

        # self.closed_trades_profit = None
        self.step_reward = None
        self.position = None  # 1 long, -1 short, 0 None
        self.real_action = None
        self.trade_steps = None
        # self.r_ticket = None
        # self.price_list = None
        self.actual_action_mask = None
        self.end_window = None

        self.cd = DataCollector(initial_size=self.memory_size)
        # erre azért van szükség, mert a tréning elején a reset 2x hívódik meg
        # egyszer, hogy megkapja a kezdő értékeket, és egyszer már a PPO hívja meg a observationért
        # ezt azért csináltam így, hogy a kezdő értékek csak 1 helyen legyenek megadva
        # így kerülöm el, hogy az init és reset eltérő értékekkel indít
        # az end_window ez alól kivétel

        self.set_init_values()
        print(f"Start worket: {self.worker_id +1}, window manager: {self.window_type} {self.window_size} {self.df_rows / self.window_size}")

        # self.reset(options={'from': 'init'})
        # self.reset()

        # if self.worker_id == 0:
        #     df_visualize = pd.DataFrame()
        #     df_visualize['date'] = self.df['date']
        #     df_visualize['open'] = self.df['open']
        #     df_visualize['high'] = self.df['high']
        #     df_visualize['low'] = self.df['low']
        #     df_visualize['close'] = self.df['close']
        #     df_visualize['volume'] = self.df['qtn'] * self.df['close']
        #     df_visualize['action'] = 0
        #     df_visualize['date'] = pd.to_datetime(df_visualize['date'])
        #     df_visualize.set_index('date', inplace=True)
        #
        #     self.visualizer = RLVisualizer(df_visualize, 'Crypto')
        #     # self.visualizer.refresh()

    # def add_rticket(self, ticket):
    #     self.r_ticket.append(ticket)
    #
    # def calculate_rtickets(self, last_trade_profit, portfolio_value):
    #     ret_reward = 0
    #     for t in self.r_ticket:
    #         ticket = self.price_list[t]
    #         t_amount = 0
    #         if ticket["type"] == "fix":
    #             t_amount = float(ticket["value"])
    #         elif ticket["type"] == "percent":
    #             if ticket["base"] == "trade_value":
    #                 if last_trade_profit == 0:
    #                     print("calculate_reward Error (trade_value = 0)", t)
    #                 else:
    #                     t_amount = last_trade_profit * float(ticket["value"]) / 100
    #             elif ticket["base"] == "portfolio_value":
    #                 if portfolio_value == 0:
    #                     print("calculate_reward Error (tradportfolio_valuee_value = 0)", t)
    #                 else:
    #                     t_amount = portfolio_value * float(ticket["value"]) / 100
    #             else:
    #                 print("calculate_reward Error (base)", t)
    #         else:
    #             print("calculate_reward Error (type)", t)
    #
    #         self.price_list[t]["count"] += 1
    #
    #         if ticket["reinforcement"] == "panelty":
    #             self.price_list[t]["amount"] -= t_amount
    #             ret_reward -= t_amount
    #         elif ticket["reinforcement"] == "reward":
    #             self.price_list[t]["amount"] += t_amount
    #             ret_reward += t_amount
    #     self.r_ticket = []
    #     return ret_reward

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

    def seed(self, seed=42):
        seed = random.randint(0,2000)
        # print("Set seed.", seed)
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)

        # Ha CUDA-t használsz
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    def dround(self, value):
        factor = 10 ** self.asset_precision_decimal
        return math.floor(value * factor) / factor

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

    def long(self, price):
        self.position = 1
        self.position_enter_price = price
        self.stop_loss_long_price = price * (1 - (self.stop_loss_long_percent / 100))
        self.trailer_long_activation_price = price * (1 + (self.trailer_long_activation_percent / 100))
        self.position_enter_qty = self.dround(self.cash / price)
        self.trade_steps = 1
        self.real_action = 1

        self.trailer_long_active = False
        self.trailer_short_active = False
        self.trailer_long_stop_price = 0
        self.trailer_short_stop_price = 0
        self.trailer_long_highest_price = 0
        self.trailer_short_lowest_price = 100_000_000

    def short(self, price):
        self.position = -1
        self.position_enter_price = price
        self.stop_loss_short_price = price * (1 + (self.stop_loss_short_percent / 100))
        self.trailer_short_activation_price = price * (1 - (self.trailer_short_activation_percent / 100))
        self.position_enter_qty = self.dround(self.cash / price)
        self.trade_steps = 1
        self.real_action = 3

        self.trailer_long_active = False
        self.trailer_short_active = False
        self.trailer_long_stop_price = 0
        self.trailer_short_stop_price = 0
        self.trailer_long_highest_price = 0
        self.trailer_short_lowest_price = 100_000_000

    def close_all(self, price, source=""):
        # print("close_all", source, self.position)
        fee = 0
        profit = 0
        if self.position == 1:  # close long
            profit = (price - self.position_enter_price) * self.position_enter_qty
            fee = (self.position_enter_qty * price) * (self.fee_percent / 100)
            fee += (self.position_enter_qty * self.position_enter_price) * (self.fee_percent / 100)
            self.real_action = 2

        elif self.position == -1:  # close short
            profit = (self.position_enter_price - price) * self.position_enter_qty
            fee = (self.position_enter_qty * price) * (self.fee_percent / 100)
            fee += (self.position_enter_qty * self.position_enter_price) * (self.fee_percent / 100)
            self.real_action = 4
            # print("close_all -1 profitcalc: ",profit, fee )

        else:
            print("BUG: position not 1 or -1.")

        # self.fee_total += fee
        # self.fee_total = round(self.fee_total, 2)

        self.last_trade_gross_profit = profit
        self.last_trade_fee = fee
        # self.closed_trades_profit += self.last_trade_profit
        self.cash += (profit - fee)
        self.cash = round(self.cash, 2)

        self.position = 0
        self.closed_trades_count += 1
        self.position_enter_qty = 0
        self.position_enter_price = 0
        self.trade_steps = 0

        self.stop_loss_long_price = 0
        self.stop_loss_short_price = 0
        self.trailer_long_activation_price = 0
        self.trailer_short_activation_price = 0
        self.trailer_long_highest_price = 0
        self.trailer_short_lowest_price = 100_000_000

    def create_state_2d(self):
        idf = pd.DataFrame(self.df[self.step_count-self.memory_size:self.step_count])

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
                raise ValueError(f"A dictionary {key} kulcshoz tartozó értéke nem DataFrame vagy 1D numpy array", type(value))

        # ret = result_df.to_numpy(dtype=np.float64).flatten().round(data_precision_decimal).tolist()
        # ret = np.reshape(ret,(self.features_space, self.memory_size))
        # ret = result_df.to_numpy(dtype=np.float64).T
        # ret = result_df.to_numpy(dtype=np.float64)
        # ret = np.expand_dims(ret.T, axis=0)
        ret = result_df.T.values.flatten().tolist()
        return ret

    # @staticmethod
    # def covert_1d(d: dict):
    #     flat_list = []
    #
    #     for value in d.values():
    #         if isinstance(value, list):
    #             flat_list.extend(value)
    #         elif isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
    #             flat_list.extend(value.values.flatten())
    #         else:
    #             flat_list.append(value)
    #
    #     return np.array(flat_list, dtype=np.float64)

    def get_open_position_value(self):
        if self.position == 1:
            ret = (self.market_price - self.position_enter_price) * self.position_enter_qty
        elif self.position == -1:
            ret = (self.position_enter_price - self.market_price) * self.position_enter_qty
        else:
            ret = 0.0
        return round(ret, 2)

    # def create_state_1d(self):
    #     state_dict = {
    #         "position": self.position,
    #         "trade_steps": self.trade_steps,
    #         "data": self.data.drop(["date"]),
    #         "potfolio_value": self.get_portfolio_value() / self.initial_cash,
    #         "open_position_value": self.get_open_position_value() / self.initial_cash,
    #     }
    #     return self.covert_1d(state_dict)

    def get_portfolio_value(self):
        ret = 0.0
        if self.position == 0:
            ret = self.cash
        elif self.position == 1:
            ret = self.cash + ((self.market_price - self.position_enter_price) * self.position_enter_qty)
        elif self.position == -1:
            ret = self.cash + ((self.position_enter_price - self.market_price) * self.position_enter_qty)
        return round(ret, 2)

    def get_portfolio_value_end(self):
        return round(self.portfolio_value_end, 4)

    def get_reward_end(self):
        return round(self.reward_end, 4)

    def get_closed_trade_end(self):
        return round(self.closed_trade_end, 4)

    def get_netprofit_per_closed_trades_end(self):
        return round(self.netprofit_per_closed_trades_end, 4)

    # def get_netprofit_per_closed_trades(self):
    #     return round(self.collect.last_trade_net_profit.sum() / (self.trades_count + 0.01), 4)
    #
    # def get_closed_trades(self):
    #     return self.trades_count
    #
    # def get_reward_spct(self):
    #     return round(self.collect.reward.sum() / (self.trades_count + 0.01), 4)
    #
    # def get_reward(self):
    #     return self.collect.reward.sum()

    def trade(self, long_signal, short_signal):

        self.trade_steps = 1

        # isleep = 0
        ibreak = False
        while True:
            # print(self.step_count)

            # data = self.df.loc[self.step_count, :]
            close_price = self.market_price = self.df.close[self.step_count]
            # if self.worker_id == 0:
            #     print(data['date'], self.step_count, step_count)

            open_price = self.df.open[self.step_count]

            # print("- " * 25)
            # print("trade loop", data['date'], self.step_count, step_count,
            #       f"long: {long_signal}, short: {short_signal} position: {self.position}")
            #
            # print(f"close_price {close_price}, open_price {open_price}")
            # print(f"trailer_short_active {self.trailer_short_active}, trailer_short_active {self.trailer_short_active}")
            # print(f"stop_loss_long_price {self.stop_loss_long_price}, stop_loss_short_price {self.stop_loss_short_price}")
            # print(f"trailer_long_activation_price {self.trailer_long_activation_price}, trailer_short_activation_price {self.trailer_short_activation_price}")
            # print(f"trailer_long_stop_price {self.trailer_long_stop_price}, trailer_short_stop_price {self.trailer_short_stop_price}")

            if long_signal:
                self.long(open_price)
                self.position = 1
                long_signal = False
                short_signal = False
            elif self.position == 1 and close_price < self.stop_loss_long_price:
                self.close_all(close_price, "stop_loss_long")
                return_type = "stop_loss_long"
                ibreak = True
            elif self.position == 1 and self.trailer_long_active and close_price < self.trailer_long_stop_price:
                self.close_all(close_price, "trailer_long_stop")
                return_type = "trailer_long_stop"
                ibreak = True

            if not ibreak and self.position == 1 and close_price > self.trailer_long_activation_price and not self.trailer_long_active:
                self.trailer_long_active = True

            if not ibreak and self.position == 1 and self.trailer_long_active:
                self.trailer_long_highest_price = max(self.trailer_long_highest_price, close_price)
                self.trailer_long_stop_price = self.trailer_long_highest_price * (
                            1 - (self.trailer_long_offset_percent / 100))

            if short_signal:
                self.short(open_price)
                long_signal = False
                short_signal = False
                self.position = -1
            elif self.position == -1 and close_price > self.stop_loss_short_price:
                self.close_all(close_price, "stop_loss_short")
                return_type = "stop_loss_short"
                ibreak = True
            elif self.position == -1 and self.trailer_short_active and close_price > self.trailer_short_stop_price:
                self.close_all(close_price, "trailer short stop")
                return_type = "trailer_short_stop"
                ibreak = True

            if not ibreak and self.position == -1 and close_price < self.trailer_short_activation_price and not self.trailer_short_active:
                self.trailer_short_active = True

            if not ibreak and self.position == -1 and self.trailer_short_active:
                self.trailer_short_lowest_price = min(self.trailer_short_lowest_price, close_price)
                self.trailer_short_stop_price = self.trailer_short_lowest_price * (1 + (self.trailer_short_offset_percent / 100))

            self.cd.trade_steps.add(self.trade_steps)
            self.cd.position.add(self.position)
            # self.cd.current_date.add(data['date'])
            self.cd.portfolio_value.add(self.get_portfolio_value())
            self.cd.open_position.add(self.get_open_position_value())

            self.cd.real_action.add(self.real_action)
            self.cd.action.add(self.real_action)
            self.real_action = 0

            self.cd.close_price.add(close_price)
            self.cd.open_price.add(open_price)

            self.cd.last_trade_net_profit.add(self.last_trade_gross_profit - self.last_trade_fee)
            # self.step_reward = self.last_trade_gross_profit - self.last_trade_fee
            self.step_reward = self.calc_step_reward()
            self.cd.reward.add(self.step_reward)

            self.cd.last_trade_gross_profit.add(self.last_trade_gross_profit)
            self.cd.last_trade_fee.add(self.last_trade_fee)
            self.last_trade_gross_profit = 0
            self.last_trade_fee = 0

            self.step_count += 1
            self.trade_steps += 1

            if ibreak:
                break

            if self.step_count >= min(self.end_window, self.df_rows - 1):
                return_type = "end_window"
                break

        # print(" trade loop end - --------")

        return return_type

    def get_rolling_net_profit_mean(self):

        n = 0
        sub_profit = 0
        for i in self.rolling_net_profit:
            n += 1
            sub_profit += self.rolling_net_profit[i]

        if n > 0:
            return sub_profit / n
        else:
            return n


    def end_job(self):
        # self.portfolio_value_end = self.get_portfolio_value()
        # self.closed_trade_end = self.closed_trades_count
        # self.netprofit_per_closed_trades_end = self.cd.last_trade_net_profit.sum() / (
        #         self.closed_trades_count + 0.01)
        # self.reward_end = (self.cd.reward.sum())
        self.rolling_net_profit[str(self.end_window)] = self.cd.last_trade_net_profit.sum()

        mply = self.df_rows / self.window_size
        self.portfolio_value_end = (self.cd.last_trade_net_profit.sum() * mply) + self.cash
        self.closed_trade_end = self.closed_trades_count * mply
        self.netprofit_per_closed_trades_end = self.cd.last_trade_net_profit.sum() / (
                self.closed_trades_count + 0.01)
        self.reward_end = self.cd.reward.sum() * mply
        self.render()

    def step(self, action):
        # if self.worker_id == 0:
        #     print(f"step, {self.step_count},  {self.end_window}, {self.window_size}")

        while True:

            # if self.reset_count == 1:
            #     print("reset == 1", self.step_count)

            if not self.nn_request:

                while not self.long_signal and not self.short_signal and self.step_count < self.end_window:
                    # print(self.step_count)
                    # data = self.df.loc[self.step_count, :]

                    # if self.worker_id == 2:
                    #     print(data['date'], self.step_count)
                    close_p = self.market_price = self.df.close[self.step_count]
                    open_p = self.df.open[self.step_count]
                    # print("step loop, not nn_request", data['date'], self.step_count)
                    p0 = self.step_count
                    p1 = p0 - 1

                    # self.long_signal = (
                    #             round(self.df.ema_fast_long_down_shift[p0], 8) > round(self.df.ema_slow_long[p0], 8) and
                    #             round(self.df.ema_fast_long_down_shift[p1], 8) <= round(self.df.ema_slow_long[p1], 8))
                    #
                    # self.short_signal = (
                    #             round(self.df.ema_fast_short_up_shift[p0], 8) < round(self.df.ema_slow_short[p0], 8) and
                    #             round(self.df.ema_fast_short_up_shift[p1], 8) >= round(self.df.ema_slow_short[p1], 8))

                    self.long_signal = (
                            self.df.ema_fast_long_down_shift[p0] > self.df.ema_slow_long[p0] and
                            self.df.ema_fast_long_down_shift[p1] <= self.df.ema_slow_long[p1])

                    self.short_signal = (
                            self.df.ema_fast_short_up_shift[p0] < self.df.ema_slow_short[p0] and
                            self.df.ema_fast_short_up_shift[p1] >= self.df.ema_slow_short[p1])

                    # self.short_signal = False

                    # if self.long_signal:
                    #     print("Long " * 5)

                    self.step_count += 1
                    self.cd.trade_steps.add(0.0)
                    self.cd.position.add(0.0)
                    # self.cd.current_date.add(data['date'])
                    self.cd.portfolio_value.add(self.get_portfolio_value())
                    self.cd.open_position.add(0.0)
                    self.cd.last_trade_net_profit.add(0.0)
                    self.cd.last_trade_gross_profit.add(0.0)
                    self.cd.last_trade_fee.add(0.0)
                    self.cd.real_action.add(0.0)
                    self.cd.close_price.add(close_p)
                    self.cd.open_price.add(open_p)
                    self.cd.reward.add(0.0)
                    self.cd.real_action.add(0.0)
                    self.cd.action.add(0.0)

                if self.step_count >= min(self.end_window, self.df_rows - 1):
                    self.terminal = True
                    self.truncated = True
                    self.end_job()

                    # if self.worker_id == 0:
                    #     # print("step loop end")
                    #     self.end_job()

                else:
                    if self.long_signal:
                        ret_actual_action_mask = np.array([1, 1, 0], dtype=np.int32)
                        # print(ret_actual_action_mask)
                    elif self.short_signal:
                        ret_actual_action_mask = np.array([1, 0, 1], dtype=np.int32)
                    else:
                        ret_actual_action_mask = np.array([1, 0, 0], dtype=np.int32)
                    self.actual_action_mask = ret_actual_action_mask

                self.nn_request = True
                self.rl_steps += 1
                # print(f"Real steps: {self.rl_steps}")

                return (self.create_state_2d(), self.cd.reward.sum(), self.terminal, self.truncated,
                        {
                            # "date": pd.to_datetime(self.current_date),
                            "real_action": self.real_action,
                            "action": action,
                            "step_count": self.step_count,
                            "end_window": self.end_window,
                            "portfolio_value": self.get_portfolio_value(),
                            "last_trade_gross_profit": self.last_trade_gross_profit,
                            "last_trade_fee": self.last_trade_fee,
                            "last_trade_net_profit": self.last_trade_gross_profit - self.last_trade_fee,
                            "market_price": self.market_price,
                            "'action_mask'": self.actual_action_mask,
                        })

            else:

                # print("action", action)

                ret_type = "none"
                if action == 0:  # pass
                    # print("pass", self.worker_id)
                    pass
                elif action == 1:  # go long
                    # print("start long", self.step_count)
                    ret_type = self.trade(self.long_signal, self.short_signal)
                    # if self.worker_id == 0 and self.reset_count == 1:
                    #     print("profit long", ret_type, self.worker_id, self.cd.last_trade_net_profit.sum())
                        # time.sleep(5)
                elif action == 2:  # go short
                    # print("start short", self.step_count)

                    ret_type = self.trade(self.long_signal, self.short_signal)
                    # if self.worker_id == 0 and self.reset_count == 1:
                    #     print("profit short", ret_type, self.worker_id, self.cd.last_trade_net_profit.sum())
                        # time.sleep(5)

                    # print("return trade")
                    # time.sleep(10)

                self.long_signal = False
                self.short_signal = False
                self.nn_request = False

                # print("step loop", ret_type, self.step_count, self.end_window, self.df_rows - 1)

                if self.step_count >= min(self.end_window, self.df_rows - 1):

                    # if self.worker_id == 0:
                    #     # print(f"end trade end, {self.df_rows} {self.step_count}, {self.end_window}, {self.df_rows}")
                    #     # print(self.cd.last_trade_net_profit.sum())
                    #     # print(ret_type)
                    #     # print("sleep")
                    #     # time.sleep(10)
                    self.end_job()

                    self.terminal = True
                    self.truncated = True
                    return (self.create_state_2d(), self.cd.reward.sum(), self.terminal, self.truncated,
                            {
                                # "date": pd.to_datetime(self.current_date),
                                "real_action": self.real_action,
                                "action": action,
                                "step_count": self.step_count,
                                "end_window": self.end_window,
                                "portfolio_value": self.get_portfolio_value(),
                                "last_trade_gross_profit": self.last_trade_gross_profit,
                                "last_trade_fee": self.last_trade_fee,
                                "last_trade_net_profit": self.last_trade_gross_profit - self.last_trade_fee,
                                "market_price": self.market_price,
                                "'action_mask'": self.actual_action_mask,
                            })



            # if self.worker_id == 0:
            #     self.visualizer.set_action(self.collect.real_action.get(numpy=True))
            #     self.visualizer.refresh()


            # if self.predict_mode:
            # df = pd.DataFrame()
            # df["real_action"] = self.collect.real_action.get(numpy=True)[10:]
            # df["last_trade_gross_profit"] = self.collect.last_trade_gross_profit.get(numpy=True)[10:]
            # df["current_date"] = self.collect.current_date.get()[10:]
            # df["market_price"] = self.collect.market_price.get()[10:]
            # df["action"] = self.collect.action.get()[10:]
            # df.to_hdf("RL_Futures/learn_train.hdf5", key='df', mode='w')
            # self.collect.state.save_np("RL_Futures/states")

            # np.save('RL_Futures/real_action_memory.npy', self.collect.real_action.get(numpy=True))
            # np.save('RL_Futures/last_trade_gross_profit_memory.npy', self.collect.last_trade_gross_profit.get(numpy=True))
            # np.save('RL_Futures/current_date_memory.npy', self.collect.current_date.get(numpy=True))

        # if not self.terminal:
        #
        #     if self.position != 0:
        #         self.trade_steps += 1
        #
        #     self.last_trade_gross_profit = 0.0
        #     self.last_trade_fee = 0.0
        #
        #     self.data = self.df.loc[self.step_count, :]
        #     self.current_date = self.data["date"]
        #     # row = self.df.values[self.step_count]
        #     self.market_price = self.data["close"]
        #
        #     if action == 0:  # hold/
        #         if self.position == 1:
        #             self.add_rticket("hold_position")
        #         elif self.position == 0:
        #             self.add_rticket("hold_position")
        #         elif self.position == -1:
        #             self.add_rticket("hold_position")
        #     if action == 1:  # long
        #         if self.position == 1:
        #             self.add_rticket("repeat_long")
        #         elif self.position == 0:
        #             self.long()
        #         elif self.position == -1:
        #             self.add_rticket("turn_over")
        #     elif action == 2:  # short
        #         if self.position == 1:
        #             self.add_rticket("turn_over")
        #         elif self.position == 0:
        #             self.short()
        #         elif self.position == -1:
        #             self.add_rticket("repeat_short")
        #     elif action == 3:  # close
        #         if self.position == 1:
        #             self.close_all()
        #             self.add_rticket("close_trade")
        #         elif self.position == 0:
        #             self.add_rticket("close_empty")
        #         elif self.position == -1:
        #             self.close_all()
        #             self.add_rticket("close_trade")

            # reward calculator
            # self.step_reward = self.calc_step_reward()

            # memory
            # self.collect.last_trade_net_profit.add(self.last_trade_gross_profit - self.last_trade_fee)
            # self.collect.last_trade_gross_profit.add(self.last_trade_gross_profit)
            # self.collect.last_trade_fee.add(self.last_trade_fee)
            # self.cd.real_action.add(self.real_action)
            # self.cd.market_price.add(self.market_price)
            # # self.cd.state.add(self.create_state_2d())
            # self.cd.action.add(action)
            # self.position_funding_panelty = 0.0
            #
            # self.cd.reward.add(self.step_reward)


            # print(self.collect.portfolio_value.get_last())

            #
            # print(action, self.real_action, self.step_count, self.market_price, self.current_date, np.sum(self.create_state_2d()))
            # sys.exit(0)

        # ACTION MASKING
        # self.action_space = Discrete(4)  # 0 hold, 1 long, 2 short, 3 close (lond and close)
        # self.position = None  # 1 long, -1 short, 0 None
        # if self.worker_id == 0:
        #     print(self.position, self.trade_steps)

        # # ret_actual_action_mask = np.array([0, 0, 0, 0])
        # if self.position in [1, -1]:
        #     if self.position == 1:
        #         if self.market_price < self.stop_loss_price_long:
        #             ret_actual_action_mask = np.array([0, 0, 0, 1], dtype=np.int32)
        #         elif self.market_price > self.take_profit_price_long:
        #             ret_actual_action_mask = np.array([0, 0, 0, 1], dtype=np.int32)
        #         else:
        #             ret_actual_action_mask = np.array([1, 0, 0, 1], dtype=np.int32)
        #     else:
        #         if self.market_price > self.stop_loss_price_short:
        #             ret_actual_action_mask = np.array([0, 0, 0, 1], dtype=np.int32)
        #         elif self.market_price < self.take_profit_price_short:
        #             ret_actual_action_mask = np.array([0, 0, 0, 1], dtype=np.int32)
        #         else:
        #             ret_actual_action_mask = np.array([1, 0, 0, 1], dtype=np.int32)
        # else:
        #     ret_actual_action_mask = np.array([1, 1, 1, 0], dtype=np.int32)
        # self.actual_action_mask = ret_actual_action_mask



    def action_mask(self, env):
        return self.actual_action_mask

    @staticmethod
    def exponential_decay(value, distance):
        k = 0.2
        reduced_value = value * math.exp(-k * distance)

        return reduced_value

    def calc_step_reward(self):
        net_profit = self.last_trade_gross_profit - self.last_trade_fee
        ret = 0
        if self.real_action in [2, 4]:
            ret = self.exponential_decay(net_profit, self.last_trade_steps)
            self.last_trade_steps = 0
        return ret

    def set_init_values(self):
        self.nn_request = False
        self.real_action = 0
        self.trade_steps = 0
        self.last_trade_steps = 0
        self.position_enter_qty = 0
        self.position_enter_price = 0
        self.cash = self.initial_cash
        self.market_price = 0
        self.actual_action_mask = np.array([1, 1, 1], dtype=np.int32)

        # if self.predict_mode:
        #     self.step_count = self.memory_size+1
        # else:

        self.terminal = False
        self.truncated = False
        self.closed_trades_count = 0
        self.step_reward = 0.0
        self.position = 0  # 1 long, -1 short, 0 None
        # self.r_ticket = []
        self.last_trade_gross_profit = 0.0
        self.last_trade_fee = 0.0
        # self.position_funding_panelty = 0.0
        # self.price_list = copy.deepcopy(self.price_list_base)

        self.stop_loss_long_price = 0
        self.stop_loss_short_price = 0

        self.trailer_long_activation_price = 0  # %
        self.trailer_long_highest_price = 0
        self.trailer_long_stop_price = 0
        self.trailer_long_active = False

        self.trailer_short_activation_price = 0  # %
        self.trailer_short_lowest_price = 0
        self.trailer_short_stop_price = 0
        self.trailer_short_active = False

    # def print_state(self, text):
    #     print(f"worker: {self.worker_id} {text} stepc: {self.step_count} "
    #           f"window size: {self.window_size} end_window: {self.end_window} "
    #           f"df_row: {self.df_rows} \n")

    def reset(self, seed=None, options=None):
        # print(f"{self.worker_id}, {self.reset_count} \n\n")
        # print("reset", self.worker_id, self.step_count, self.end_window, self.reset_count)

        self.reset_count += 1

        # if options is None:
            # print("options: None")
        # else:
            # print("options:", options)

        self.set_init_values()

        if self.predict_mode:
            self.end_window = self.df_rows - 1
            self.step_count = self.memory_size
        else:
            if self.window_type == "growing":
                if self.step_count is None:
                    self.step_count = self.memory_size
                    self.end_window = self.window_size
                elif self.end_window == self.df_rows - 1:
                    self.step_count = self.memory_size
                    self.end_window = self.window_size
                else:
                    self.step_count = self.memory_size
                    self.end_window += self.window_size
                    self.end_window = min(self.end_window, self.df_rows - 1)
            elif self.window_type == "slideing":
                if self.step_count is None:
                    rnd_start = random.randint(0, 3) * self.window_size
                    self.step_count = self.memory_size + rnd_start
                    self.end_window = self.window_size + rnd_start
                elif self.end_window == self.df_rows - 1:
                    self.step_count = self.memory_size
                    self.end_window = self.window_size
                else:
                    self.end_window += self.window_size
                    self.end_window = min(self.end_window, self.df_rows - 1)
            elif self.window_type == "slideing-overlay":
                # self.print_state("slideing-overlay")

                if self.step_count is None:
                    # self.print_state("None")
                    rnd_start = random.randint(0, 30) * self.window_size
                    # self.xinfo[1] = rnd_start
                    self.step_count = self.memory_size + rnd_start
                    self.end_window = self.window_size + rnd_start
                elif self.end_window == self.df_rows - 1:
                    # self.print_state(" self.df_rows - 1")
                    self.step_count = self.memory_size
                    self.end_window = self.window_size
                else:
                    # self.print_state("else")

                    # print(f"{self.worker_id}, else \n\n")
                    self.step_count -= int(self.window_size / 2)
                    self.end_window = self.step_count + self.window_size
                    self.end_window = min(self.end_window, self.df_rows - 1)

            elif self.window_type == "full":
                self.step_count = self.memory_size
                self.end_window = self.df_rows - 1

        self.cd.reset()
        # self.print_state("end of reset")
        self.data = self.df.loc[self.step_count, :]
        # if self.worker_id == 0:
        #     print(f"reset count {self.reset_count}")

        # print("reset end", self.step_count, self.end_window, self.df_rows)
        return self.create_state_2d(), {}

    def render(self, mode='human'):
        # qc = self.check_trade_sequence_quality(self.cd.real_action.get())

        # rtickets = "\n".join([f"  {key}: {value['count']} / {value['amount']:,.2f}" for key, value in self.price_list.items()])
        gross_profit = self.cd.last_trade_gross_profit.sum()
        total_fee = self.cd.last_trade_fee.sum()
        net_profit = gross_profit - total_fee

        count_1 = np.count_nonzero(self.cd.real_action.get() == 1)
        count_2 = np.count_nonzero(self.cd.real_action.get() == 2)
        count_3 = np.count_nonzero(self.cd.real_action.get() == 3)
        count_4 = np.count_nonzero(self.cd.real_action.get() == 4)

        # print(self.collect.last_trade_net_profit.describe())
        # print('- ' * 50)

        self.qprint(f"Render: {self.worker_id} {self.window_type}",
                    f"               Datetime: {datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')}",
                    f"       Mean sub profits: {self.get_rolling_net_profit_mean():,.2f}",
                    f"                 Reward: {self.cd.reward.sum():,.2f}",
                    f"    Portfolio value end: {self.portfolio_value_end:,.2f}",
                    f"     Total gross profit: {gross_profit:,.2f}",
                    f"              Total fee: {total_fee:,.2f}",
                    f"             Net profit: {net_profit:,.2f}",
                    f"          Closed trades: {self.closed_trades_count} / {int(self.df_rows / (self.closed_trades_count + 0.01))} min",
                    f"AVG net profit / trades: {net_profit / (self.closed_trades_count + 0.01):,.2f}",
                    f"             Step count: {self.step_count}",
                    f"             End window: {self.end_window}",
                    f"       Num of open long: {count_1}",
                    f"      Num of close long: {count_2}",
                    f"      Num of open short: {count_3}",
                    f"     Num of close short: {count_4}",
                    # f"       Real action Qchk: {'OK' if qc else 'Action order BUG'}",
                    )
        return self.create_state_2d()

