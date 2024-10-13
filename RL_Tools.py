# import random
import time
import datetime
import os
import pandas as pd
from binance.client import Client
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import pandas_ta as ta
from scipy import stats

# from numba import int32, float32
# from numba.experimental import jitclass


# spec = [
#     ('initial_size', int32),
#     ('initial_value', float32),
#     ('data', float32[:]),
#     ('memory_size', int32),
# ]
#
#
# @jitclass(spec)
class Collector:
    def __init__(self, initial_size, initial_value, length=1000):
        self.length = np.int32(length)
        self.initial_size = np.int32(initial_size)
        self.initial_value = np.float32(initial_value)

        self.data = np.full(self.length, None)
        self.data[:self.initial_size] = self.initial_value
        self.step_count = self.initial_size
        # self.data = np.full(self.initial_size, self.initial_value, dtype=np.float32)
        self.memory_size = np.int32(initial_size)

    def add(self, value):
        self.data[self.step_count] = value
        self.step_count += 1

    def reset(self):
        # self.data = np.full(self.initial_size, self.initial_value, dtype=np.float32)

        self.data = np.full(self.length, None)
        self.data[:self.initial_size] = self.initial_value

    def get(self):
        return self.data[:self.step_count]

    def sum(self):
        return np.sum(self.data[:self.step_count])

    def mean(self):
        return np.mean(self.data[:self.step_count])

    def save_np(self, fname):
        np.save(fname, self.data[:self.step_count])
        return

    def get_last(self, precision=6):
        return self.data[self.step_count-self.memory_size:self.step_count].round(precision)

    def describe(self):
        stats = {
            'count': len(self.data[:self.step_count]),
            'mean': np.mean(self.data[:self.step_count]),
            'std': np.std(self.data[:self.step_count], ddof=1),  # Use sample standard deviation
            'min': np.min(self.data[:self.step_count]),
            '25%': np.percentile(self.data[:self.step_count], 25),
            '50%': np.percentile(self.data[:self.step_count], 50),  # This is the median
            '75%': np.percentile(self.data[:self.step_count], 75),
            'max': np.max(self.data[:self.step_count])
        }

        return stats


class DataCollector:
    def __init__(self, initial_size=10, initial_value=0.0, initial_cash=100_000.0, length=100000):
        self.reward = Collector(initial_size, initial_value, length)
        self.portfolio_value = Collector(initial_size, initial_cash, length)
        self.position = Collector(initial_size, initial_value, length)
        self.trade_steps = Collector(initial_size, initial_value, length)
        self.open_position = Collector(initial_size, initial_value, length)
        self.last_trade_net_profit = Collector(initial_size, initial_value, length)
        self.last_trade_gross_profit = Collector(initial_size, initial_value, length)
        self.last_trade_fee = Collector(initial_size, initial_value, length)
        self.action = Collector(initial_size, initial_value, length)
        self.real_action = Collector(initial_size, initial_value, length)
        # self.current_date = Collector(initial_size, initial_date)
        self.close_price = Collector(initial_size, initial_value, length)
        self.open_price = Collector(initial_size, initial_value, length)
        # self.state = Collector(0, [])

    def reset(self):
        self.reward.reset()
        self.portfolio_value.reset()
        self.position.reset()
        self.trade_steps.reset()
        self.open_position.reset()
        self.last_trade_net_profit.reset()
        self.last_trade_gross_profit.reset()
        self.last_trade_fee.reset()
        self.real_action.reset()
        # self.current_date.reset()
        self.close_price.reset()
        self.open_price.reset()
        # self.state.reset()
        self.action.reset()


class TensorboardCallback(BaseCallback):
    """
    Egyedi callback a TensorBoard-hoz, amely további értékeket naplóz.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Egyedi értékek naplózása

        # custom_env = self.training_env.envs[0]

        # portfolio_value = self.training_env.get_attr('portfolio_value', 0)[0]

        n = 0
        total_subs = 0
        for sub_net_profit in self.training_env.env_method('get_rolling_net_profit_mean', indices=0):
            n += 1
            total_subs += sub_net_profit

        if n > 0:
            mean_subs = total_subs / n
        else:
            mean_subs = 0

        portfolio_value = self.training_env.env_method('get_portfolio_value_end', indices=0)[0]  # Az első környezet
        reward = self.training_env.env_method('get_reward_end', indices=0)[0]  # Az első környezet
        netprofit_per_closed_trades = self.training_env.env_method('get_netprofit_per_closed_trades_end', indices=0)[0]  # Az első környezet
        closed_trades = self.training_env.env_method('get_closed_trade_end', indices=0)[0]  # Az első környezet

        # Elérjük a custom_env attribútumát: last_action
        # portfolio_value = custom_env.get_portfolio_value()

        # Naplózzuk az utolsó akciót a TensorBoard-ba
        if mean_subs is not None:
            self.logger.record('custom_metric/mean_sub_net_profits', mean_subs)

        if closed_trades is not None:
            self.logger.record('custom_metric/closed_trade_end', closed_trades)

        if netprofit_per_closed_trades is not None:
            self.logger.record('custom_metric/netprofit_per_closed_trades_end', netprofit_per_closed_trades)

        if portfolio_value is not None:
            self.logger.record('custom_metric/portfolio_value_at_dataset_end', portfolio_value)

        if reward is not None:
            self.logger.record('custom_metric/reward_end', reward)

        return True


class TimeEstimatorCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super(TimeEstimatorCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.elapsed_time = time.time()
        self.estimated_remaining_time = None

    def _on_training_start(self):
        self.start_time = time.time()  # Start timer at the beginning of training

    def _on_step(self) -> bool:
        self.elapsed_time = time.time() - self.start_time
        current_timesteps = self.model.num_timesteps
        fps = current_timesteps / self.elapsed_time if self.elapsed_time > 0 else float('inf')
        remaining_timesteps = self.total_timesteps - current_timesteps
        self.estimated_remaining_time = remaining_timesteps / fps if fps > 0 else float('inf')
        return True

    def _on_rollout_end(self):
        if self.verbose > 0:
            remaining_time = datetime.timedelta(seconds=self.estimated_remaining_time)
            estimated_end_time = datetime.datetime.now() + remaining_time
            print(f"Estimated remaining time: {estimated_end_time}")
        return


def df_split(df, start, end, target_date_col="date"):
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    # data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def binance_download(symbol, from_dt, cutoff_dt=None, refresh=False, futures=False, interval='1h'):
    possible_interval = ['1m', '3m', '5m', '15m', '30m',
                         '1h', '2h', '4h', '6h', '8h', '12h',
                         '1d', '3d',
                         '1w',
                         '1M']

    if interval not in possible_interval:
        print("Selected intervall not exist.", interval)
        return

    print("")
    print("Binance_download->", from_dt.strftime("%Y-%m-%d %H:%M:%S"), symbol, interval)
    if futures:
        futures_text = "_futures"
    else:
        futures_text = ""

    file_name = "binance_data/" + symbol + interval + futures_text + "_" + from_dt.strftime("%Y-%m-%d_%H-%M-%S")
    if refresh and os.path.exists(file_name):
        os.remove(file_name)

    if os.path.exists(file_name):
        print("   Load data from cache.")
        df_ret = pd.read_hdf(file_name, "df")
    else:
        from_dt = from_dt.strftime("%Y-%m-%d %H:%M:%S")
        api_key = 'YOUR_API_KEY'
        api_secret = 'YOUR_API_SECRET'
        client = Client(api_key, api_secret)
        # adatok letöltéséhez kell a client objektum, de nem kell belépni

        if futures:
            klines = client.futures_historical_klines(symbol, interval, from_dt)
        else:
            klines = client.get_historical_klines(symbol, interval, from_dt)

        df = pd.DataFrame(klines, columns=[
            'date', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        # data.set_index('date', inplace=True)
        # df['qtn'] = df["volume"].astype(np.float64) / df["close"].astype(np.float64)
        # df_ret = pd.DataFrame()
        df_ret = df[['date', 'open', 'high', 'low', 'close', 'volume',
                     'number_of_trades',
                     # 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
                     ]].astype(np.float64)

        df_ret.rename(columns={
            'date': 'date',
            'open': symbol + '_open',
            'high': symbol + '_high',
            'low': symbol + '_low',
            'close': symbol + '_close',
            'volume': symbol + '_volume',
            'number_of_trades': symbol + '_number_of_trades',
            # 'taker_buy_base_asset_volume': symbol + 'taker_buy_base_asset_volume',
            # 'taker_buy_quote_asset_volume': symbol + 'taker_buy_quote_asset_volume',
        }, inplace=True)

        # data['volume'] = data['volume'].astype(np.int64)

        # df.drop(['volume',
        #          'open',
        #          'quote_asset_volume',
        #          'taker_buy_base_asset_volume',
        #          'taker_buy_quote_asset_volume']
        #         , axis=1, inplace=True)

        df_ret['date'] = pd.to_datetime(df_ret['date'], unit='ms')
        # data['Volume'] = data['Volume'].astype(np.float32)
        df_ret["date"] = df_ret.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        # data['tic'] = symbol
        df_ret = df_ret.dropna()
        # df_ret = df_ret.reset_index(drop=True)
        df_ret = df_ret.sort_values(by=["date"]).reset_index(drop=True)
        df_ret.to_hdf(file_name, key='df', mode='w')

        client.close_connection()
        del client

    if cutoff_dt:

        print("   Cutoff from:", cutoff_dt)
        df_ret['date2'] = pd.to_datetime(df_ret['date'])

        # cutoff_dt = pd.Timestamp(cutoff_dt.strftime("%Y-%m-%d %H:%M:%S"), unit='ms')
        # df_ret.set_index('date2', inplace=True)

        df_ret = df_ret[df_ret.date2 <= cutoff_dt]
        # df_ret = df_ret.reset_index(drop=True)
        df_ret.drop(columns='date2', inplace=True)
        df_ret = df_ret.sort_values(by=["date"]).reset_index(drop=True)

    return df_ret


def get_dd_du(arr):
    cumulative_sum = 0
    min_sum = 10000000
    max_sum = 0
    for value in arr:
        cumulative_sum += value  # Add the current value to the cumulative sum
        min_sum = min(min_sum, cumulative_sum)
        max_sum = max(max_sum, cumulative_sum)

    return min_sum, max_sum


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


def count_long_short(df):
    l_count = 0
    s_count = 0
    for step_count in range(len(df)):

        p0 = step_count
        p1 = p0 - 1
        long_signal = (
                df.ema_fast_long_down_shift[p0] > df.ema_slow_long[p0] and
                df.ema_fast_long_down_shift[p1] <= df.ema_slow_long[p1])

        short_signal = (
                df.ema_fast_short_up_shift[p0] < df.ema_slow_short[p0] and
                df.ema_fast_short_up_shift[p1] >= df.ema_slow_short[p1])

        if long_signal:
            l_count += 1

        if short_signal:
            s_count += 1

    return l_count, s_count


def add_diff_ratio(df, columns):
    for col in columns:
        if col in df.columns:
            df[f"{col}_diff"] = df[col].diff()
            df[f"{col}_diff_ratio"] = df[f"{col}_diff"] / df[col]
            df.drop(columns=[f"{col}_diff"], inplace=True)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame")

    return df


def get_dataset(symbol, from_dt, cutoff_dt, ta_config, dropna=False, refresh=False):
    # from_dt = datetime.datetime(year=2024, month=6, day=1, hour=0, minute=0)
    # from_dt = datetime.datetime(year=2024, month=9, day=3, hour=0, minute=0)
    # cutoff_dt = datetime.datetime(year=2024, month=9, day=1, hour=0, minute=0)
    # cutoff_dt = None
    interval = "1m"
    # interval = "5m"
    # interval = "15m"
    # refresh = True
    futures = True
    # contras = True
    contras = False
    contra_symbol1 = 'ETHUSDT'
    contra_symbol2 = 'BTCUSDT'
    add_stat = False
    add_ta = True
    diff_set = False

    df_ori = binance_download(symbol,
                              from_dt=from_dt,
                              cutoff_dt=cutoff_dt,
                              refresh=refresh,
                              futures=futures,
                              interval=interval)  # in minute from now()

    if contras:

        df_contra1 = binance_download(contra_symbol1,
                                      from_dt=from_dt,
                                      cutoff_dt=cutoff_dt,
                                      refresh=refresh,
                                      futures=futures,
                                      interval=interval)  # in minute from now()

        df_contra2 = binance_download(contra_symbol2,
                                      from_dt=from_dt,
                                      cutoff_dt=cutoff_dt,
                                      refresh=refresh,
                                      futures=futures,
                                      interval=interval)  # in minute from now()

        df = pd.merge(df_ori, df_contra1, on='date', how='inner')
        df = pd.merge(df, df_contra2, on='date', how='inner')

    else:
        df = df_ori

    df.rename(columns={
        symbol + '_close': 'close',
        symbol + '_open': 'open',
        symbol + '_high': 'high',
        symbol + '_low': 'low',
        symbol + '_volume': 'volume',
        symbol + '_number_of_trades': 'number_of_trades',
    }, inplace=True)

    if diff_set:

        df = add_diff_ratio(df, [
            'open', 'high', 'low', 'close',
            contra_symbol1 + '_open', contra_symbol1 + '_high', contra_symbol1 + '_low', contra_symbol1 + '_close',
            contra_symbol2 + '_open', contra_symbol2 + '_high', contra_symbol2 + '_low', contra_symbol2 + '_close',
            'qtn', contra_symbol1 + '_qtn', contra_symbol2 + '_qtn',
            'number_of_trades', contra_symbol1 + '_number_of_trades', contra_symbol2 + '_number_of_trades',
        ])

        df['hpl'] = df['high'] / df['low']
        df[contra_symbol1 + '_hpl'] = (df[contra_symbol1 + '_high'] / df[contra_symbol1 + '_low']) - 1
        df[contra_symbol2 + '_hpl'] = (df[contra_symbol2 + '_high'] / df[contra_symbol2 + '_low']) - 1

        df.drop([
            contra_symbol1 + '_open', contra_symbol1 + '_high', contra_symbol1 + '_low', contra_symbol1 + '_close',
            contra_symbol1 + '_qtn',
            contra_symbol2 + '_open', contra_symbol2 + '_high', contra_symbol2 + '_low', contra_symbol2 + '_close',
            contra_symbol2 + '_qtn',
            contra_symbol1 + '_number_of_trades', contra_symbol2 + '_number_of_trades'

        ], axis=1, inplace=True)

    if add_ta:
        # for lgt in range(3, 15, 7):
        #     df['EMA_'+str(lgt)] = ta.ema(df['close'], length=lgt)

        # base = "AVAX"
        # quote = "USDT"
        # config = [['base', base],
        #           ['quote', quote],
        #           ['is_long', 1], ['ema_fast_long', 53], ['ema_slow_long', 265], ['ema_fast_long_down_shift', 99],
        #           ['is_stop_loss_long', 1], ['stop_loss_percent_long', 64],
        #           ['is_trailer_long', 1], ['trailer_long_enter_percent', 49], ['trailer_long_offset', 47],
        #           ['is_short', 1],
        #           ['ema_fast_short', 39], ['ema_slow_short', 397],
        #           ['ema_fast_short_up_shift', 101], ['is_stop_loss_short', 1], ['stop_loss_percent_short', 48],
        #           ['is_trailer_short', 1], ['trailer_short_enter_percent', 38],
        #           ['trailer_short_offset', 85], ['max_trade_steps', 47]
        #           ]

        # df['ema_fast_long_down_shift'] = ta.ema(df['close'], length=53) * 0.99
        # df['ema_slow_long'] = ta.ema(df['close'], length=265)
        #
        # df['ema_fast_short_up_shift'] = ta.ema(df['close'], length=39) * 1.01
        # df['ema_slow_short'] = ta.ema(df['close'], length=397)

        # base = "ADA"
        # quote = "USDT"
        # config = [['base', base],
        #           ['quote', quote],
        #           ['is_long', 1], ['ema_fast_long', 65], ['ema_slow_long', 346], ['ema_fast_long_down_shift', 99],
        #           ['is_stop_loss_long', 1], ['stop_loss_percent_long', 168], ['is_trailer_long', 1],
        #           ['trailer_long_enter_percent', 96], ['trailer_long_offset', 87], ['is_short', 1],
        #           ['ema_fast_short', 35],
        #           ['ema_slow_short', 331], ['ema_fast_short_up_shift', 101], ['is_stop_loss_short', 1],
        #           ['stop_loss_percent_short', 138], ['is_trailer_short', 1], ['trailer_short_enter_percent', 72],
        #           ['trailer_short_offset', 15], ['max_trade_steps', 59]
        #
        #           ]

        ta_fll = ta_config[symbol]['ema_fast_long_down_shift_length']
        ta_fls = ta_config[symbol]['ema_fast_long_down_shift']
        ta_sll = ta_config[symbol]['ema_slow_long_length']
        ta_fsl = ta_config[symbol]['ema_fast_short_up_shift_length']
        ta_fss = ta_config[symbol]['ema_fast_short_up_shift']
        ta_ssl = ta_config[symbol]['ema_slow_short_length']

        df['ema_fast_long_down_shift'] = ta.ema(df['close'], length=ta_fll) * ta_fls
        df['ema_slow_long'] = ta.ema(df['close'], length=ta_sll)

        df['ema_fast_short_up_shift'] = ta.ema(df['close'], length=ta_fsl) * ta_fss
        df['ema_slow_short'] = ta.ema(df['close'], length=ta_ssl)

        print("Num of long, short:", count_long_short(df))

        # df['EMA_25'] = ta.ema(df['close'], length=25) * 0.95
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        df[['MACD', 'MACD_S', 'MACD_H']] = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df.drop(['MACD', 'MACD_S'], axis=1, inplace=True)  # ez nem kell mert a MACD_H kifejezi a másik kettőt is
        df['MACD_H'] = df['MACD_H'] * 100

        # df['ATRr_14'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)

    if add_stat:
        # statisztikai elemzés

        # Typical Price
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3

        window_size = 200
        df['tp_Std_Dev'] = df['tp'].rolling(window=window_size).std()

        df['Variance'] = df['tp'].rolling(window=window_size).var()
        # Átlagos Abszolút Eltérés (MAD)
        df['tp_MAD'] = df['tp'].rolling(window=window_size).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

        # Tartomány
        df['tp_Range'] = df['tp'].rolling(window=window_size).apply(lambda x: np.max(x) - np.min(x))

        # Interkvartilis Terjedelem (IQR)
        df['tp_IQR'] = df['tp'].rolling(window=window_size).apply(lambda x: stats.iqr(x))

        # Autokorreláció (lag=1)
        def rolling_autocorr(x):
            if len(x) < 2:
                return np.nan
            return x.autocorr(lag=1)
        df['tp_Autocorrelation'] = df['tp'].rolling(window=window_size).apply(rolling_autocorr, raw=False)

        # Kurtózis
        df['tp_Kurtosis'] = df['tp'].rolling(window=window_size).apply(lambda x: stats.kurtosis(x), raw=False)

        # Ferdeség
        df['tp_Skewness'] = df['tp'].rolling(window=window_size).apply(lambda x: stats.skew(x), raw=False)

        df.drop(['tp'], axis=1, inplace=True)  # ez nem kell mert a MACD_H kifejezi a másik kettőt is

    if dropna:
        df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.sort_values(by=["date"]).reset_index(drop=True)

    # mindent ami nem date float64 re konvertálok
    cols_to_convert = df.columns.difference(['date'])
    df[cols_to_convert] = df[cols_to_convert].astype(np.float64)
    df[cols_to_convert] = df[cols_to_convert].clip(upper=300000.0)
    return df
