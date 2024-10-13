import sys
import time
import numpy as np
import pandas as pd
import datetime
import multiprocessing as mp
from RL_Tools import get_dataset
from cytatu import CythonTaTuning


def parmeter_combinations():
    settings = {}
    settings["ema_fast_long"] = np.random.permutation(np.arange(3, 120, 1, dtype=np.int16))
    settings["ema_fast_long_down_shift"] = np.random.permutation(np.arange(98, 100, 1, dtype=np.int16))  # / 100
    settings["ema_slow_long"] = np.random.permutation(np.arange(121, 500, 1, dtype=np.int16))

    settings["ema_fast_short"] = np.random.permutation(np.arange(3, 120, 1, dtype=np.int16))
    settings["ema_fast_short_up_shift"] = np.random.permutation(np.arange(100, 102, 1, dtype=np.int16))  # / 100
    settings["ema_slow_short"] = np.random.permutation(np.arange(121, 500, 1, dtype=np.int16))

    params_array = []
    settings_name = []

    for k in settings:
        params_array.append(settings[k])
        settings_name.append(k)

    total_combinations = 1
    for a in params_array:
        total_combinations *= len(a)

    return params_array, settings_name, total_combinations


def nth_combination(arrays, n):
    indices = []
    for array in arrays:
        size = len(array)
        index = n % size
        indices.append(index)
        n //= size

    combination = [arrays[i][indices[i]] for i in range(len(arrays))]
    combination = np.array(combination, dtype=np.int64)
    return combination


def worker(worker_id, params_array, shared_dict, is_long=True, is_short=True):
    if worker_id == 14:
        print(f"Workers has started.")

    if not is_long:
        params_array[0] = [10]
        params_array[1] = [1]
        params_array[2] = [100]

    if not is_short:
        params_array[3] = [10]
        params_array[4] = [1]
        params_array[5] = [100]

    open_data = np.load("binance_data/open_data.npy")
    close_data = np.load("binance_data/close_data.npy")
    ctt = CythonTaTuning(open_data, close_data,
                         is_long=is_long, is_short=is_short)

    turns = 50
    step_from = worker_id * turns
    step_to = step_from + turns

    for i in range(step_from, step_to):
        comb = nth_combination(params_array, i)
        p1 = comb[0]
        p2 = comb[1] / 100
        p3 = comb[2]
        p4 = comb[3]
        p5 = comb[4] / 100
        p6 = comb[5]

        ctt.set_data(p1, p2, p3, p4, p5, p6)
        meter, params = ctt.run_backtest()
        shared_dict['last_meter'] = meter
        if meter > shared_dict['meter']:
            shared_dict['meter'] = meter
            shared_dict['params'] = params
            ctt.render()
        ctt.reset()
        shared_dict["worker" + str(worker_id)] = round((((i - step_from) + 1) / turns) * 100, 2)


if __name__ == "__main__":
    params_array, settings_name, total_combinations = parmeter_combinations()

    # params_array = [[11], [99], [147], [14], [100], [484]]
    # ADA
    # ema_fast_long_down_shift_length
    # 11
    # ema_fast_long_down_shift
    # 0.99
    # ema_slow_long_length
    # 147
    #
    # ema_fast_short_up_shift_length
    # 14
    # ema_fast_short_up_shift
    # 1.0
    # ema_slow_short_length
    # 484
    #

    print("total_combinations", total_combinations)

    symbol = "FTMUSDT"
    trans_data = False

    if not trans_data:

        from_dt = datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0)
        cutoff_dt = None
        refresh = False

        ta_config = {
            'FTMUSDT': {
                'ema_fast_long_down_shift_length': 3,
                'ema_fast_long_down_shift': 0.98,
                'ema_slow_long_length': 283,
                'ema_fast_short_up_shift_length': 3,
                'ema_fast_short_up_shift': 1.0,
                'ema_slow_short_length': 315,
            },

            # 'SEIUSDT': {
            #     'ema_fast_long_down_shift_length': 27,
            #     'ema_fast_long_down_shift': 0.99,
            #     'ema_slow_long_length': 370,
            #     'ema_fast_short_up_shift_length': 40,
            #     'ema_fast_short_up_shift': 1.01,
            #     'ema_slow_short_length': 282,
            # },

            'AVAXUSDT': {
                'ema_fast_long_down_shift_length': 53,
                'ema_fast_long_down_shift': 0.99,
                'ema_slow_long_length': 265,
                'ema_fast_short_up_shift_length': 39,
                'ema_fast_short_up_shift': 1.01,
                'ema_slow_short_length': 397,
            },
            'ADAUSDT': {
                'ema_fast_long_down_shift_length': 65,
                'ema_fast_long_down_shift': 0.99,
                'ema_slow_long_length': 346,
                'ema_fast_short_up_shift_length': 35,
                'ema_fast_short_up_shift': 1.01,
                'ema_slow_short_length': 331,
            },

        }
        #
        # p1 = ta_config[symbol]['ema_fast_long_down_shift_length']
        # p2 = ta_config[symbol]['ema_fast_long_down_shift']
        # p3 = ta_config[symbol]['ema_slow_long_length']
        # p4 = ta_config[symbol]['ema_fast_short_up_shift_length']
        # p5 = ta_config[symbol]['ema_fast_short_up_shift']
        # p6 = ta_config[symbol]['ema_slow_short_length']

        df = get_dataset(symbol, from_dt, cutoff_dt, ta_config, refresh=refresh)
        print(symbol)
        print(df.T)
        print("shape", df.shape)
        open_data = np.array(df.open)
        close_data = np.array(df.close)

        np.save("binance_data/open_data", open_data)
        np.save("binance_data/close_data", close_data)

    manager = mp.Manager()
    shared_dict = manager.dict()

    selected_params = [0] * 6
    num_workers = mp.cpu_count() - 1


    # RUN LONG search
    shared_dict['meter'] = -1000000
    shared_dict['last_meter'] = -1000000
    shared_dict['params'] = []
    shared_dict['worker0'] = 0
    shared_dict['worker1'] = 0.0
    shared_dict['worker2'] = 0.0
    shared_dict['worker3'] = 0.0
    shared_dict['worker4'] = 0.0
    shared_dict['worker5'] = 0.0
    shared_dict['worker6'] = 0.0
    shared_dict['worker7'] = 0.0
    shared_dict['worker8'] = 0.0
    shared_dict['worker9'] = 0.0
    shared_dict['worker10'] = 0.0
    shared_dict['worker11'] = 0.0
    shared_dict['worker12'] = 0.0
    shared_dict['worker13'] = 0.0
    shared_dict['worker14'] = 0.0

    is_long = True
    is_short = False
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(worker_id, params_array, shared_dict, is_long, is_short))
        p.daemon = True
        p.start()

    sp = 0
    while sp != 100 * num_workers:
        print(shared_dict)
        time.sleep(20)

        sp = 0
        for i in range(num_workers):
            sp += shared_dict['worker' + str(i)]

    selected_params[0:3] = shared_dict['params'][0:3]

    # RUN SHORT search
    shared_dict['meter'] = -1000000
    shared_dict['last_meter'] = -1000000
    shared_dict['params'] = []
    shared_dict['worker0'] = 0
    shared_dict['worker1'] = 0.0
    shared_dict['worker2'] = 0.0
    shared_dict['worker3'] = 0.0
    shared_dict['worker4'] = 0.0
    shared_dict['worker5'] = 0.0
    shared_dict['worker6'] = 0.0
    shared_dict['worker7'] = 0.0
    shared_dict['worker8'] = 0.0
    shared_dict['worker9'] = 0.0
    shared_dict['worker10'] = 0.0
    shared_dict['worker11'] = 0.0
    shared_dict['worker12'] = 0.0
    shared_dict['worker13'] = 0.0
    shared_dict['worker14'] = 0.0

    is_long = False
    is_short = True
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(worker_id, params_array, shared_dict, is_long, is_short))
        p.daemon = True
        p.start()

    sp = 0
    while sp != 100 * num_workers:
        print(shared_dict)
        time.sleep(20)

        sp = 0
        for i in range(num_workers):
            sp += shared_dict['worker' + str(i)]

    selected_params[3:6] = shared_dict['params'][3:6]

    print(selected_params)

    print(f"""
            '{symbol}': {{
                        'ema_fast_long_down_shift_length': {selected_params[0]},
                        'ema_fast_long_down_shift': {selected_params[1] / 100},
                        'ema_slow_long_length': {selected_params[2]},
                        'ema_fast_short_up_shift_length': {selected_params[3]},
                        'ema_fast_short_up_shift': {selected_params[4] / 100},
                        'ema_slow_short_length': {selected_params[5]},
                    }},
            
            
            """)


