import os
import sys
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np


from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from RL_Futures_Visualization import RLVisualizer
from RL_Tools import TensorboardCallback, TimeEstimatorCallback, get_dataset
from RL_Tools import df_split, binance_download, check_trade_sequence_quality, get_dd_du
from RL_Futures_Env import SingleFuturesEnv
from RL_NNs import CustomLinear
from RL_NNs import CustomCNN3

rl_path = "RL_Futures/"
np.set_printoptions(precision=2, suppress=True)
log_dir = "tensorboard_logs/"


def make_env(symbol, rank, seed=0, predict_mode=False):
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


if __name__ == "__main__":

    # symbol = 'ETHUSDT'
    symbol = 'ADAUSDT'
    # symbol = 'AVAXUSDT'

    ta_config = {
        'ETHUSDT': {
            'ema_fast_long_down_shift_length': 27,
            'ema_fast_long_down_shift': 0.99,
            'ema_slow_long_length': 370,
            'ema_fast_short_up_shift_length': 40,
            'ema_fast_short_up_shift': 1.01,
            'ema_slow_short_length': 282,
        },

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

    from_dt = datetime.datetime(year=2023, month=1, day=1, hour=0, minute=0)
    cutoff_dt = datetime.datetime(year=2024, month=8, day=1, hour=0, minute=0)
    # cutoff_dt = None

    run_mode = "train"
    # run_mode = "predict"

    load_model = False
    # load_model = True

    new_trans_dataset = True
    # new_trans_dataset = False

    # num_cpu = 1
    num_cpu = 32

    # total_timesteps = 1_000_000
    total_timesteps = 20_000

    if new_trans_dataset:
        df = get_dataset(symbol, from_dt, cutoff_dt, ta_config)
        file_name = "binance_data/" + symbol + "_trans_data"
        print("Create new trans_data.")
        df.to_hdf(file_name, key='df', mode='w')

    model_base = "RL_Futures_model_Ta"
    model_ver = "dev_01l"
    model_name = symbol + '_' + model_base + '_' + model_ver
    norm_model_name = 'Vecnormalize_' + model_name

    learning_rate = 0.00018
    clip_range = 0.2
    ent_coef = 0.005
    batch_size = 128 * 1
    n_steps = 64 * 1
    verbose = 1

    if run_mode == "train":
        print("Start:", datetime.datetime.now())

        if load_model:

            print("Load model.")
            envs = [make_env(symbol=symbol, rank=i, predict_mode=False) for i in range(num_cpu)]
            #
            # env_test = DummyVecEnv(envs)
            envs = SubprocVecEnv(envs)
            envs = VecNormalize.load(rl_path + norm_model_name + '.pkl', envs)
            envs.training = True
            envs.norm_reward = True
            model = MaskablePPO.load(rl_path + model_name,
                                     env=envs,
                                     )
            # model.env.reset()
        else:
            print("Create new model.")
            envs = [make_env(symbol=symbol, rank=i, predict_mode=False) for i in range(num_cpu)]
            envs = SubprocVecEnv(envs)
            envs = VecNormalize(envs, norm_obs=True, norm_reward=True)

            policy_kwargs = dict(
                net_arch=dict(pi=[16, 16], vf=[8, 8])
            )

            # policy_kwargs = dict(
            #     features_extractor_class=CustomLinear,
            #     features_extractor_kwargs=dict(features_dim=64)
            # )

            # policy_kwargs = dict(
            #     features_extractor_class=CustomLSTM,
            #     features_extractor_kwargs=dict(features_dim=32)
            # )

            # policy_kwargs = dict(
            #     features_extractor_class=CustomCNN3,
            #     features_extractor_kwargs=dict(features_dim=32)
            # )

            model = MaskablePPO(
                policy='MlpPolicy',
                # policy='CnnPolicy',
                env=envs,
                batch_size=batch_size,
                n_steps=n_steps,
                verbose=verbose,
                tensorboard_log=log_dir,
                learning_rate=learning_rate,
                clip_range=clip_range,
                policy_kwargs=policy_kwargs,
                ent_coef=ent_coef,
                device="cuda"
            )

        # set_random_seed(seed_value)

        time_estimator_callback = TimeEstimatorCallback(total_timesteps=total_timesteps)
        tensorboard_callback = TensorboardCallback()

        callback_list = CallbackList([time_estimator_callback, tensorboard_callback])

        model.learn(total_timesteps=total_timesteps, callback=callback_list, tb_log_name=model_name)
        print(datetime.datetime.now())
        print("Save models.")
        model.save(rl_path + model_name)
        envs.save(rl_path + norm_model_name + '.pkl')


        # visualizer = RLVisualizer(predict_df, symbol)
        # visualizer.show()
