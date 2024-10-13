import datetime
import math
import pandas as pd
import numpy as np
import json
import time
import os
import sys
import asyncio

from binance import Client, ThreadedWebsocketManager
from binance.enums import *
from binance.exceptions import BinanceAPIException

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class RL_Broker:
    def __init__(self, traded_symbols):

        self.traded_symbols = traded_symbols
        self.api_key, self.secure_key = self.get_api_key()
        self.binance = Client(self.api_key, self.secure_key, testnet=False, requests_params={'timeout': (20, 30)})
        self.set_all_leverage(1)
        self.exchange_info = self.get_futures_exchange_info()

        self.step_size = {}
        self.min_order = {}
        self.tick_size = {}
        self.min_order_in_target = {}
        self.set_filters()

        self.binance_socket = ThreadedWebsocketManager(self.api_key, self.secure_key)
        self.binance_socket.daemon = True
        self.binance_socket.start()

        self.position = {}
        self.account_balance = {}

    @staticmethod
    def get_api_key():
        # api_acces_key.json file is:
        #
        # {
        #     "api_key": "xxxxx",
        #     "secure_key": "yyyyy"
        # }

        json_file_path = 'tokens/api_acces_key.json'
        with open(json_file_path, 'r') as file:
            keys = json.load(file)

        api_key = keys['api_key']
        secure_key = keys['secure_key']

        return api_key, secure_key

    def get_futures_exchange_info(self, filename="binance_data/futures_exchange_info.json", max_age_days=3):
        if os.path.exists(filename):
            current_time = time.time()
            file_mod_time = os.path.getmtime(filename)
            file_age_seconds = current_time - file_mod_time
            file_age_days = file_age_seconds / (24 * 3600)

            if file_age_days <= max_age_days:
                with open(filename, 'r') as file:
                    exchange_info = json.load(file)
                return exchange_info

        exchange_info = self.binance.futures_exchange_info()
        with open(filename, 'w') as file:
            json.dump(exchange_info, file, indent=4)
        return exchange_info

    def get_symbol_info(self, ssymbol):
        symbol_info = next((symbol for symbol in self.exchange_info['symbols'] if symbol['symbol'] == ssymbol), None)
        return symbol_info

    def set_filters(self):
        for symbol in self.traded_symbols:
            symbol_info = self.get_symbol_info(symbol)
            for f in symbol_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    self.step_size[symbol] = f['stepSize']
                    self.min_order[symbol] = f['minQty']
                elif f['filterType'] == 'PRICE_FILTER':
                    self.tick_size[symbol] = f['tickSize']
                elif f['filterType'] == 'NOTIONAL':
                    self.min_order_in_target[symbol] = f['minNotional']

    def get_futures_account_position(self):

        # a =[{
        #      'symbol': 'ADAUSDT',
        #      'positionAmt': '87',
        #      'entryPrice': '0.34',
        #      'breakEvenPrice': '0.340153',
        #      'markPrice': '0.33980000',
        #      'unRealizedProfit': '-0.01740000',
        #      'liquidationPrice': '0',
        #      'leverage': '1',
        #      'maxNotionalValue': '5.0E7',
        #      'marginType': 'cross',
        #      'isolatedMargin': '0.00000000',
        #      'isAutoAddMargin': 'false',
        #      'positionSide': 'BOTH',
        #      'notional': '29.56260000',
        #      'isolatedWallet': '0',
        #      'updateTime': 1728493271907,
        #      'isolated': False, 'adlQuantile': 2
        #  }, .......
        #  ]

        positions = self.binance.futures_position_information()
        self.position = {position['symbol']: position for position in positions}
        self.futures_account_position_callback()

    def futures_account_position_callback(self):
        pass

    def get_symbol_position(self, symbol):
        return float(self.position[symbol]['positionAmt'])

    def get_futures_account_balance(self):
        # [{
        #      'accountAlias': 'SgSgXqmYoCoCuXfW',
        #      'asset': 'BNFCR',
        #      'balance': '26.68120888',
        #      'crossWalletBalance': '26.68120888',
        #      'crossUnPnl': '-0.01740000',
        #      'availableBalance': '705.87720552',
        #      'maxWithdrawAmount': '26.68120888',
        #      'marginAvailable': True,
        #      'updateTime': 1728493258439
        #  }, {
        #      'accountAlias': 'SgSgXqmYoCoCuXfW',
        #      'asset': 'BNB', 'balance': '0.03406456',
        #      'crossWalletBalance': '0.03406456',
        #      'crossUnPnl': '0.00000000',
        #      'availableBalance': '1.16469437',
        #      'maxWithdrawAmount': '0.03406456',
        #      'marginAvailable': True,
        #      'updateTime': 1728493271907
        #  }......]

        balances = self.binance.futures_account_balance()
        self.account_balance = {balance['asset']: balance for balance in balances}

    def get_portfolio_value(self):
        self.get_futures_account_balance()
        value = float(self.account_balance['BNFCR']['availableBalance'])

        #      'symbol': 'SNTUSDT',
        #      'positionAmt': '0',
        #      'entryPrice': '0.0',
        #      'breakEvenPrice': '0.0',
        #      'markPrice': '0.00000000',
        #      'unRealizedProfit': '0.00000000',
        #      'liquidationPrice': '0', 'leverage': '20',
        #      'maxNotionalValue': '25000',
        #      'marginType': 'cross',
        #      'isolatedMargin': '0.00000000',
        #      'isAutoAddMargin': 'false',
        #      'positionSide': 'BOTH',
        #      'notional': '0',
        #      'isolatedWallet': '0',
        #      'updateTime': 0,
        #      'isolated': False,
        #      'adlQuantile': 0
        #  },

        for symbol in self.traded_symbols:
            value += (float(self.position[symbol]['positionAmt']) *
                      float(self.position[symbol]['markPrice']))
        return value

    def start_user_socket(self):
        self.binance_socket.start_futures_user_socket(self._handle_user_socket_message)
        time.sleep(1)
        print("start start_user_socket")

    def start_market_socket(self):
        for isymbol in self.traded_symbols:
            self.binance_socket.start_kline_futures_socket(
                self._handle_kline_socket_message,
                symbol=isymbol,
                interval='1m')
        # self.binance_socket.join()

    def _handle_user_socket_message(self, msg):
        # # long ADA
        # a = {
        #     'e': 'ORDER_TRADE_UPDATE', 'T': 1728489422555, 'E': 1728489422555, 'o': {
        #         's': 'ADAUSDT', 'c': 'H18qjvJJyW0osJbjB6jN2g', 'S': 'BUY', 'o': 'MARKET', 'f': 'GTC', 'q': '87',
        #         'p': '0',
        #         'ap': '0', 'sp': '0', 'x': 'NEW', 'X': 'NEW', 'i': 44758775074, 'l': '0', 'z': '0', 'L': '0', 'n': '0',
        #         'N': 'BNFCR', 'T': 1728489422555, 't': 0, 'b': '0', 'a': '0', 'm': False, 'R': False,
        #         'wt': 'CONTRACT_PRICE', 'ot': 'MARKET', 'ps': 'BOTH', 'cp': False, 'rp': '0', 'pP': False, 'si': 0,
        #         'ss': 0,
        #         'V': 'NONE', 'pm': 'NONE', 'gtd': 0
        #     }
        # }
        # a = {
        #     'e': 'TRADE_LITE', 'E': 1728489422555, 'T': 1728489422555, 's': 'ADAUSDT', 'q': '87', 'p': '0.00000',
        #     'm': False, 'c': 'H18qjvJJyW0osJbjB6jN2g', 'S': 'BUY', 'L': '0.34350', 'l': '87', 't': 1184868868,
        #     'i': 44758775074
        # }
        # a = {
        #     'e': 'ACCOUNT_UPDATE', 'T': 1728489422555, 'E': 1728489422555, 'a': {
        #         'B': [{'a': 'BNFCR', 'wb': '26.73130888', 'cw': '26.73130888', 'bc': '0'},
        #               {'a': 'BNB', 'wb': '0.03419736', 'cw': '0.03419736', 'bc': '0'}], 'P': [{
        #             's': 'ADAUSDT',
        #             'pa': '87',
        #             'ep': '0.3435',
        #             'cr': '12.55649999',
        #             'up': '0', 'mt': 'cross',
        #             'iw': '0', 'ps': 'BOTH',
        #             'ma': 'BNFCR',
        #             'bep': '0.343654575'
        #         }], 'm': 'ORDER'
        #     }
        # }
        # a = {
        #     'e': 'ORDER_TRADE_UPDATE', 'T': 1728489422555, 'E': 1728489422555, 'o': {
        #         's': 'ADAUSDT', 'c': 'H18qjvJJyW0osJbjB6jN2g', 'S': 'BUY', 'o': 'MARKET', 'f': 'GTC', 'q': '87',
        #         'p': '0',
        #         'ap': '0.34350', 'sp': '0', 'x': 'TRADE', 'X': 'FILLED', 'i': 44758775074, 'l': '87', 'z': '87',
        #         'L': '0.34350', 'n': '0.00002312', 'N': 'BNB', 'T': 1728489422555, 't': 1184868868, 'b': '0', 'a': '0',
        #         'm': False, 'R': False, 'wt': 'CONTRACT_PRICE', 'ot': 'MARKET', 'ps': 'BOTH', 'cp': False, 'rp': '0',
        #         'pP': False, 'si': 0, 'ss': 0, 'V': 'NONE', 'pm': 'NONE', 'gtd': 0
        #     }
        # }
        #
        # # long AVAXUSDT
        #
        # a = {
        #     'e': 'TRADE_LITE', 'E': 1728489427853, 'T': 1728489427852, 's': 'AVAXUSDT', 'q': '1', 'p': '0.0000',
        #     'm': False, 'c': 'Xz5233podxm0p7MEQ55cOV', 'S': 'BUY', 'L': '26.2240', 'l': '1', 't': 886858426,
        #     'i': 23767402990
        # }
        # a = {
        #     'e': 'ORDER_TRADE_UPDATE', 'T': 1728489427852, 'E': 1728489427853, 'o': {
        #         's': 'AVAXUSDT', 'c': 'Xz5233podxm0p7MEQ55cOV', 'S': 'BUY', 'o': 'MARKET', 'f': 'GTC', 'q': '1',
        #         'p': '0',
        #         'ap': '0', 'sp': '0', 'x': 'NEW', 'X': 'NEW', 'i': 23767402990, 'l': '0', 'z': '0', 'L': '0', 'n': '0',
        #         'N': 'BNFCR', 'T': 1728489427852, 't': 0, 'b': '0', 'a': '0', 'm': False, 'R': False,
        #         'wt': 'CONTRACT_PRICE', 'ot': 'MARKET', 'ps': 'BOTH', 'cp': False, 'rp': '0', 'pP': False, 'si': 0,
        #         'ss': 0,
        #         'V': 'NONE', 'pm': 'NONE', 'gtd': 0
        #     }
        # }
        # a = {
        #     'e': 'ACCOUNT_UPDATE', 'T': 1728489427852, 'E': 1728489427853, 'a': {
        #         'B': [{'a': 'BNFCR', 'wb': '26.73130888', 'cw': '26.73130888', 'bc': '0'},
        #               {'a': 'BNB', 'wb': '0.03417707', 'cw': '0.03417707', 'bc': '0'}], 'P': [{
        #             's': 'AVAXUSDT',
        #             'pa': '1', 'ep': '26.224',
        #             'cr': '10.46800006',
        #             'up': '-0.00400000',
        #             'mt': 'cross', 'iw': '0',
        #             'ps': 'BOTH',
        #             'ma': 'BNFCR',
        #             'bep': '26.2358008'
        #         }], 'm': 'ORDER'
        #     }
        # }
        # a = {
        #     'e': 'ORDER_TRADE_UPDATE', 'T': 1728489427852, 'E': 1728489427853, 'o': {
        #         's': 'AVAXUSDT', 'c': 'Xz5233podxm0p7MEQ55cOV', 'S': 'BUY', 'o': 'MARKET', 'f': 'GTC', 'q': '1',
        #         'p': '0',
        #         'ap': '26.2240', 'sp': '0', 'x': 'TRADE', 'X': 'FILLED', 'i': 23767402990, 'l': '1', 'z': '1',
        #         'L': '26.2240', 'n': '0.00002029', 'N': 'BNB', 'T': 1728489427852, 't': 886858426, 'b': '0', 'a': '0',
        #         'm': False, 'R': False, 'wt': 'CONTRACT_PRICE', 'ot': 'MARKET', 'ps': 'BOTH', 'cp': False, 'rp': '0',
        #         'pP': False, 'si': 0, 'ss': 0, 'V': 'NONE', 'pm': 'NONE', 'gtd': 0
        #     }
        # }
        #
        # # close ADA
        # a = {
        #     'e': 'ORDER_TRADE_UPDATE', 'T': 1728489525516, 'E': 1728489525516, 'o': {
        #         's': 'ADAUSDT', 'c': 'android_yzMdmZxSdBmIocK3mGoN', 'S': 'SELL', 'o': 'MARKET', 'f': 'GTC', 'q': '87',
        #         'p': '0', 'ap': '0', 'sp': '0', 'x': 'NEW', 'X': 'NEW', 'i': 44758792034, 'l': '0', 'z': '0', 'L': '0',
        #         'n': '0', 'N': 'BNFCR', 'T': 1728489525516, 't': 0, 'b': '0', 'a': '0', 'm': False, 'R': True,
        #         'wt': 'CONTRACT_PRICE', 'ot': 'MARKET', 'ps': 'BOTH', 'cp': False, 'rp': '0', 'pP': False, 'si': 0,
        #         'ss': 0,
        #         'V': 'NONE', 'pm': 'NONE', 'gtd': 0
        #     }
        # }
        # a = {
        #     'e': 'TRADE_LITE', 'E': 1728489525516, 'T': 1728489525516, 's': 'ADAUSDT', 'q': '87', 'p': '0.00000',
        #     'm': False, 'c': 'android_yzMdmZxSdBmIocK3mGoN', 'S': 'SELL', 'L': '0.34320', 'l': '23', 't': 1184869055,
        #     'i': 44758792034
        # }
        # a = {
        #     'e': 'TRADE_LITE', 'E': 1728489525516, 'T': 1728489525516, 's': 'ADAUSDT', 'q': '87', 'p': '0.00000',
        #     'm': False, 'c': 'android_yzMdmZxSdBmIocK3mGoN', 'S': 'SELL', 'L': '0.34320', 'l': '18', 't': 1184869056,
        #     'i': 44758792034
        # }
        # a = {
        #     'e': 'TRADE_LITE', 'E': 1728489525516, 'T': 1728489525516, 's': 'ADAUSDT', 'q': '87', 'p': '0.00000',
        #     'm': False, 'c': 'android_yzMdmZxSdBmIocK3mGoN', 'S': 'SELL', 'L': '0.34320', 'l': '46', 't': 1184869057,
        #     'i': 44758792034
        # }
        # a = {
        #     'e': 'ACCOUNT_UPDATE', 'T': 1728489525516, 'E': 1728489525516, 'a': {
        #         'B': [{'a': 'BNFCR', 'wb': '26.72440888', 'cw': '26.72440888', 'bc': '0'},
        #               {'a': 'BNB', 'wb': '0.03417096', 'cw': '0.03417096', 'bc': '0'}], 'P': [{
        #             's': 'ADAUSDT',
        #             'pa': '64',
        #             'ep': '0.3435',
        #             'cr': '12.54959999',
        #             'up': '-0.01019904',
        #             'mt': 'cross', 'iw': '0',
        #             'ps': 'BOTH',
        #             'ma': 'BNFCR',
        #             'bep': '0.3438734397656'
        #         }], 'm': 'ORDER'
        #     }
        # }
        # a = {
        #     'e': 'ORDER_TRADE_UPDATE', 'T': 1728489525516, 'E': 1728489525516, 'o': {
        #         's': 'ADAUSDT', 'c': 'android_yzMdmZxSdBmIocK3mGoN', 'S': 'SELL', 'o': 'MARKET', 'f': 'GTC', 'q': '87',
        #         'p': '0', 'ap': '0.34320', 'sp': '0', 'x': 'TRADE', 'X': 'PARTIALLY_FILLED', 'i': 44758792034,
        #         'l': '23',
        #         'z': '23', 'L': '0.34320', 'n': '0.00000611', 'N': 'BNB', 'T': 1728489525516, 't': 1184869055, 'b': '0',
        #         'a': '0', 'm': False, 'R': True, 'wt': 'CONTRACT_PRICE', 'ot': 'MARKET', 'ps': 'BOTH', 'cp': False,
        #         'rp': '-0.00690000', 'pP': False, 'si': 0, 'ss': 0, 'V': 'NONE', 'pm': 'NONE', 'gtd': 0
        #     }
        # }
        # a = {
        #     'e': 'ACCOUNT_UPDATE', 'T': 1728489525516, 'E': 1728489525516, 'a': {
        #         'B': [{'a': 'BNFCR', 'wb': '26.71900888', 'cw': '26.71900888', 'bc': '0'},
        #               {'a': 'BNB', 'wb': '0.03416618', 'cw': '0.03416618', 'bc': '0'}], 'P': [{
        #             's': 'ADAUSDT',
        #             'pa': '46',
        #             'ep': '0.3435',
        #             'cr': '12.54419999',
        #             'up': '-0.00733056',
        #             'mt': 'cross', 'iw': '0',
        #             'ps': 'BOTH',
        #             'ma': 'BNFCR',
        #             'bep': '0.3441973927174'
        #         }], 'm': 'ORDER'
        #     }
        # }
        # a = {
        #     'e': 'ORDER_TRADE_UPDATE', 'T': 1728489525516, 'E': 1728489525516, 'o': {
        #         's': 'ADAUSDT', 'c': 'android_yzMdmZxSdBmIocK3mGoN', 'S': 'SELL', 'o': 'MARKET', 'f': 'GTC', 'q': '87',
        #         'p': '0', 'ap': '0.34320', 'sp': '0', 'x': 'TRADE', 'X': 'PARTIALLY_FILLED', 'i': 44758792034,
        #         'l': '18',
        #         'z': '41', 'L': '0.34320', 'n': '0.00000478', 'N': 'BNB', 'T': 1728489525516, 't': 1184869056, 'b': '0',
        #         'a': '0', 'm': False, 'R': True, 'wt': 'CONTRACT_PRICE', 'ot': 'MARKET', 'ps': 'BOTH', 'cp': False,
        #         'rp': '-0.00540000', 'pP': False, 'si': 0, 'ss': 0, 'V': 'NONE', 'pm': 'NONE', 'gtd': 0
        #     }
        # }
        # a = {
        #     'e': 'ACCOUNT_UPDATE', 'T': 1728489525516, 'E': 1728489525516, 'a': {
        #         'B': [{'a': 'BNFCR', 'wb': '26.70520888', 'cw': '26.70520888', 'bc': '0'},
        #               {'a': 'BNB', 'wb': '0.03415396', 'cw': '0.03415396', 'bc': '0'}], 'P': [{
        #             's': 'ADAUSDT', 'pa': '0',
        #             'ep': '0',
        #             'cr': '12.53039999',
        #             'up': '0', 'mt': 'cross',
        #             'iw': '0', 'ps': 'BOTH',
        #             'ma': 'BNFCR', 'bep': '0'
        #         }], 'm': 'ORDER'
        #     }
        # }
        # a = {
        #     'e': 'ORDER_TRADE_UPDATE', 'T': 1728489525516, 'E': 1728489525516, 'o': {
        #         's': 'ADAUSDT', 'c': 'android_yzMdmZxSdBmIocK3mGoN', 'S': 'SELL', 'o': 'MARKET', 'f': 'GTC', 'q': '87',
        #         'p': '0', 'ap': '0.34320', 'sp': '0', 'x': 'TRADE', 'X': 'FILLED', 'i': 44758792034, 'l': '46',
        #         'z': '87',
        #         'L': '0.34320', 'n': '0.00001222', 'N': 'BNB', 'T': 1728489525516, 't': 1184869057, 'b': '0', 'a': '0',
        #         'm': False, 'R': True, 'wt': 'CONTRACT_PRICE', 'ot': 'MARKET', 'ps': 'BOTH', 'cp': False,
        #         'rp': '-0.01380000', 'pP': False, 'si': 0, 'ss': 0, 'V': 'NONE', 'pm': 'NONE', 'gtd': 0
        #     }
        # }
        # a = {
        #     'e': 'TRADE_LITE', 'E': 1728489551451, 'T': 1728489551450, 's': 'AVAXUSDT', 'q': '1', 'p': '0.0000',
        #     'm': False, 'c': 'android_JnosJoM1q0kwbaDdgbrN', 'S': 'SELL', 'L': '26.2000', 'l': '1', 't': 886859194,
        #     'i': 23767433533
        # }
        #
        # # close AVAX
        # a = {
        #     'e': 'ORDER_TRADE_UPDATE', 'T': 1728489551450, 'E': 1728489551451, 'o': {
        #         's': 'AVAXUSDT', 'c': 'android_JnosJoM1q0kwbaDdgbrN', 'S': 'SELL', 'o': 'MARKET', 'f': 'GTC', 'q': '1',
        #         'p': '0', 'ap': '0', 'sp': '0', 'x': 'NEW', 'X': 'NEW', 'i': 23767433533, 'l': '0', 'z': '0', 'L': '0',
        #         'n': '0', 'N': 'BNFCR', 'T': 1728489551450, 't': 0, 'b': '0', 'a': '0', 'm': False, 'R': True,
        #         'wt': 'CONTRACT_PRICE', 'ot': 'MARKET', 'ps': 'BOTH', 'cp': False, 'rp': '0', 'pP': False, 'si': 0,
        #         'ss': 0,
        #         'V': 'NONE', 'pm': 'NONE', 'gtd': 0
        #     }
        # }
        # a = {
        #     'e': 'ACCOUNT_UPDATE', 'T': 1728489551450, 'E': 1728489551451, 'a': {
        #         'B': [{'a': 'BNFCR', 'wb': '26.68120888', 'cw': '26.68120888', 'bc': '0'},
        #               {'a': 'BNB', 'wb': '0.03413367', 'cw': '0.03413367', 'bc': '0'}], 'P': [{
        #             's': 'AVAXUSDT',
        #             'pa': '0', 'ep': '0',
        #             'cr': '10.44400006',
        #             'up': '0', 'mt': 'cross',
        #             'iw': '0', 'ps': 'BOTH',
        #             'ma': 'BNFCR', 'bep': '0'
        #         }], 'm': 'ORDER'
        #     }
        # }
        # a = {
        #     'e': 'ORDER_TRADE_UPDATE', 'T': 1728489551450, 'E': 1728489551451, 'o': {
        #         's': 'AVAXUSDT', 'c': 'android_JnosJoM1q0kwbaDdgbrN', 'S': 'SELL', 'o': 'MARKET', 'f': 'GTC', 'q': '1',
        #         'p': '0', 'ap': '26.2000', 'sp': '0', 'x': 'TRADE', 'X': 'FILLED', 'i': 23767433533, 'l': '1', 'z': '1',
        #         'L': '26.2000', 'n': '0.00002029', 'N': 'BNB', 'T': 1728489551450, 't': 886859194, 'b': '0', 'a': '0',
        #         'm': False, 'R': True, 'wt': 'CONTRACT_PRICE', 'ot': 'MARKET', 'ps': 'BOTH', 'cp': False,
        #         'rp': '-0.02400000', 'pP': False, 'si': 0, 'ss': 0, 'V': 'NONE', 'pm': 'NONE', 'gtd': 0
        #     }
        # }

        if msg['e'] == 'error':
            raise msg
        elif msg['e'] == 'ORDER_TRADE_UPDATE' and msg['o']['X'] != 'FILLED':
            self.get_futures_account_position()

    def _handle_kline_socket_message(self, msg):
        if msg['e'] == 'continuous_kline' and msg['k']['x']:
            # {
            #     'e': 'continuous_kline',
            #     'E': 1728624697400,
            #     'ps': 'ADAUSDT',
            #     'ct': 'PERPETUAL',
            # if msg['k']['x']:  # Is closed
            # {'t': 1717606380000,
            # 'T': 1717606439999,
            # 'i': '1m',
            # 'f': 4739669067560,
            # 'L': 4739673419912,
            # 'o': '71484.70',
            # 'c': '71451.30',
            # 'h': '71484.80',
            # 'l': '71407.10',
            # 'v': '104.280',
            # 'n': 2125,
            # 'x': True,
            # 'q': '7449588.38820',
            # 'V': '39.741',
            # 'Q': '2838844.68270',
            # 'B': '0'}

            kline_df = self.parser_to_kline(msg['k'])
            self.handle_kline_socket_message(msg['ps'], kline_df)

    def handle_kline_socket_message(self, symbol, kline_df):
        pass

    def handle_account_update(self, msg):
        pass

    @staticmethod
    def parser_to_kline(kline):
        df = pd.DataFrame([[kline['t'], kline['o'], kline['h'], kline['l'], kline['c'], kline['v'], kline['n']]],
                          columns=['date', 'open', 'high', 'low', 'close', 'volume', 'number_of_trades']).astype(
            np.float64)

        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df['date'] = df.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        return df

    def set_all_leverage(self, leverage):
        for symbol in self.traded_symbols:
            self.set_symbol_leverage(symbol, leverage)

    def set_symbol_leverage(self, symbol, leverage):
        self.binance.futures_change_leverage(symbol=symbol, leverage=leverage)

    def long(self, symbol, amount, price):
        qty = amount / price
        qty = self.format_quantity(symbol, qty)
        if qty > 0:
            self.create_order(symbol, side='BUY', qty=qty)
        else:
            print(f"Long failed: symbol {symbol}, amount {amount}, price {price} ")

        return qty


    def short(self, symbol, amount, price):
        qty = amount / price
        qty = self.format_quantity(symbol, qty)
        if qty > 0:
            self.create_order(symbol, side='SELL', qty=qty)
        else:
            print(f"Short failed: symbol {symbol}, amount {amount}, price {price} ")

        return qty

    def close(self, symbol):
        qty = float(self.position[symbol]['positionAmt'])
        if qty != 0:
            if qty < 0:
                self.create_order(symbol, side='BUY', qty=abs(qty))
            else:
                self.create_order(symbol, side='SELL', qty=abs(qty))

    def create_order(self, symbol, side, qty):
        ret = self.binance.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=qty)

    @staticmethod
    def _format_value(value, step):
        precision = step.find('1') - 1
        if precision > 0:
            return '{:0.0{}f}'.format(float(value), precision)
        return math.floor(int(value))

    def format_price(self, symbol, price):
        return self._format_value(price, self.tick_size[symbol])

    def format_quantity(self, symbol, size):
        return self._format_value(size, self.step_size[symbol])

    def get_futures_account_trades(self, limit=3):
        all_trades = []
        for symbol in self.traded_symbols:
            all_trades += self.binance.futures_account_trades(symbol=symbol, limit=limit)

        df = pd.DataFrame(all_trades)
        df = df[['time', 'symbol', 'side', 'qty', 'price', 'quoteQty',
                 'realizedPnl', 'commission', 'commissionAsset', 'id',
                 ]]
        columns_to_convert = df.columns.difference(['symbol', 'side', 'commissionAsset'])
        df[columns_to_convert] = df[columns_to_convert].astype(np.float64)

        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df["time"] = df.time.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        df = df.sort_values(by=["time"]).reset_index(drop=True)

        excel_file_path = 'riports/RL_trades_data.xlsx'  # You can specify any file path you want
        df.to_excel(excel_file_path, index=False)


if __name__ == "__main__":

    rlb = RL_Broker(['ADAUSDT', 'AVAXUSDT'])

    rlb.start_user_socket()
    rlb.get_futures_account_balance()
    rlb.get_futures_account_position()
    print("portfolio value:", rlb.get_portfolio_value())

    for si in rlb.exchange_info['symbols']:
        symbol = si['symbol']
        symbol_info = rlb.get_symbol_info(symbol)
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                # print(symbol, f['filterType'], f['stepSize'])
                if f['stepSize'] == "1" and "USDT" in symbol:
                    print(symbol, f['stepSize'])

    print(rlb.exchange_info['symbols'][0])
    # time.sleep(5)
    # print(rlb.position["ADAUSDT"])
    # print("ADA long")
    # rlb.long("ADAUSDT", 30, 0.34340)
    # # rlb.short("ADAUSDT", 30, 0.34340)
    # time.sleep(1)
    # print("most")
    # print(rlb.position["ADAUSDT"])
    #
    # rlb.get_futures_account_balance()
    #
    # print("portfolio value:",rlb.get_portfolio_value())
    #
    # time.sleep(5)
    # rlb.stop("ADAUSDT")
    # time.sleep(1)
    # print(rlb.position["ADAUSDT"])
    # time.sleep(5)
    # print("portfolio value:",rlb.get_portfolio_value())

    # rlb.get_futures_account_trades()

    # rlb.get_symbol_info("ADAUSDT")


    # print("balance")
    # rlb.get_futures_account_balance()

    # time.sleep(5)
    # print("AVAX")
    # rlb.long("AVAXUSDT", 30, 26.2190)

    # rlb.start_market_socket()
    while True:
        time.sleep(1200)


