import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import mplfinance as mpf
from matplotlib.widgets import Button
import tkinter as tk  # A képernyő méretének lekérdezéséhez
import matplotlib as mpl
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 0

import matplotlib.style as mplstyle
mplstyle.use('fast')


class RLVisualizer:
    def __init__(self, df, symbol):

        self.top_max = 4000
        self.zoom_sticks = 100
        self.zoom_shift = self.zoom_sticks

        self.df = df
        self.symbol = symbol
        # Generálj tesztadatokat
        # self.df = self.generate_test_data(25000)
        self.current_index = 0  # Az alsó ablak induló indexe

        # Inicializáljuk a figurát és a tengelyeket
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()  # Képernyő szélessége
        screen_height = root.winfo_screenheight()  # Képernyő magassága
        root.withdraw()  # Elrejtjük a tkinter ablakot

        # Ábra méretének beállítása a képernyőhöz igazodva
        self.fig = plt.figure(figsize=(screen_width / 100 * 2, screen_height / 100 * 2))  # Az ábra méretének beállítása

        self.fig.subplots_adjust(hspace=0.0)
        gs = plt.GridSpec(3, 1, height_ratios=[1, 2, 0.5])
        print(gs[0])
        self.ax1 = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1])  # Azonos x tengely, hogy ne legyen eltérés
        self.ax3 = plt.subplot(gs[2], sharex=self.ax2)  # Azonos x tengely

        pos1 = self.ax1.get_position()
        pos2 = self.ax2.get_position()
        pos3 = self.ax3.get_position()

        # Set custom positions: Keep hspace between 1 and 2 as 0.03, and between 2 and 3 as 0
        self.ax1.set_position([pos1.x0, pos1.y0 + 0.03, pos1.width, pos1.height])
        self.ax2.set_position([pos2.x0, pos2.y0, pos2.width, pos2.height])
        self.ax3.set_position([pos3.x0, pos3.y0, pos3.width, pos3.height])

        sns.set_style("whitegrid")

        self.fig.canvas.manager.window.wm_geometry('+0+0')

        # Gombok létrehozása az előre és hátra mozgáshoz
        ax_forward = plt.axes([0.8, 0.02, 0.1, 0.04])  # Gomb pozíciója
        ax_backward = plt.axes([0.6, 0.02, 0.1, 0.04])  # Gomb pozíciója
        self.btn_forward = Button(ax_forward, 'Előre')
        self.btn_backward = Button(ax_backward, 'Hátra')

        # Gomb események összekötése a metódusokkal
        self.btn_forward.on_clicked(self.shift_window_forward)
        self.btn_backward.on_clicked(self.shift_window_backward)


        self.plot_line_chart(self.ax1)
        self.plot_candlestick_chart(self.ax2, self.ax3, self.current_index)
        # plt.draw()
        plt.pause(0.01)

        # self.refresh()

    def refresh(self):
        self.plot_line_chart(self.ax1)
        self.plot_candlestick_chart(self.ax2, self.ax3, self.current_index)
        plt.show()
        # self.fig.update()
        # # plt.draw()
        plt.pause(0.01)

    def set_df(self, df):
        self.df = df

    def set_action(self, action):
        n = len(action)
        self.df.iloc[:n, self.df.columns.get_loc('action')] = np.array(action, dtype=np.int8)

    def plot_line_chart(self, ax1):
        ax1.plot(self.df.index[-self.top_max:], self.df['close'][-self.top_max:], color='green',
                      label='close Price')

        little = 0.001

        # LONG
        mark_array = np.array([np.nan] * abs(len(self.df[-self.top_max:])))
        buy_mask = np.where(self.df['real_action'][-self.top_max:] == 1)[0]
        mark_array[buy_mask] = self.df['low'].iloc[-self.top_max:].iloc[buy_mask] * (1 - little)
        ax1.plot(self.df.index[-self.top_max:], mark_array, color="blue",
                 marker=(3, 0, 0),
                 markersize=7,
                 linestyle='None')

        # LONG CLOSE
        mark_array = np.array([np.nan] * abs(len(self.df[-self.top_max:])))
        buy_mask = np.where(self.df['real_action'][-self.top_max:] == 2)[0]
        mark_array[buy_mask] = self.df['high'].iloc[-self.top_max:].iloc[buy_mask] * (1 + little)
        ax1.plot(self.df.index[-self.top_max:], mark_array, color="blue",
                 marker=(10, 0, 180),
                 markersize=7,
                 linestyle='None')

        # SHORT
        mark_array = np.array([np.nan] * abs(len(self.df[-self.top_max:])))
        buy_mask = np.where(self.df['real_action'][-self.top_max:] == 3)[0]
        mark_array[buy_mask] = self.df['high'].iloc[-self.top_max:].iloc[buy_mask] * (1 + little)
        ax1.plot(self.df.index[-self.top_max:], mark_array, color="red",
                 marker=(3, 0, 180),
                 markersize=7,
                 linestyle='None')

        # SHORT CLOSE
        mark_array = np.array([np.nan] * abs(len(self.df[-self.top_max:])))
        buy_mask = np.where(self.df['real_action'][-self.top_max:] == 4)[0]
        mark_array[buy_mask] = self.df['low'].iloc[-self.top_max:].iloc[buy_mask] * (1 - little)
        ax1.plot(self.df.index[-self.top_max:], mark_array, color="red",
                      marker=(10, 0, 0),
                      markersize=7,
                      linestyle='None')

        ax1.set_title(self.symbol + ' Záróárak (close)', fontsize=12)
        ax1.set_xlabel('Dátum')
        ax1.set_ylabel('Ár')
        ax1.legend()

    def plot_candlestick_chart(self, ax_candle, ax_volume, start_idx):
        """Candlestick és volume chart egy adott szeletre."""
        end_idx = start_idx + self.zoom_sticks
        data_slice = self.df.iloc[start_idx:end_idx]
        c_length = end_idx - start_idx

        # Candlestick chart és volume megjelenítés
        ax_candle.clear()  # Töröljük az előző candlestick chartot
        ax_volume.clear()  # Töröljük az előző volume chartot

        sty = ['binance',
               'blueskies',
               'brasil',
               'charles',
               'checkers',
               'classic',
               'default',
               'ibd',
               'kenan',
               'mike',
               'nightclouds',
               'sas',
               'starsandstripes',
               'yahoo']

        mpf.plot(data_slice,
                 type='candle',
                 ax=ax_candle,
                 volume=ax_volume,
                 show_nontrading=True,
                 style=sty[3],
                 scale_width_adjustment=dict(volume=0.7, candle=1.55)
                 )

        markersize = 15
        markersize2 = 8
        little = 0.0005
        # LONG
        mark_array = np.array([np.nan] * abs(len(self.df[start_idx:end_idx])))
        buy_mask = np.where(self.df['real_action'][start_idx:end_idx] == 1)[0]
        mark_array[buy_mask] = self.df['low'].iloc[start_idx:end_idx].iloc[buy_mask] * (1 - little)
        self.ax2.plot(self.df.index[start_idx:end_idx], mark_array, color="blue",
                      marker=(3, 0, 0),
                      markersize=markersize,
                      linestyle='None')

        # LONG CLOSE
        mark_array = np.array([np.nan] * abs(len(self.df[start_idx:end_idx])))
        buy_mask = np.where(self.df['real_action'][start_idx:end_idx] == 2)[0]
        mark_array[buy_mask] = self.df['high'].iloc[start_idx:end_idx].iloc[buy_mask] * (1 + little)
        self.ax2.plot(self.df.index[start_idx:end_idx], mark_array, color="blue",
                      marker=(10, 0, 0),
                      markersize=markersize2,
                      linestyle='None')

        # SHORT
        mark_array = np.array([np.nan] * abs(len(self.df[start_idx:end_idx])))
        buy_mask = np.where(self.df['real_action'][start_idx:end_idx] == 3)[0]
        mark_array[buy_mask] = self.df['high'].iloc[start_idx:end_idx].iloc[buy_mask] * (1 + little)
        self.ax2.plot(self.df.index[start_idx:end_idx], mark_array, color="red",
                      marker=(3, 0, 180),
                      markersize=markersize,
                      linestyle='None')

        # SHORT CLOSE
        mark_array = np.array([np.nan] * abs(len(self.df[start_idx:end_idx])))
        buy_mask = np.where(self.df['real_action'][start_idx:end_idx] == 4)[0]
        mark_array[buy_mask] = self.df['low'].iloc[start_idx:end_idx].iloc[buy_mask] * (1 - little)
        self.ax2.plot(self.df.index[start_idx:end_idx], mark_array, color="red",
                      marker=(10, 0, 0),
                      markersize=markersize2,
                      linestyle='None')
        self.fig.canvas.draw()

    def shift_window_forward(self, event):
        """150 sorral előre mozgatja az alsó blokkot."""
        if self.current_index + self.zoom_sticks < len(self.df):
            self.current_index += self.zoom_shift
        self.plot_candlestick_chart(self.ax2, self.ax3, self.current_index)

    def shift_window_backward(self, event):
        """150 sorral hátra mozgatja az alsó blokkot."""
        if self.current_index - self.zoom_shift >= 0:
            self.current_index -= self.zoom_shift
        self.plot_candlestick_chart(self.ax2, self.ax3, self.current_index)

    def show(self):
        plt.show()

if __name__ == "__main__":

    
  print(":)")