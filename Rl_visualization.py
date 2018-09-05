import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import matplotlib.ticker as ticker

pause = False
def onClick(event):
    global pause
    pause ^= True

class RLVisualizer(object):
    # path = '/Users/zhiji/GitHub/ccts/Reinforcement_Learninng/cartpole-keras-qlearning/Data/Prices.csv'
    # action_list = np.random.randint(3, size=l)
    def __init__(self, df, action_list, correction=True):
        # self.path = path
        self.hold_action = 0
        self.buy_action = 1
        self.sell_action = 2
        self.action_list = action_list
        # self.df = self.read_csv(self.path)
        self.df = df
        self.correction = correction


    def get_animation(self, figsize=(50, 10), title='Stock trend RQ Learner', size=40):


        fig = plt.figure(figsize=figsize)
        fig.canvas.mpl_connect('button_press_event', onClick)
        ax = plt.gca()
        l = len(self.df)
        ln1, = ax.plot([], [], 'b-', animated=False)
        ln2, = ax.plot([], [], 'b-', animated=False)



        def data():
            result = 0
            t = 0
            while t < len(self.df) - 1:
                if not pause:
                    result = t
                    t += 1
                yield result

        def init():
            ln1.set_data(self.df.index, self.df['Close'])
            return ln1,

        def update(frame):
            index = self.df.index
            date = self.df['Date']
            close = self.df['Close']
            index_next = index[frame + 1]
            close_next = close[frame + 1]
            x = [index[frame], index_next]
            y = [close[frame], close_next]
            up = (close_next - close[frame]) > 0
            if self.action_list[frame] == self.buy_action:
                mk = '*'
                if self.correction:
                    if up:
                        c = 'g'
                        label = 'True'
                    else:
                        c = 'r'
                        label = 'False'
                else:
                    c = 'r'
                    label = 'Buy'
            elif self.action_list[frame] == self.sell_action:
                mk = 'X'
                if self.correction:
                    if not up:
                        c = 'g'
                        label = 'True'
                    else:
                        c = 'r'
                        label = 'False'
                else:
                    c = 'g'
                    label = 'Sell'
            else:
                c = 'y'
                mk = 'o'
                label = 'Hold'
            # Define right bound and left bound
            bound = int(size/2)
            if frame < bound:
                lb, rb = 0, frame + bound
            elif frame > l - bound:
                lb, rb = frame - bound, l - 1
            else:
                lb, rb = frame - bound, frame + bound
            ax.set_xticklabels(date[lb:rb])
            for tick in ax.get_xticklabels():
                tick.set_rotation(-45)
            # set variables for axes
            ax.set_xlim(index[lb], index[rb])
            ax.set_ylim(y[0] / 1.1, y[1] * 1.1)
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price')
            # set all variables for each frame for line
            ln2.set_data(x, y)
            ln2.set_lw(10)
            ln2.set_color(c)
            ln2.set_marker(mk)
            ln2.set_markersize(30)
            ln2.set_markevery(2)
            ln2.set_label(label)
            ax.legend()
            return ln2,

        ani = FuncAnimation(fig, update, frames=data(),
                            init_func=init)
        plt.show()

    def get_plot(self, figsize=(50, 10), title='Stock trend RQ Learner (Total)', sample = 50):
        plt.figure(figsize=figsize)
        date = self.df['Date']
        close = self.df['Close']
        ax = plt.gca()
        for i in range(len(self.df) - 1):
            x_next = date[i + 1]
            y_next = close[i + 1]
            if self.action_list[i] == self.buy_action:
                c = 'r'
                label = 'Buy'
            elif self.action_list[i] == self.sell_action:
                c = 'g'
                label = 'Sell'
            else:
                c = 'y'
                label = 'Hold'
            x = [date[i], x_next]
            y = [close[i], y_next]
            ax.plot(x, y, color=c, label=label)
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Close Price')
        tick_spacing = int(len(date)/sample)
        for tick in ax.get_xticklabels():
            tick.set_rotation(-45)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.show()



