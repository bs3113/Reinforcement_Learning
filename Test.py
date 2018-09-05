from Rl_visualization import *

path = '/Users/zhiji/GitHub/ccts/Reinforcement_Learninng/cartpole-keras-qlearning/Data/Prices.csv'


def read_csv(path, sep=','):
    dtypes = {'Date': str}
    df = pd.read_csv(path, sep=sep, header=0, names=['Date', 'Open', 'Close', 'High',
                                                     'Low', 'Total_turnover', 'Volume'],
                     dtype=dtypes)
    return df
df = read_csv(path)
action_list = np.random.randint(3, size=len(df))
rqv = RLVisualizer(df, action_list)
# rqv.get_plot()
rqv.get_animation()