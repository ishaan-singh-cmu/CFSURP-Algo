from ib_insync import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

contract = Stock('NVDA', 'SMART', 'USD')
bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='30 D',
    barSizeSetting='1 hour', whatToShow='TRADES',
    useRTH=True, formatDate=1
)
df = util.df(bars).set_index('date')

df['return'] = np.log(df['close'] / df['close'].shift(1))
mu    = df['return'].mean()
sigma = df['return'].std()
S0    = df['close'].iloc[-1]

RSI_WINDOW = 14
T          = 100  
N          = T
dt         = 1
n_sims     = 100

simulations = np.zeros((n_sims, N))
for i in range(n_sims):
    path = [S0]
    for _ in range(N-1):
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * np.random.normal()
        path.append(path[-1] * np.exp(drift + shock))
    simulations[i] = path

def run_strategy(prices):
    df_sim = pd.DataFrame({'close': prices})
    df_sim['rsi']      = RSIIndicator(df_sim['close'], window=RSI_WINDOW).rsi()
    df_sim['position'] = 0
    df_sim.loc[df_sim['rsi'] < 30, 'position'] = 1   # buy
    df_sim.loc[df_sim['rsi'] > 70, 'position'] = 0   # sell
    df_sim['position'] = df_sim['position'].ffill().fillna(0)
    df_sim['ret']      = df_sim['close'].pct_change()
    df_sim['strat']    = df_sim['position'].shift(1) * df_sim['ret']
    cum = (1 + df_sim['strat'].fillna(0)).cumprod()
    total_return = cum.iloc[-1] - 1
    max_dd       = (cum / cum.cummax() - 1).min()
    return total_return, max_dd

results    = [run_strategy(path) for path in simulations]
returns, drawdowns = map(np.array, zip(*results))

print(f"Simulations run      : {n_sims}")
print(f"Forecast horizon (T) : {T} steps")
print()
print(f"Mean Return          : {returns.mean()*100:.2f}%")
print(f"Median Return        : {np.median(returns)*100:.2f}%")
print(f"Std Dev of Returns   : {returns.std()*100:.2f}%")
print(f"P(Loss)              : {(returns<0).mean()*100:.2f}%")
print(f"5th / 95th Percentile: {np.percentile(returns,5)*100:.2f}% / {np.percentile(returns,95)*100:.2f}%")
print(f"Best / Worst Return  : {returns.max()*100:.2f}% / {returns.min()*100:.2f}%")
print()
print(f"Avg Max Drawdown     : {drawdowns.mean()*100:.2f}%")
print(f"Worst Max Drawdown   : {drawdowns.min()*100:.2f}%")

plt.figure(figsize=(10,5))
for i in range(n_sims):  
    plt.plot(simulations[i], alpha=0.5)
plt.title("Monte Carlo GBM Price Paths")
plt.xlabel("Step")
plt.ylabel("Price")
plt.grid(True)
plt.show()

ib.disconnect()