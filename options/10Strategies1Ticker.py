from ib_insync import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ta

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

initial_cash = 100000
bar_size = "1 min"
duration = "5 D"
use_rth = False

nvda_contract = Stock("NVDA", "SMART", "USD")
bars = ib.reqHistoricalData(
    nvda_contract,
    endDateTime='',
    durationStr=duration,
    barSizeSetting=bar_size,
    whatToShow='TRADES',
    useRTH=use_rth,
    formatDate=1
)

df = util.df(bars)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
df['sma50'] = df['close'].rolling(window=50).mean()
df['sma200'] = df['close'].rolling(window=200).mean()
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['rsi'] = ta.momentum.rsi(df['close'])
df['boll_upper'] = ta.volatility.bollinger_hband(df['close'])
df['boll_lower'] = ta.volatility.bollinger_lband(df['close'])
df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
df['willr'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
df['vol_sma20'] = df['volume'].rolling(window=20).mean()
df['high20'] = df['high'].rolling(window=20).max()
df['low20'] = df['low'].rolling(window=20).min()

df = df.between_time("09:30", "16:00")
df = df[df.index.dayofweek < 5]

strategies = {}
metrics_table = []

def run_strategy(name, buy_cond, sell_cond):
    cash = initial_cash
    position = 0
    trades = 0
    portfolio = []
    for i in range(20, len(df)):
        price = df['close'].iloc[i]
        if buy_cond(i) and position == 0:
            position = cash // price
            cash -= position * price
            trades += 1
        elif sell_cond(i) and position > 0:
            cash += position * price
            position = 0
        value = cash + position * price
        portfolio.append({'time': df.index[i], 'value': value})
    perf = pd.DataFrame(portfolio).set_index('time')
    perf['returns'] = perf['value'].pct_change().dropna()
    sharpe = (perf['returns'].mean() / perf['returns'].std()) * np.sqrt(252 * 390) if perf['returns'].std() > 0 else 0
    drawdown = (perf['value'].cummax() - perf['value']).max()
    total_return = perf['value'].iloc[-1] - initial_cash
    strategies[name] = perf
    metrics_table.append([
        name,
        total_return,
        sharpe,
        drawdown,
        trades
    ])

run_strategy("EMA-MACD-RSI", 
    lambda i: df['close'].iloc[i] > df['ema20'].iloc[i] and df['macd'].iloc[i] > df['macd_signal'].iloc[i] and df['rsi'].iloc[i] < 70,
    lambda i: df['close'].iloc[i] < df['ema20'].iloc[i] and df['macd'].iloc[i] < df['macd_signal'].iloc[i] and df['rsi'].iloc[i] > 30
)
run_strategy("EMA20 Crossover", 
    lambda i: df['close'].iloc[i] > df['ema20'].iloc[i],
    lambda i: df['close'].iloc[i] < df['ema20'].iloc[i]
)
run_strategy("MACD Crossover", 
    lambda i: df['macd'].iloc[i] > df['macd_signal'].iloc[i],
    lambda i: df['macd'].iloc[i] < df['macd_signal'].iloc[i]
)
run_strategy("RSI Reversion", 
    lambda i: df['rsi'].iloc[i] < 30,
    lambda i: df['rsi'].iloc[i] > 70
)
run_strategy("Golden Cross", 
    lambda i: df['sma50'].iloc[i] > df['sma200'].iloc[i],
    lambda i: df['sma50'].iloc[i] < df['sma200'].iloc[i]
)
run_strategy("Bollinger Mean Reversion", 
    lambda i: df['close'].iloc[i] < df['boll_lower'].iloc[i],
    lambda i: df['close'].iloc[i] > df['boll_upper'].iloc[i]
)
run_strategy("CCI Dip Buying", 
    lambda i: df['cci'].iloc[i] < -100,
    lambda i: df['cci'].iloc[i] > 100
)
run_strategy("20-Bar Breakout", 
    lambda i: df['close'].iloc[i] > df['high20'].iloc[i],
    lambda i: df['close'].iloc[i] < df['low20'].iloc[i]
)
run_strategy("Williams %R", 
    lambda i: df['willr'].iloc[i] < -80,
    lambda i: df['willr'].iloc[i] > -20
)
run_strategy("Volume Momentum Spike", 
    lambda i: df['volume'].iloc[i] > 1.5 * df['vol_sma20'].iloc[i] and df['close'].iloc[i] > df['close'].iloc[i-1],
    lambda i: df['volume'].iloc[i] > 1.5 * df['vol_sma20'].iloc[i] and df['close'].iloc[i] < df['close'].iloc[i-1]
)

ib.disconnect()

fig, ax = plt.subplots(figsize=(18, 10))

all_times = sorted({ts for perf in strategies.values() for ts in perf.index})
fake_x = {ts: i for i, ts in enumerate(all_times)}
for name, perf in strategies.items():
    x_vals = [fake_x[ts] for ts in perf.index]
    ax.plot(x_vals, perf['value'], label=name)

ax.set_xticks([])
ax.set_xlabel("")
ax.set_title("NVDA Strategy Comparison (Trading Hours Only — Days Compressed)")
ax.set_ylabel("Portfolio Value ($)", labelpad=10)
ax.grid(True)
plt.tight_layout(rect=[0.07, 0.55, 0.85, 0.95])

returns = np.array([row[1] for row in metrics_table])
sharpes = np.array([row[2] for row in metrics_table])
drawdowns = np.array([row[3] for row in metrics_table])
trades = [row[4] for row in metrics_table]

norm_return = (returns - returns.min()) / (returns.max() - returns.min() + 1e-8)
norm_sharpe = (sharpes - sharpes.min()) / (sharpes.max() - sharpes.min() + 1e-8)
norm_drawdown = 1 - (drawdowns - drawdowns.min()) / (drawdowns.max() - drawdowns.min() + 1e-8)

cmap = plt.get_cmap('RdYlGn')

table_data = [["Strategy", "Total Return", "Sharpe", "Max Drawdown", "Total Trades"]]
for i, row in enumerate(metrics_table):
    table_data.append([
        row[0],
        f"${row[1]:,.2f}",
        f"{row[2]:.2f}",
        f"${row[3]:,.2f}",
        str(row[4])
    ])

table = plt.table(
    cellText=table_data,
    colLabels=None,
    loc='bottom',
    cellLoc='center',
    bbox=[0, -0.75, 1, 0.55]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 5.0)

for key, cell in table.get_celld().items():
    row, col = key
    if row == 0:
        cell.set_fontsize(12)
        cell.set_text_props(weight='bold')
    elif row > 0:
        if col == 1:
            cell.set_facecolor(cmap(norm_return[row - 1]))
        elif col == 2:
            cell.set_facecolor(cmap(norm_sharpe[row - 1]))
        elif col == 3:
            cell.set_facecolor(cmap(norm_drawdown[row - 1]))

ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9)

plt.show()