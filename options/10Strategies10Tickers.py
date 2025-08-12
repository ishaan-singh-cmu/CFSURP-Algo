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
use_rth = True
symbols = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC"]

strategies_perf = {}
strategy_stats = {}

for symbol in symbols:
    contract = Stock(symbol, "SMART", "USD")
    bars = ib.reqHistoricalData(contract, '', duration, bar_size, 'TRADES',
                                useRTH=use_rth, formatDate=1)
    df = util.df(bars)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').between_time("09:30", "16:00")
    df['ema20'] = ta.trend.ema_indicator(df['close'], 20)
    df['ema50'] = ta.trend.ema_indicator(df['close'], 50)
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['rsi'] = ta.momentum.rsi(df['close'])
    df['boll_upper'] = ta.volatility.bollinger_hband(df['close'])
    df['boll_lower'] = ta.volatility.bollinger_lband(df['close'])
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
    df['willr'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['high20'] = df['high'].rolling(20).max()
    df['low20'] = df['low'].rolling(20).min()

    strategy_defs = {
        "EMA-MACD-RSI": (
            lambda i: (df['close'].iloc[i] > df['ema20'].iloc[i]) and
                      (df['macd'].iloc[i] > df['macd_signal'].iloc[i]) and
                      (df['rsi'].iloc[i] < 70),
            lambda i: (df['close'].iloc[i] < df['ema20'].iloc[i]) and
                      (df['macd'].iloc[i] < df['macd_signal'].iloc[i]) and
                      (df['rsi'].iloc[i] > 30)
        ),
        "EMA20 Crossover": (
            lambda i: df['close'].iloc[i] > df['ema20'].iloc[i],
            lambda i: df['close'].iloc[i] < df['ema20'].iloc[i]
        ),
        "MACD Crossover": (
            lambda i: df['macd'].iloc[i] > df['macd_signal'].iloc[i],
            lambda i: df['macd'].iloc[i] < df['macd_signal'].iloc[i]
        ),
        "RSI Reversion": (
            lambda i: df['rsi'].iloc[i] < 30,
            lambda i: df['rsi'].iloc[i] > 70
        ),
        "Golden Cross": (
            lambda i: df['sma50'].iloc[i] > df['sma200'].iloc[i],
            lambda i: df['sma50'].iloc[i] < df['sma200'].iloc[i]
        ),
        "Bollinger Mean Reversion": (
            lambda i: df['close'].iloc[i] < df['boll_lower'].iloc[i],
            lambda i: df['close'].iloc[i] > df['boll_upper'].iloc[i]
        ),
        "CCI Dip Buying": (
            lambda i: df['cci'].iloc[i] < -100,
            lambda i: df['cci'].iloc[i] > 100
        ),
        "20-Bar Breakout": (
            lambda i: df['close'].iloc[i] > df['high20'].iloc[i],
            lambda i: df['close'].iloc[i] < df['low20'].iloc[i]
        ),
        "Williams %R": (
            lambda i: df['willr'].iloc[i] < -80,
            lambda i: df['willr'].iloc[i] > -20
        ),
        "Volume Momentum Spike": (
            lambda i: (df['volume'].iloc[i] > 1.5 * df['vol_sma20'].iloc[i]) and
                      (df['close'].iloc[i] > df['close'].iloc[i-1]),
            lambda i: (df['volume'].iloc[i] > 1.5 * df['vol_sma20'].iloc[i]) and
                      (df['close'].iloc[i] < df['close'].iloc[i-1])
        )
    }

    for name, (buy, sell) in strategy_defs.items():
        cash, position = initial_cash, 0
        vals, trades = [], 0
        for i in range(20, len(df)):
            price = df['close'].iloc[i]
            if buy(i) and position == 0:
                position = cash // price
                cash -= position * price
                trades += 1
            elif sell(i) and position > 0:
                cash += position * price
                position = 0
                trades += 1
            vals.append(cash + position * price)

        if not vals:
            continue

        s = pd.Series(vals)
        r = s.pct_change().dropna()
        sharpe = (r.mean()/r.std()) * np.sqrt(252*390) if r.std()>0 else 0
        ret = s.iloc[-1] - initial_cash

        if name not in strategies_perf:
            strategies_perf[name] = s.copy()
            strategy_stats[name] = {'Return ($)': ret,
                                     'Sharpe': sharpe,
                                     'Total Trades': trades}
        else:
            strategies_perf[name] = strategies_perf[name].add(s, fill_value=0)
            stats = strategy_stats[name]
            stats['Return ($)']   += ret
            stats['Sharpe']       += sharpe
            stats['Total Trades'] += trades

ib.disconnect()

combined_initial = initial_cash * len(symbols)

for name, stats in strategy_stats.items():
    series = strategies_perf[name]
    dd = combined_initial - series.min()
    stats['Max Drawdown ($)'] = float(dd)

metrics_df = pd.DataFrame(strategy_stats).T[
    ['Return ($)', 'Sharpe', 'Max Drawdown ($)', 'Total Trades']
].round(1)

norm_ret = mcolors.Normalize(metrics_df['Return ($)'].min(),
                             metrics_df['Return ($)'].max())
norm_sh  = mcolors.Normalize(metrics_df['Sharpe'].min(),
                             metrics_df['Sharpe'].max())
norm_dd  = mcolors.Normalize(metrics_df['Max Drawdown ($)'].min(),
                             metrics_df['Max Drawdown ($)'].max())
cmap     = plt.cm.RdYlGn

cell_colors = []
for _, row in metrics_df.iterrows():
    cell_colors.append([
        cmap(norm_ret(row['Return ($)'])),
        cmap(norm_sh(row['Sharpe'])),
        plt.cm.RdYlGn_r(norm_dd(row['Max Drawdown ($)'])),
        'white'
    ])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                               gridspec_kw={'height_ratios': [3, 1]})

for name, series in strategies_perf.items():
    ax1.plot(series.values, label=name)
ax1.set_title("Strategy Portfolio Value (Aggregated over 10 Stocks)")
ax1.set_ylabel("Portfolio Value ($)")
ax1.set_xticks([])
ax1.grid(True)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)
ax1.ticklabel_format(useOffset=False, style='plain', axis='y')
ax1.yaxis.get_major_formatter().set_scientific(False)

table = ax2.table(cellText=metrics_df.values,
                  rowLabels=metrics_df.index,
                  colLabels=metrics_df.columns,
                  cellColours=cell_colors,
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(6)
table.scale(1, 2)
ax2.axis('off')

plt.tight_layout()
plt.show()