import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from ib_insync import IB, Stock, util
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def bs_price(S, K, T, r, sigma, optionType='C'):
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if optionType == 'C':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def _fetch_greek_bars(symbol, start, end, bar_size, vol_window):
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=999)
    stock = Stock(symbol, 'SMART', 'USD')
    bars = ib.reqHistoricalData(
        stock,
        endDateTime=end.strftime('%Y%m%d %H:%M:%S'),
        durationStr=f"{(end - start).days+1} D",
        barSizeSetting=bar_size,
        whatToShow='MIDPOINT',
        useRTH=True,
        formatDate=1
    )
    ib.disconnect()
    if not bars:
        return pd.DataFrame()

    df = util.df(bars).set_index('date').sort_index()
    df.index = df.index.tz_localize(None)
    df['close'] = df['close'].astype(float)
    df['return'] = df['close'].pct_change()
    df['sigma']  = df['return'].rolling(vol_window).std() * np.sqrt(252)
    final_expiry = end
    df['T'] = (final_expiry - df.index).total_seconds() / (252 * 24*3600)

    def greek_row(r):
        S, T, sig = r['close'], max(r['T'], 1e-6), r['sigma']
        K = round(S * 2) / 2
        d1 = (math.log(S/K) + 0.5 * sig**2 * T) / (sig * math.sqrt(T))
        d2 = d1 - sig * math.sqrt(T)
        return pd.Series({
            'strike': K,
            'delta':  norm.cdf(d1),
            'gamma':  norm.pdf(d1)/(S*sig*math.sqrt(T)),
            'theta':  - (S*norm.pdf(d1)*sig)/(2*math.sqrt(T)),
            'vega':   S*norm.pdf(d1)*math.sqrt(T)
        })

    greeks = df.apply(greek_row, axis=1)
    return df.join(greeks)

def backtest_delta_long_signal(symbol, start, end,
                               bar_size='15 mins',
                               vol_window=20,
                               entry_delta=0.6,
                               exit_delta=0.4):
    df = _fetch_greek_bars(symbol, start, end, bar_size, vol_window)
    trades, in_trade = [], False
    for ts, r in df.iterrows():
        if not in_trade and r['delta'] > entry_delta:
            entry_price      = bs_price(r['close'], r['strike'], r['T'], 0, r['sigma'], 'C')
            entry_underlying = r['close']
            entry_strike     = r['strike']
            entry_time       = ts
            in_trade         = True
        elif in_trade and r['delta'] < exit_delta:
            exit_price = bs_price(r['close'], entry_strike, r['T'], 0, r['sigma'], 'C')
            pnl        = exit_price - entry_price
            trades.append({
                'entry_time':       entry_time,
                'exit_time':        ts,
                'underlying_price': entry_underlying,
                'strike':           entry_strike,
                'entry_price':      entry_price,
                'exit_price':       exit_price,
                'pnl':              pnl
            })
            in_trade = False
    return pd.DataFrame(trades)

def backtest_theta_short_signal(symbol, start, end,
                                bar_size='15 mins',
                                vol_window=20,
                                entry_theta_ratio=1.2,
                                exit_theta_ratio=1.1):
    df = _fetch_greek_bars(symbol, start, end, bar_size, vol_window)
    df['theta_ma'] = df['theta'].abs().rolling(vol_window).mean()
    trades, in_trade = [], False
    for ts, r in df.iterrows():
        ratio = (abs(r['theta']) / r['theta_ma']) if r['theta_ma'] else 0
        if not in_trade and ratio > entry_theta_ratio:
            call_p = bs_price(r['close'], r['strike'], r['T'], 0, r['sigma'], 'C')
            put_p  = bs_price(r['close'], r['strike'], r['T'], 0, r['sigma'], 'P')
            entry_credit     = call_p + put_p
            entry_underlying = r['close']
            entry_strike     = r['strike']
            entry_time       = ts
            in_trade         = True
        elif in_trade and ratio < exit_theta_ratio:
            call_c = bs_price(r['close'], entry_strike, r['T'], 0, r['sigma'], 'C')
            put_c  = bs_price(r['close'], entry_strike, r['T'], 0, r['sigma'], 'P')
            exit_cost = call_c + put_c
            pnl = entry_credit - exit_cost
            trades.append({
                'entry_time':       entry_time,
                'exit_time':        ts,
                'underlying_price': entry_underlying,
                'strike':           entry_strike,
                'entry_price':      entry_credit,
                'exit_price':       exit_cost,
                'pnl':              pnl
            })
            in_trade = False
    return pd.DataFrame(trades)

def backtest_vega_squeeze_signal(symbol, start, end,
                                 bar_size='15 mins',
                                 vol_window=20,
                                 entry_vega_ratio=1.2,
                                 exit_vega_ratio=1.1,
                                 offset_pct=0.05):
    df = _fetch_greek_bars(symbol, start, end, bar_size, vol_window)
    df['vega_ma'] = df['vega'].rolling(vol_window).mean()
    trades, in_trade = [], False
    for ts, r in df.iterrows():
        ma = r['vega_ma'] if not np.isnan(r['vega_ma']) else 1
        ratio = r['vega'] / ma
        if not in_trade and ratio > entry_vega_ratio:
            Kc = r['strike'] * (1 + offset_pct)
            Kp = r['strike'] * (1 - offset_pct)
            call_p = bs_price(r['close'], Kc, r['T'], 0, r['sigma'], 'C')
            put_p  = bs_price(r['close'], Kp, r['T'], 0, r['sigma'], 'P')
            entry_cost       = call_p + put_p
            entry_underlying = r['close']
            entry_strikes    = (Kc, Kp)
            entry_time       = ts
            in_trade         = True
        elif in_trade and ratio < exit_vega_ratio:
            ST = r['close']
            Kc, Kp = entry_strikes
            payoff = max(ST - Kc, 0) + max(Kp - ST, 0)
            pnl = payoff - entry_cost
            trades.append({
                'entry_time':       entry_time,
                'exit_time':        ts,
                'underlying_price': entry_underlying,
                'strike':           f"{Kc:.2f}/{Kp:.2f}",
                'entry_price':      entry_cost,
                'exit_price':       payoff,
                'pnl':              pnl
            })
            in_trade = False
    return pd.DataFrame(trades)

def backtest_gamma_scalping_signal(symbol, start, end,
                                   bar_size='15 mins',
                                   vol_window=20,
                                   entry_gamma_ratio=1.5,
                                   exit_gamma_ratio=1.2):
    df = _fetch_greek_bars(symbol, start, end, bar_size, vol_window)
    df['gamma_ma'] = df['gamma'].rolling(vol_window).mean()
    trades, in_trade = [], False
    for ts, r in df.iterrows():
        ma = r['gamma_ma'] if not np.isnan(r['gamma_ma']) else 1
        ratio = r['gamma'] / ma
        if not in_trade and ratio > entry_gamma_ratio:
            call_p = bs_price(r['close'], r['strike'], r['T'], 0, r['sigma'], 'C')
            put_p  = bs_price(r['close'], r['strike'], r['T'], 0, r['sigma'], 'P')
            entry_cost       = call_p + put_p
            entry_underlying = r['close']
            entry_strike     = r['strike']
            entry_time       = ts
            in_trade         = True
        elif in_trade and ratio < exit_gamma_ratio:
            ST = r['close']
            payoff = max(ST - entry_strike, 0) + max(entry_strike - ST, 0)
            pnl = payoff - entry_cost
            trades.append({
                'entry_time':       entry_time,
                'exit_time':        ts,
                'underlying_price': entry_underlying,
                'strike':           entry_strike,
                'entry_price':      entry_cost,
                'exit_price':       payoff,
                'pnl':              pnl
            })
            in_trade = False
    return pd.DataFrame(trades)

class BacktestApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Options Signal Backtester")
        self.geometry("900x700")

        ctrl = ttk.Frame(self); ctrl.pack(fill='x', pady=5)
        ttk.Label(ctrl, text="Symbol:").pack(side='left', padx=5)
        self.symbol_var = tk.StringVar(value="SPY")
        ttk.Entry(ctrl, width=6, textvariable=self.symbol_var).pack(side='left')

        ttk.Label(ctrl, text="Start:").pack(side='left', padx=5)
        self.start_var = tk.StringVar(value="2024-01-01")
        ttk.Entry(ctrl, width=12, textvariable=self.start_var).pack(side='left')

        ttk.Label(ctrl, text="End:").pack(side='left', padx=5)
        self.end_var = tk.StringVar(value="2024-06-01")
        ttk.Entry(ctrl, width=12, textvariable=self.end_var).pack(side='left')

        ttk.Label(ctrl, text="Strategy:").pack(side='left', padx=5)
        self.strategy_var = tk.StringVar(value="Delta Long")
        self.strategy_cb = ttk.Combobox(
            ctrl,
            textvariable=self.strategy_var,
            values=["Delta Long", "Theta Short", "Vega Squeeze", "Gamma Scalping"],
            state='readonly', width=15
        )
        self.strategy_cb.pack(side='left', padx=(0,10))

        ttk.Button(ctrl, text="Run Backtest", command=self.run_backtest).pack(side='left')

        self.summary_txt = tk.Text(self, height=4, state='disabled', bg=self.cget('bg'), bd=0)
        self.summary_txt.pack(fill='x', padx=10, pady=5)

        tbl_frm = ttk.Frame(self); tbl_frm.pack(fill='both', expand=True, padx=10, pady=5)
        cols = ('entry_time','exit_time','underlying_price','strike','entry_price','exit_price','pnl')
        self.tree = ttk.Treeview(tbl_frm, columns=cols, show='headings')
        for c in cols:
            self.tree.heading(c, text=c.replace('_',' ').title())
            self.tree.column(c, anchor='center')
        vsb = ttk.Scrollbar(tbl_frm, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        self.tree.pack(fill='both', expand=True)

        plot_frm = ttk.Frame(self); plot_frm.pack(fill='x', padx=10)
        self.fig, self.ax = plt.subplots(figsize=(6,2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frm)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.get_tk_widget().pack_forget()

    def run_backtest(self):
        try:
            sym   = self.symbol_var.get().strip().upper()
            start = datetime.fromisoformat(self.start_var.get())
            end   = datetime.fromisoformat(self.end_var.get())
        except Exception:
            messagebox.showerror("Input Error", "Invalid dates. Use YYYY-MM-DD.")
            return

        strat = self.strategy_var.get()
        if strat == "Delta Long":
            df = backtest_delta_long_signal(sym, start, end)
        elif strat == "Theta Short":
            df = backtest_theta_short_signal(sym, start, end)
        elif strat == "Vega Squeeze":
            df = backtest_vega_squeeze_signal(sym, start, end)
        else:
            df = backtest_gamma_scalping_signal(sym, start, end)

        if df.empty:
            messagebox.showinfo("No Trades", "No signals generated.")
            return

        total = df['pnl'].sum()
        n     = len(df)
        wins  = (df['pnl'] > 0).sum()
        wr    = wins / n
        summary = (
            f"Symbol: {sym}    Period: {start.date()} → {end.date()}\n"
            f"Strategy: {strat}    Trades: {n}    "
            f"Total P&L: {total:.2f}    Win Rate: {wr:.0%}"
        )
        self.summary_txt.config(state='normal')
        self.summary_txt.delete('1.0','end')
        self.summary_txt.insert('1.0', summary)
        self.summary_txt.config(state='disabled')

        for row in self.tree.get_children():
            self.tree.delete(row)
        for _, r in df.iterrows():
            self.tree.insert('', 'end', values=(
                r.entry_time, r.exit_time,
                f"{r.underlying_price:.2f}", r.strike,
                f"{r.entry_price:.2f}", f"{r.exit_price:.2f}",
                f"{r.pnl:.2f}"
            ))

        cum = df['pnl'].cumsum()
        self.ax.clear()
        self.ax.plot(cum.index, cum.values)
        self.ax.set_title("Cumulative P&L")
        self.ax.set_xlabel("Trade #")
        self.ax.set_ylabel("P&L")
        self.fig.tight_layout()

        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.draw()

if __name__ == '__main__':
    app = BacktestApp()
    app.mainloop()