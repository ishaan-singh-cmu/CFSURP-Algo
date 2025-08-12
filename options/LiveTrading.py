from ib_insync import *
import pandas as pd
import datetime
import time

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  

symbols = ['SPY', 'AAPL', 'MSFT']
contracts = {
    symbol: Stock(symbol, 'SMART', 'USD') for symbol in symbols
}

for contract in contracts.values():
    ib.qualifyContracts(contract)

account_values = ib.accountSummary()
net_liq = float(next(row.value for row in account_values if row.tag == 'NetLiquidation'))
cash_at_risk = 0.1

last_action = {symbol: None for symbol in symbols}

while True:
    print("\n---", datetime.datetime.now(), "---")
    for symbol in symbols:
        contract = contracts[symbol]

        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='2 D',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            df = util.df(bars)
            if df.empty or 'close' not in df:
                print(f"{symbol}: No data.")
                continue

            df['EMA20'] = df['close'].ewm(span=20).mean()
            df['RSI'] = df['close'].rolling(14).apply(
                lambda x: 100 - 100 / (1 + x.pct_change().mean() / x.pct_change().std()) if x.pct_change().std() else 50
            )
            df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            df['Signal'] = df['MACD'].ewm(span=9).mean()

            latest = df.iloc[-1]
            price = latest['close']
            ema = latest['EMA20']
            rsi = latest['RSI']
            macd = latest['MACD']
            signal = latest['Signal']

            quantity = int((net_liq * cash_at_risk) // price)

            position = ib.positions()
            pos_qty = next((p.position for p in position if p.contract.symbol == symbol), 0)

            if price > ema and macd > signal and rsi < 70 and last_action[symbol] != 'buy':
                if pos_qty < 0:
                    ib.placeOrder(contract, MarketOrder('BUY', abs(pos_qty)))
                ib.placeOrder(contract, MarketOrder('BUY', quantity))
                last_action[symbol] = 'buy'
                print(f"{symbol}: BUY {quantity} @ {price}")

            elif price < ema and macd < signal and rsi > 30 and last_action[symbol] != 'sell':
                if pos_qty > 0:
                    ib.placeOrder(contract, MarketOrder('SELL', pos_qty))
                ib.placeOrder(contract, MarketOrder('SELL', quantity))
                last_action[symbol] = 'sell'
                print(f"{symbol}: SELL {quantity} @ {price}")
            else:
                print(f"{symbol}: HOLD | Price: {price:.2f}, EMA: {ema:.2f}, MACD: {macd:.2f}, Signal: {signal:.2f}, RSI: {rsi:.2f}")

        except Exception as e:
            print(f"{symbol}: Error fetching or processing data: {e}")

    time.sleep(60)