# Options & High-Frequency Trading Strategies  

This folder contains research code, documentation, and tools for **high-frequency trading**, **options trading**, and **derivatives strategies** developed during the **CFSURP 2025** program.  
It includes backtesting utilities, autonomous live trading scripts, Monte Carlo simulations, and educational material on the **Black-Scholes model** and **options strategies**.  

---

## 📂 Files & Descriptions  

### 📈 High-Frequency Trading Scripts  
- [`10Strategies10Tickers.py`](./10Strategies10Tickers.py) — Runs **10 high-frequency trading strategies** over the **last 5 trading days** across **10 different tickers**, with performance visualization.  
- [`10Strategies1Ticker.py`](./10Strategies1Ticker.py) — Runs the **same 10 strategies** but on **a single ticker**, allowing focused performance analysis.  

### 📜 Documentation  
- [`Black_Scholes.pdf`](./Black_Scholes.pdf) — Explains the **Black-Scholes engine** used in this project for options pricing.  
  - Designed to provide **all required knowledge ahead of CMU’s 21-270** class to understand the code implementation.  
- [`Options_Strategies.pdf`](./Options_Strategies.pdf) — A detailed reference on **10 options trading strategies** based on the **Greeks**, including payoff diagrams and risk profiles.  

### 🤖 Live Trading & Backtesting  
- [`LiveTrading.py`](./LiveTrading.py) — An **autonomous live trading system** using **Interactive Brokers TWS** in **Paper Trading mode**.  
- [`OptionsBacktester.py`](./OptionsBacktester.py) — A **GUI-based backtesting tool** for multiple options trading strategies.  
  - Modular design allows adding new strategies easily.  

### 📊 Simulation Tools  
- [`montecarlo.py`](./montecarlo.py) — Implements **Monte Carlo simulations** for option pricing and strategy performance evaluation.  

---

## 📌 Notes  
- All scripts are in **Python 3.10+**.  
- Market data is sourced via **Interactive Brokers API** and/or **Yahoo Finance**.  
- Risk management features are included in several strategies using **Greeks-based hedging**.  

---

## 📬 Contact  
For questions, feedback, or collaboration inquiries, email: **ishaansingh@cmu.edu**  
