# MisuModels-ML

This project is mainly a predictive software, running a sophisticated ensemble model 
with advanced backtesting to make real-time predictions. It takes in user-inputted
parameters for the model to grow off of, and allows for graphical and numerical analysis
of each model results.

The model uses a weighted combination of Random Forest, Gradient Boosting, and Ridge Regression
to formulate its final predictions. Employing over 40 technical indicators, the code conducts
automated selection of indicators with backtesting and dynamic retraining.

The GUI allows for the user to input various parameters of the model, and then outputs 4 key graphs
depicting model performance measurements, alongside numerical historical gains. Then, it
caches the model to offer a within 30 day prediction of the stock.

## Files Description

- `main.py` - Generates the GUI. Run this to activate the model.
- `backtest.py` - Transmits the user-entered data from the GUI into core_model.
- `core_model.py` - Contains code retraining the ensemble model, with also backtesting.

## Installation

Run 

pip install -r requirements.txt

to install all the key requirements for this code to work. 

## Detailed Description of core_model

The core generation system involves ensemble model architecture. The code runs three models: Random Forest Regression, Gradient Boosting Regression, and Ridge Regression.

Random Forest Regression -- Primarily captures non-linear relationships and helps filter noisy financial data
Gradient Boosting Regression -- Introduced to reduce prediction errors, leveraging sequential tree construction to improve wea learners
Ridge Regression -- Baseline, provides regularized linear predictions to reduce overfitting (common in RFR)

The models are individually trained over the same feature set across each lookback horizon, and their outputs are subsequently combined with weighted averaging. The weights are dynamically determined based on cross-validated performance of each model.

The feature set is self-calculated via pulling data from the yfinance library, including key financial indicators such as Bollinger Bands,
RSI, MACD, various moving averages, and Volume measures. Users can select to enable or disable Feature Selection, which determines
whether all indicators must be used, or if only top predictive ones are employed, to potentially reduce noise.

For performance measurement, the model simulates trading over past historical data, retraining at the interval desired by the user. 
Key performance metrics that get returned are Sharpe ratio, maximum drawdown, win rate, and annualized return. Graphically, the GUI
provides graphs comparing the model's performance versus the stock (denoted as "Buy and Hold Strategy"), alongside various bar charts
depicting trade success and P/L performance.

## Prediction

Upon running any model, the latest trained ensemble is cached. Then, we just leverage most recent data with this model to calculate 
a predicted price. 

## Previous Models and Progress

For the progress and intermediary failures/models that led to this, feel free to look at the following slideshow for key milestones.

https://docs.google.com/presentation/d/12wMapa389-2oXqekd83qkprUOqrkWdvMGo_gs15ECn8/edit?usp=sharing

For code regarding these models, reach out to ezhang24@andrew.cmu.edu for the actual files, alongside any elaboration on code logic or
slideshow content if needed. 
