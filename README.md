# Irish Residential Property Price Forecast

**Live app:** <[Streamlit Link](https://irish-property-price-forecast-swxbjj9xmqeacc9m9fihlu.streamlit.app/)>

Interactive dashboard that forecasts the Irish Residential Property Price Index
12 months ahead, built from CSO data (table HPM09) with a SARIMA model.

![Dashboard screenshot](<img width="1914" height="907" alt="Dashboard" src="https://github.com/user-attachments/assets/6ddf028a-6aff-4b54-b4cf-38cc90876ee9"/>
>
)

## What it does
- Shows the index history and a 12-month forecast with an uncertainty band
- Lets you change the history window, forecast horizon and band width
- Compares the models I tested, with backtest results
- Lists the model's limitations

## Method
1. Downloaded the monthly Residential Property Price Index from the CSO PxStat API.
2. Compared Naive, Naive + drift, Seasonal naive, SARIMA, Prophet and XGBoost.
3. Evaluated with rolling-origin backtesting (train only on data before each test window).
4. Re-tuned SARIMA inside every window so tuning never saw the test data.
5. Checked whether the 95% prediction band was honest, then recalibrated it.

## Results (8 rolling 12-month windows, average MAPE)

| Model | MAPE % |
|---|---|
| SARIMA | 2.14 |
| Prophet + SARIMA | 2.24 |
| Naive + drift | 2.91 |
| Prophet | 3.88 |

- SARIMA beat the trend-aware baseline by about 26%.
- Prophet looked best over the latest 4 windows (1.70%) but fell behind over 8.
  The shorter test was misleading, which is why I extended it.
- XGBoost (lagged changes only) was tested on the shorter 4-window backtest and
  did not beat Prophet or SARIMA (2.44% vs 1.70% and 2.09%).
- Error was small in calm periods (0.3 to 2.5%) but reached about 6% for the
  windows starting September 2020 and 5% for September 2023, both turning points.
- SARIMA's own 95% band covered 84% of months. Widening it by 25% brought
  backtest coverage to about 95% (measured on the same windows, so slightly optimistic).

## Limitations
- Statistical model only: it learns past patterns and does not know about interest
  rates, supply or policy.
- National index, not county level.
- Eight test windows is useful but not conclusive.
- Not financial advice.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Data
Central Statistics Office Ireland, Residential Property Price Index (HPM09),
licensed CC BY 4.0.

