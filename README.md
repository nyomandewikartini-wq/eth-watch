# ETH Watch: Real-Time Ethereum Market Monitoring and Forecasting

**ETH Watch** is a Streamlit-based dashboard designed to monitor Ethereum (ETH-USD) market movements, forecast daily closing prices using Long Short-Term Memory (LSTM), and identify unusual market behavior through Complex Event Processing (CEP).

The dashboard combines time-series forecasting, market analytics, and rule-based anomaly monitoring in an interactive interface with a modern dark-purple visual theme. It is developed for statistical analysis, time-series modeling, and Big Data learning purposes.

## Dashboard Features

ETH Watch consists of six main pages:

### 1. Market Overview

Provides an overview of Ethereum market conditions through key market indicators, historical price visualization, and recent market movements.

### 2. LSTM Forecasting

Forecasts the next day's Ethereum closing price using an LSTM neural network trained on historical closing-price data. The page presents forecasting results and relevant model evaluation information.

### 3. CEP Monitoring

Monitors unusual price movements and market events using predefined CEP rules. It supports daily anomaly analysis and intraday monitoring based on available market data.

### 4. Market Analytics

Provides additional statistical insights into Ethereum price movements, returns, volatility, and market behavior.

### 5. Model & Methodology

Explains the forecasting model, model configuration, preprocessing approach, and analytical methods implemented in the dashboard.

### 6. Data Explorer

Allows users to inspect historical Ethereum market data and explore the dataset used in the analysis.

## Methods

### LSTM Price Forecasting

The forecasting component uses a Long Short-Term Memory (LSTM) neural network to model temporal patterns in Ethereum closing prices.

The current model configuration is:

| Parameter | Value |
|---|---|
| Target variable | Daily closing price (`Close`) |
| Lookback window | 14 days |
| LSTM units | 128 |
| Best epoch | 12 |
| Forecast horizon | 1 day ahead |
| Model selection criterion | Lowest validation RMSE |
| Validation RMSE | Approximately 153.3688 USD |

The model uses a training-only fitted scaler to help prevent information leakage during preprocessing. Forecasting results are estimates based on learned historical patterns and do not guarantee future market prices.

### Complex Event Processing (CEP)

CEP is used to identify unusual market behavior based on predefined event-detection rules.

The monitoring component can evaluate price anomalies using rolling statistical thresholds and monitor intraday market activity using available lower-frequency market data.

The detection results depend on the selected parameters, available observations, and rules implemented in the application. An anomaly indicates an observation that meets the specified detection criteria; it does not necessarily indicate a market error or a future price movement.

### Market Analytics

The dashboard provides statistical visualizations to support the exploration of Ethereum price dynamics, returns, and volatility. These analyses complement the forecasting and event-monitoring components.

## Data Source

Historical Ethereum market data is obtained from **Yahoo Finance** using the `ETH-USD` ticker through the `yfinance` Python library.

The dashboard uses daily historical data for price analysis and forecasting. Historical data is refreshed when requested by the application.

For intraday monitoring, the dashboard uses available one-minute market candles from Yahoo Finance, with periodic refresh while the relevant page is active. This is not a direct exchange tick feed, and data availability and update timing depend on the data provider.

## Project Structure

The project structure may vary depending on the final repository configuration. The main application and model artifacts are expected to be organized as follows:

```text
ETH_Watch/
├── ETH_Watch_refined.py
├── requirements.txt
├── .python-version
├── .gitignore
├── README.md
└── models/
    ├── eth_lstm_best.keras
    ├── eth_close_scaler.joblib
    └── eth_lstm_config.json
```

The model artifacts must be stored at the paths expected by the application. If the current implementation uses a different directory structure, update this section to reflect the actual repository.

## Requirements

The project uses Python and the libraries required for data processing, visualization, forecasting, and dashboard development.

Key dependencies include:

- Python 3.11
- Streamlit
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- Joblib
- Plotly
- yfinance

For the complete and reproducible dependency list, refer to `requirements.txt`.

## Run Locally

### 1. Clone the repository

Replace `<REPOSITORY_URL>` with the URL of your GitHub repository.

```bash
git clone <REPOSITORY_URL>
cd ETH_Watch
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run ETH_Watch_refined.py
```

Streamlit will provide a local URL that you can open in your browser.

## Deployment

ETH Watch can be deployed using **Streamlit Community Cloud**.

General deployment steps:

1. Push the project files to a GitHub repository.
2. Open Streamlit Community Cloud and connect the repository.
3. Select `ETH_Watch_refined.py` as the main application file.
4. Ensure that `requirements.txt` and the required model artifacts are available in the expected locations.
5. Deploy the application and verify that all dashboard pages and data-dependent features work correctly.

Before deployment, confirm that the model files are included or otherwise accessible to the application and that the repository does not contain unnecessary large files or sensitive information.

## Limitations

- **Forecast uncertainty:** LSTM predictions are estimates and may not capture sudden market changes or unforeseen events.
- **Changing market conditions:** Model performance may vary across different market regimes.
- **Data availability:** Data freshness, completeness, and intraday coverage depend on Yahoo Finance.
- **Rule-based monitoring:** CEP anomaly detection depends on the implemented rules and selected thresholds.
- **Forecasting scope:** The current LSTM configuration forecasts one day ahead rather than providing a guaranteed long-term outlook.
- **No investment advice:** The dashboard is an analytical and educational tool, not a financial recommendation.

## Disclaimer

ETH Watch is developed for educational purposes, statistical analysis, and demonstration of time-series forecasting and event-monitoring techniques. Its outputs should not be interpreted as guarantees of future market performance or as recommendations to buy, hold, or sell cryptocurrency.

Users are responsible for conducting their own research and evaluating the risks associated with cryptocurrency markets.

## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) — market data source
- [Streamlit](https://streamlit.io/) — interactive dashboard framework
- [TensorFlow](https://www.tensorflow.org/) — LSTM model implementation
- [Plotly](https://plotly.com/python/) — interactive data visualization