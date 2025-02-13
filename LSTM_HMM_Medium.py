# Import all necessary libraries
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import logging
import re
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from datetime import datetime
import yfinance as yf
import mplfinance as mpf
from scipy.stats import norm
from plotly.subplots import make_subplots
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from hmmlearn import hmm
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.units import inch
import vectorbt as vbt
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


###############################
# Data download and preprocessing
###############################
def validate_date(date_str):
    """
    Validates if the date is in YYYY-MM-DD format.
    Returns True if valid, otherwise False.
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def get_end_date():
    """
    Prompts the user to enter a valid end date.
    """
    while True:
        end_date = input("Enter the end date (format YYYY-MM-DD): ").strip()
        if validate_date(end_date):
            return end_date
        else:
            logging.error("Invalid date format. Please try again using YYYY-MM-DD format.")


def download_data(ticker, start_date, end_date):
    """
    Downloads historical data for the specified ticker using yfinance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        logging.error("No data available for the specified ticker and date range.")
        exit(1)
    logging.info(f"Data downloaded for {ticker} from {start_date} to {end_date}.")
    logging.info(f"Available columns: {data.columns.tolist()}")
    return data


def preprocess_data(data, lag_p=7):
    """
    Preprocesses the data by calculating returns, lagged features, and technical indicators.
    """
    # If columns have a MultiIndex, remove the second level.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        logging.info("Removed second level of MultiIndex from columns.")

    if 'Close' in data.columns:
        close_prices = data['Close']
    else:
        logging.error("The 'Close' column is not present in the data.")
        exit(1)

    # Calculate 5-day returns.
    returns = close_prices.pct_change(5).dropna()
    data['Return'] = returns

    # Calculate lagged features.
    for l in range(1, lag_p + 1):
        data[f'lag_{l}'] = close_prices.shift(l)
    data = data.dropna().copy()

    # Calculate technical indicators (SMA and RSI).
    data.loc[:, 'SMA_20'] = data['Close'].rolling(window=20).mean()
    data.loc[:, 'SMA_50'] = data['Close'].rolling(window=50).mean()

    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    RS = avg_gain / avg_loss
    data.loc[:, 'RSI'] = 100 - (100 / (1 + RS))

    data = data.dropna()
    
    expected_features = ['Return']
    missing_features = [feat for feat in expected_features if feat not in data.columns]
    if missing_features:
        logging.error(f"Missing expected features: {missing_features}")
        exit(1)

    X = data[expected_features].values
    y = returns.loc[data.index].values

    return X, y, data


###############################
# Visualization: Technical Indicators
###############################
def plot_technical_indicators(data):
    """
    Displays and saves candlestick charts with moving averages and RSI using mplfinance.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # First plot: Candlestick with SMA_20 and SMA_50.
    df = data[['Open', 'High', 'Low', 'Close']]
    addplots = [
        mpf.make_addplot(data['SMA_20'], color='blue', width=2),
        mpf.make_addplot(data['SMA_50'], color='red', width=2)
    ]
    fig, ax = mpf.plot(df, type='candle', addplot=addplots, returnfig=True, figsize=(14, 7),
                        title='Technical Indicators (Candlestick Chart + SMA_20 - SMA_50)', style='yahoo')
    plt.savefig('technical_indicators.png')
    plt.show()
    plt.close()

    # Second plot: Candlestick with SMA and RSI.
    addplots = [
        mpf.make_addplot(data['SMA_20'], panel=0, color='blue', width=2),
        mpf.make_addplot(data['SMA_50'], panel=0, color='red', width=2),
        mpf.make_addplot(data['RSI'], panel=1, color='purple', width=1)
    ]
    fig, axes = mpf.plot(df, type='candle', addplot=addplots, returnfig=True, figsize=(14, 7),
                           title='Technical Indicators (Candlestick + RSI)', panel_ratios=(3,1), style='yahoo')
    axes[0].set_ylabel('Price')
    axes[1].set_ylabel('RSI')
    axes[1].axhline(30, color='red', linestyle='--')
    axes[1].axhline(70, color='green', linestyle='--')
    plt.savefig('rsi.png')
    plt.show()
    plt.close()


###############################
# HMM Modeling
###############################
def determine_optimal_states(X, max_states=5):
    """
    Determines the optimal number of HMM states using BIC and AIC.
    Returns the optimal number of states (based on the lowest BIC) and the scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    bic = []
    aic = []
    n_components_range = range(1, max_states + 1)
    for n in n_components_range:
        model = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=1000, random_state=0)
        model.fit(X_scaled)
        bic.append(model.bic(X_scaled))
        aic.append(model.aic(X_scaled))

    plt.figure(figsize=(10, 5))
    plt.plot(n_components_range, bic, label='BIC')
    plt.plot(n_components_range, aic, label='AIC')
    plt.xlabel('Number of States')
    plt.ylabel('Value')
    plt.title('Optimal Number of States for HMM Selection')
    plt.legend()
    plt.grid(True)
    plt.close()

    optimal_states = n_components_range[np.argmin(bic)]
    logging.info(f"Optimal number of states determined by BIC: {optimal_states}")
    return optimal_states, scaler


def apply_hmm(X, hmm_model):
    """
    Applies the HMM to preprocessed data.
    Returns hidden states and membership probabilities.
    """
    X_scaled = hmm_model.scaler.transform(X)
    hidden_states = hmm_model.hmm_model.predict(X_scaled)
    state_probs = hmm_model.hmm_model.predict_proba(X_scaled)
    logging.info(f"HMM clustering completed with {hmm_model.hmm_model.n_components} states.")
    return hidden_states, state_probs


class HMMModel:
    """
    Manages the Hidden Markov Model and its associated scaler.
    """
    def __init__(self, hmm_model, scaler):
        self.hmm_model = hmm_model
        self.scaler = scaler


def plot_hmm_transition_matrix(hmm_model):
    """
    Plots the transition matrix of the HMM.
    """
    if not hasattr(hmm_model.hmm_model, "transmat_"):
        logging.error("HMM model does not have a transition matrix.")
        return

    import seaborn as sns
    transition_matrix = hmm_model.hmm_model.transmat_
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=[f"State {i}" for i in range(transition_matrix.shape[0])],
                yticklabels=[f"State {i}" for i in range(transition_matrix.shape[0])])
    plt.title("HMM Transition Matrix")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.show()


def save_hmm_model(hmm_model, filename):
    """
    Saves the HMMModel using joblib.
    """
    dump(hmm_model, filename)
    logging.info(f"HMM model saved to {filename}.")


def load_hmm_model(filename):
    """
    Loads the HMMModel using joblib.
    """
    model = load(filename)
    logging.info(f"HMM model loaded from {filename}.")
    return model


###############################
# LSTM Modeling
###############################
def build_lstm_model(input_shape, num_steps):
    """
    Builds and compiles the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(num_steps))  # Output with num_steps future values
    model.compile(optimizer='nadam', loss='mse', metrics=['mae'])
    return model


def plot_training_history(history):
    """
    Plots and saves training history.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Progression During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.show()
    plt.close()


def split_data(X, y, train_ratio):
    """
    Splits data into training and testing sets.
    """
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logging.info(f"Data split: {train_ratio*100:.2f}% training, {100-train_ratio*100:.2f}% testing.")
    logging.info(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}.")
    return X_train, X_test, y_train, y_test


def prepare_data_multi_step(X, y, time_step, num_steps):
    """
    Prepares data for multi-step LSTM model.
    """
    X_data, y_data = [], []
    for i in range(len(X) - time_step - num_steps + 1):
        X_data.append(X[i:i + time_step])
        y_data.append(y[i + time_step:i + time_step + num_steps])
    return np.array(X_data), np.array(y_data)


###############################
# Vectorbt Backtesting Functions
###############################
def save_optimized_m(m_value, ticker, filename_prefix="optimized_m_"):
    """
    Saves the optimized multiplier value to a file including the ticker in its name.
    """
    filename = f"{filename_prefix}{ticker}.txt"
    with open(filename, "w") as f:
        f.write(str(m_value))
    logging.info(f"File {filename} updated with optimized m: {m_value}")


def load_optimized_m(ticker, filename_prefix="optimized_m_"):
    """
    Loads the optimized multiplier value from a file. Returns None if not found.
    """
    filename = f"{filename_prefix}{ticker}.txt"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            m_value = float(f.read().strip())
        logging.info(f"Optimized m loaded from {filename}: {m_value}")
        return m_value
    else:
        logging.info(f"File {filename} not found. No optimized m value.")
        return None


def compute_dynamic_signals_with_multiplier(prices, y_pred_multi, y_test_multi, std_return_train, window_size=20, multiplier=0.5):
    """
    Computes dynamic trading signals based on multi-step predictions.
    Returns:
      - signals: array of signals (1=LONG, -1=SHORT, 0=Neutral)
      - dynamic_thresholds: dynamic thresholds for each day
      - weighted_mean_returns: weighted mean returns for each day
    """
    n = y_pred_multi.shape[0]
    num_steps = y_pred_multi.shape[1]
    signals = np.zeros(n, dtype=int)
    dynamic_thresholds = np.zeros(n)
    weighted_mean_returns = np.zeros(n)
    
    for i in range(n):
        start_idx = i - window_size if i >= window_size else 0
        recent_prices = prices.iloc[start_idx:i+1]
        recent_returns = recent_prices.pct_change().dropna()
        recent_std = np.std(recent_returns) if len(recent_returns) > 0 else std_return_train

        recent_y_true = y_test_multi[start_idx:i+1, :]
        recent_y_pred = y_pred_multi[start_idx:i+1, :]
        
        if recent_y_true.shape[0] > 0:
            mse = np.zeros(num_steps)
            for step in range(num_steps):
                mse[step] = mean_squared_error(recent_y_true[:, step], recent_y_pred[:, step])
            rmse = np.sqrt(mse)
            mean_rmse = np.mean(rmse)
            weights = 1 / (rmse + 1e-8)
            normalized_weights = weights / np.sum(weights)
        else:
            mean_rmse = 0
            normalized_weights = np.ones(num_steps) / num_steps

        max_threshold = multiplier * recent_std
        dynamic_threshold = min(max_threshold, recent_std / (1 + mean_rmse))
        dynamic_thresholds[i] = dynamic_threshold

        weighted_mean_return = np.dot(normalized_weights, y_pred_multi[i, :])
        weighted_mean_returns[i] = weighted_mean_return
        
        if weighted_mean_return > dynamic_threshold:
            signals[i] = 1
        elif weighted_mean_return < -dynamic_threshold:
            signals[i] = -1
        else:
            signals[i] = 0

    return signals, dynamic_thresholds, weighted_mean_returns


def optimize_max_threshold(prices, y_pred_multi, y_test_multi, y_train, window_size=20, multipliers=np.arange(0.1, 1.05, 0.1)):
    """
    Optimizes the multiplier for the dynamic threshold using backtesting.
    Returns the best multiplier, its metric (e.g., Sharpe Ratio), and a dictionary of results.
    """
    best_metric = -np.inf
    best_multiplier = None
    results = {}
    std_return_train = np.std(y_train)
    
    for m in multipliers:
        signals, dynamic_thresholds, _ = compute_dynamic_signals_with_multiplier(
            prices, y_pred_multi, y_test_multi, std_return_train, window_size, multiplier=m
        )
        signals_series = pd.Series(signals, index=prices.index)
        entries = signals_series == 1
        exits = signals_series == -1
        
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=100000,
            freq='1D'
        )
        sharpe = portfolio.sharpe_ratio()
        results[m] = sharpe
        if sharpe > best_metric:
            best_metric = sharpe
            best_multiplier = m

    return best_multiplier, best_metric, results


def aggregate_optimal_multiplier(prices, y_pred_multi, y_test_multi, y_train, window_size=20, segment_length=20, step=10, multipliers=np.arange(0.1, 1.05, 0.1)):
    """
    Splits the period into segments, optimizes multiplier m for each, and aggregates (e.g., using median).
    Returns aggregated m and list of optimal m values per segment.
    """
    optimal_m_values = []
    for start in range(0, len(prices) - segment_length + 1, step):
        segment_prices = prices.iloc[start:start+segment_length]
        segment_y_pred_multi = y_pred_multi[start:start+segment_length]
        segment_y_test_multi = y_test_multi[start:start+segment_length]
        std_return_train_seg = np.std(y_train)
        best_m, best_metric, _ = optimize_max_threshold(segment_prices, segment_y_pred_multi, segment_y_test_multi, y_train, window_size, multipliers)
        optimal_m_values.append(best_m)
    aggregated_m = np.median(optimal_m_values)
    return aggregated_m, optimal_m_values


def backtest_vectorbt_full(prices, y_pred_multi, y_test_multi, std_return_train, window_size=20,
                           initial_capital=100000, freq='1D', multiplier=0.5):
    """
    Performs backtesting with vectorbt using dynamic signals.
    Orders are executed at closing price for evaluation.
    """
    signals, dynamic_thresholds, weighted_mean_returns = compute_dynamic_signals_with_multiplier(
        prices, y_pred_multi, y_test_multi, std_return_train, window_size, multiplier=multiplier
    )
    
    signals_series = pd.Series(signals, index=prices.index)
    entries = signals_series == 1
    exits = signals_series == -1

    portfolio = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=initial_capital,
        freq=freq
    )

    # Save vectorbt-generated plots
    try:
        fig_orders = portfolio.plot_orders()
        fig_orders.write_image("vectorbt_orders.png")
    except Exception as e:
        print(f"Error generating orders plot: {e}")
    
    try:
        fig_equity = portfolio.plot()
        fig_equity.write_image("vectorbt_equity_curve.png")
    except Exception as e:
        print(f"Error generating equity curve plot: {e}")
    
    return portfolio, signals_series, dynamic_thresholds, weighted_mean_returns


def plot_vectorbt_portfolio(portfolio):
    """
    Displays the portfolio's equity curve interactively.
    """
    portfolio.plot(title="Equity Curve").show()


def print_vectorbt_portfolio_metrics(portfolio):
    """
    Prints portfolio performance metrics calculated by vectorbt.
    """
    stats = portfolio.stats()
    print("=== Portfolio Metrics (vectorbt) ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Calculate CAGR manually if not present.
    cagr = stats.get("CAGR")
    if cagr is None:
        initial_value = portfolio.value().iloc[0]
        final_value = portfolio.value().iloc[-1]
        n = len(portfolio.value())
        cagr = (final_value / initial_value) ** (252 / n) - 1
    
    total_return = portfolio.total_return()
    sharpe_ratio = portfolio.sharpe_ratio()
    max_drawdown = portfolio.max_drawdown()
    
    print("\n=== Specific Metrics ===")
    print(f"Total Return [%]: {total_return:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown [%]: {max_drawdown:.2%}")


def assign_state_labels(data_with_features, num_states):
    """
    Analyzes each state and assigns a label based on average return.
    """
    state_labels = {}
    state_stats = {}
    for i in range(num_states):
        state_data = data_with_features[data_with_features['hidden_state'] == i]
        returns = state_data['Close'].pct_change().dropna()
        average_return = returns.mean()
        state_stats[i] = average_return

    percentile_25 = np.percentile(list(state_stats.values()), 25)
    percentile_75 = np.percentile(list(state_stats.values()), 75)

    for i, avg_return in state_stats.items():
        if avg_return > percentile_75:
            label = 'Bullish Phase'
        elif avg_return < percentile_25:
            label = 'Bearish Phase'
        else:
            label = 'Neutral Phase'
        state_labels[i] = label

    return state_labels


def plot_hmm_probabilities(data_with_features, hmm_model):
    """
    Colors the closing price chart based on dominant HMM state labels.
    Saves the plot as an image.
    """
    num_states = hmm_model.hmm_model.n_components
    state_prob_columns = [f'state_prob_{i}' for i in range(num_states)]
    missing_cols = [col for col in state_prob_columns if col not in data_with_features.columns]
    if missing_cols:
        logging.error(f"Missing state probability columns: {missing_cols}")
        return

    data_with_features['dominant_state'] = data_with_features[state_prob_columns].idxmax(axis=1)
    data_with_features['dominant_state'] = data_with_features['dominant_state'].apply(lambda x: int(x.split('_')[-1]))

    state_labels = assign_state_labels(data_with_features, num_states)
    label_color_map = {'Bullish Phase': 'green', 'Bearish Phase': 'red', 'Neutral Phase': 'lightblue'}
    color_map = {state: label_color_map.get(label, 'gray') for state, label in state_labels.items()}
    data_with_features['color'] = data_with_features['dominant_state'].map(color_map)

    plt.figure(figsize=(14, 7))
    plt.scatter(data_with_features.index, data_with_features['Close'], c=data_with_features['color'], label='Closing Price')
    plt.title('Closing Price Colored by Market Phase')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)

    handles = []
    for state, label in state_labels.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color_map[state], markersize=10))
    plt.legend(handles=handles, title='Market Phase', loc='upper left')
    plt.savefig('hmm_probabilities.png')
    plt.close()


def calculate_performance_metrics(backtest_results):
    """
    Calculates performance metrics from backtest results.
    """
    total_return = (backtest_results['Total_Value'].iloc[-1] - backtest_results['Total_Value'].iloc[0]) / backtest_results['Total_Value'].iloc[0]
    num_years = (backtest_results.index[-1] - backtest_results.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / num_years) - 1
    cumulative_returns = backtest_results['Total_Value']
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    daily_returns = backtest_results['Portfolio_Returns']
    average_daily_return = daily_returns.mean()
    daily_return_std = daily_returns.std()
    trading_days = 252
    annualized_return_perf = (1 + average_daily_return) ** trading_days - 1
    annualized_volatility = daily_return_std * np.sqrt(trading_days)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return_perf - risk_free_rate) / annualized_volatility

    metrics = {
        'Total Return': f"{total_return * 100:.2f}%",
        'Annualized Return': f"{annualized_return * 100:.2f}%",
        'Maximum Drawdown': f"{max_drawdown * 100:.2f}%",
        'Annualized Sharpe Ratio': f"{sharpe_ratio:.2f}"
    }
    return metrics


###############################
# PDF Report Generation
###############################
def generate_pdf_report(portfolio_metrics_text, ticker, portfolio=None, output_filename=None, suggested_action=None):
    """
    Generates a PDF report that includes:
      - A header with the current date and ticker.
      - A section with performance metrics (passed as a multiline string).
      - (Optional) A paragraph with the Suggested Action.
      - A full-page image of the cumulative equity curve (vectorbt_equity_curve.png).
      
    The PDF file name will include the ticker if not provided.
    """
    # Set output filename with ticker if not provided
    if output_filename is None:
        output_filename = f"backtest_report_{ticker}.pdf"
    else:
        if ticker not in output_filename:
            base, ext = os.path.splitext(output_filename)
            output_filename = f"{base}_{ticker}{ext}"

    current_dir = os.getcwd()
    current_date = datetime.now().strftime('%Y-%m-%d')

    doc = SimpleDocTemplate(
        output_filename,
        pagesize=LETTER,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )

    elements = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenterTitle', alignment=TA_CENTER, fontSize=20, spaceAfter=20))
    styles.add(ParagraphStyle(name='Right', alignment=TA_RIGHT, fontSize=10))
    styles.add(ParagraphStyle(name='Left', alignment=TA_LEFT, fontSize=12))

    # Header
    title_text = f"Backtest Report for {ticker}"
    header_text = f"Date: {current_date} <br/> Ticker: {ticker}"
    elements.append(Paragraph(title_text, styles['CenterTitle']))
    elements.append(Paragraph(header_text, styles['Right']))
    elements.append(Spacer(1, 12))

    # Performance Metrics Section
    metrics_html = portfolio_metrics_text.replace('\n', '<br/>')
    elements.append(Paragraph(metrics_html, styles['Left']))
    elements.append(Spacer(1, 24))
    
    # Optional Suggested Action Section
    if suggested_action:
        action_text = f"Suggested action: {suggested_action}"
        elements.append(Paragraph(action_text, styles['Left']))
        elements.append(Spacer(1, 24))

    # Page break before image
    elements.append(PageBreak())

    # Generate or use existing equity curve image
    if portfolio is not None:
        try:
            fig_equity = portfolio.plot()
            equity_file = os.path.join(current_dir, "vectorbt_equity_curve.png")
            fig_equity.write_image(equity_file)
        except Exception as e:
            logging.error(f"Error generating equity curve image: {e}")
            equity_file = os.path.join(current_dir, "vectorbt_equity_curve.png")
    else:
        equity_file = os.path.join(current_dir, "vectorbt_equity_curve.png")

    try:
        equity_file_abs = os.path.abspath(equity_file)
        # Set image dimensions (6.0" x 9.0") to fit the available area.
        elements.append(Image(equity_file_abs, width=6.0*inch, height=9.0*inch))
        elements.append(Spacer(1, 12))
    except Exception as e:
        logging.error(f"Error adding equity curve image: {e}")

    # Conclusion (optional)
    conclusion_text = "This report was automatically generated using vectorbt and other analysis tools."
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(conclusion_text, styles['Left']))

    try:
        doc.build(elements)
        logging.info(f"PDF report successfully generated: {output_filename}")
    except Exception as e:
        logging.error(f"Error generating PDF: {e}")


def sanitize_filename(filename):
    """
    Removes or replaces invalid characters in file names.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


###############################
# Main function: Execution flow
###############################
def main():
    # Get ticker input from user
    while True:
        ticker = input("Enter the stock ticker: ").strip().upper()
        if ticker:
            break
        else:
            logging.error("Invalid ticker. Please enter a valid ticker.")
    logging.info(f"Accepted ticker: {ticker}")
    ticker_sanitized = sanitize_filename(ticker)
    start_date = '2000-01-01'
    scaler_filename = f'scaler_LSTM_{ticker}.joblib'
    model_filename = f'LSTM_model_{ticker}.keras'
    hmm_model_filename = f'HMM_model_{ticker}.joblib'
    time_step = 25  # Input sequence length
    num_steps = 5   # Number of future steps to predict

    # Get end date input
    end_date = get_end_date()

    # Download historical data
    data = download_data(ticker, start_date, end_date)
    print(data.tail())

    # Preprocess data
    X, y, data_with_features = preprocess_data(data)
    if 'Date' not in data_with_features.columns:
        data_with_features = data_with_features.reset_index().rename(columns={'index': 'Date'})

    # Get training split percentage from user
    while True:
        split_percentage = input("Enter the percentage of data to use for training (0-100): ").strip()
        try:
            split_percentage = float(split_percentage)
            if 0 <= split_percentage <= 100:
                break
            else:
                logging.error("Percentage must be between 0 and 100.")
        except ValueError:
            logging.error("Please enter a valid numeric value.")
    train_ratio = split_percentage / 100

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio)

    # HMM Modeling
    try:
        hmm_model = load_hmm_model(hmm_model_filename)
        logging.info("HMM model loaded successfully.")
        hidden_states_train, state_probs_train = apply_hmm(X_train, hmm_model)
        hidden_states_test, state_probs_test = apply_hmm(X_test, hmm_model)
    except FileNotFoundError:
        logging.info("HMM model not found. Training a new one.")
        optimal_states, scaler_hmm = determine_optimal_states(X_train, max_states=10)
        optimal_states = 3  # Force optimal states to 3 (as per your logic)
        hmm_instance = hmm.GaussianHMM(n_components=optimal_states, covariance_type="full", n_iter=1000, random_state=0)
        X_train_scaled_hmm = scaler_hmm.transform(X_train)
        hmm_instance.fit(X_train_scaled_hmm)
        hidden_states_train = hmm_instance.predict(X_train_scaled_hmm)
        state_probs_train = hmm_instance.predict_proba(X_train_scaled_hmm)
        X_test_scaled_hmm = scaler_hmm.transform(X_test)
        hidden_states_test = hmm_instance.predict(X_test_scaled_hmm)
        state_probs_test = hmm_instance.predict_proba(X_test_scaled_hmm)
        hmm_model = HMMModel(hmm_model=hmm_instance, scaler=scaler_hmm)
        save_hmm_model(hmm_model, hmm_model_filename)
        plot_hmm_transition_matrix(hmm_model)

    # Add HMM states and probabilities to training and testing data
    data_with_features_train = data_with_features.iloc[:len(X_train)].copy()
    data_with_features_train['hidden_state'] = hidden_states_train
    for i in range(hmm_model.hmm_model.n_components):
        data_with_features_train[f'state_prob_{i}'] = state_probs_train[:, i]

    data_with_features_test = data_with_features.iloc[len(X_train):].copy()
    data_with_features_test['hidden_state'] = hidden_states_test
    for i in range(hmm_model.hmm_model.n_components):
        data_with_features_test[f'state_prob_{i}'] = state_probs_test[:, i]

    data_with_features = pd.concat([data_with_features_train, data_with_features_test])

    # Visualization: technical indicators and HMM probabilities
    plot_technical_indicators(data_with_features)
    plot_hmm_probabilities(data_with_features, hmm_model)

    # Prepare features for LSTM: 'Return' and state probabilities
    feature_columns_extended = ['Return'] + [f'state_prob_{i}' for i in range(hmm_model.hmm_model.n_components)]
    X_train_extended = data_with_features_train[feature_columns_extended].values
    X_test_extended = data_with_features_test[feature_columns_extended].values

    # Log descriptive statistics
    logging.info(f"Mean 5-day returns: {np.mean(y_train):.4f}")
    logging.info(f"Standard deviation of 5-day returns: {np.std(y_train):.4f}")
    logging.info(f"Maximum return: {np.max(y_train):.4f}")
    logging.info(f"Minimum return: {np.min(y_train):.4f}")
    logging.info(f"25th percentile: {np.percentile(y_train, 25):.4f}")
    logging.info(f"75th percentile: {np.percentile(y_train, 75):.4f}")

    # Neural model training decision
    while True:
        choice = input("Do you want to retrain the neural model? [Y/N]: ").strip().upper()
        if choice in ['Y', 'N']:
            break
        else:
            logging.error("Invalid input. Please enter 'Y' or 'N'.")

    if choice == 'Y':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_extended)
        X_test_scaled = scaler.transform(X_test_extended)
        dump(scaler, scaler_filename)
        logging.info(f"Scaler saved to {scaler_filename}.")

        X_train_reshaped, y_train_multi = prepare_data_multi_step(X_train_scaled, y_train, time_step, num_steps)
        X_test_reshaped, y_test_multi = prepare_data_multi_step(X_test_scaled, y_test, time_step, num_steps)

        model = build_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]), num_steps)
        model.summary()

        checkpoint = ModelCheckpoint(filepath=model_filename, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

        history = model.fit(
            X_train_reshaped, y_train_multi,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_reshaped, y_test_multi),
            callbacks=[reduce_lr, checkpoint, early_stop],
            verbose=2
        )

        plot_training_history(history)
        best_model = load_model(model_filename)
        logging.info("Neural network trained and saved successfully.")
    elif choice == 'N':
        try:
            best_model = load_model(model_filename)
            scaler = load(scaler_filename)
            hmm_model = load_hmm_model(hmm_model_filename)
            logging.info("Model, scaler, and HMM loaded successfully.")
            plot_hmm_transition_matrix(hmm_model)
        except (OSError, IOError) as e:
            logging.error(f"Error loading model, scaler, or HMM: {e}")
            exit(1)
        
        data_with_features_train = data_with_features.iloc[:len(X_train)].copy()
        data_with_features_train['hidden_state'] = hidden_states_train
        for i in range(hmm_model.hmm_model.n_components):
            data_with_features_train[f'state_prob_{i}'] = state_probs_train[:, i]

        data_with_features_test = data_with_features.iloc[len(X_train):].copy()
        data_with_features_test['hidden_state'] = hidden_states_test
        for i in range(hmm_model.hmm_model.n_components):
            data_with_features_test[f'state_prob_{i}'] = state_probs_test[:, i]

        data_with_features = pd.concat([data_with_features_train, data_with_features_test])
        feature_columns_lstm = ['Return'] + [f'state_prob_{i}' for i in range(hmm_model.hmm_model.n_components)]
        X_train_extended = data_with_features_train[feature_columns_lstm].values
        X_test_extended = data_with_features_test[feature_columns_lstm].values
        X_train_scaled = scaler.transform(X_train_extended)
        X_test_scaled = scaler.transform(X_test_extended)
        X_train_reshaped, y_train_multi = prepare_data_multi_step(X_train_scaled, y_train, time_step, num_steps)
        X_test_reshaped, y_test_multi = prepare_data_multi_step(X_test_scaled, y_test, time_step, num_steps)

        y_pred_multi = best_model.predict(X_test_reshaped)
        mse_per_step = [mean_squared_error(y_test_multi[:, step], y_pred_multi[:, step]) for step in range(num_steps)]
        logging.info(f"MSE for each future step: {mse_per_step}")

        future_input = X_test_reshaped[-1].reshape((1, time_step, X_test_reshaped.shape[2]))
        future_predictions = best_model.predict(future_input).flatten()
        logging.info(f"Future Predictions: {future_predictions}")
        y_test_single = y_test_multi[:, 0]

        plt.figure(figsize=(14, 7))
        plt.plot(y_test_single, label='Actual', color='blue', marker='o', linestyle='-', alpha=0.7)
        plt.plot(y_pred_multi[:, 0], label='Predicted', color='green', marker='o', linestyle='--', alpha=0.7)
        future_steps = range(len(y_test_single), len(y_test_single) + len(future_predictions))
        plt.plot(future_steps, future_predictions, label='Future Predictions', color='red', marker='o', linestyle=':')
        plt.title('Actual vs. Predicted vs. Future Predictions')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('actual_vs_predicted_future.png')
        plt.show()
        plt.close()

        errors = y_test_multi - y_pred_multi
        mse_per_step = np.mean(errors**2, axis=0)
        std_dev_per_step = np.sqrt(mse_per_step)
        last_close_price = data_with_features['Close'].iloc[-1]
        future_prices = [last_close_price]
        lower_bounds = []
        upper_bounds = []
        confidence_interval = 0.95
        z_score = norm.ppf(1 - (1 - confidence_interval) / 2)
        cumulative_std_dev = []

        for i, predicted_return in enumerate(future_predictions):
            next_price = future_prices[-1] * np.exp(predicted_return)
            future_prices.append(next_price)
            if i == 0:
                cumulative_variance = std_dev_per_step[i]**2
            else:
                cumulative_variance += std_dev_per_step[i]**2
            cumulative_std = np.sqrt(cumulative_variance)
            cumulative_std_dev.append(cumulative_std)
            lower_bound = next_price * np.exp(-z_score * cumulative_std)
            upper_bound = next_price * np.exp(z_score * cumulative_std)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        future_prices = future_prices[1:]
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(future_prices) + 1), future_prices, 'o-', label='Future Prices')
        plt.fill_between(range(1, len(future_prices) + 1), lower_bounds, upper_bounds, color='gray', alpha=0.3,
                         label=f'{int(confidence_interval*100)}% Confidence Interval')
        for i, (price, lower, upper) in enumerate(zip(future_prices, lower_bounds, upper_bounds), 1):
            plt.text(i, price, f'{price:.2f}', fontsize=8, ha='center', va='bottom', color='blue')
            plt.text(i, lower, f'{lower:.2f}', fontsize=8, ha='center', va='top', color='green')
            plt.text(i, upper, f'{upper:.2f}', fontsize=8, ha='center', va='bottom', color='red')
        plt.title('Projected Future Prices with Confidence Intervals (Numeric Values)')
        plt.xlabel('Steps Ahead')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('projected_future_prices.png')
        plt.close()

        start_index = len(X_train) + time_step + num_steps - 1
        test_dates_bt = data_with_features['Date'].iloc[start_index : start_index + len(y_pred_multi)]
        test_close_prices_bt = data_with_features['Close'].iloc[start_index : start_index + len(y_pred_multi)].values
        prices_close_bt = pd.Series(test_close_prices_bt, index=test_dates_bt)

        best_multiplier, best_metric, optimization_results = optimize_max_threshold(
            prices_close_bt, y_pred_multi, y_test_multi, y_train, window_size=20
        )
        logging.info(f"Best multiplier found: {best_multiplier} with Sharpe Ratio: {best_metric}")

        # Decision for future days: Optimize or choose multiplier m.
        start_index = len(X_train) + time_step + num_steps - 1
        test_dates_bt = data_with_features['Date'].iloc[start_index : start_index + len(y_pred_multi)]
        test_close_prices_bt = data_with_features['Close'].iloc[start_index : start_index + len(y_pred_multi)].values
        prices_close_bt = pd.Series(test_close_prices_bt, index=test_dates_bt)
        std_return_train = np.std(y_train)
        choice = input("Do you want to reoptimize the multiplier m? [Y/N/C] (C to enter a custom value): ").strip().upper()

        if choice == 'Y':
            aggregated_m, optimal_m_values = aggregate_optimal_multiplier(
                prices_close_bt, y_pred_multi, y_test_multi, y_train,
                window_size=20, segment_length=20, step=10, multipliers=np.arange(0.1, 1.05, 0.1)
            )
            save_optimized_m(aggregated_m, ticker)
            logging.info(f"Aggregated optimal multiplier (reoptimized): {aggregated_m}")
            logging.info(f"Optimal multipliers per segment: {optimal_m_values}")
        elif choice == 'C':
            try:
                custom_m = float(input("Enter your desired m value (e.g., 0 to eliminate threshold): ").strip())
            except ValueError:
                logging.error("Invalid value. Default m (0.5) will be used.")
                custom_m = 0.5
            aggregated_m = custom_m
            logging.info(f"Custom m value: {aggregated_m}")
        elif choice == 'N':
            aggregated_m = load_optimized_m(ticker)
            if aggregated_m is None:
                aggregated_m = 0.5
                logging.info("No optimized value found, using default m: 0.5")
            else:
                logging.info(f"Using previously optimized m: {aggregated_m}")
        else:
            logging.info("Invalid choice. Using default m: 0.5")
            aggregated_m = 0.5

        signals, dynamic_thresholds, weighted_mean_returns = compute_dynamic_signals_with_multiplier(
            prices_close_bt, y_pred_multi, y_test_multi, std_return_train, window_size=20, multiplier=aggregated_m
        )
        signal_mapping = {1: "LONG", -1: "SHORT", 0: "SELL LONG COVER SHORT"}
        suggested_action_numeric = signals[-1]
        suggested_action_text = signal_mapping[suggested_action_numeric]
        logging.info(f"Suggested action: {suggested_action_text}")

        plt.figure(figsize=(12,6))
        plt.plot(prices_close_bt.index, weighted_mean_returns, label="Weighted Mean Return", marker='o')
        plt.plot(prices_close_bt.index, dynamic_thresholds, label="Dynamic Threshold", linestyle='--')
        plt.plot(prices_close_bt.index, -dynamic_thresholds, label="-Dynamic Threshold", linestyle='--')
        plt.title("Dynamic Thresholds and Weighted Mean Returns")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Backtesting with vectorbt using optimized signals
        start_index = len(X_train) + time_step + num_steps - 1
        test_dates_bt = data_with_features['Date'].iloc[start_index : start_index + len(y_pred_multi)]
        test_close_prices_bt = data_with_features['Close'].iloc[start_index : start_index + len(y_pred_multi)].values
        prices_close_bt = pd.Series(test_close_prices_bt, index=test_dates_bt)

        portfolio_vbt, signals_series, dynamic_thresholds_back, weighted_mean_returns_back = backtest_vectorbt_full(
            prices=prices_close_bt,
            y_pred_multi=y_pred_multi,
            y_test_multi=y_test_multi,
            std_return_train=std_return_train,
            window_size=20,
            initial_capital=100000,
            freq='1D',
            multiplier=aggregated_m
        )

        plot_vectorbt_portfolio(portfolio_vbt)
        print_vectorbt_portfolio_metrics(portfolio_vbt)

        stats = portfolio_vbt.stats()
        portfolio_metrics_text = f"""=== Portfolio Metrics (vectorbt) ===
        Start: {stats['Start']}
        End: {stats['End']}
        Period: {stats['Period']}
        Start Value: {stats['Start Value']}
        End Value: {stats['End Value']}
        Total Return [%]: {stats['Total Return [%]']}
        Benchmark Return [%]: {stats['Benchmark Return [%]']}
        Max Gross Exposure [%]: {stats['Max Gross Exposure [%]']}
        Total Fees Paid: {stats['Total Fees Paid']}
        Max Drawdown [%]: {stats['Max Drawdown [%]']}
        Max Drawdown Duration: {stats['Max Drawdown Duration']}
        Total Trades: {stats['Total Trades']}
        Total Closed Trades: {stats['Total Closed Trades']}
        Total Open Trades: {stats['Total Open Trades']}
        Open Trade PnL: {stats['Open Trade PnL']}
        Win Rate [%]: {stats['Win Rate [%]']}
        Best Trade [%]: {stats['Best Trade [%]']}
        Worst Trade [%]: {stats['Worst Trade [%]']}
        Avg Winning Trade [%]: {stats['Avg Winning Trade [%]']}
        Avg Losing Trade [%]: {stats['Avg Losing Trade [%]']}
        Avg Winning Trade Duration: {stats['Avg Winning Trade Duration']}
        Avg Losing Trade Duration: {stats['Avg Losing Trade Duration']}
        Profit Factor: {stats['Profit Factor']}
        Expectancy: {stats['Expectancy']}
        Sharpe Ratio: {stats['Sharpe Ratio']}
        Calmar Ratio: {stats['Calmar Ratio']}
        Omega Ratio: {stats['Omega Ratio']}
        Sortino Ratio: {stats['Sortino Ratio']}

        === Specific Metrics ===
        Total Return [%]: {stats['Total Return [%]']}  
        Sharpe Ratio: {stats['Sharpe Ratio']}  
        Max Drawdown [%]: {stats['Max Drawdown [%]']}  
        """
        generate_pdf_report(
            portfolio_metrics_text=portfolio_metrics_text,
            ticker=ticker, 
            portfolio=portfolio_vbt,
            suggested_action=suggested_action_text
        )

    else:
        logging.error("Error: invalid choice.")
        exit(1)


if __name__ == "__main__":
    main()
