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
from datetime import datetime
from scipy.stats import norm
from plotly.subplots import make_subplots
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from hmmlearn import hmm
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.units import inch

import vectorbt as vbt

import os





# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    logging.info(f"Available columns in data DataFrame: {data.columns.tolist()}")
    return data


def preprocess_data(data, lag_p=7):
   
    # Preprocesses the data by calculating returns, lagged features, and technical indicators.
    
    # If columns have a MultiIndex, remove the second level
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        logging.info("Columns after removing the second level of MultiIndex.")

    # Access closing prices
    if 'Close' in data.columns:
        close_prices = data['Close']
    else:
        logging.error("The 'Close' column is not present in the data.")
        exit(1)

    # Calculate LAG-day returns
    returns = close_prices.pct_change(5).dropna()

    
    data['Return'] = returns
   
   # Calculate lagged features
    for l in range(1, lag_p + 1):
        data[f'lag_{l}'] = close_prices.shift(l)
    data = data.dropna()

    data = data.copy()

    # Calculate technical indicators
    data.loc[:, 'SMA_20'] = data['Close'].rolling(window=20).mean()
    data.loc[:, 'SMA_50'] = data['Close'].rolling(window=50).mean()

    # Calculate RSI
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
        logging.error(f"The following expected features are missing in the dataset: {missing_features}")
        exit(1)

    X = data[expected_features].values
    y = returns.loc[data.index].values


    return X, y, data




def plot_technical_indicators(data):
    """
    Displays a candlestick chart with the SMA_20 and SMA_50 moving averages using mplfinance,
    and saves the figure as an image.
    """


    # Ensure that the DataFrame index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # First plot: candlestick chart with SMA_20 and SMA_50
    # Create a DataFrame with the columns required by mplfinance
    # mplfinance expects columns: Open, High, Low, Close, indexed by datetime.
    df = data[['Open', 'High', 'Low', 'Close']]

    # Prepare additional plots for the moving averages
    addplots = [
        mpf.make_addplot(data['SMA_20'], color='blue', width=2),
        mpf.make_addplot(data['SMA_50'], color='red', width=2)
    ]

    # Create the candlestick chart with the additional SMA lines
    fig, ax = mpf.plot(
        df,
        type='candle',
        addplot=addplots,
        returnfig=True,
        figsize=(14, 7),
        title='Technical Indicators (Candlestick Chart + SMA_20 - SMA_50)',
        style='yahoo'  # Add this line to use colored candlesticks
    )

    # Save the first plot as an image
    plt.savefig('technical_indicators.png')
    plt.show()
    plt.close()

    """
    Displays a candlestick chart with SMA_20 and SMA_50 on the top panel
    and RSI on the bottom panel, and then saves the figure as an image.
    """

    # Create a DataFrame again for the second plot
    df = data[['Open', 'High', 'Low', 'Close']]

    # Create the addplots for SMA (panel=0) and RSI (panel=1)
    addplots = [
        mpf.make_addplot(data['SMA_20'], panel=0, color='blue', width=2),
        mpf.make_addplot(data['SMA_50'], panel=0, color='red', width=2),
        # RSI on the lower panel (panel=1)
        mpf.make_addplot(data['RSI'], panel=1, color='purple', width=1)
    ]

    fig, axes = mpf.plot(
        df,
        type='candle',
        addplot=addplots,
        returnfig=True,
        figsize=(14, 7),
        title='Technical Indicators (Candlestick + RSI)',
        panel_ratios=(3,1),
        style='yahoo'
    )

    axes[0].set_ylabel('Price')
    axes[1].set_ylabel('RSI')
    axes[1].axhline(30, color='red', linestyle='--')
    axes[1].axhline(70, color='green', linestyle='--')

    # Save the second plot as an image
    plt.savefig('rsi.png')
    plt.show()
    plt.close()
    


def determine_optimal_states(X, max_states=5):
    """
    Determines the optimal number of states for the HMM using BIC and AIC.
    Returns the optimal number based on the lowest BIC.
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

    # Plot BIC and AIC
    plt.figure(figsize=(10, 5))
    plt.plot(n_components_range, bic, label='BIC')
    plt.plot(n_components_range, aic, label='AIC')
    plt.xlabel('Number of States')
    plt.ylabel('Value')
    plt.title('Optimal Number of States for HMM Selection')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image
    # plt.savefig('hmm_bic_aic.png')
    plt.close()

    # Choose the number of states with the lowest BIC
    optimal_states = n_components_range[np.argmin(bic)]
    logging.info(f"Optimal number of states determined by BIC: {optimal_states}")
    return optimal_states, scaler


def apply_hmm(X, hmm_model):
    """
    Applies the Hidden Markov Model to the preprocessed data.
    Returns hidden states and membership probabilities.
    """
    X_scaled = hmm_model.scaler.transform(X)
    hidden_states = hmm_model.hmm_model.predict(X_scaled)
    state_probs = hmm_model.hmm_model.predict_proba(X_scaled)
    logging.info(f"Hidden Markov Model clustering completed with {hmm_model.hmm_model.n_components} states.")
    return hidden_states, state_probs


class HMMModel:
    """
    Class to manage the Hidden Markov Model and its associated scaler.
    """
    def __init__(self, hmm_model, scaler):
        self.hmm_model = hmm_model
        self.scaler = scaler


def plot_hmm_transition_matrix(hmm_model):
    """
    Plots the transition matrix of the HMM model.
    """
    if not hasattr(hmm_model.hmm_model, "transmat_"):
        logging.error("The HMM model does not have a transition matrix.")
        return
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    transition_matrix = hmm_model.hmm_model.transmat_

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=[f"State {i}" for i in range(transition_matrix.shape[0])],
        yticklabels=[f"State {i}" for i in range(transition_matrix.shape[0])]
    )
    plt.title("HMM Transition Matrix")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.show()




def save_hmm_model(hmm_model, filename):
    """
    Saves the HMMModel using joblib.
    """
    dump(hmm_model, filename)
    logging.info(f"Hidden Markov Model saved to {filename}.")


def load_hmm_model(filename):
    """
    Loads the HMMModel using joblib.
    """
    hmm_model = load(filename)
    logging.info(f"Hidden Markov Model loaded from {filename}.")
    return hmm_model


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
    Displays the loss progression during training.
    Saves the plot as an image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Progression During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image
    plt.savefig('training_history.png')

    plt.show()

    plt.close()


def split_data(X, y, train_ratio):
    """
    Splits the data into training and testing sets.
    """
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logging.info(f"Data split with {train_ratio * 100:.2f}% for training and {100 - train_ratio * 100:.2f}% for testing.")
    logging.info(f"Number of samples in training set: {len(X_train)}")
    logging.info(f"Number of samples in testing set: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def prepare_data_multi_step(X, y, time_step, num_steps):
    """
    Prepares the data for input into the multi-step LSTM model.
    """
    X_data, y_data = [], []
    for i in range(len(X) - time_step - num_steps + 1):
        X_data.append(X[i:i + time_step])
        y_data.append(y[i + time_step:i + time_step + num_steps])
    return np.array(X_data), np.array(y_data)



    #####################################
    # Nuove funzioni per il backtest con vectorbt
    #####################################

import os

def save_optimized_m(m_value, ticker, filename_prefix="optimized_m_"):
    """
    Salva il valore ottimizzato di m in un file che include il ticker nel nome.
    Ad esempio, per ticker 'AAPL' il file sarà 'optimized_m_AAPL.txt'.
    """
    filename = f"{filename_prefix}{ticker}.txt"
    with open(filename, "w") as f:
        f.write(str(m_value))
    logging.info(f"File {filename} aggiornato con il valore ottimizzato di m: {m_value}")

def load_optimized_m(ticker, filename_prefix="optimized_m_"):
    """
    Carica il valore ottimizzato di m da un file il cui nome include il ticker.
    Restituisce il valore salvato se il file esiste, altrimenti None.
    """
    filename = f"{filename_prefix}{ticker}.txt"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            m_value = float(f.read().strip())
        logging.info(f"Valore ottimizzato di m caricato dal file {filename}: {m_value}")
        return m_value
    else:
        logging.info(f"Il file {filename} non esiste. Nessun valore ottimizzato trovato.")
        return None




def compute_dynamic_signals_with_multiplier(prices, y_pred_multi, y_test_multi, std_return_train, window_size=20, multiplier=0.5):
    """
    Calcola, per ogni giorno, il segnale dinamico basato sui multi-step predictions,
    utilizzando il parametro 'multiplier' per definire il max_threshold.
    
    Restituisce:
      - signals: array di segnali (1 = LONG, -1 = SHORT, 0 = Neutro)
      - dynamic_thresholds: array delle soglie dinamiche per ogni giorno
      - weighted_mean_returns: array dei rendimenti pesati per ogni giorno
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
        if len(recent_returns) > 0:
            recent_std = np.std(recent_returns)
        else:
            recent_std = std_return_train

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

        # Usa il parametro multiplier per definire il max_threshold
        max_threshold = multiplier * recent_std
        dynamic_threshold = min(max_threshold, recent_std * (1 / (1 + mean_rmse)))
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
    Ottimizza il moltiplicatore per il max_threshold eseguendo backtest su un range di valori.
    
    Restituisce:
      - best_multiplier: il valore ottimale per questo segmento,
      - best_metric: la metrica (es. Sharpe Ratio) corrispondente,
      - results: un dizionario con i risultati per ogni moltiplicatore testato.
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
    Suddivide il periodo in segmenti e ottimizza il parametro m per ciascun segmento,
    aggregando poi i valori ottimali (ad esempio con la mediana) per ottenere un valore robusto.
    
    Restituisce:
      - aggregated_m: il valore aggregato (es. mediana) dei moltiplicatori ottimali,
      - optimal_m_values: lista dei moltiplicatori ottimali per ciascun segmento.
    """
    optimal_m_values = []
    for start in range(0, len(prices) - segment_length + 1, step):
        segment_prices = prices.iloc[start:start+segment_length]
        segment_y_pred_multi = y_pred_multi[start:start+segment_length]
        segment_y_test_multi = y_test_multi[start:start+segment_length]
        std_return_train_seg = np.std(y_train)  # Puoi anche usare i dati del segmento, se lo preferisci
        best_m, best_metric, _ = optimize_max_threshold(segment_prices, segment_y_pred_multi, segment_y_test_multi, y_train, window_size, multipliers)
        optimal_m_values.append(best_m)
    aggregated_m = np.median(optimal_m_values)
    return aggregated_m, optimal_m_values




def backtest_vectorbt_full(prices, y_pred_multi, y_test_multi, std_return_train, window_size=20,
                           initial_capital=100000, freq='1D', multiplier=0.5):
    """
    Esegue il backtest con vectorbt utilizzando i segnali dinamici.
    Gli ordini vengono eseguiti al prezzo di chiusura per la valutazione, ma
    la simulazione dell'ordine utilizza il prezzo di apertura del giorno successivo.
    """
    # Calcola i segnali dinamici usando la versione che accetta il parametro multiplier
    signals, dynamic_thresholds, weighted_mean_returns = compute_dynamic_signals_with_multiplier(
        prices, y_pred_multi, y_test_multi, std_return_train, window_size, multiplier=multiplier
    )
    
    signals_series = pd.Series(signals, index=prices.index)
    
    # Definisci ingressi ed uscite in base ai segnali
    entries = signals_series == 1
    exits = signals_series == -1

    # Esegui il backtest con vectorbt, specificando anche il parametro open
    portfolio = vbt.Portfolio.from_signals(
        close=prices,
        #open=prices.shift(-1),  # Ordini eseguiti al prezzo di apertura del giorno successivo
        entries=entries,
        exits=exits,
        init_cash=initial_capital,
        freq=freq
    )


    # Salva i grafici generati da vectorbt
    try:
        fig_orders = portfolio.plot_orders()  # Grafico degli ordini
        fig_orders.write_image("vectorbt_orders.png")
    except Exception as e:
        print(f"Errore nella generazione del grafico degli ordini: {e}")
    
    try:
        fig_equity = portfolio.plot()  # Grafico della performance cumulata
        fig_equity.write_image("vectorbt_equity_curve.png")
    except Exception as e:
        print(f"Errore nella generazione del grafico della performance cumulata: {e}")
    
    return portfolio, signals_series, dynamic_thresholds, weighted_mean_returns



def plot_vectorbt_portfolio(portfolio):
    """
    Visualizza l'equity curve (valore del portafoglio) del backtest eseguito con vectorbt.
    """
    # Utilizziamo portfolio.plot() per una visualizzazione interattiva
    portfolio.plot(title="Equity Curve").show()





def print_vectorbt_portfolio_metrics(portfolio):
    """
    Calcola e stampa le metriche di performance del portafoglio utilizzando vectorbt.
    
    Parametri:
      - portfolio (vbt.Portfolio): oggetto Portfolio ottenuto dal backtest con vectorbt.
    """
    # Ottieni il dizionario delle statistiche
    stats = portfolio.stats()
    
    print("=== Portfolio Metrics (vectorbt) ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Prova a ottenere il CAGR dal dizionario delle statistiche
    cagr = stats.get("CAGR")
    
    # Se non esiste, calcolalo manualmente
    if cagr is None:
        # Calcolo manuale del CAGR: assumiamo 252 giorni di trading in un anno.
        # Chiamando portfolio.value() otteniamo la serie dei valori del portafoglio.
        initial_value = portfolio.value().iloc[0]
        final_value = portfolio.value().iloc[-1]
        n = len(portfolio.value())
        cagr = (final_value / initial_value) ** (252 / n) - 1
    
    total_return = portfolio.total_return()
    sharpe_ratio = portfolio.sharpe_ratio()
    max_drawdown = portfolio.max_drawdown()
    
    print("\n=== Specific Metrics ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")







def assign_state_labels(data_with_features, num_states):
    """
    Analyzes each state and assigns a meaningful label based on average return.
    Uses percentiles to dynamically handle any number of states.
    """
    state_labels = {}
    state_stats = {}

    for i in range(num_states):
        state_data = data_with_features[data_with_features['hidden_state'] == i]
        returns = state_data['Close'].pct_change().dropna()
        average_return = returns.mean()
        state_stats[i] = average_return

    # Calculate percentiles for average returns
    percentile_25 = np.percentile(list(state_stats.values()), 25)
    percentile_75 = np.percentile(list(state_stats.values()), 75)

    # Assign labels to states based on percentiles
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
    Colors the closing price chart based on the dominant state using meaningful labels.
    Colors:
        - Green for 'Bullish Phase'
        - Red for 'Bearish Phase'
        - Light Blue for 'Neutral Phase'
    Saves the plot as an image.
    """
    num_states = hmm_model.hmm_model.n_components
    state_prob_columns = [f'state_prob_{i}' for i in range(num_states)]

    # Verify that state probability columns are present
    missing_cols = [col for col in state_prob_columns if col not in data_with_features.columns]
    if missing_cols:
        logging.error(f"The following state probability columns are missing: {missing_cols}")
        return

    # Find the dominant state for each date
    data_with_features['dominant_state'] = data_with_features[state_prob_columns].idxmax(axis=1)
    data_with_features['dominant_state'] = data_with_features['dominant_state'].apply(lambda x: int(x.split('_')[-1]))

    # Assign meaningful labels to states
    state_labels = assign_state_labels(data_with_features, num_states)

    # Define a color map based on labels
    label_color_map = {
        'Bullish Phase': 'green',
        'Bearish Phase': 'red',
        'Neutral Phase': 'lightblue'
    }

    # Create a map from state to color using labels
    color_map = {}
    for state, label in state_labels.items():
        color = label_color_map.get(label, 'gray')  # 'gray' as default color if label is unrecognized
        color_map[state] = color

    # Assign colors based on dominant state
    data_with_features['color'] = data_with_features['dominant_state'].map(color_map)

    plt.figure(figsize=(14, 7))
    plt.scatter(data_with_features.index, data_with_features['Close'], c=data_with_features['color'], label='Closing Price')
    plt.title('Closing Price Colored by Market Phase')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)

    # Create legend with meaningful labels and specified colors
    handles = []
    for state, label in state_labels.items():
        color = color_map[state]
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color, markersize=10))
    plt.legend(handles=handles, title='Market Phase', loc='upper left')

    # Save the plot as an image
    plt.savefig('hmm_probabilities.png')
    plt.close()


def calculate_performance_metrics(backtest_results):
    """
    Calculates performance metrics from the backtest results.
    """
    total_return = (backtest_results['Total_Value'].iloc[-1] - backtest_results['Total_Value'].iloc[0]) / backtest_results['Total_Value'].iloc[0]
    num_years = (backtest_results.index[-1] - backtest_results.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / num_years) - 1

    cumulative_returns = backtest_results['Total_Value']
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Sharpe Ratio
    daily_returns = backtest_results['Portfolio_Returns']
    average_daily_return = daily_returns.mean()
    daily_return_std = daily_returns.std()
    trading_days = 252
    annualized_return_perf = (1 + average_daily_return) ** trading_days - 1
    annualized_volatility = daily_return_std * np.sqrt(trading_days)
    risk_free_rate = 0.02  # 2% annual
    sharpe_ratio = (annualized_return_perf - risk_free_rate) / annualized_volatility

    metrics = {
        'Total Return': f"{total_return * 100:.2f}%",
        'Annualized Return': f"{annualized_return * 100:.2f}%",
        'Maximum Drawdown': f"{max_drawdown * 100:.2f}%",
        'Annualized Sharpe Ratio': f"{sharpe_ratio:.2f}"
    }

    return metrics


def generate_pdf_report(portfolio_metrics_text, ticker, portfolio=None, output_filename=None, suggested_action=None):
    """
    Genera un report PDF che include:
      - Un'intestazione con la data corrente e il ticker.
      - Una sezione con le metriche (passate come stringa multilinea) prodotte da vectorbt.
      - Se fornito, un paragrafo con la Suggested Action (es. "Suggested action: SELL LONG COVER SHORT").
      - Una sola immagine su pagina intera: la curva cumulata (vectorbt_equity_curve.png).
      
    Il nome del file PDF generato includerà il ticker (se non viene passato un nome specifico).

    Parametri:
      - portfolio_metrics_text: stringa multilinea contenente tutte le metriche prodotte da vectorbt.
      - ticker: il simbolo del titolo.
      - portfolio: (opzionale) oggetto Portfolio di vectorbt; se fornito verrà usato per generare l'immagine.
      - output_filename: nome del file PDF da generare. Se None, verrà creato un nome che include il ticker.
      - suggested_action: (opzionale) stringa contenente la "Suggested Action" da inserire nel report.
    """
    

    # Se non viene specificato un output_filename, crealo includendo il ticker
    if output_filename is None:
        output_filename = f"backtest_report_{ticker}.pdf"
    else:
        if ticker not in output_filename:
            base, ext = os.path.splitext(output_filename)
            output_filename = f"{base}_{ticker}{ext}"

    current_dir = os.getcwd()
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Creazione del documento PDF
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

    # Titolo e intestazione
    title_text = f"Backtest Report for {ticker}"
    header_text = f"Date: {current_date} <br/> Ticker: {ticker}"
    elements.append(Paragraph(title_text, styles['CenterTitle']))
    elements.append(Paragraph(header_text, styles['Right']))
    elements.append(Spacer(1, 12))

    # Sezione Metriche: converto i ritorni a capo in <br/> per ReportLab (HTML semplice)
    metrics_html = portfolio_metrics_text.replace('\n', '<br/>')
    elements.append(Paragraph(metrics_html, styles['Left']))
    elements.append(Spacer(1, 24))
    
    # Inserisce il paragrafo della Suggested Action se presente
    if suggested_action:
        action_text = f"Suggested action: {suggested_action}"
        elements.append(Paragraph(action_text, styles['Left']))
        elements.append(Spacer(1, 24))

    # Inserisci un PageBreak per posizionare la figura su una nuova pagina
    elements.append(PageBreak())

    # Genera (o usa) l'immagine della curva cumulata (vectorbt_equity_curve.png)
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
        # Imposta le dimensioni a 6.0" x 9.0", per non superare l'area disponibile
        elements.append(Image(equity_file_abs, width=6.0*inch, height=9.0*inch))
        elements.append(Spacer(1, 12))
    except Exception as e:
        logging.error(f"Error adding equity curve image: {e}")

    # Testo conclusivo (opzionale)
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




def main():
    # Initial parameters
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
    hmm_model_filename = f'HMM_model_{ticker}.joblib'  # Filename to save the HMM model
    time_step = 25  # Length of the input sequence
    num_steps = 5  # Number of future steps to predict
    

    # Prompt for end date and validate format
    end_date = get_end_date()

    # Download historical data using yfinance
    data = download_data(ticker, start_date, end_date)

    print(data.tail())

    # Preprocess the data
    X, y, data_with_features = preprocess_data(data)

    # Reset index to have 'Date' as a column, if necessary
    if 'Date' not in data_with_features.columns:
        data_with_features = data_with_features.reset_index().rename(columns={'index': 'Date'})

    # Ask the user for the training data split percentage
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

    # Convert percentage to proportion
    train_ratio = split_percentage / 100

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio)

    # Apply Hidden Markov Model Clustering
    # Determine if the HMM model already exists
    try:
        hmm_model = load_hmm_model(hmm_model_filename)
        logging.info("Hidden Markov Model loaded successfully.")
        # Apply the HMM model
        hidden_states_train, state_probs_train = apply_hmm(X_train, hmm_model)
        hidden_states_test, state_probs_test = apply_hmm(X_test, hmm_model)
    except FileNotFoundError:
        logging.info("Hidden Markov Model not found. Proceeding with training.")
        # Determine the optimal number of states using only X_train
        optimal_states, scaler_hmm = determine_optimal_states(X_train, max_states=10)

        optimal_states = 3

        # Apply HMM with the optimal number of states on X_train
        hmm_instance = hmm.GaussianHMM(n_components=optimal_states, covariance_type="full", n_iter=1000, random_state=0)
        X_train_scaled_hmm = scaler_hmm.transform(X_train)
        hmm_instance.fit(X_train_scaled_hmm)
        hidden_states_train = hmm_instance.predict(X_train_scaled_hmm)
        state_probs_train = hmm_instance.predict_proba(X_train_scaled_hmm)

        # Apply HMM to the test set using the trained scaler
        X_test_scaled_hmm = scaler_hmm.transform(X_test)
        hidden_states_test = hmm_instance.predict(X_test_scaled_hmm)
        state_probs_test = hmm_instance.predict_proba(X_test_scaled_hmm)

        # Create an instance of HMMModel and save it
        hmm_model = HMMModel(hmm_model=hmm_instance, scaler=scaler_hmm)
        save_hmm_model(hmm_model, hmm_model_filename)

        plot_hmm_transition_matrix(hmm_model)

    # Add states to the DataFrame
    # After HMM training
    # Add states to the training set DataFrame
    data_with_features_train = data_with_features.iloc[:len(X_train)].copy()
    data_with_features_train['hidden_state'] = hidden_states_train

    # Add state probabilities as new features for the training set
    for i in range(hmm_model.hmm_model.n_components):
        data_with_features_train[f'state_prob_{i}'] = state_probs_train[:, i]

    # Add states to the test set DataFrame
    data_with_features_test = data_with_features.iloc[len(X_train):].copy()
    data_with_features_test['hidden_state'] = hidden_states_test

    # Add state probabilities as new features for the test set
    for i in range(hmm_model.hmm_model.n_components):
        data_with_features_test[f'state_prob_{i}'] = state_probs_test[:, i]

    # Combine training and test sets
    data_with_features = pd.concat([data_with_features_train, data_with_features_test])

    # Display technical indicators and save plots
    plot_technical_indicators(data_with_features)

    # Plot HMM probabilities and save plot
    plot_hmm_probabilities(data_with_features, hmm_model)

    # Define extended features: 'Return' and state probabilities
    # feature_columns_extended = ['Return'] + [f'state_prob_{i}' for i in range(hmm_model.hmm_model.n_components)]
    feature_columns_extended = ['Return'] + [f'state_prob_{i}' for i in range(hmm_model.hmm_model.n_components)]
    X_train_extended = data_with_features_train[feature_columns_extended].values
    X_test_extended = data_with_features_test[feature_columns_extended].values

    # Descriptive statistics
    mean_return = np.mean(y_train)
    std_return = np.std(y_train)
    median_return = np.median(y_train)
    percentile_25 = np.percentile(y_train, 25)
    percentile_75 = np.percentile(y_train, 75)
    max_return = np.max(y_train)
    min_return = np.min(y_train)

    logging.info(f"Mean 5-day returns: {mean_return:.4f}")
    logging.info(f"Standard deviation of 5-day returns: {std_return:.4f}")
    logging.info(f"Maximum return: {max_return:.4f}")
    logging.info(f"Minimum return: {min_return:.4f}")
    logging.info(f"25th percentile: {percentile_25:.4f}")
    logging.info(f"75th percentile: {percentile_75:.4f}")

    # Ask the user if they want to retrain the neural model
    while True:
        choice = input("Do you want to retrain the neural model? [Y/N]: ").strip().upper()
        if choice in ['Y', 'N']:
            break
        else:
            logging.error("Invalid input. Please enter 'Y' for Yes or 'N' for No.")

    if choice == 'Y':
        # Initialize and fit the scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_extended)
        X_test_scaled = scaler.transform(X_test_extended)

        # Save the scaler
        dump(scaler, scaler_filename)
        logging.info(f"Scaler saved to {scaler_filename}.")

        # Prepare data for the model
        X_train_reshaped, y_train_multi = prepare_data_multi_step(X_train_scaled, y_train, time_step, num_steps)
        X_test_reshaped, y_test_multi = prepare_data_multi_step(X_test_scaled, y_test, time_step, num_steps)

        # Build the LSTM model
        model = build_lstm_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]), num_steps)
        model.summary()

        # Configure callbacks
        checkpoint = ModelCheckpoint(filepath=model_filename, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

        # Train the model
        history = model.fit(
            X_train_reshaped, y_train_multi,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_reshaped, y_test_multi),
            callbacks=[reduce_lr, checkpoint, early_stop],
            verbose=2
        )

        # Plot training history and save plot
        plot_training_history(history)

        # Load the best saved model
        best_model = load_model(model_filename)
        logging.info("Neural network trained and saved successfully.")

    elif choice == 'N':
        # Load the model, scaler, and HMM
        try:
            best_model = load_model(model_filename)
            scaler = load(scaler_filename)
            hmm_model = load_hmm_model(hmm_model_filename)
            logging.info("Model, scaler, and HMM loaded successfully.")

            plot_hmm_transition_matrix(hmm_model)

        except (OSError, IOError) as e:
            logging.error(f"Error loading the model, scaler, or HMM: {e}")
            exit(1)

        # Use the already split datasets
        # Ensure that X_train, X_test, y_train, y_test are already defined

        # Add states and probabilities to the training set
        data_with_features_train = data_with_features.iloc[:len(X_train)].copy()
        data_with_features_train['hidden_state'] = hidden_states_train
        for i in range(hmm_model.hmm_model.n_components):
            data_with_features_train[f'state_prob_{i}'] = state_probs_train[:, i]

        # Add states and probabilities to the test set
        data_with_features_test = data_with_features.iloc[len(X_train):].copy()
        data_with_features_test['hidden_state'] = hidden_states_test
        for i in range(hmm_model.hmm_model.n_components):
            data_with_features_test[f'state_prob_{i}'] = state_probs_test[:, i]

        # Combine training and test sets
        data_with_features = pd.concat([data_with_features_train, data_with_features_test])

        # Define extended features for LSTM: 'Return' and HMM state probabilities
        # feature_columns_lstm = ['Return'] + [f'state_prob_{i}' for i in range(hmm_model.hmm_model.n_components)]
        feature_columns_lstm = ['Return'] + [f'state_prob_{i}' for i in range(hmm_model.hmm_model.n_components)]
        X_train_extended = data_with_features_train[feature_columns_lstm].values
        X_test_extended = data_with_features_test[feature_columns_lstm].values

        # Scale the data
        X_train_scaled = scaler.transform(X_train_extended)
        X_test_scaled = scaler.transform(X_test_extended)

        # Prepare data for the LSTM model
        X_train_reshaped, y_train_multi = prepare_data_multi_step(X_train_scaled, y_train, time_step, num_steps)
        X_test_reshaped, y_test_multi = prepare_data_multi_step(X_test_scaled, y_test, time_step, num_steps)

        # Make predictions on the test set
        y_pred_multi = best_model.predict(X_test_reshaped)

        # Calculate mean squared error for each future step
        mse_per_step = [mean_squared_error(y_test_multi[:, step], y_pred_multi[:, step]) for step in range(num_steps)]
        logging.info(f"MSE for each future step: {mse_per_step}")

        # Use the last test sample as input for future predictions
        future_input = X_test_reshaped[-1].reshape((1, time_step, X_test_reshaped.shape[2]))
        future_predictions = best_model.predict(future_input).flatten()
        logging.info(f"Future Predictions: {future_predictions}")

        # Use only the first step for comparison
        y_test_single = y_test_multi[:, 0]

        # Create "Actual vs. Predicted vs. Future Predictions" plot and save plot
        plt.figure(figsize=(14, 7))

        # Plot actual values (y_test_single)
        plt.plot(y_test_single, label='Actual', color='blue', marker='o', linestyle='-', alpha=0.7)

        # Plot model predictions (y_pred_multi)
        plt.plot(y_pred_multi[:, 0], label='Predicted', color='green', marker='o', linestyle='--', alpha=0.7)  # Only first prediction step

        # Plot future predictions
        future_steps = range(len(y_test_single), len(y_test_single) + len(future_predictions))
        plt.plot(future_steps, future_predictions, label='Future Predictions', color='red', marker='o', linestyle=':')

        # Labels and legend
        plt.title('Actual vs. Predicted vs. Future Predictions')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # Save the plot as an image
        plt.savefig('actual_vs_predicted_future.png')
        plt.show()
        plt.close()

        # Calculate confidence intervals
        errors = y_test_multi - y_pred_multi  # Difference between target and predictions
        mse_per_step = np.mean(errors**2, axis=0)  # MSE for each future step
        std_dev_per_step = np.sqrt(mse_per_step)  # Standard deviation for each future step

        # Most recent closing price
        last_close_price = data_with_features['Close'].iloc[-1]

        # Calculate future prices with confidence intervals
        future_prices = [last_close_price]  # List of future prices
        lower_bounds = []  # Lower limits
        upper_bounds = []  # Upper limits
        confidence_interval = 0.95  # Confidence interval (e.g., 95%)
        z_score = norm.ppf(1 - (1 - confidence_interval) / 2)  # z-score for the confidence interval

        # Cumulative standard deviation of predicted returns
        cumulative_std_dev = []

        for i, predicted_return in enumerate(future_predictions):
            # Calculate future price
            next_price = future_prices[-1] * np.exp(predicted_return)
            future_prices.append(next_price)

            # Calculate cumulative standard deviation
            if i == 0:
                cumulative_variance = std_dev_per_step[i]**2
            else:
                cumulative_variance += std_dev_per_step[i]**2
            cumulative_std = np.sqrt(cumulative_variance)
            cumulative_std_dev.append(cumulative_std)

            # Calculate confidence intervals
            lower_bound = next_price * np.exp(-z_score * cumulative_std)
            upper_bound = next_price * np.exp(z_score * cumulative_std)

            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)

        # Remove the first element of future_prices (the last known value)
        future_prices = future_prices[1:]

        # Plot future prices with confidence intervals and numeric values and save plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(future_prices) + 1), future_prices, 'o-', label='Future Prices')
        plt.fill_between(
            range(1, len(future_prices) + 1),
            lower_bounds,
            upper_bounds,
            color='gray',
            alpha=0.3,
            label=f'{int(confidence_interval*100)}% Confidence Interval'
        )

        # Add numeric values to the plot
        for i, (price, lower, upper) in enumerate(zip(future_prices, lower_bounds, upper_bounds), 1):
            plt.text(i, price, f'{price:.2f}', fontsize=8, ha='center', va='bottom', color='blue')
            plt.text(i, lower, f'{lower:.2f}', fontsize=8, ha='center', va='top', color='green')
            plt.text(i, upper, f'{upper:.2f}', fontsize=8, ha='center', va='bottom', color='red')

        plt.title('Projected Future Prices with Confidence Intervals (Numeric Values)')
        plt.xlabel('Steps Ahead')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Save the plot as an image
        plt.savefig('projected_future_prices.png')
        plt.close()



        # Assicurati di avere prices_close_bt, y_pred_multi, y_test_multi e y_train definiti.
        # Per esempio, prices_close_bt potrebbe essere definito così:
        start_index = len(X_train) + time_step + num_steps - 1
        test_dates_bt = data_with_features['Date'].iloc[start_index : start_index + len(y_pred_multi)]
        test_close_prices_bt = data_with_features['Close'].iloc[start_index : start_index + len(y_pred_multi)].values
        prices_close_bt = pd.Series(test_close_prices_bt, index=test_dates_bt)

        # Ottimizza il moltiplicatore
        best_multiplier, best_metric, optimization_results = optimize_max_threshold(prices_close_bt, y_pred_multi, y_test_multi, y_train, window_size=20)
        logging.info(f"Best multiplier found: {best_multiplier} with Sharpe Ratio: {best_metric}")




        ################################### DECISION FOR FUTURE DAYS ##############################################

        # Assicurati che la Series dei prezzi di chiusura per il periodo di test sia definita:
        start_index = len(X_train) + time_step + num_steps - 1
        test_dates_bt = data_with_features['Date'].iloc[start_index : start_index + len(y_pred_multi)]
        test_close_prices_bt = data_with_features['Close'].iloc[start_index : start_index + len(y_pred_multi)].values
        prices_close_bt = pd.Series(test_close_prices_bt, index=test_dates_bt)

        # Calcola la deviazione standard dei rendimenti sul training
        std_return_train = np.std(y_train)

        # Chiedi all'utente se desidera riottimizzare il parametro m, usare il valore salvato o inserire un valore personalizzato
        choice = input("Vuoi riottimizzare il parametro m? [Y/N/C] (C per inserire un valore personalizzato): ").strip().upper()

        if choice == 'Y':
            # Ottimizza m su più segmenti e aggrega il risultato
            aggregated_m, optimal_m_values = aggregate_optimal_multiplier(
                prices_close_bt, y_pred_multi, y_test_multi, y_train,
                window_size=20, segment_length=20, step=10, multipliers=np.arange(0.1, 1.05, 0.1)
            )
            save_optimized_m(aggregated_m, ticker)
            logging.info(f"Aggregated optimal multiplier (riottimizzato): {aggregated_m}")
            logging.info(f"Optimal multipliers per segment: {optimal_m_values}")
        elif choice == 'C':
            try:
                custom_m = float(input("Inserisci il valore di m desiderato (es. 0 per eliminare la soglia): ").strip())
            except ValueError:
                logging.error("Valore non valido. Verrà utilizzato il valore predefinito di m: 0.5")
                custom_m = 0.5
            aggregated_m = custom_m
            logging.info(f"Valore di m personalizzato: {aggregated_m}")
        elif choice == 'N':
            aggregated_m = load_optimized_m(ticker)
            if aggregated_m is None:
                aggregated_m = 0.5
                logging.info("Nessun valore ottimizzato trovato, utilizzo il valore predefinito di m: 0.5")
            else:
                logging.info(f"Utilizzo il valore ottimizzato di m precedentemente salvato: {aggregated_m}")
        else:
            logging.info("Scelta non valida. Verrà utilizzato il valore predefinito di m: 0.5")
            aggregated_m = 0.5

        # Usa il valore di m (aggregated_m) per calcolare i segnali dinamici
        signals, dynamic_thresholds, weighted_mean_returns = compute_dynamic_signals_with_multiplier(
            prices_close_bt, y_pred_multi, y_test_multi, std_return_train, window_size=20, multiplier=aggregated_m
        )

        # Mappatura dei segnali numerici in testo
        signal_mapping = {
            1: "LONG",
            -1: "SHORT",
            0: "SELL LONG COVER SHORT"
        }
        suggested_action_numeric = signals[-1]
        suggested_action_text = signal_mapping[suggested_action_numeric]
        logging.info(f"Suggested action: {suggested_action_text}")

        # Grafico: visualizza Weighted Mean Return e Dynamic Threshold
        
        plt.figure(figsize=(12,6))
        plt.plot(prices_close_bt.index, weighted_mean_returns, label="Weighted Mean Return", marker='o')
        plt.plot(prices_close_bt.index, dynamic_thresholds, label="Dynamic Threshold", linestyle='--')
        plt.plot(prices_close_bt.index, -dynamic_thresholds, label="-Dynamic Threshold", linestyle='--')
        plt.title("Dynamic Thresholds e Weighted Mean Returns")
        plt.xlabel("Data")
        plt.ylabel("Valore")
        plt.legend()
        plt.grid(True)
        plt.show()

        ################################## BACKTESTING ##############################################

        # Ricalcola (o conferma) prices_close_bt per il backtest (se non è già definito)
        start_index = len(X_train) + time_step + num_steps - 1
        test_dates_bt = data_with_features['Date'].iloc[start_index : start_index + len(y_pred_multi)]
        test_close_prices_bt = data_with_features['Close'].iloc[start_index : start_index + len(y_pred_multi)].values
        prices_close_bt = pd.Series(test_close_prices_bt, index=test_dates_bt)

        # Esegui il backtest con vectorbt utilizzando i segnali calcolati dalla funzione di ottimizzazione
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


        # Visualizza l'equity curve
        plot_vectorbt_portfolio(portfolio_vbt)
        # Stampa le metriche di portafoglio
        print_vectorbt_portfolio_metrics(portfolio_vbt)


        stats = portfolio_vbt.stats()
        #print(stats.keys())

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
            portfolio=portfolio_vbt,  # qui usi portfolio_vbt come parametro
            suggested_action=suggested_action_text
        )



    else:
        logging.error("Error: invalid choice.")
        exit(1)


if __name__ == "__main__":
    main()
