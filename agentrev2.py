import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import ta
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta
import time
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import logging
import shap
import nltk
from robin_stocks import robinhood as r
import yfinance

nltk.download('vader_lexicon')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_debug.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add logging functions without modifying core logic
def add_logging_to_script():
    # Example logging at different levels
    logger.debug("This is a debug message for detailed troubleshooting.")
    logger.info("Application has started.")
    logger.warning("This is a warning message, please check the configuration.")
    logger.error("An error has occurred during script execution.")
    logger.critical("Critical issue encountered.")

# Assuming the rest of your existing script is invoked here
if __name__ == "__main__":
    try:
        add_logging_to_script()
        # Place your code functions or classes here as required, e.g., main()
        logger.info("All tasks completed successfully.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

# Securely load Robinhood credentials
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
try:
    with open(config_path) as config_file:
        config = json.load(config_file)
        username = config.get('username')
        password = config.get('password')
    logger.info("Successfully loaded Robinhood credentials.")
except FileNotFoundError:
    logger.error(f"Configuration file not found at {config_path}", exc_info=True)
    raise FileNotFoundError(f"Configuration file not found at {config_path}")

if not username or not password:
    logger.error("Robinhood credentials not found in the configuration file.")
    raise ValueError("Robinhood credentials not found in the configuration file.")

# Robinhood login
try:
    r.login(username=username, password=password)
    logger.info("Successfully logged into Robinhood.")
except Exception as e:
    logger.error(f"Failed to login to Robinhood: {e}", exc_info=True)
    raise

class StockDataFetcher:
    @staticmethod
    def get_stock_data(symbol, interval='5minute', span='day'):
        logger.info(f"Fetching stock data for {symbol} with interval {interval} and span {span}.")
        try:
            # Fetch intraday data
            intraday_data = r.stocks.get_stock_historicals(symbol, interval=interval, span=span)
            if intraday_data is None or len(intraday_data) == 0:
                logger.error(f"No intraday data available for {symbol}", exc_info=True)
                return None

            df = pd.DataFrame(intraday_data)
            df['close_price'] = df['close_price'].astype(float)
            df['high_price'] = df['high_price'].astype(float)
            df['low_price'] = df['low_price'].astype(float)
            df['volume'] = df['volume'].astype(float)

            logger.debug(f"Calculating technical indicators for {symbol}.")

            # Technical indicators using 'ta' library
            # RSI
            rsi_indicator = ta.momentum.RSIIndicator(close=df['close_price'])
            df['rsi'] = rsi_indicator.rsi()

            # MACD
            macd_indicator = ta.trend.MACD(close=df['close_price'])
            df['MACD'] = macd_indicator.macd()
            df['MACD_signal'] = macd_indicator.macd_signal()

            # ATR
            atr_indicator = ta.volatility.AverageTrueRange(
                high=df['high_price'],
                low=df['low_price'],
                close=df['close_price']
            )
            df['ATR'] = atr_indicator.average_true_range()

            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(close=df['close_price'])
            df['BB_upper'] = bb_indicator.bollinger_hband()
            df['BB_middle'] = bb_indicator.bollinger_mavg()
            df['BB_lower'] = bb_indicator.bollinger_lband()

            # Price and volume changes
            df['price_change'] = df['close_price'].pct_change()
            df['volume_change'] = df['volume'].pct_change()

            # Add sentiment score
            df['sentiment'] = StockDataFetcher.get_sentiment_score(symbol)

            # Add historical context
            logger.debug(f"Fetching historical data for {symbol}.")
            historical_data = r.stocks.get_stock_historicals(symbol, interval='day', span='year')
            if historical_data is None or len(historical_data) == 0:
                logger.error(f"No historical data available for {symbol}", exc_info=True)
                return None

            hist_df = pd.DataFrame(historical_data)
            hist_df['close_price'] = hist_df['close_price'].astype(float)

            # Compute yearly metrics
            df['yearly_high'] = hist_df['close_price'].max()
            df['yearly_low'] = hist_df['close_price'].min()
            df['price_to_yearly_high'] = df['close_price'] / df['yearly_high']
            df['price_to_yearly_low'] = df['close_price'] / df['yearly_low']
            df['yearly_trend'] = (
                hist_df['close_price'].iloc[-1] - hist_df['close_price'].iloc[0]
            ) / hist_df['close_price'].iloc[0]

            # Drop rows with NaN values
            df = df.dropna()

            if df.empty:
                logger.error(f"No valid data after processing for {symbol}", exc_info=True)
                return None

            logger.info(f"Successfully fetched and processed data for {symbol}.")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e, exc_info=True)}", exc_info=True)
            return None

    @staticmethod
    def get_sentiment_score(symbol):
        logger.debug(f"Calculating sentiment score for {symbol}.")
        try:
            # Placeholder for sentiment analysis
            sia = SentimentIntensityAnalyzer()
            # Simulate sentiment score
            sentiment_score = np.random.uniform(-1, 1)
            logger.debug(f"Sentiment score for {symbol}: {sentiment_score}")
            return sentiment_score
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {str(e, exc_info=True)}", exc_info=True)
            return 0

class StockSelector:
    @staticmethod
    def identify_workable_stocks():
        logger.info("Identifying workable stocks.")
        try:
            # Fetch the list of S&P 500 companies from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            all_stocks = sp500_table['Symbol'].tolist()

            # Correct symbols for processing
            symbol_corrections = {
                'BRK.B': 'BRK-B',
                'BF.B': 'BF-B',
                'FB': 'META',  # Update FB to META
                'AMTM': None,  # Invalid ticker
                'SW': None     # Invalid ticker
            }
            all_stocks = [symbol_corrections.get(symbol, symbol) for symbol in all_stocks if symbol_corrections.get(symbol, symbol)]

            workable_stocks = []
            for symbol in all_stocks:
                if not is_valid_ticker(symbol):
                    logger.warning(f"Invalid or unsupported ticker: {symbol}")
                    continue
                try:
                    logger.debug(f"Processing stock: {symbol}")
                    # Use 'ytd' or 'max' instead of '1y' for valid period
                    hist_df = yf.download(symbol, period='ytd', interval='1d')
                    if hist_df.empty:
                        logger.warning(f"No historical data available for {symbol}")
                        continue
                    hist_df = hist_df.rename(columns={'Close': 'close_price', 'Volume': 'volume'})
                    hist_df['close_price'] = hist_df['close_price'].astype(float)
                    hist_df['volume'] = hist_df['volume'].astype(float)

                    # Calculate historical metrics
                    avg_volume = hist_df['volume'].mean()
                    volatility = hist_df['close_price'].pct_change().std()
                    trend = (hist_df['close_price'].iloc[-1] - hist_df['close_price'].iloc[0]) / hist_df['close_price'].iloc[0]

                    if avg_volume > 1_000_000 and volatility > 0.02 and abs(trend) > 0.1:
                        workable_stocks.append({
                            'symbol': symbol,
                            'avg_volume': avg_volume,
                            'volatility': volatility,
                            'trend': trend
                        })
                        logger.info(f"Stock {symbol} added to workable stocks.")
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {str(e, exc_info=True)}", exc_info=True)
                    continue

            # Sort stocks by a combined score
            workable_stocks.sort(
                key=lambda x: x['avg_volume'] * x['volatility'] * abs(x['trend']),
                reverse=True
            )
            top_stocks = [stock['symbol'] for stock in workable_stocks[:10]]
            logger.info(f"Top 10 workable stocks: {top_stocks}")
            return top_stocks  # Return top 10 stocks
        except Exception as e:
            logger.error(f"Error identifying workable stocks: {str(e, exc_info=True)}", exc_info=True)
            return []

class AdaptiveEnsembleModel:
    def __init__(self, symbol):
        self.symbol = symbol
        self.lstm_model = None
        self.rf_model = None
        self.gb_model = None
        self.scaler = None
        self.last_train_time = None
        logger.info(f"Initialized AdaptiveEnsembleModel for {symbol}.")

    def train(self, df):
        logger.info(f"Training models for {self.symbol}.")
        try:
            features = ['rsi', 'MACD', 'MACD_signal', 'ATR', 'price_change', 'volume_change',
                        'sentiment', 'price_to_yearly_high', 'price_to_yearly_low', 'yearly_trend']
            X = df[features].values
            y = np.where(df['close_price'].shift(-1) > df['close_price'], 1, 0)[:-1]

            X_train, X_test, y_train, y_test = train_test_split(X[:-1], y, test_size=0.2, random_state=42)

            self.scaler = MinMaxScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # LSTM model
            X_lstm_train = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_lstm_test = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

            self.lstm_model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.lstm_model.fit(X_lstm_train, y_train, epochs=50, batch_size=32, validation_data=(X_lstm_test, y_test), verbose=0)

            # Random Forest model
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.rf_model.fit(X_train, y_train)

            # Gradient Boosting model
            self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            self.gb_model.fit(X_train, y_train)

            self.last_train_time = datetime.now()

            # Evaluate models
            lstm_accuracy = self.lstm_model.evaluate(X_lstm_test, y_test, verbose=0)[1]
            rf_accuracy = self.rf_model.score(X_test, y_test)
            gb_accuracy = self.gb_model.score(X_test, y_test)

            logger.info(f"{self.symbol} model accuracies - LSTM: {lstm_accuracy:.4f}, RF: {rf_accuracy:.4f}, GB: {gb_accuracy:.4f}")
        except Exception as e:
            logger.error(f"Error training models for {self.symbol}: {e}", exc_info=True)

    def predict(self, df):
        logger.info(f"Predicting price movement for {self.symbol}.")
        try:
            if self.last_train_time is None or (datetime.now() - self.last_train_time) > timedelta(hours=1):
                self.train(df)

            features = ['rsi', 'MACD', 'MACD_signal', 'ATR', 'price_change', 'volume_change',
                        'sentiment', 'price_to_yearly_high', 'price_to_yearly_low', 'yearly_trend']
            latest_features = df[features].iloc[-1].values.reshape(1, -1)

            lstm_input = self.scaler.transform(latest_features).reshape(1, 1, -1)
            lstm_pred = self.lstm_model.predict(lstm_input)[0][0]

            rf_pred = self.rf_model.predict_proba(latest_features)[0][1]
            gb_pred = self.gb_model.predict_proba(latest_features)[0][1]

            ensemble_pred = (lstm_pred + rf_pred + gb_pred) / 3
            logger.debug(f"Prediction for {self.symbol}: {ensemble_pred}")
            return ensemble_pred
        except Exception as e:
            logger.error(f"Error predicting for {self.symbol}: {e}", exc_info=True)
            return 0.5  # Neutral prediction on error

    def explain_prediction(self, df):
    logger.info(f"Generating SHAP explanation for {self.symbol}.")
    try:
        logger.info(f"Generating SHAP explanation for {self.symbol}.")
        try:
            features = ['rsi', 'MACD', 'MACD_signal', 'ATR', 'price_change', 'volume_change',
                        'sentiment', 'price_to_yearly_high', 'price_to_yearly_low', 'yearly_trend']
            X = df[features].iloc[-1].values.reshape(1, -1)

            explainer = shap.TreeExplainer(self.rf_model)
            shap_values = explainer.shap_values(X)

            # Save the SHAP values and features for later plotting
            np.save(f'{self.symbol}_shap_values.npy', shap_values)
            np.save(f'{self.symbol}_X.npy', X)
            logger.info(f"SHAP values saved for {self.symbol}.")
        except Exception as e:
            logger.error(f"Error generating SHAP explanation for {self.symbol}: {e}", exc_info=True)


class TradingStrategy:
    def __init__(self):
        self.models = {}
        self.portfolio = []
        self.cash_balance = float(r.profiles.load_account_profile()['buying_power'])
        self.positions = r.account.build_holdings()
        self.daily_profit = 0
        self.trade_log = []
        self.max_daily_trades = 20  # Maximum number of trades per day
        self.max_daily_loss = -50  # Maximum allowed loss per day in dollars
        self.trades_today = 0  # Counter for trades made today
        self.start_balance = self.cash_balance  # Starting balance for the day
        self.last_trade_date = datetime.now().date()  # Initialize last_trade_date
        self.exceptional_growth_threshold = 0.05  # 5% growth in a short period
        self.exceptional_growth_timeframe = timedelta(minutes=30)  # Time frame to measure exceptional growth
        self.max_daily_profit = 100  # Absolute maximum daily profit, even for exceptional growth
        self.stop_loss_percentage = 0.02  # 2% stop loss
        self.trailing_stop_percentage = 0.01  # 1% trailing stop
        logger.info("TradingStrategy initialized.")

    def predict_price_movement(self, symbol, df):
        logger.debug(f"Predicting price movement for {symbol}.")
        if symbol not in self.models:
            self.models[symbol] = AdaptiveEnsembleModel(symbol)

        prediction = self.models[symbol].predict(df)
        self.models[symbol].explain_prediction(df)  # Generate SHAP explanation

        if prediction > 0.6:
            return 'up'
        elif prediction < 0.4:
            return 'down'
        else:
            return 'neutral'

    def is_exceptional_growth(self, symbol):
        logger.debug(f"Checking for exceptional growth in {symbol}.")
        df = StockDataFetcher.get_stock_data(symbol, interval='5minute', span='hour')
        if df is None or len(df) < 2:
            return False

        timeframe = int(self.exceptional_growth_timeframe.total_seconds() / 300)
        if timeframe >= len(df):
            timeframe = len(df) - 1

        start_price = df['close_price'].iloc[-timeframe]
        end_price = df['close_price'].iloc[-1]
        growth_rate = (end_price - start_price) / start_price

        logger.debug(f"Growth rate for {symbol}: {growth_rate}")
        return growth_rate > self.exceptional_growth_threshold

    def trading_decision(self, symbol, prediction, current_price, holdings, df):
        logger.info(f"Making trading decision for {symbol}.")
        # Check if we've hit the absolute maximum daily profit
        if self.daily_profit >= self.max_daily_profit:
            logger.info("Maximum daily profit reached.")
            return 'hold', 0

        # Check if we've reached the maximum number of trades for the day
        if self.trades_today >= self.max_daily_trades:
            logger.info("Maximum daily trades reached.")
            return 'hold', 0

        # Check if we've hit the maximum daily loss
        if self.cash_balance - self.start_balance <= self.max_daily_loss:
            logger.warning(f"Maximum daily loss of ${abs(self.max_daily_loss)} reached. Stopping trading for the day.")
            return 'hold', 0

        # Check if we've reached the normal daily profit target
        if self.daily_profit >= 20:
            # If we've reached the target, only continue if there's exceptional growth
            if not self.is_exceptional_growth(symbol):
                logger.info(f"Daily profit target reached. No exceptional growth in {symbol}.")
                return 'hold', 0
            logger.info(f"Continuing to trade {symbol} due to exceptional growth despite reaching daily target")

        max_position_size = min(1000, 0.1 * self.cash_balance)  # Max 10% of cash balance per position
        target_profit_per_trade = 5  # Target $5 profit per trade

        # Use ATR for dynamic position sizing
        atr = df['ATR'].iloc[-1]
        risk_per_trade = min(0.01 * self.cash_balance, 50)  # Risk 1% of balance per trade, max $50
        shares = int(risk_per_trade / (atr * 2))  # Use 2 * ATR as stop loss

        # Ensure we don't exceed our cash balance
        shares = min(shares, int(self.cash_balance / current_price))

        # Check portfolio correlation
        if len(self.portfolio) >= 5 and self.is_highly_correlated(symbol):
            logger.info(f"Stock {symbol} is highly correlated with existing portfolio. Skipping.")
            return 'hold', 0  # Avoid overexposure to correlated stocks

        if prediction == 'up' and self.cash_balance > current_price * shares:
            logger.info(f"Decision: Buy {shares} shares of {symbol}.")
            return 'buy', min(shares, int(max_position_size / current_price))
        elif prediction == 'down' and holdings > 0:
            logger.info(f"Decision: Sell holdings of {symbol}.")
            return 'sell', holdings
        else:
            logger.info(f"Decision: Hold {symbol}.")
            return 'hold', 0

    def is_highly_correlated(self, symbol):
        logger.debug(f"Checking correlation for {symbol}.")
        if len(self.portfolio) < 5:
            return False

        correlations = []
        for other_symbol in self.portfolio:
            if other_symbol != symbol:
                df1 = StockDataFetcher.get_stock_data(symbol, interval='day', span='month')
                df2 = StockDataFetcher.get_stock_data(other_symbol, interval='day', span='month')
                if df1 is not None and df2 is not None:
                    correlation = df1['close_price'].corr(df2['close_price'])
                    correlations.append(correlation)

        if correlations:
            avg_correlation = np.mean(correlations)
            logger.debug(f"Average correlation for {symbol}: {avg_correlation}")
            return avg_correlation > 0.7  # Consider highly correlated if average correlation > 0.7
        else:
            return False

    def check_stop_loss(self, symbol, current_price):
        if symbol in self.positions:
            entry_price = float(self.positions[symbol]['average_buy_price'])
            if current_price <= entry_price * (1 - self.stop_loss_percentage):
                logger.info(f"Stop loss triggered for {symbol}.")
                return True
        return False

    def check_trailing_stop(self, symbol, current_price):
        if symbol in self.positions:
            highest_price = float(self.positions[symbol].get('highest_price', 0))
            if current_price <= highest_price * (1 - self.trailing_stop_percentage):
                logger.info(f"Trailing stop triggered for {symbol}.")
                return True
            elif current_price > highest_price:
                self.positions[symbol]['highest_price'] = current_price
        return False

    def execute_trade(self, symbol, action, quantity, current_price):
        logger.info(f"Executing trade: {action} {quantity} shares of {symbol} at ${current_price}.")
        try:
            if action == 'buy':
                r.orders.order_buy_market(symbol, quantity)
                self.cash_balance -= current_price * quantity
                self.portfolio.append(symbol)
                self.trade_log.append({'symbol': symbol, 'action': 'buy', 'price': current_price, 'quantity': quantity})
                logger.info(f"Bought {quantity} shares of {symbol}")
                if symbol not in self.positions:
                    self.positions[symbol] = {'average_buy_price': current_price, 'quantity': quantity}
                self.positions[symbol]['highest_price'] = current_price
            elif action == 'sell':
                r.orders.order_sell_market(symbol, quantity)
                profit = (current_price - float(self.positions[symbol]['average_buy_price'])) * quantity
                self.daily_profit += profit
                self.cash_balance += current_price * quantity
                if symbol in self.portfolio:
                    self.portfolio.remove(symbol)
                self.trade_log.append({'symbol': symbol, 'action': 'sell', 'price': current_price, 'quantity': quantity, 'profit': profit})
                logger.info(f"Sold {quantity} shares of {symbol}. Profit: ${profit:.2f}")
                del self.positions[symbol]

            self.trades_today += 1
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e, exc_info=True)}", exc_info=True)

    def process_stock(self, symbol):
        logger.debug(f"Processing stock: {symbol}")
        df = StockDataFetcher.get_stock_data(symbol)
        if df is not None:
            current_price = float(r.stocks.get_latest_price(symbol)[0])
            if self.check_stop_loss(symbol, current_price) or self.check_trailing_stop(symbol, current_price):
                holdings = int(float(self.positions.get(symbol, {}).get('quantity', 0)))
                self.execute_trade(symbol, 'sell', holdings, current_price)
                return

            prediction = self.predict_price_movement(symbol, df)
            holdings = int(float(self.positions.get(symbol, {}).get('quantity', 0)))
            action, quantity = self.trading_decision(symbol, prediction, current_price, holdings, df)

            if action != 'hold' and quantity > 0:
                self.execute_trade(symbol, action, quantity, current_price)
        else:
            logger.error(f"Failed to fetch data for {symbol}.", exc_info=True)

    def run(self):
    logger.info("Starting trading strategy.")
    workable_stocks = StockSelector.identify_workable_stocks()
    processed_symbols = []
        logger.info("Starting trading strategy.")
        workable_stocks = StockSelector.identify_workable_stocks()

        while self.daily_profit < self.max_daily_profit and self.trades_today < self.max_daily_trades:
            if self.cash_balance - self.start_balance <= self.max_daily_loss:
                logger.warning(f"Maximum daily loss of ${abs(self.max_daily_loss)} reached. Stopping trading for the day.")
                break

            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(self.process_stock, workable_stocks)
        processed_symbols.extend(workable_stocks)

            # Reset daily counters if a new day has started
            if datetime.now().date() > self.last_trade_date:
                self.trades_today = 0
                self.daily_profit = 0
                self.start_balance = self.cash_balance
                self.last_trade_date = datetime.now().date()
                logger.info("New trading day started. Counters reset.")

            # Wait for a short period before next iteration
            logger.info("Waiting for next iteration.")
            time.sleep(300)  # Wait for 5 minutes

    self.generate_shap_plots(processed_symbols)
        self.analyze_performance()

def generate_shap_plots(self, symbols):
    logger.info("Generating SHAP plots in the main thread.")
    for symbol in symbols:
        try:
            shap_values = np.load(f"{symbol}_shap_values.npy", allow_pickle=True)
            X = np.load(f"{symbol}_X.npy", allow_pickle=True)
            features = ["rsi", "MACD", "MACD_signal", "ATR", "price_change", "volume_change",
                        "sentiment", "price_to_yearly_high", "price_to_yearly_low", "yearly_trend"]
            shap.summary_plot(shap_values, X, feature_names=features, plot_type="bar")
            plt.savefig(f"{symbol}_shap_summary.png")
            plt.close()
            logger.info(f"SHAP summary plot saved for {symbol}.")
        except Exception as e:
            logger.error(f"Error generating SHAP plot for {symbol}: {e}", exc_info=True)

    def analyze_performance(self):
        logger.info("Analyzing trading performance.")
        df = pd.DataFrame(self.trade_log)
        total_trades = len(df)
        profitable_trades = len(df[df.get('profit', 0) > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        total_profit = df.get('profit', pd.Series(0)).sum()

        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Profitable trades: {profitable_trades}")
        logger.info(f"Win rate: {win_rate:.2%}")
        logger.info(f"Total profit: ${total_profit:.2f}")

        # Calculate Sharpe ratio
        returns = df.groupby('symbol')['profit'].sum() / df.groupby('symbol')['price'].first()
        if returns.std() != 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Assuming 252 trading days in a year
        else:
            sharpe_ratio = 0
        logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")

        # Plot profit over time
        cumulative_profit = df.get('profit', pd.Series(0)).cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_profit.index, cumulative_profit.values)
        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative Profit ($)')
        logger.info("Cumulative profit plot saved.")

        # Plot trade distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df.get('profit', pd.Series(0)), kde=True)
        plt.title('Distribution of Trade Profits')
        plt.xlabel('Profit ($)')
        plt.ylabel('Frequency')
        logger.info("Profit distribution plot saved.")

        # Save trade log to file
        df.to_csv('trade_log.csv', index=False)
        logger.info("Trade log saved to trade_log.csv.")

        # Calculate additional metrics
        max_drawdown = (cumulative_profit.cummax() - cumulative_profit).max()
        total_positive_profit = df[df.get('profit', 0) > 0]['profit'].sum()
        total_negative_profit = abs(df[df.get('profit', 0) < 0]['profit'].sum())
        profit_factor = total_positive_profit / total_negative_profit if total_negative_profit != 0 else np.inf

        logger.info(f"Max Drawdown: ${max_drawdown:.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")

        # Save performance metrics to file
        performance_metrics = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }
        with open('performance_metrics.json', 'w') as f:
            json.dump(performance_metrics, f, indent=4)
        logger.info("Performance metrics saved to performance_metrics.json.")

def is_valid_ticker(symbol):
    logger.debug(f"Validating ticker: {symbol}")
    try:
        ticker = yf.Ticker(symbol)
        # Attempt to fetch data for a minimal period
        data = ticker.history(period='1d')
        is_valid = not data.empty
        logger.debug(f"Ticker {symbol} is valid: {is_valid}")
        return is_valid
    except Exception:
        logger.warning(f"Ticker {symbol} is invalid.")
        return False

def main():
    try:
        strategy = TradingStrategy()
        strategy.run()
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e, exc_info=True)}", exc_info=True)

if __name__ == "__main__":
    main()

