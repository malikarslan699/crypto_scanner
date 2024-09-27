from pybit.unified_trading import HTTP
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import asyncio
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# Configuration
API_KEY = "qN3MG3kfIWhfeLIldB"
API_SECRET = "7Uk5wQKAshUeYilFwytz7AhAQU0hLwrtba30"
TESTNET = True

TIMEFRAMES = ['5', '15', '60', '240', 'D']
SCAN_INTERVAL = 300  # 5 minutes

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize session
session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET)

# Machine Learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)

def fetch_ohlcv(symbol, interval, limit=1000):
    try:
        end_time = int(time.time() * 1000)
        start_time = end_time - (limit * get_interval_ms(interval))
        
        kline_data = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            start=start_time,
            end=end_time,
            limit=limit
        )
        
        if 'retCode' in kline_data and kline_data['retCode'] != 0:
            logger.error(f"Error fetching OHLCV data for {symbol} on {interval}: {kline_data['retMsg']}")
            return None
        
        df = pd.DataFrame(kline_data['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df.iloc[::-1]
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol} on {interval}: {str(e)}")
        return None

def get_interval_ms(interval):
    if interval == 'D':
        return 24 * 60 * 60 * 1000
    return int(interval) * 60 * 1000

def calculate_indicators(df):
    if df is None or len(df) < 50:
        return None
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    if macd is not None:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
    else:
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['volume_change'] = df['volume'].pct_change()
    df['price_change'] = df['close'].pct_change()
    return df

def prepare_ml_data(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    features = ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 'atr', 'volume_change', 'price_change']
    
    df_clean = df.dropna()
    
    X = df_clean[features].values
    y = df_clean['target'].values
    
    return X[:-1], y[:-1]

async def train_model(symbol, interval):
    df = fetch_ohlcv(symbol, interval)
    if df is None or len(df) < 200:
        logger.warning(f"Insufficient data for {symbol} on {interval}")
        return False
    
    df = calculate_indicators(df)
    if df is None:
        logger.warning(f"Unable to calculate indicators for {symbol} on {interval}")
        return False
    
    X, y = prepare_ml_data(df)
    
    logger.info(f"Prepared data for {symbol} on {interval}. X shape: {X.shape}, y shape: {y.shape}")
    
    if len(X) < 100:
        logger.warning(f"Not enough samples for {symbol} on {interval} after preprocessing")
        return False
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    logger.info(f"Model trained for {symbol} on {interval}. Accuracy: {accuracy:.2f}")
    return True

def predict_signal(df):
    features = ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 'atr', 'volume_change', 'price_change']
    X = df[features].iloc[-1].values.reshape(1, -1)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)  # Handle NaN and infinity values
    prediction = model.predict(X)
    probability = model.predict_proba(X)[0]
    return "BUY" if prediction[0] == 1 else "SELL", max(probability)

def analyze_market_structure(df):
    current_price = df['close'].iloc[-1]
    sma_20 = df['sma_20'].iloc[-1]
    sma_50 = df['sma_50'].iloc[-1]
    
    if pd.isna(sma_20) or pd.isna(sma_50):
        return "Neutral"
    
    if current_price > sma_20 > sma_50:
        return "Bullish"
    elif current_price < sma_20 < sma_50:
        return "Bearish"
    else:
        return "Neutral"

def calculate_risk_reward(df, signal):
    atr = df['atr'].iloc[-1]
    current_price = df['close'].iloc[-1]
    
    if signal == "BUY":
        stop_loss = current_price - 2 * atr
        take_profit = current_price + 3 * atr
    else:
        stop_loss = current_price + 2 * atr
        take_profit = current_price - 3 * atr
    
    return stop_loss, take_profit

def detect_fake_move(df):
    volume_change = df['volume_change'].iloc[-1]
    price_change = df['price_change'].iloc[-1]
    
    if abs(price_change) > 0.05 and volume_change < 2:
        return True
    return False

async def analyze_coin(symbol, interval):
    df = fetch_ohlcv(symbol, interval)
    if df is None:
        return None
    
    df = calculate_indicators(df)
    if df is None:
        return None
    
    signal, probability = predict_signal(df)
    market_structure = analyze_market_structure(df)
    stop_loss, take_profit = calculate_risk_reward(df, signal)
    
    current_price = df['close'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    macd = df['macd'].iloc[-1]
    
    is_fake_move = detect_fake_move(df)
    
    analysis = {
        "symbol": symbol,
        "interval": interval,
        "price": current_price,
        "signal": signal,
        "probability": probability,
        "market_structure": market_structure,
        "rsi": rsi,
        "macd": macd,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "is_fake_move": is_fake_move
    }
    
    return analysis

def format_price(price):
    if price >= 10:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.3f}"
    elif price >= 0.1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.5f}"
    elif price >= 0.001:
        return f"{price:.6f}"
    elif price >= 0.0001:
        return f"{price:.7f}"
    else:
        return f"{price:.8f}"

def format_analysis(analysis):
    fake_move_warning = "WARNING: Potential fake move detected!" if analysis['is_fake_move'] else ""
    return f"""
{analysis['symbol']} | {analysis['interval']} | Signal: {analysis['signal']} (Prob: {analysis['probability']:.2f})
Price: {format_price(analysis['price'])} | Structure: {analysis['market_structure']}
RSI: {analysis['rsi']:.2f} | MACD: {format_price(analysis['macd'])}
Stop Loss: {format_price(analysis['stop_loss'])} | Take Profit: {format_price(analysis['take_profit'])}
{fake_move_warning}
"""

async def analyze_symbol_all_timeframes(symbol):
    tasks = [analyze_coin(symbol, interval) for interval in TIMEFRAMES]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

async def scan_market():
    print("Scanning entire market... This may take a while.")
    all_symbols = session.get_instruments_info(category="linear")
    symbols = [symbol_info['symbol'] for symbol_info in all_symbols['result']['list']]
    
    very_strong_signals = []
    chunk_size = 20  # Analyze 20 symbols at a time
    
    with tqdm(total=len(symbols), desc="Analyzing Symbols") as pbar:
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            tasks = [analyze_symbol_all_timeframes(symbol) for symbol in chunk]
            chunk_results = await asyncio.gather(*tasks)
            
            for results in chunk_results:
                very_strong_signals.extend([r for r in results if r and r['probability'] > 0.85 and not r['is_fake_move']])
            
            pbar.update(len(chunk))
            
            # Print intermediate results
            if very_strong_signals:
                print("\nIntermediate Results:")
                for signal in sorted(very_strong_signals, key=lambda x: x['probability'], reverse=True)[:3]:
                    print(format_analysis(signal))
    
    if very_strong_signals:
        logger.info("Very strong signals detected:")
        for signal in sorted(very_strong_signals, key=lambda x: x['probability'], reverse=True)[:6]:
            logger.info(format_analysis(signal))
    else:
        logger.info("No very strong signals detected in this scan.")

async def manual_search():
    while True:
        symbol = input("Enter a symbol to analyze (or 'q' to quit): ").upper()
        if symbol == 'Q':
            break
        
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        
        print(f"Analyzing {symbol} on all timeframes...")
        try:
            results = await analyze_symbol_all_timeframes(symbol)
            
            if results:
                for analysis in results:
                    print(format_analysis(analysis))
            else:
                print(f"Unable to analyze {symbol}. The symbol might not exist or there's insufficient data.")
        except Exception as e:
            print(f"An error occurred while analyzing {symbol}: {str(e)}")
        
        print("\nPress Enter to continue...")
        input()

async def main():
    logger.info("Starting Advanced Crypto Scanner (Testnet)")
    
    # Initial model training
    logger.info("Training initial models...")
    training_tasks = [train_model(symbol, '60') for symbol in ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT']]
    await asyncio.gather(*training_tasks)
    
    while True:
        print("\n1. Auto Scan Market")
        print("2. Manual Search")
        print("3. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            await scan_market()
        elif choice == '2':
            await manual_search()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())
