import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from telegram.ext import ApplicationBuilder, CommandHandler
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import time
import signal
import sys
import asyncio
from pybit.unified_trading import HTTP
import requests
from datetime import datetime
import warnings
from textblob import TextBlob
import concurrent.futures

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
TELEGRAM_BOT_TOKEN = '7378817782:AAEXhA3akmTXzfvqUjBG7Ew8XROe6-V_8rQ'
TELEGRAM_CHANNEL_ID = '@EtelCryptoScanner'
BYBIT_API_KEY = 'qN3MG3kfIWhfeLIldB'
BYBIT_SECRET = '7Uk5wQKAshUeYilFwytz7AhAQU0hLwrtba30'
SCAN_INTERVAL_MINUTES = 5
MAX_CONCURRENT_REQUESTS = 5

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Bybit session for live trading
session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_SECRET)

# Initialize scheduler
scheduler = BackgroundScheduler()

# Timeframes
TIMEFRAMES = [('5m', '5'), ('15m', '15'), ('30m', '30'), ('1H', '60'), ('4H', '240'), ('1D', 'D')]

# Machine Learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)

def fetch_ohlcv_data(symbol, timeframe, limit=1000):
    try:
        logger.info(f"Fetching OHLCV data for {symbol} on {timeframe} timeframe")
        kline_data = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        if 'retCode' in kline_data and kline_data['retCode'] != 0:
            logger.error(f"Error fetching OHLCV data: {kline_data['retMsg']}")
            return None
        if 'result' not in kline_data or 'list' not in kline_data['result']:
            logger.error(f"Unexpected response format for {symbol} on {timeframe}: {kline_data}")
            return None
        df = pd.DataFrame(kline_data['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        if df.empty:
            logger.warning(f"Received empty dataframe for {symbol} on {timeframe}")
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        logger.info(f"Successfully fetched OHLCV data for {symbol} on {timeframe}. Got {len(df)} candles.")
        return df.iloc[::-1]
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol} on {timeframe}: {str(e)}", exc_info=True)
        return None

def calculate_indicators(df):
    try:
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        df['rsi'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        bb = ta.bbands(df['close'], length=20)
        df['upper_bb'] = bb['BBU_20_2.0']
        df['middle_bb'] = bb['BBM_20_2.0']
        df['lower_bb'] = bb['BBL_20_2.0']
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx['ADX_14']
        df['obv'] = ta.obv(df['close'], df['volume'])
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
        return None

def prepare_ml_data(df):
    try:
        X = df[['rsi', 'macd', 'cci', 'mfi', 'adx', 'obv']].dropna().values
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)[:-1]
        return X[:-1], y
    except Exception as e:
        logger.error(f"Error preparing ML data: {str(e)}", exc_info=True)
        return None, None

def train_ml_model(symbol, timeframe):
    global model
    try:
        df = fetch_ohlcv_data(symbol, timeframe, limit=5000)
        if df is None or len(df) < 1000:
            logger.warning(f"Insufficient data for ML model training: {symbol} on {timeframe}")
            return False
        
        df = calculate_indicators(df)
        if df is None:
            logger.warning(f"Failed to calculate indicators for ML model training: {symbol} on {timeframe}")
            return False
        
        X, y = prepare_ml_data(df)
        if X is None or y is None:
            logger.warning(f"Failed to prepare ML data for training: {symbol} on {timeframe}")
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        logger.info(f"ML model trained for {symbol} on {timeframe}. Accuracy: {accuracy:.2f}")
        return True
    except Exception as e:
        logger.error(f"Error training ML model: {str(e)}", exc_info=True)
        return False

def get_ml_prediction(df):
    try:
        X = df[['rsi', 'macd', 'cci', 'mfi', 'adx', 'obv']].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(X)
        probability = model.predict_proba(X)[0]
        return "BUY" if prediction[0] == 1 else "SELL", max(probability)
    except Exception as e:
        logger.error(f"Error in ML prediction: {str(e)}", exc_info=True)
        return "N/A", 0.0

def analyze_sentiment(symbol):
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token=YOUR_CRYPTOPANIC_API_KEY&currencies={symbol}"
        response = requests.get(url)
        data = response.json()
        if 'results' in data:
            recent_news = data['results'][:5]
            sentiment_scores = []
            for news in recent_news:
                analysis = TextBlob(news['title'])
                sentiment_scores.append(analysis.sentiment.polarity)
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            return "Bullish" if avg_sentiment > 0 else "Bearish" if avg_sentiment < 0 else "Neutral"
    except Exception as e:
        logger.error(f"Error in sentiment analysis for {symbol}: {str(e)}")
    return "Neutral"

def analyze_candles(df, n=10):
    recent_candles = df.tail(n)
    bearish_count = sum(recent_candles['close'] < recent_candles['open'])
    bullish_count = n - bearish_count
    trend = "Bearish" if bearish_count > bullish_count else "Bullish"
    strength = max(bearish_count, bullish_count)
    return trend, strength

def find_key_levels(df):
    pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
    s1 = 2 * pivot - df['high'].iloc[-1]
    s2 = pivot - (df['high'].iloc[-1] - df['low'].iloc[-1])
    r1 = 2 * pivot - df['low'].iloc[-1]
    r2 = pivot + (df['high'].iloc[-1] - df['low'].iloc[-1])
    return s2, s1, r1, r2

def calculate_24h_change(df):
    current_price = df['close'].iloc[-1]
    price_24h_ago = df['close'].iloc[-24]  # Assuming 1-hour timeframe
    change = (current_price - price_24h_ago) / price_24h_ago * 100
    return f"{change:.1f}%"

def analyze_volume(df):
    avg_volume = df['volume'].tail(20).mean()
    current_volume = df['volume'].iloc[-1]
    volume_change = (current_volume - avg_volume) / avg_volume * 100
    return volume_change

def analyze_order_book(symbol, depth=50):
    try:
        order_book = session.get_orderbook(
            category="linear",
            symbol=symbol,
            limit=depth
        )
        bids = order_book['result']['b']
        asks = order_book['result']['a']
        
        total_bid_volume = sum(float(bid[1]) for bid in bids)
        total_ask_volume = sum(float(ask[1]) for ask in asks)
        total_volume = total_bid_volume + total_ask_volume
        
        buy_percentage = (total_bid_volume / total_volume) * 100
        sell_percentage = (total_ask_volume / total_volume) * 100
        
        bid_wall = max(bids, key=lambda x: float(x[1]))
        ask_wall = max(asks, key=lambda x: float(x[1]))
        
        return (f"{buy_percentage:.0f}% buy | {sell_percentage:.0f}% sell | "
                f"Buy wall @ {bid_wall[0]} | Sell wall @ {ask_wall[0]}")
    except Exception as e:
        logger.error(f"Error analyzing order book for {symbol}: {str(e)}")
        return "Order book analysis unavailable"

def detect_whale_activity(df, threshold=100):
    large_transactions = df[df['volume'] * df['close'] > threshold * 1000000]
    if not large_transactions.empty:
        latest_whale = large_transactions.iloc[-1]
        return f"{latest_whale['volume']:.0f} BTC {'sell' if latest_whale['close'] < latest_whale['open'] else 'buy'} @ {latest_whale['close']:.0f}"
    return None

def generate_entry_strategy(df, trend):
    current_price = df['close'].iloc[-1]
    atr = ta.atr(df['high'], df['low'], df['close']).iloc[-1]
    
    if trend == "Bearish":
        entry = current_price - atr * 0.5
        sl = entry + atr * 2
        tp1 = entry - atr * 1.5
        tp2 = entry - atr * 3
    else:
        entry = current_price + atr * 0.5
        sl = entry - atr * 2
        tp1 = entry + atr * 1.5
        tp2 = entry + atr * 3
    
    return entry, sl, tp1, tp2

def calculate_risk(entry, stop_loss, account_balance, risk_percentage=1):
    risk_amount = account_balance * (risk_percentage / 100)
    position_size = risk_amount / abs(entry - stop_loss)
    return position_size

def analyze_coin(symbol, timeframe):
    df = fetch_ohlcv_data(symbol, timeframe)
    if df is None or len(df) < 50:
        logger.warning(f"Insufficient data for {symbol} on {timeframe} timeframe")
        return None

    df = calculate_indicators(df)
    if df is None:
        logger.warning(f"Failed to calculate indicators for {symbol} on {timeframe}")
        return None

    trend, strength = analyze_candles(df)
    s2, s1, r1, r2 = find_key_levels(df)
    
    last_row = df.iloc[-1]
    current_price = last_row['close']
    
    rsi = last_row['rsi']
    macd = last_row['macd']
    
    try:
        ml_signal, ml_probability = get_ml_prediction(df)
    except Exception as e:
        logger.error(f"Error in ML prediction for {symbol} on {timeframe}: {str(e)}")
        ml_signal, ml_probability = "N/A", 0.0

    sentiment = analyze_sentiment(symbol.replace('USDT', ''))
    
    if trend == "Bullish" and rsi > 50 and macd > 0 and ml_signal == "BUY" and sentiment == "Bullish":
        signal = "STRONG BUY"
    elif trend == "Bearish" and rsi < 50 and macd < 0 and ml_signal == "SELL" and sentiment == "Bearish":
        signal = "STRONG SELL"
    elif (trend == "Bullish" and (rsi > 50 or macd > 0)) or ml_signal == "BUY":
        signal = "BUY"
    elif (trend == "Bearish" and (rsi < 50 or macd < 0)) or ml_signal == "SELL":
        signal = "SELL"
    else:
        signal = "NEUTRAL"
    
    confidence = (abs(rsi - 50) / 50 + abs(macd) / 100 + ml_probability) / 3
    quality = int(confidence * 10)
    
    volume_change = analyze_volume(df)
    order_book_analysis = analyze_order_book(symbol)
    whale_activity = detect_whale_activity(df)
    
    entry, sl, tp1, tp2 = generate_entry_strategy(df, trend)
    
    account_balance = 10000  # Assume a balance of 10,000 USDT
    position_size = calculate_risk(entry, sl, account_balance)
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "quality": quality,
        "signal": signal,
        "price": current_price,
        "change_24h": calculate_24h_change(df),
        "trend": f"{trend} ({strength}/10 last candles {'red' if trend == 'Bearish' else 'green'})",
        "key_levels": {"S2": s2, "S1": s1, "R1": r1, "R2": r2},
        "indicators": {
            "RSI": f"{rsi:.2f} {'↑' if rsi > 50 else '↓'}",
            "MACD": f"{macd:.4f} {'↑' if macd > 0 else '↓'}",
            "BB": "Upper band touch" if last_row['close'] > last_row['upper_bb'] else "Lower band bounce" if last_row['close'] < last_row['lower_bb'] else "Inside bands",
            "ADX": f"{last_row['adx']:.2f}",
            "CCI": f"{last_row['cci']:.2f}",
            "MFI": f"{last_row['mfi']:.2f}"
        },
        "ml_prediction": f"{ml_signal} (Probability: {ml_probability:.2f})",
        "sentiment": sentiment,
        "volume": f"{volume_change:.0f}% vs avg",
        "order_book": order_book_analysis,
        "whale_activity": whale_activity,
        "entry": entry,
        "stop_loss": sl,
        "take_profit1": tp1,
        "take_profit2": tp2,
        "position_size": position_size
    }

def format_analysis(analysis):
    return f"""
{analysis['symbol']} | {analysis['timeframe']} | Quality: {analysis['quality']}/10 | {analysis['signal']}

Price: {analysis['price']:.2f} ({analysis['change_24h']})
Trend: {analysis['trend']}
Sentiment: {analysis['sentiment']}

Key Levels:
S2: {analysis['key_levels']['S2']:.0f} | S1: {analysis['key_levels']['S1']:.0f} | R1: {analysis['key_levels']['R1']:.0f} | R2: {analysis['key_levels']['R2']:.0f}

Indicators:
RSI: {analysis['indicators']['RSI']} | MACD: {analysis['indicators']['MACD']}
BB: {analysis['indicators']['BB']}
ADX: {analysis['indicators']['ADX']} | CCI: {analysis['indicators']['CCI']} | MFI: {analysis['indicators']['MFI']}

ML Prediction: {analysis['ml_prediction']}

Volume: {analysis['volume']}
OB: {analysis['order_book']}

Whale: {analysis['whale_activity'] if analysis['whale_activity'] else 'No significant activity'}

Entry: {analysis['entry']:.0f}
SL: {analysis['stop_loss']:.0f} | TP1: {analysis['take_profit1']:.0f} | TP2: {analysis['take_profit2']:.0f}
Position Size: {analysis['position_size']:.2f} USDT

Action: {'BUY' if 'BUY' in analysis['signal'] else 'SELL'} {f"at {analysis['entry']:.0f}" if 'STRONG' not in analysis['signal'] else 'now'}
"""

async def send_telegram_message(message):
    try:
        await application.bot.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=message)
    except Exception as e:
        logger.error(f"Error sending Telegram message: {str(e)}")

def analyze_market():
    try:
        symbols = session.get_instruments_info(category="linear")
        all_symbols = [symbol['symbol'] for symbol in symbols['result']['list'] if symbol['symbol'].endswith('USDT')]
        strong_signals = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            futures = []
            for symbol in all_symbols:
                for _, tf in TIMEFRAMES:
                    futures.append(executor.submit(analyze_coin, symbol, tf))
            
            for future in concurrent.futures.as_completed(futures):
                analysis = future.result()
                if analysis and 'STRONG' in analysis['signal']:
                    strong_signals.append(analysis)

        return strong_signals
    except Exception as e:
        logger.error(f"Error in analyze_market: {str(e)}")
        return []

def auto_scan():
    try:
        strong_signals = analyze_market()
        if strong_signals:
            message = "Strong signals detected:\n\n"
            for signal in strong_signals:
                message += format_analysis(signal) + "\n"
            asyncio.create_task(send_telegram_message(message))
        else:
            logger.info("No strong signals detected in this scan.")
    except Exception as e:
        logger.error(f"Error in auto_scan: {str(e)}")

async def manual_search(update, context):
    if len(context.args) < 2:
        await update.message.reply_text("Please provide a symbol and timeframe(s). Example: /scan BTCUSDT 1,3,4")
        return
    
    symbol = context.args[0].upper()
    timeframe_inputs = context.args[1].split(',')
    
    valid_timeframes = []
    for tf in timeframe_inputs:
        if tf.isdigit() and 1 <= int(tf) <= len(TIMEFRAMES):
            valid_timeframes.append(TIMEFRAMES[int(tf) - 1][1])
        else:
            await update.message.reply_text(f"Invalid timeframe selection: {tf}. Skipping.")
    
    if not valid_timeframes:
        await update.message.reply_text("No valid timeframes selected. Please try again.")
        return
    
    for timeframe in valid_timeframes:
        analysis = analyze_coin(symbol, timeframe)
        if analysis:
            await update.message.reply_text(format_analysis(analysis))
        else:
            await update.message.reply_text(f"Unable to analyze {symbol} on {timeframe} timeframe.")

async def setup_telegram_bot():
    global application
    try:
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        await application.bot.delete_webhook()
        application.add_handler(CommandHandler('scan', manual_search))
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
    except Exception as e:
        logger.error(f"Error setting up Telegram bot: {str(e)}", exc_info=True)
        raise

def signal_handler(signum, frame):
    logger.info("Received shutdown signal. Stopping...")
    scheduler.shutdown()
    sys.exit(0)

async def main():
    try:
        await setup_telegram_bot()
        scheduler.add_job(auto_scan, 'interval', minutes=SCAN_INTERVAL_MINUTES, max_instances=1)
        scheduler.start()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("Advanced Crypto Scanner started. Press Ctrl+C to exit.")

        # Initial ML model training
        logger.info("Starting initial ML model training...")
        if train_ml_model('BTCUSDT', '1H'):
            logger.info("Initial ML model training completed successfully.")
        else:
            logger.warning("Initial ML model training failed. ML predictions may not be accurate.")

        while True:
            symbol = input("Enter a coin symbol (e.g., BTC) or 'q' to quit: ").strip().upper()
            if symbol.lower() == 'q':
                break
            if not symbol:
                continue
            
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
            
            print("Select timeframe(s):")
            for i, (name, code) in enumerate(TIMEFRAMES, 1):
                print(f"{i}. {name}")

            timeframe_input = input("Enter timeframe number(s) separated by commas: ")
            selected_timeframes = []

            for tf in timeframe_input.split(','):
                if tf.strip().isdigit() and 1 <= int(tf) <= len(TIMEFRAMES):
                    selected_timeframes.append(TIMEFRAMES[int(tf.strip()) - 1][1])
                else:
                    print(f"Invalid timeframe selection: {tf}. Skipping.")
            
            if not selected_timeframes:
                print("No valid timeframes selected. Please try again.")
                continue
            
            for timeframe in selected_timeframes:
                analysis = analyze_coin(symbol, timeframe)
                if analysis:
                    print(format_analysis(analysis))
                else:
                    print(f"Unable to analyze {symbol} on {timeframe} timeframe.")
            
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
    finally:
        await application.stop()
        scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
