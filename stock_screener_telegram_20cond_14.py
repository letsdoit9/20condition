import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

# Configuration
TELEGRAM_BOT_TOKEN = "7923075723:AAGL5-DGPSU0TLb68vOLretVwioC6vK0fJk"
TELEGRAM_CHAT_ID = "457632002"
BASE_URL = "https://api.upstox.com/v2"

# Page config
st.set_page_config(page_title="Nifty 500 Screener", page_icon="📈", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               padding: 1rem; border-radius: 10px; color: white; text-align: center; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_instruments():
    try:
        return pd.read_csv('nifty500_instrument_keys.csv')
    except FileNotFoundError:
        st.error("❌ CSV file not found!")
        return pd.DataFrame()

@st.cache_data(ttl=300, max_entries=1000)
def get_stock_data(instrument_key, headers, days_data):
    """Optimized single stock data fetch with caching"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_data)).strftime('%Y-%m-%d')
    
    try:
        url = f"{BASE_URL}/historical-candle/{instrument_key}/day/{end_date}/{start_date}"
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'candles' in data['data']:
                candles = data['data']['candles']
                if candles:
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    return df.sort_values('timestamp').reset_index(drop=True)
    except:
        pass
    return None

# Optimized vectorized indicator calculations
@functools.lru_cache(maxsize=500)
def calculate_ema(data_tuple, span):
    """Cached EMA calculation"""
    data = np.array(data_tuple)
    alpha = 2 / (span + 1)
    ema_vals = np.zeros_like(data)
    ema_vals[0] = data[0]
    for i in range(1, len(data)):
        ema_vals[i] = alpha * data[i] + (1 - alpha) * ema_vals[i-1]
    return ema_vals

def calculate_all_indicators(df):
    """Vectorized calculation of all 20+ indicators"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    open_prices = df['open'].values
    
    # Convert to tuple for caching
    close_tuple = tuple(close)
    
    # EMAs & SMAs (vectorized)
    df['ema5'] = calculate_ema(close_tuple, 5)
    df['ema13'] = calculate_ema(close_tuple, 13)
    df['ema26'] = calculate_ema(close_tuple, 26)
    
    # SMAs using pandas rolling (optimized)
    close_series = pd.Series(close)
    df['sma50'] = close_series.rolling(50, min_periods=1).mean().values
    df['sma100'] = close_series.rolling(100, min_periods=1).mean().values
    df['sma200'] = close_series.rolling(200, min_periods=1).mean().values
    
    # RSI (vectorized)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use exponential moving average for RSI (faster than rolling)
    alpha = 1/14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = np.mean(gain[:14]) if len(gain) >= 14 else gain[0]
    avg_loss[0] = np.mean(loss[:14]) if len(loss) >= 14 else loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # StochRSI
    rsi_series = pd.Series(df['rsi'])
    rsi_min = rsi_series.rolling(14, min_periods=1).min()
    rsi_max = rsi_series.rolling(14, min_periods=1).max()
    df['stoch_rsi'] = ((rsi_series - rsi_min) / (rsi_max - rsi_min + 1e-8) * 100).values
    
    # MACD
    ema12 = calculate_ema(close_tuple, 12)
    ema26 = calculate_ema(close_tuple, 26)
    macd = ema12 - ema26
    df['macd'] = macd
    df['macd_signal'] = calculate_ema(tuple(macd), 9)
    
    # Bollinger Bands
    sma20 = close_series.rolling(20, min_periods=1).mean()
    std20 = close_series.rolling(20, min_periods=1).std()
    df['bb_upper'] = (sma20 + std20 * 2).values
    df['bb_lower'] = (sma20 - std20 * 2).values
    
    # ATR and ADX (vectorized)
    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    tr = np.maximum(hl, np.maximum(hc, lc))
    df['atr'] = pd.Series(tr).rolling(14, min_periods=1).mean().values
    
    # DI+ and DI- (vectorized)
    plus_dm = np.where((np.diff(high, prepend=high[0]) > np.abs(np.diff(low, prepend=low[0]))) & 
                      (np.diff(high, prepend=high[0]) > 0), np.diff(high, prepend=high[0]), 0)
    minus_dm = np.where((np.abs(np.diff(low, prepend=low[0])) > np.diff(high, prepend=high[0])) & 
                       (np.diff(low, prepend=low[0]) < 0), np.abs(np.diff(low, prepend=low[0])), 0)
    
    atr_series = pd.Series(tr).rolling(14, min_periods=1).mean()
    di_plus = 100 * pd.Series(plus_dm).rolling(14, min_periods=1).mean() / (atr_series + 1e-8)
    di_minus = 100 * pd.Series(minus_dm).rolling(14, min_periods=1).mean() / (atr_series + 1e-8)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)
    df['adx'] = dx.rolling(14, min_periods=1).mean().values
    df['di_plus'] = di_plus.values
    df['di_minus'] = di_minus.values
    
    # OBV (vectorized)
    price_change = np.diff(close, prepend=close[0])
    obv_multiplier = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    df['obv'] = np.cumsum(obv_multiplier * volume)
    
    # VWAP
    vwap = (close_series * pd.Series(volume)).rolling(20, min_periods=1).sum() / pd.Series(volume).rolling(20, min_periods=1).sum()
    df['vwap'] = vwap.values
    
    # Ichimoku (simplified calculation)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    high9 = high_series.rolling(9, min_periods=1).max()
    low9 = low_series.rolling(9, min_periods=1).min()
    high26 = high_series.rolling(26, min_periods=1).max()
    low26 = low_series.rolling(26, min_periods=1).min()
    high52 = high_series.rolling(52, min_periods=1).max()
    low52 = low_series.rolling(52, min_periods=1).min()
    
    df['tenkan'] = ((high9 + low9) / 2).values
    df['kijun'] = ((high26 + low26) / 2).values
    df['senkou_a'] = (((high9 + low9) / 2 + (high26 + low26) / 2) / 2).shift(26).fillna(method='bfill').values
    df['senkou_b'] = ((high52 + low52) / 2).shift(26).fillna(method='bfill').values
    
    return df

def check_all_20_conditions(df, thresholds):
    """All 20 original conditions with optimized checks"""
    if len(df) < 50:
        return 0, {}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Pre-calculate common values
    min_periods = min(200, len(df))
    high_52w = df['high'].rolling(min_periods, min_periods=1).max().iloc[-1]
    vol_avg_20 = df['volume'].rolling(20, min_periods=1).mean().iloc[-1]
    obv_6_ago = df['obv'].iloc[-6] if len(df) >= 6 else df['obv'].iloc[0]
    
    conditions = {
        'C1_EMA5_Above_EMA13': latest['ema5'] > latest['ema13'],
        'C2_EMA13_Above_EMA26': latest['ema13'] > latest['ema26'],
        'C3_SMA50_Above_SMA100': latest['sma50'] > latest['sma100'],
        'C4_SMA100_Above_SMA200': latest['sma100'] > latest['sma200'],
        'C5_DIPlus_Above_DIMinus': latest['di_plus'] > latest['di_minus'],
        'C6_ADX_Above_Threshold': latest['adx'] > thresholds['adx'],
        'C7_MACD_Above_Signal': latest['macd'] > latest['macd_signal'],
        'C8_RSI_In_Range': thresholds['rsi_min'] < latest['rsi'] < thresholds['rsi_max'],
        'C9_StochRSI_Below_Threshold': latest['stoch_rsi'] < thresholds['stoch_rsi_max'],
        'C10_Close_Above_BB_Upper': latest['close'] > latest['bb_upper'],
        'C11_Bullish_Candle': latest['close'] > latest['open'],
        'C12_Volume_Above_Min': latest['volume'] > thresholds['volume_min'],
        'C13_Price_Near_52W_High': latest['close'] > high_52w * thresholds['high_threshold'],
        'C14_Higher_Close': latest['close'] > prev['close'],
        'C15_Volume_Above_Avg': latest['volume'] > vol_avg_20,
        'C16_Close_Above_VWAP': latest['close'] > latest['vwap'],
        'C17_RSI_Rising_3Days': (len(df) >= 4 and 
                                all(df['rsi'].iloc[-i] > df['rsi'].iloc[-i-1] 
                                    for i in range(1, min(4, len(df))))),
        'C18_OBV_Rising': latest['obv'] > obv_6_ago,
        'C19_Ichimoku_Bullish': (latest['close'] > latest['tenkan'] and 
                                latest['tenkan'] > latest['kijun']),
        'C20_Cloud_Above': latest['senkou_a'] > latest['senkou_b']
    }
    
    score = sum(conditions.values())
    return score, conditions

def process_single_stock(args):
    """Optimized single stock processing"""
    symbol, instrument_key, thresholds, headers, days_data = args
    
    df = get_stock_data(instrument_key, headers, days_data)
    if df is not None and len(df) >= 20:
        try:
            df = calculate_all_indicators(df)
            score, conditions = check_all_20_conditions(df, thresholds)
            
            if score >= 5:  # Only return promising stocks
                latest = df.iloc[-1]
                return {
                    'Stock': symbol,
                    'Price': latest['close'],
                    'RSI': latest['rsi'],
                    'Volume': latest['volume'],
                    'ADX': latest['adx'],
                    'ATR': latest['atr'],
                    'Score': score,
                    'Conditions': conditions,
                    'Data': df.tail(50)
                }
        except Exception as e:
            pass
    return None

def send_telegram_message(message):
    """Send message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
        response = requests.post(url, data=payload, timeout=10)
        return response.status_code == 200
    except:
        return False

def format_telegram_message(results):
    """Format top 10 results for Telegram"""
    message = f"🚀 <b>NIFTY 500 SCREENER</b>\n📅 {datetime.now().strftime('%d-%m-%Y %H:%M')}\n\n"
    
    for i, stock in enumerate(results[:10], 1):
        cmp = stock['Price']
        atr = stock['ATR']
        target1 = cmp + (1.5 * atr)
        target2 = cmp + (2.0 * atr)
        stoploss = cmp - (1.0 * atr)
        
        message += f"<b>#{i} {stock['Stock']}</b>\n"
        message += f"💰 CMP: ₹{cmp:.2f} | 🎯 T1: ₹{target1:.2f} | T2: ₹{target2:.2f}\n"
        message += f"🛡 SL: ₹{stoploss:.2f} | 📊 Score: {stock['Score']}/20\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    return message

def main():
    st.markdown('<h1 class="main-header">🚀 Nifty 500 Stock Screener</h1>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    with st.sidebar.expander("🔐 API Settings", expanded=True):
        access_token = st.text_input("Access Token", type="password")
    
    # Parameters
    scan_limit = st.sidebar.slider("Stocks to scan", 10, 500, 50)
    days_data = st.sidebar.slider("Historical days", 50, 200, 100)
    
    # Thresholds
    st.sidebar.header("📊 Thresholds")
    with st.sidebar.expander("Technical Settings"):
        adx_threshold = st.slider("ADX Min", 15.0, 40.0, 25.0)
        rsi_min = st.slider("RSI Min", 20, 40, 30)
        rsi_max = st.slider("RSI Max", 60, 80, 70)
        stoch_rsi_max = st.slider("StochRSI Max", 20, 40, 30)
        volume_min = st.number_input("Min Volume", 50000, 500000, 100000)
        high_threshold = st.slider("52W High %", 0.8, 0.99, 0.95, 0.01)
    
    send_to_telegram = st.sidebar.checkbox("Send to Telegram", True)
    
    # Load instruments
    instruments = load_instruments()
    if instruments.empty:
        return
    
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Start Screening", type="primary"):
            if not access_token:
                st.error("Please enter access token")
                return
            
            thresholds = {
                'adx': adx_threshold,
                'rsi_min': rsi_min,
                'rsi_max': rsi_max,
                'stoch_rsi_max': stoch_rsi_max,
                'volume_min': volume_min,
                'high_threshold': high_threshold
            }
            
            # Prepare stocks for processing
            stocks_to_process = instruments.head(scan_limit)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            # Process with optimized threading
            args_list = [
                (row['tradingsymbol'], row['instrument_key'], thresholds, headers, days_data)
                for _, row in stocks_to_process.iterrows()
            ]
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_single_stock, args) for args in args_list]
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    progress = (i + 1) / len(futures)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {i+1}/{len(futures)} stocks...")
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                # Sort by score
                results.sort(key=lambda x: x['Score'], reverse=True)
                
                # Send to Telegram
                if send_to_telegram:
                    telegram_message = format_telegram_message(results)
                    with st.spinner("📱 Sending to Telegram..."):
                        if send_telegram_message(telegram_message):
                            st.success("✅ Sent to Telegram!")
                        else:
                            st.error("❌ Telegram failed!")
                
                # Display results
                st.header("📊 Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Scanned", len(futures))
                with col2:
                    st.metric("Qualified", len(results))
                with col3:
                    st.metric("Best Score", f"{results[0]['Score']}/20")
                with col4:
                    avg_score = sum(r['Score'] for r in results) / len(results)
                    st.metric("Avg Score", f"{avg_score:.1f}/20")
                
                # Results table
                df_results = pd.DataFrame([{
                    'Stock': r['Stock'],
                    'Price': f"₹{r['Price']:.2f}",
                    'Score': f"{r['Score']}/20",
                    'RSI': f"{r['RSI']:.1f}",
                    'ADX': f"{r['ADX']:.1f}",
                    'Volume': f"{r['Volume']:,.0f}"
                } for r in results])
                
                st.dataframe(df_results, use_container_width=True)
                
                # Trading signals
                st.header("📋 Trading Signals")
                signals_text = ""
                for i, stock in enumerate(results[:10], 1):
                    cmp = stock['Price']
                    atr = stock['ATR']
                    signals_text += f"{i}. {stock['Stock']}: CMP ₹{cmp:.2f} | T1 ₹{cmp+1.5*atr:.2f} | T2 ₹{cmp+2*atr:.2f} | SL ₹{cmp-atr:.2f}\n"
                
                st.text_area("Top 10 Signals", signals_text, height=200)
                
                # Top 5 detailed view
                st.header("🏆 Top 5 Analysis")
                for i, stock in enumerate(results[:5]):
                    with st.expander(f"#{i+1} {stock['Stock']} - Score: {stock['Score']}/20"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Chart
                            df_chart = stock['Data']
                            fig = go.Figure(data=[go.Candlestick(
                                x=df_chart.index,
                                open=df_chart['open'],
                                high=df_chart['high'],
                                low=df_chart['low'],
                                close=df_chart['close']
                            )])
                            fig.update_layout(title=stock['Stock'], height=300, xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Trading setup
                            cmp = stock['Price']
                            atr = stock['ATR']
                            st.markdown(f"""
                            **Trading Setup:**
                            - CMP: ₹{cmp:.2f}
                            - Target 1: ₹{cmp + 1.5*atr:.2f}
                            - Target 2: ₹{cmp + 2*atr:.2f}
                            - Stoploss: ₹{cmp - atr:.2f}
                            """)
                            
                            # Conditions summary
                            passed = sum(stock['Conditions'].values())
                            st.metric("Conditions Met", f"{passed}/20")
            else:
                st.warning("No qualifying stocks found. Try adjusting thresholds.")
    
    with col2:
        st.metric("Available Stocks", len(instruments))
    
    with col3:
        if st.button("📱 Test Telegram"):
            test_msg = f"🤖 Test from Screener\n{datetime.now().strftime('%d-%m-%Y %H:%M')}"
            if send_telegram_message(test_msg):
                st.success("✅ Telegram OK!")
            else:
                st.error("❌ Telegram failed!")

if __name__ == "__main__":
    main()