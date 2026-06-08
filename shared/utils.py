import uuid
import random
from datetime import datetime, timedelta
from typing import List, Tuple
import numpy as np
from config.settings import settings


def generate_id() -> str:
    return str(uuid.uuid4())


def generate_timestamp() -> datetime:
    return datetime.now()


def calculate_rolling_correlation(
    sentiment_scores: List[float],
    prices: List[float],
    window: int = 30
) -> float:
    if len(sentiment_scores) < window or len(prices) < window:
        return 0.0
    
    s = np.array(sentiment_scores[-window:])
    p = np.array(prices[-window:])
    
    if len(s) < 2 or len(p) < 2:
        return 0.0
    
    if np.std(s) == 0 or np.std(p) == 0:
        return 0.0
    
    return float(np.corrcoef(s, p)[0, 1])


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    periods: int = 252
) -> float:
    if len(returns) == 0:
        return 0.0
    
    returns_array = np.array(returns)
    daily_returns = returns_array
    
    if len(daily_returns) < 2:
        return 0.0
    
    excess_returns = daily_returns - (risk_free_rate / periods)
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.sqrt(periods) * (np.mean(excess_returns) / np.std(excess_returns))
    return float(sharpe)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    return float(max_dd)


def generate_price_series(
    start_price: float,
    days: int,
    volatility: float = 0.02,
    drift: float = 0.0005
) -> List[float]:
    prices = [start_price]
    for _ in range(days - 1):
        change = np.random.normal(drift, volatility)
        next_price = prices[-1] * (1 + change)
        prices.append(max(next_price, 0.01))
    return prices


def generate_symbol_price(symbol: str) -> float:
    base_prices = {
        "AAPL": 190.0,
        "GOOGL": 140.0,
        "MSFT": 380.0,
        "AMZN": 150.0,
        "META": 500.0,
        "TSLA": 250.0,
        "NVDA": 800.0,
        "JPM": 195.0,
        "BAC": 33.0,
        "V": 275.0
    }
    base = base_prices.get(symbol, 100.0)
    return base * (1 + random.uniform(-0.02, 0.02))


def generate_order_book(symbol: str, current_price: float, levels: int = 10) -> Tuple[List, List]:
    bids = []
    asks = []
    
    for i in range(1, levels + 1):
        bid_price = current_price * (1 - i * 0.001)
        ask_price = current_price * (1 + i * 0.001)
        bid_size = random.uniform(100, 5000)
        ask_size = random.uniform(100, 5000)
        
        bids.append({'price': round(bid_price, 2), 'size': round(bid_size, 2), 'side': 'bid'})
        asks.append({'price': round(ask_price, 2), 'size': round(ask_size, 2), 'side': 'ask'})
    
    return bids, asks
