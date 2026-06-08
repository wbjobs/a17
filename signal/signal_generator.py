import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from shared.models import TradingSignal, WindowSentimentAggregate
from streaming.spark_processor import StreamDataStore
from shared.utils import calculate_rolling_correlation


class TradingSignalGenerator:
    def __init__(self, data_store: StreamDataStore):
        self.data_store = data_store
        self.last_signal_time: dict = defaultdict(lambda: datetime.min)
        self.min_signal_interval = 5
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        now = datetime.now()
        
        last_time = self.last_signal_time.get(symbol, datetime.min)
        if (now - last_time).total_seconds() < self.min_signal_interval:
            return None
        
        window_aggs = list(self.data_store.window_aggregates.get(symbol, []))
        if not window_aggs:
            return None
        
        latest_agg = window_aggs[-1]
        
        sentiment_scores = self.data_store.get_sentiment_scores(
            symbol, limit=settings.CORRELATION_WINDOW
        )
        prices = self.data_store.get_price_values(
            symbol, limit=settings.CORRELATION_WINDOW
        )
        
        correlation = calculate_rolling_correlation(
            sentiment_scores, prices, settings.CORRELATION_WINDOW
        )
        
        signal_strength, signal_type, reason = self._calculate_signal(
            latest_agg.avg_sentiment,
            correlation,
            latest_agg.sentiment_momentum,
            latest_agg.positive_ratio,
            latest_agg.negative_ratio
        )
        
        if signal_type is None:
            return None
        
        confidence = self._calculate_confidence(
            latest_agg, correlation, len(sentiment_scores)
        )
        
        signal = TradingSignal(
            symbol=symbol,
            signal=signal_type,
            strength=abs(signal_strength),
            sentiment_score=latest_agg.avg_sentiment,
            price_correlation=correlation,
            timestamp=now,
            confidence=confidence,
            reason=reason
        )
        
        self.last_signal_time[symbol] = now
        return signal
    
    def _calculate_signal(
        self,
        avg_sentiment: float,
        correlation: float,
        momentum: float,
        positive_ratio: float,
        negative_ratio: float
    ) -> Tuple[float, Optional[str], str]:
        
        correlation_factor = 1 + abs(correlation) * 0.5
        momentum_factor = 1 + momentum * 0.3
        
        if correlation >= 0:
            adjusted_score = avg_sentiment * correlation_factor * momentum_factor
        else:
            adjusted_score = avg_sentiment * (1 - abs(correlation) * 0.3) * momentum_factor
        
        buy_threshold = settings.SIGNAL_THRESHOLD_BUY
        sell_threshold = settings.SIGNAL_THRESHOLD_SELL
        
        if adjusted_score >= buy_threshold and correlation > 0.2:
            signal_type = "BUY"
            reason = (f"Strong positive sentiment ({avg_sentiment:.3f}) "
                     f"with positive price correlation ({correlation:.3f}), "
                     f"momentum ({momentum:.3f})")
        elif adjusted_score <= sell_threshold and correlation < -0.2:
            signal_type = "SELL"
            reason = (f"Strong negative sentiment ({avg_sentiment:.3f}) "
                     f"with negative price correlation ({correlation:.3f}), "
                     f"momentum ({momentum:.3f})")
        elif adjusted_score >= buy_threshold * 0.8 and positive_ratio > 0.6:
            signal_type = "BUY"
            reason = (f"High positive news ratio ({positive_ratio:.2f}), "
                     f"sentiment ({avg_sentiment:.3f})")
        elif adjusted_score <= sell_threshold * 0.8 and negative_ratio > 0.6:
            signal_type = "SELL"
            reason = (f"High negative news ratio ({negative_ratio:.2f}), "
                     f"sentiment ({avg_sentiment:.3f})")
        else:
            signal_type = None
            reason = "No strong signal"
        
        return adjusted_score, signal_type, reason
    
    def _calculate_confidence(
        self,
        agg: WindowSentimentAggregate,
        correlation: float,
        data_points: int
    ) -> float:
        
        components = []
        
        sentiment_strength = min(abs(agg.avg_sentiment) * 1.5, 1.0)
        components.append(sentiment_strength * 0.35)
        
        correlation_confidence = min(abs(correlation) * 1.5, 1.0)
        components.append(correlation_confidence * 0.25)
        
        news_count_confidence = min(agg.news_count / 20.0, 1.0)
        components.append(news_count_confidence * 0.2)
        
        ratio_confidence = max(agg.positive_ratio, agg.negative_ratio)
        components.append(ratio_confidence * 0.1)
        
        data_confidence = min(data_points / settings.CORRELATION_WINDOW, 1.0)
        components.append(data_confidence * 0.1)
        
        return float(sum(components))


class Position:
    def __init__(self, symbol: str, entry_price: float, quantity: float, signal: TradingSignal):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_signal = signal
        self.entry_time = signal.timestamp
        self.current_price = entry_price
        self.pnl = 0.0
    
    def update(self, current_price: float):
        self.current_price = current_price
        self.pnl = (current_price - self.entry_price) * self.quantity


class Portfolio:
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict = {}
        self.trades: List[TradingSignal] = []
        self.equity_curve: List[float] = [initial_cash]
        self.returns: List[float] = []
    
    def execute_signal(self, signal: TradingSignal, current_price: float, position_size: float = 0.1):
        symbol = signal.symbol
        position_value = self.total_equity() * position_size
        quantity = position_value / current_price
        
        if signal.signal == "BUY":
            if symbol not in self.positions:
                cost = quantity * current_price
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[symbol] = Position(symbol, current_price, quantity, signal)
                    self.trades.append(signal)
                    return True
        elif signal.signal == "SELL":
            if symbol in self.positions:
                position = self.positions[symbol]
                revenue = position.quantity * current_price
                self.cash += revenue
                pnl_pct = (current_price - position.entry_price) / position.entry_price
                self.returns.append(pnl_pct)
                del self.positions[symbol]
                self.trades.append(signal)
                return True
        
        return False
    
    def update_prices(self, prices: dict):
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update(price)
        
        current_equity = self.total_equity()
        if len(self.equity_curve) > 0:
            daily_return = (current_equity - self.equity_curve[-1]) / self.equity_curve[-1]
            if daily_return != 0:
                self.returns.append(daily_return)
        self.equity_curve.append(current_equity)
    
    def total_equity(self) -> float:
        position_value = sum(
            pos.current_price * pos.quantity 
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)


class SignalExecutor:
    def __init__(self, data_store: StreamDataStore, initial_cash: float = 100000.0):
        self.data_store = data_store
        self.portfolio = Portfolio(initial_cash)
        self.generator = TradingSignalGenerator(data_store)
    
    def process_signals(self):
        prices = {}
        for symbol in settings.SYMBOLS:
            recent_prices = self.data_store.get_recent_prices(symbol, limit=1)
            if recent_prices:
                prices[symbol] = recent_prices[-1].price
        
        self.portfolio.update_prices(prices)
        
        for symbol in settings.SYMBOLS:
            signal = self.generator.generate_signal(symbol)
            if signal and symbol in prices:
                executed = self.portfolio.execute_signal(signal, prices[symbol])
                if executed:
                    print(f"Executed {signal.signal} signal for {symbol} "
                          f"at ${prices[symbol]:.2f}, "
                          f"strength: {signal.strength:.3f}")
        
        return self.portfolio
