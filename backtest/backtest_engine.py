import os
import sys
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from shared.models import NewsArticle, PriceData, TradingSignal, BacktestResult
from sentiment.finbert_analyzer import SentimentAnalyzer
from signal.signal_generator import TradingSignalGenerator
from shared.utils import (
    calculate_sharpe_ratio, calculate_max_drawdown,
    generate_price_series, generate_id
)


@dataclass
class BacktestConfig:
    execution_delay_ms: int = 100
    slippage_bps: float = 5.0
    commission_bps: float = 1.0
    position_size_pct: float = 0.1
    max_positions: int = 5
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    enable_short: bool = False
    lookahead_check_level: str = "strict"


@dataclass
class Order:
    id: str
    symbol: str
    side: str
    quantity: float
    create_time: float
    execute_time: Optional[float] = None
    execute_price: Optional[float] = None
    status: str = "PENDING"
    reason: str = ""


@dataclass
class BacktestPosition:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: float
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class TemporalDataGuard:
    def __init__(self, check_level: str = "strict"):
        self.check_level = check_level
        self.current_time: Optional[float] = None
        self._access_log: List[Tuple[str, float, float]] = []
        self._violations: List[str] = []
    
    def set_current_time(self, current_time: float):
        self.current_time = current_time
    
    def check_access(self, data_type: str, data_time: float, operation: str = "read") -> bool:
        if self.current_time is None:
            raise RuntimeError("Current time not set for temporal guard")
        
        time_diff = data_time - self.current_time
        
        if self.check_level == "strict":
            if time_diff > 1e-9:
                violation = (
                    f"LOOKAHEAD VIOLATION: {operation} {data_type} at "
                    f"{datetime.fromtimestamp(data_time)} but current time is "
                    f"{datetime.fromtimestamp(self.current_time)} (ahead by {time_diff:.6f}s)"
                )
                self._violations.append(violation)
                return False
        elif self.check_level == "warn":
            if time_diff > 1.0:
                print(f"WARNING: Potential lookahead: {data_type} ahead by {time_diff:.1f}s")
        
        self._access_log.append((data_type, data_time, self.current_time))
        return True
    
    def get_violations(self) -> List[str]:
        return self._violations.copy()
    
    def has_violations(self) -> bool:
        return len(self._violations) > 0


class HistoricalDataGenerator:
    def __init__(self, use_mock_sentiment: bool = True):
        self.analyzer = SentimentAnalyzer(use_mock=use_mock_sentiment)
        self._news_templates = [
            "{symbol} reports strong quarterly earnings growth",
            "{symbol} misses earnings expectations",
            "{symbol} announces new product launch",
            "{symbol} faces regulatory scrutiny",
            "{symbol} secures major client contract",
            "{symbol} CEO resigns unexpectedly",
            "{symbol} expands into emerging markets",
            "{symbol} stock downgraded by analysts",
            "{symbol} announces stock split",
            "{symbol} receives patent approval",
            "{symbol} partners with leading tech firm",
            "{symbol} recalls product due to defects",
            "{symbol} increases dividend payout",
            "{symbol} reports declining sales",
            "{symbol} acquires competitor in $5B deal",
            "{symbol} investigation into accounting practices",
            "{symbol} beats revenue estimates",
            "{symbol} warns of supply chain issues",
            "{symbol} enters new market segment",
            "{symbol} stock upgraded to strong buy"
        ]
    
    def generate_historical_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        news_per_day: int = 50
    ) -> List[NewsArticle]:
        news_list = []
        current_date = start_date
        
        while current_date <= end_date:
            num_news = max(1, int(np.random.normal(news_per_day, news_per_day * 0.3)))
            
            for _ in range(num_news):
                hour = random.randint(8, 20)
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                timestamp = current_date.replace(hour=hour, minute=minute, second=second)
                
                template = random.choice(self._news_templates)
                content = template.format(symbol=symbol)
                
                news = NewsArticle(
                    id=generate_id(),
                    symbol=symbol,
                    title=content[:60] + "..." if len(content) > 60 else content,
                    content=content,
                    source=random.choice(["Bloomberg", "Reuters", "WSJ", "CNBC"]),
                    timestamp=timestamp,
                    url=f"https://example.com/news/{generate_id()}"
                )
                news_list.append(news)
            
            current_date += timedelta(days=1)
        
        return sorted(news_list, key=lambda x: x.timestamp)
    
    def generate_historical_prices(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        start_price: float = None
    ) -> List[PriceData]:
        from shared.utils import generate_symbol_price
        
        if start_price is None:
            start_price = generate_symbol_price(symbol)
        
        days = (end_date - start_date).days + 1
        prices = generate_price_series(start_price, days, volatility=0.02, drift=0.0005)
        
        price_data = []
        current_date = start_date
        
        for i, price in enumerate(prices):
            for hour in [9, 10, 11, 12, 13, 14, 15, 16]:
                for minute in [0, 15, 30, 45]:
                    variation = np.random.normal(0, 0.003)
                    intraday_price = price * (1 + variation)
                    timestamp = current_date.replace(hour=hour, minute=minute, second=0)
                    
                    spread = intraday_price * 0.0005
                    bid_size = random.uniform(100, 5000)
                    ask_size = random.uniform(100, 5000)
                    
                    price_data.append(PriceData(
                        symbol=symbol,
                        price=round(intraday_price, 2),
                        volume=random.uniform(10000, 500000),
                        timestamp=timestamp,
                        bid=round(intraday_price - spread, 2),
                        ask=round(intraday_price + spread, 2),
                        bid_size=bid_size,
                        ask_size=ask_size
                    ))
            
            current_date += timedelta(days=1)
        
        return price_data


class OrderExecutionSimulator:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.pending_orders: deque = deque()
        self.executed_orders: List[Order] = []
    
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_time: float
    ) -> Order:
        order = Order(
            id=generate_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            create_time=current_time
        )
        self.pending_orders.append(order)
        return order
    
    def process_orders(
        self,
        current_time: float,
        price_data: Dict[str, PriceData]
    ) -> List[Order]:
        executed = []
        
        while self.pending_orders:
            order = self.pending_orders[0]
            
            delay_seconds = self.config.execution_delay_ms / 1000.0
            if current_time < order.create_time + delay_seconds:
                break
            
            order = self.pending_orders.popleft()
            
            if order.symbol not in price_data:
                order.status = "REJECTED"
                order.reason = "No price data available"
                self.executed_orders.append(order)
                executed.append(order)
                continue
            
            price = price_data[order.symbol]
            
            slippage = (self.config.slippage_bps / 10000.0) * price.price
            commission = (self.config.commission_bps / 10000.0) * price.price
            
            if order.side == "BUY":
                execute_price = price.ask + slippage if price.ask else price.price + slippage
            else:
                execute_price = price.bid - slippage if price.bid else price.price - slippage
            
            order.execute_price = round(execute_price + commission, 2)
            order.execute_time = current_time
            order.status = "EXECUTED"
            
            self.executed_orders.append(order)
            executed.append(order)
        
        return executed


class ProtectedBacktestDataStore:
    def __init__(self, temporal_guard: TemporalDataGuard):
        self.temporal_guard = temporal_guard
        self.sentiment_results: List = []
        self.price_data: List = []
        self.window_aggregates: dict = {}
    
    def add_sentiment(self, result):
        self.sentiment_results.append(result)
    
    def add_price(self, price):
        self.price_data.append(price)
    
    def add_window_aggregate(self, agg):
        if agg.symbol not in self.window_aggregates:
            self.window_aggregates[agg.symbol] = []
        self.window_aggregates[agg.symbol].append(agg)
    
    def get_sentiment_scores(self, symbol: str, limit: int = 100) -> List[float]:
        current_time = self.temporal_guard.current_time
        
        scores = []
        for s in self.sentiment_results:
            if s.symbol == symbol:
                if not self.temporal_guard.check_access(
                    "sentiment", s.timestamp.timestamp(), "read"
                ):
                    continue
                scores.append(s.sentiment_score)
        
        return scores[-limit:]
    
    def get_price_values(self, symbol: str, limit: int = 100) -> List[float]:
        current_time = self.temporal_guard.current_time
        
        prices = []
        for p in self.price_data:
            if p.symbol == symbol:
                if not self.temporal_guard.check_access(
                    "price", p.timestamp.timestamp(), "read"
                ):
                    continue
                prices.append(p.price)
        
        return prices[-limit:]


class BacktestEngine:
    def __init__(self, use_mock_sentiment: bool = True, config: BacktestConfig = None):
        self.data_generator = HistoricalDataGenerator(use_mock_sentiment=use_mock_sentiment)
        self.use_mock_sentiment = use_mock_sentiment
        self.config = config or BacktestConfig()
        self.temporal_guard = TemporalDataGuard(check_level=self.config.lookahead_check_level)
    
    def _process_window_aggregation(
        self,
        symbol: str,
        data_store: ProtectedBacktestDataStore,
        current_time: datetime,
        window_duration: int = 5,
        sentiment_buffer: List = None
    ):
        from shared.models import WindowSentimentAggregate
        from streaming.backpressure_controller import (
            SlidingWindowIncrementalAggregator
        )
        
        if sentiment_buffer is None:
            sentiment_buffer = []
        
        agg = None
        
        window_aggregator = SlidingWindowIncrementalAggregator(
            window_duration_seconds=window_duration,
            slide_interval_seconds=settings.SLIDE_DURATION
        )
        
        for s in sentiment_buffer:
            window_aggregator.add(
                s.symbol,
                s.sentiment_score,
                s.timestamp.timestamp()
            )
        
        agg_data = window_aggregator.emit(symbol, current_time.timestamp())
        
        if agg_data:
            agg = WindowSentimentAggregate(
                symbol=agg_data['symbol'],
                window_start=datetime.fromtimestamp(agg_data['window_start']),
                window_end=datetime.fromtimestamp(agg_data['window_end']),
                avg_sentiment=agg_data['avg_sentiment'],
                news_count=agg_data['news_count'],
                positive_ratio=agg_data['positive_ratio'],
                negative_ratio=agg_data['negative_ratio'],
                sentiment_momentum=agg_data['sentiment_momentum']
            )
            
            data_store.add_window_aggregate(agg)
        
        return agg
    
    def _check_risk_management(
        self,
        position: BacktestPosition,
        current_price: float,
        current_time: float
    ) -> Optional[str]:
        if position.side == "LONG":
            if current_price <= position.stop_loss:
                return "STOP_LOSS"
            if current_price >= position.take_profit:
                return "TAKE_PROFIT"
        else:
            if current_price >= position.stop_loss:
                return "STOP_LOSS"
            if current_price <= position.take_profit:
                return "TAKE_PROFIT"
        
        return None
    
    def run_backtest(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        initial_cash: float = 100000.0
    ) -> BacktestResult:
        if start_date is None:
            start_date = settings.BACKTEST_START_DATE
        if end_date is None:
            end_date = settings.BACKTEST_END_DATE
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        print(f"Generating historical data for {symbol}...")
        news_data = self.data_generator.generate_historical_news(symbol, start_dt, end_dt)
        price_data = self.data_generator.generate_historical_prices(symbol, start_dt, end_dt)
        
        print(f"Generated {len(news_data)} news articles and {len(price_data)} price points")
        
        data_store = ProtectedBacktestDataStore(self.temporal_guard)
        signal_generator = TradingSignalGenerator(data_store)
        execution_sim = OrderExecutionSimulator(self.config)
        
        cash = initial_cash
        positions: Dict[str, BacktestPosition] = {}
        all_signals: List[TradingSignal] = []
        equity_curve: List[float] = [initial_cash]
        returns: List[float] = []
        trades: List[Order] = []
        
        print("Running backtest with lookahead protection...")
        
        news_idx = 0
        price_idx = 0
        current_time = start_dt
        
        sentiment_buffer = []
        last_window_time = start_dt
        
        analyzer = SentimentAnalyzer(use_mock=self.use_mock_sentiment)
        
        current_prices: Dict[str, PriceData] = {}
        
        total_events = len(news_data) + len(price_data)
        event_count = 0
        
        while news_idx < len(news_data) or price_idx < len(price_data):
            next_news_time = news_data[news_idx].timestamp if news_idx < len(news_data) else datetime.max
            next_price_time = price_data[price_idx].timestamp if price_idx < len(price_data) else datetime.max
            
            if next_news_time <= next_price_time:
                news = news_data[news_idx]
                current_time = news.timestamp
                
                self.temporal_guard.set_current_time(current_time.timestamp())
                
                sentiment = analyzer.analyze(news)
                data_store.add_sentiment(sentiment)
                sentiment_buffer.append(sentiment)
                
                if len(sentiment_buffer) > 10000:
                    sentiment_buffer = sentiment_buffer[-10000:]
                
                news_idx += 1
            else:
                price = price_data[price_idx]
                current_time = price.timestamp
                
                self.temporal_guard.set_current_time(current_time.timestamp())
                
                self.temporal_guard.check_access(
                    "price", price.timestamp.timestamp(), "process"
                )
                
                data_store.add_price(price)
                current_prices[symbol] = price
                
                price_idx += 1
            
            event_count += 1
            
            if (current_time - last_window_time).total_seconds() >= settings.SLIDE_DURATION:
                self._process_window_aggregation(
                    symbol, data_store, current_time,
                    settings.WINDOW_DURATION, sentiment_buffer
                )
                
                signal = signal_generator.generate_signal(symbol)
                if signal:
                    all_signals.append(signal)
                    
                    self.temporal_guard.set_current_time(signal.timestamp.timestamp())
                    
                    if len(positions) < self.config.max_positions:
                        current_price_obj = current_prices.get(symbol)
                        if current_price_obj:
                            position_value = cash * self.config.position_size_pct
                            quantity = position_value / current_price_obj.price
                            
                            side = signal.signal
                            
                            if side == "BUY" and symbol not in positions:
                                order = execution_sim.submit_order(
                                    symbol, "BUY", quantity, current_time.timestamp()
                                )
                            elif side == "SELL" and symbol in positions:
                                order = execution_sim.submit_order(
                                    symbol, "SELL", positions[symbol].quantity, current_time.timestamp()
                                )
                
                last_window_time = current_time
            
            executed_orders = execution_sim.process_orders(
                current_time.timestamp(), current_prices
            )
            
            for order in executed_orders:
                if order.status == "EXECUTED":
                    trades.append(order)
                    
                    if order.side == "BUY":
                        cost = order.quantity * order.execute_price
                        cash -= cost
                        
                        stop_loss = order.execute_price * (1 - self.config.stop_loss_pct)
                        take_profit = order.execute_price * (1 + self.config.take_profit_pct)
                        
                        positions[symbol] = BacktestPosition(
                            symbol=symbol,
                            side="LONG",
                            quantity=order.quantity,
                            entry_price=order.execute_price,
                            entry_time=order.execute_time,
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )
                        
                    elif order.side == "SELL" and symbol in positions:
                        revenue = order.quantity * order.execute_price
                        cash += revenue
                        
                        position = positions[symbol]
                        pnl_pct = (order.execute_price - position.entry_price) / position.entry_price
                        returns.append(pnl_pct)
                        position.realized_pnl = (order.execute_price - position.entry_price) * order.quantity
                        
                        del positions[symbol]
            
            for sym, position in list(positions.items()):
                price_obj = current_prices.get(sym)
                if price_obj:
                    position.current_price = price_obj.price
                    if position.side == "LONG":
                        position.unrealized_pnl = (price_obj.price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - price_obj.price) * position.quantity
                    
                    exit_reason = self._check_risk_management(
                        position, price_obj.price, current_time.timestamp()
                    )
                    
                    if exit_reason:
                        order = execution_sim.submit_order(
                            sym, "SELL", position.quantity, current_time.timestamp()
                        )
            
            position_value = sum(
                pos.current_price * pos.quantity for pos in positions.values()
            )
            total_equity = cash + position_value
            
            if len(equity_curve) > 0:
                ret = (total_equity - equity_curve[-1]) / equity_curve[-1] if equity_curve[-1] != 0 else 0
                if abs(ret) > 0.0001:
                    returns.append(ret)
            
            equity_curve.append(total_equity)
            
            if event_count % 1000 == 0:
                print(f"Progress: {event_count}/{total_events} events, Equity: ${total_equity:,.2f}")
        
        for sym, position in list(positions.items()):
            price_obj = current_prices.get(sym)
            if price_obj:
                order = execution_sim.submit_order(
                    sym, "SELL", position.quantity, current_time.timestamp()
                )
                
                executed = execution_sim.process_orders(
                    current_time.timestamp() + 1, current_prices
                )
                
                for order in executed:
                    if order.status == "EXECUTED":
                        revenue = order.quantity * order.execute_price
                        cash += revenue
                        trades.append(order)
        
        total_equity = cash
        equity_curve.append(total_equity)
        
        violations = self.temporal_guard.get_violations()
        if violations:
            print(f"\n⚠️  FOUND {len(violations)} LOOKAHEAD VIOLATIONS:")
            for v in violations[:10]:
                print(f"  - {v}")
        
        total_return = (total_equity - initial_cash) / initial_cash
        sharpe_ratio = calculate_sharpe_ratio(returns, settings.RISK_FREE_RATE)
        max_drawdown = calculate_max_drawdown(equity_curve)
        
        wins = len([r for r in returns if r > 0])
        win_rate = wins / len(returns) if returns else 0.0
        
        gross_profit = sum([r for r in returns if r > 0])
        gross_loss = abs(sum([r for r in returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        print(f"\nExecution Stats:")
        print(f"  Total Orders: {len(execution_sim.executed_orders)}")
        print(f"  Slippage: {self.config.slippage_bps} bps")
        print(f"  Commission: {self.config.commission_bps} bps")
        print(f"  Execution Delay: {self.config.execution_delay_ms} ms")
        
        result = BacktestResult(
            symbol=symbol,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            profit_factor=profit_factor,
            returns=returns,
            signals=all_signals
        )
        
        print("\n" + "="*50)
        print(f"Backtest Results for {symbol}")
        print("="*50)
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Total Trades: {len(trades)}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Lookahead Violations: {len(violations)}")
        print("="*50)
        
        return result
    
    def save_backtest_results(self, result: BacktestResult, output_dir: str = None):
        if output_dir is None:
            output_dir = settings.HISTORICAL_DATA_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        result_df = pd.DataFrame([result.to_dict()])
        result_path = os.path.join(output_dir, f"{result.symbol}_backtest_result.csv")
        result_df.to_csv(result_path, index=False)
        
        if result.signals:
            signals_data = []
            for sig in result.signals:
                signals_data.append({
                    'timestamp': sig.timestamp,
                    'symbol': sig.symbol,
                    'signal': sig.signal,
                    'strength': sig.strength,
                    'sentiment_score': sig.sentiment_score,
                    'price_correlation': sig.price_correlation,
                    'confidence': sig.confidence,
                    'reason': sig.reason
                })
            signals_df = pd.DataFrame(signals_data)
            signals_path = os.path.join(output_dir, f"{result.symbol}_backtest_signals.csv")
            signals_df.to_csv(signals_path, index=False)
        
        print(f"Backtest results saved to {output_dir}")
        return result_path


def run_full_backtest(symbols: List[str] = None):
    if symbols is None:
        symbols = settings.SYMBOLS
    
    config = BacktestConfig(
        execution_delay_ms=100,
        slippage_bps=5.0,
        commission_bps=1.0,
        lookahead_check_level="strict"
    )
    
    engine = BacktestEngine(use_mock_sentiment=True, config=config)
    results = []
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Running backtest for {symbol}")
        print(f"{'='*60}")
        
        result = engine.run_backtest(symbol)
        results.append(result)
        engine.save_backtest_results(result)
    
    summary_df = pd.DataFrame([r.to_dict() for r in results])
    summary_path = os.path.join(settings.HISTORICAL_DATA_DIR, "backtest_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*60)
    print("BACKTEST SUMMARY (WITH LOOKAHEAD PROTECTION)")
    print("="*60)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {summary_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run backtest for trading strategy')
    parser.add_argument('--symbol', type=str, help='Single symbol to backtest')
    parser.add_argument('--all', action='store_true', help='Backtest all symbols')
    parser.add_argument('--start', type=str, default=settings.BACKTEST_START_DATE,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=settings.BACKTEST_END_DATE,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-lookahead-check', action='store_true',
                        help='Disable lookahead bias checking')
    
    args = parser.parse_args()
    
    config = BacktestConfig(
        execution_delay_ms=100,
        slippage_bps=5.0,
        commission_bps=1.0,
        lookahead_check_level="none" if args.no_lookahead_check else "strict"
    )
    
    engine = BacktestEngine(use_mock_sentiment=True, config=config)
    
    if args.all:
        run_full_backtest()
    elif args.symbol:
        result = engine.run_backtest(args.symbol, args.start, args.end)
        engine.save_backtest_results(result)
    else:
        print("Running default backtest for AAPL with lookahead protection...")
        result = engine.run_backtest("AAPL")
        engine.save_backtest_results(result)
