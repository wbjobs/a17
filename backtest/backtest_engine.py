import os
import sys
import random
from datetime import datetime, timedelta
from typing import List, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from shared.models import NewsArticle, PriceData, TradingSignal, BacktestResult
from sentiment.finbert_analyzer import SentimentAnalyzer
from signal.signal_generator import TradingSignalGenerator, Portfolio
from shared.utils import (
    calculate_sharpe_ratio, calculate_max_drawdown,
    generate_price_series, generate_id
)


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
                timestamp = current_date.replace(hour=hour, minute=minute)
                
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
                variation = np.random.normal(0, 0.005)
                intraday_price = price * (1 + variation)
                timestamp = current_date.replace(hour=hour, minute=0)
                
                price_data.append(PriceData(
                    symbol=symbol,
                    price=round(intraday_price, 2),
                    volume=random.uniform(10000, 1000000),
                    timestamp=timestamp,
                    bid=round(intraday_price * 0.9995, 2),
                    ask=round(intraday_price * 1.0005, 2),
                    bid_size=random.uniform(100, 5000),
                    ask_size=random.uniform(100, 5000)
                ))
            
            current_date += timedelta(days=1)
        
        return price_data


class BacktestDataStore:
    def __init__(self):
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
        scores = [s.sentiment_score for s in self.sentiment_results if s.symbol == symbol]
        return scores[-limit:]
    
    def get_price_values(self, symbol: str, limit: int = 100) -> List[float]:
        prices = [p.price for p in self.price_data if p.symbol == symbol]
        return prices[-limit:]


class BacktestEngine:
    def __init__(self, use_mock_sentiment: bool = True):
        self.data_generator = HistoricalDataGenerator(use_mock_sentiment=use_mock_sentiment)
        self.use_mock_sentiment = use_mock_sentiment
    
    def _process_window_aggregation(
        self,
        symbol: str,
        data_store: BacktestDataStore,
        current_time: datetime,
        window_duration: int = 5,
        sentiment_buffer: List = None
    ):
        from shared.models import WindowSentimentAggregate
        
        if sentiment_buffer is None:
            sentiment_buffer = []
        
        window_start = current_time - timedelta(seconds=window_duration)
        
        window_results = [
            s for s in sentiment_buffer
            if window_start <= s.timestamp <= current_time
        ]
        
        if len(window_results) > 0:
            scores = [s.sentiment_score for s in window_results]
            avg_sentiment = float(np.mean(scores))
            positive_ratio = len([s for s in window_results if s.sentiment_score > 0.1]) / len(window_results)
            negative_ratio = len([s for s in window_results if s.sentiment_score < -0.1]) / len(window_results)
            
            prev_scores = data_store.get_sentiment_scores(symbol, limit=20)
            if len(prev_scores) > 10:
                sentiment_momentum = avg_sentiment - float(np.mean(prev_scores[:10]))
            else:
                sentiment_momentum = 0.0
            
            agg = WindowSentimentAggregate(
                symbol=symbol,
                window_start=window_start,
                window_end=current_time,
                avg_sentiment=avg_sentiment,
                news_count=len(window_results),
                positive_ratio=positive_ratio,
                negative_ratio=negative_ratio,
                sentiment_momentum=sentiment_momentum
            )
            
            data_store.add_window_aggregate(agg)
            return agg
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
        
        data_store = BacktestDataStore()
        signal_generator = TradingSignalGenerator(data_store)
        portfolio = Portfolio(initial_cash=initial_cash)
        
        print("Running sentiment analysis...")
        sentiment_buffer = []
        all_signals = []
        
        news_idx = 0
        price_idx = 0
        
        window_interval = timedelta(seconds=settings.SLIDE_DURATION)
        next_window_time = start_dt + window_interval
        
        analyzer = SentimentAnalyzer(use_mock=self.use_mock_sentiment)
        
        print("Processing historical data...")
        current_time = start_dt
        total_steps = len(news_data) + len(price_data)
        step = 0
        
        while news_idx < len(news_data) or price_idx < len(price_data):
            next_news_time = news_data[news_idx].timestamp if news_idx < len(news_data) else datetime.max
            next_price_time = price_data[price_idx].timestamp if price_idx < len(price_data) else datetime.max
            
            if next_news_time <= next_price_time:
                news = news_data[news_idx]
                current_time = news.timestamp
                
                sentiment = analyzer.analyze(news)
                data_store.add_sentiment(sentiment)
                sentiment_buffer.append(sentiment)
                
                if len(sentiment_buffer) > 1000:
                    sentiment_buffer = sentiment_buffer[-1000:]
                
                news_idx += 1
            else:
                price = price_data[price_idx]
                current_time = price.timestamp
                data_store.add_price(price)
                
                prices_dict = {price.symbol: price.price}
                portfolio.update_prices(prices_dict)
                
                price_idx += 1
            
            while current_time >= next_window_time:
                agg = self._process_window_aggregation(
                    symbol, data_store, next_window_time,
                    settings.WINDOW_DURATION, sentiment_buffer
                )
                
                if agg:
                    signal = signal_generator.generate_signal(symbol)
                    if signal:
                        all_signals.append(signal)
                        recent_prices = [p for p in data_store.price_data 
                                        if p.symbol == symbol and p.timestamp <= signal.timestamp]
                        if recent_prices:
                            current_price = recent_prices[-1].price
                            portfolio.execute_signal(signal, current_price)
                
                next_window_time += window_interval
            
            step += 1
            if step % 1000 == 0:
                print(f"Progress: {step}/{total_steps} steps processed")
        
        for pos in list(portfolio.positions.values()):
            recent_prices = [p for p in data_store.price_data if p.symbol == symbol]
            if recent_prices:
                final_price = recent_prices[-1].price
                signal = TradingSignal(
                    symbol=symbol,
                    signal="SELL",
                    strength=1.0,
                    sentiment_score=0.0,
                    price_correlation=0.0,
                    timestamp=end_dt,
                    confidence=1.0,
                    reason="Backtest closeout"
                )
                portfolio.execute_signal(signal, final_price)
        
        total_return = (portfolio.total_equity() - initial_cash) / initial_cash
        sharpe_ratio = calculate_sharpe_ratio(portfolio.returns, settings.RISK_FREE_RATE)
        max_drawdown = calculate_max_drawdown(portfolio.equity_curve)
        
        wins = len([r for r in portfolio.returns if r > 0])
        win_rate = wins / len(portfolio.returns) if portfolio.returns else 0.0
        
        gross_profit = sum([r for r in portfolio.returns if r > 0])
        gross_loss = abs(sum([r for r in portfolio.returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        result = BacktestResult(
            symbol=symbol,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(portfolio.trades),
            profit_factor=profit_factor,
            returns=portfolio.returns,
            signals=all_signals
        )
        
        print("\n" + "="*50)
        print(f"Backtest Results for {symbol}")
        print("="*50)
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Total Trades: {len(portfolio.trades)}")
        print(f"Profit Factor: {profit_factor:.2f}")
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
    
    engine = BacktestEngine(use_mock_sentiment=True)
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
    print("BACKTEST SUMMARY")
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
    
    args = parser.parse_args()
    
    engine = BacktestEngine(use_mock_sentiment=True)
    
    if args.all:
        run_full_backtest()
    elif args.symbol:
        result = engine.run_backtest(args.symbol, args.start, args.end)
        engine.save_backtest_results(result)
    else:
        print("Running default backtest for AAPL...")
        result = engine.run_backtest("AAPL")
        engine.save_backtest_results(result)
