import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("System Validation Test for Financial News Sentiment Trading System")
print("="*70)
print()

test_passed = 0
test_total = 0

def run_test(name, test_func):
    global test_passed, test_total
    test_total += 1
    try:
        result = test_func()
        if result:
            print(f"✅ {name} - PASSED")
            test_passed += 1
        else:
            print(f"❌ {name} - FAILED")
    except Exception as e:
        print(f"❌ {name} - FAILED with error: {e}")
        import traceback
        traceback.print_exc()
    print()

def test_config():
    from config.settings import settings
    assert len(settings.SYMBOLS) > 0, "No symbols configured"
    assert settings.WINDOW_DURATION > 0, "Invalid window duration"
    assert settings.NEWS_GENERATION_RATE >= 1000, f"News rate below 1000/s: {settings.NEWS_GENERATION_RATE}"
    return True

def test_models():
    from shared.models import NewsArticle, SentimentResult, TradingSignal
    from shared.utils import generate_id, generate_timestamp
    
    news = NewsArticle(
        id=generate_id(),
        symbol="AAPL",
        title="Test News",
        content="Apple reports record earnings",
        source="Bloomberg",
        timestamp=generate_timestamp()
    )
    
    json_str = news.to_json()
    news2 = NewsArticle.from_json(json_str)
    assert news.id == news2.id, "News serialization failed"
    assert news.symbol == news2.symbol, "News symbol mismatch"
    
    sentiment = SentimentResult(
        news_id=news.id,
        symbol="AAPL",
        positive=0.8,
        negative=0.1,
        neutral=0.1,
        sentiment_score=0.7,
        timestamp=generate_timestamp()
    )
    
    signal = TradingSignal(
        symbol="AAPL",
        signal="BUY",
        strength=0.85,
        sentiment_score=0.7,
        price_correlation=0.65,
        timestamp=generate_timestamp(),
        confidence=0.8,
        reason="Test signal"
    )
    
    return True

def test_utils():
    from shared.utils import (
        calculate_sharpe_ratio, calculate_max_drawdown,
        calculate_rolling_correlation, generate_price_series
    )
    import numpy as np
    
    returns = [0.01, -0.02, 0.015, 0.005, -0.01]
    sharpe = calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float), "Sharpe ratio should be float"
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    
    equity = [100, 110, 95, 120, 105, 115]
    max_dd = calculate_max_drawdown(equity)
    assert 0 <= max_dd <= 1, f"Invalid max drawdown: {max_dd}"
    print(f"  Max Drawdown: {max_dd:.4f}")
    
    prices = generate_price_series(100, 10)
    assert len(prices) == 10, "Price series length mismatch"
    assert all(p > 0 for p in prices), "Negative prices generated"
    print(f"  Generated {len(prices)} prices")
    
    sentiment = [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9]
    price_vals = [100, 101, 99, 102, 98, 103, 97, 104]
    corr = calculate_rolling_correlation(sentiment, price_vals, window=5)
    assert -1 <= corr <= 1, f"Invalid correlation: {corr}"
    print(f"  Correlation: {corr:.4f}")
    
    return True

def test_sentiment_analyzer():
    from sentiment.finbert_analyzer import FinBERTAnalyzer
    from shared.models import NewsArticle
    from shared.utils import generate_id, generate_timestamp
    
    analyzer = FinBERTAnalyzer(use_mock=True)
    
    news = NewsArticle(
        id=generate_id(),
        symbol="AAPL",
        title="Test",
        content="Apple reports record quarterly earnings beating estimates",
        source="Bloomberg",
        timestamp=generate_timestamp()
    )
    
    result = analyzer.analyze(news)
    assert result.symbol == "AAPL", "Symbol mismatch"
    assert -1 <= result.sentiment_score <= 1, f"Invalid sentiment score: {result.sentiment_score}"
    assert abs(result.positive + result.negative + result.neutral - 1.0) < 0.01, "Scores don't sum to 1"
    
    print(f"  Sentiment Score: {result.sentiment_score:.4f}")
    print(f"  Positive: {result.positive:.4f}, Negative: {result.negative:.4f}, Neutral: {result.neutral:.4f}")
    
    news2 = NewsArticle(
        id=generate_id(),
        symbol="MSFT",
        title="Test 2",
        content="Microsoft misses earnings expectations amid restructuring",
        source="Reuters",
        timestamp=generate_timestamp()
    )
    
    results = analyzer.analyze_batch([news, news2])
    assert len(results) == 2, "Batch analysis failed"
    print(f"  Batch analysis: {len(results)} results")
    
    return True

def test_data_store():
    from streaming.spark_processor import StreamDataStore
    from shared.models import SentimentResult, PriceData
    from shared.utils import generate_id, generate_timestamp
    
    store = StreamDataStore()
    
    for i in range(10):
        sentiment = SentimentResult(
            news_id=generate_id(),
            symbol="AAPL",
            positive=0.7,
            negative=0.1,
            neutral=0.2,
            sentiment_score=0.6,
            timestamp=generate_timestamp()
        )
        store.add_sentiment(sentiment)
        
        price = PriceData(
            symbol="AAPL",
            price=190.0 + i,
            volume=100000,
            timestamp=generate_timestamp()
        )
        store.add_price(price)
    
    assert store.news_count == 10, f"News count mismatch: {store.news_count}"
    scores = store.get_sentiment_scores("AAPL", limit=5)
    assert len(scores) == 5, f"Sentiment scores length mismatch: {len(scores)}"
    
    prices = store.get_price_values("AAPL", limit=5)
    assert len(prices) == 5, f"Prices length mismatch: {len(prices)}"
    
    print(f"  Stored {store.news_count} news items")
    print(f"  Sentiment scores: {len(scores)}, Prices: {len(prices)}")
    
    return True

def test_signal_generator():
    from streaming.spark_processor import StreamDataStore
    from signal.signal_generator import TradingSignalGenerator, Portfolio
    from shared.models import (
        SentimentResult, PriceData, WindowSentimentAggregate
    )
    from shared.utils import generate_id, generate_timestamp
    from datetime import timedelta
    import numpy as np
    
    store = StreamDataStore()
    
    base_time = generate_timestamp()
    prices = []
    sentiments = []
    
    for i in range(50):
        timestamp = base_time - timedelta(seconds=50 - i)
        
        sentiment_val = np.random.uniform(-1, 1)
        sentiment = SentimentResult(
            news_id=generate_id(),
            symbol="AAPL",
            positive=max(0, sentiment_val),
            negative=max(0, -sentiment_val),
            neutral=0.2,
            sentiment_score=sentiment_val,
            timestamp=timestamp
        )
        store.add_sentiment(sentiment)
        sentiments.append(sentiment_val)
        
        price_val = 190.0 + i * 0.1 + sentiment_val * 0.5
        price = PriceData(
            symbol="AAPL",
            price=price_val,
            volume=100000,
            timestamp=timestamp
        )
        store.add_price(price)
        prices.append(price_val)
    
    from shared.models import WindowSentimentAggregate
    agg = WindowSentimentAggregate(
        symbol="AAPL",
        window_start=base_time - timedelta(seconds=5),
        window_end=base_time,
        avg_sentiment=0.75,
        news_count=10,
        positive_ratio=0.8,
        negative_ratio=0.1,
        sentiment_momentum=0.15
    )
    store.add_window_aggregate(agg)
    
    generator = TradingSignalGenerator(store)
    signal = generator.generate_signal("AAPL")
    
    if signal:
        print(f"  Generated {signal.signal} signal with strength {signal.strength:.4f}")
        print(f"  Confidence: {signal.confidence:.4f}, Correlation: {signal.price_correlation:.4f}")
    else:
        print(f"  No strong signal generated (expected for random data)")
    
    portfolio = Portfolio(initial_cash=100000.0)
    print(f"  Initial portfolio value: ${portfolio.total_equity():,.2f}")
    
    return True

def test_backtest_engine():
    from backtest.backtest_engine import BacktestEngine
    from datetime import datetime
    
    engine = BacktestEngine(use_mock_sentiment=True)
    
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 10)
    
    news = engine.data_generator.generate_historical_news("AAPL", start, end, news_per_day=10)
    prices = engine.data_generator.generate_historical_prices("AAPL", start, end)
    
    assert len(news) > 0, "No historical news generated"
    assert len(prices) > 0, "No historical prices generated"
    
    print(f"  Generated {len(news)} news articles")
    print(f"  Generated {len(prices)} price points")
    
    return True

def test_mock_data_generator():
    from data_source.websocket_server import MockDataGenerator
    from config.settings import settings
    
    generator = MockDataGenerator()
    
    news = generator.generate_news("AAPL")
    assert news.symbol == "AAPL", "News symbol mismatch"
    assert len(news.content) > 0, "Empty news content"
    
    price = generator.generate_price("AAPL")
    assert price.price > 0, "Invalid price"
    
    order_book = generator.generate_order_book("AAPL")
    assert len(order_book.bids) == 10, "Incorrect bid levels"
    assert len(order_book.asks) == 10, "Incorrect ask levels"
    
    print(f"  News: {news.title[:50]}...")
    print(f"  Price: ${price.price:.2f}")
    print(f"  Order Book: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
    
    return True

def test_python_stream_processor():
    from streaming.spark_processor import StreamDataStore, PythonStreamProcessor
    import time
    
    store = StreamDataStore()
    processor = PythonStreamProcessor(store, use_mock_sentiment=True)
    processor.start()
    
    from shared.models import NewsArticle
    from shared.utils import generate_id, generate_timestamp
    
    for i in range(100):
        news = NewsArticle(
            id=generate_id(),
            symbol="AAPL",
            title=f"Test News {i}",
            content="Apple reports record earnings beating estimates",
            source="Bloomberg",
            timestamp=generate_timestamp()
        )
        processor.queue_news(news.to_json())
    
    from shared.models import PriceData
    for i in range(20):
        price = PriceData(
            symbol="AAPL",
            price=190.0 + i * 0.1,
            volume=100000,
            timestamp=generate_timestamp()
        )
        processor.queue_price(price.to_json())
    
    time.sleep(2)
    
    print(f"  Processed {processor.processed_count} news articles")
    print(f"  Stored {store.news_count} sentiment results")
    
    processor.stop()
    
    return store.news_count > 0

def main():
    print("Running validation tests...\n")
    
    run_test("1. Configuration Settings", test_config)
    run_test("2. Data Models (Serialization)", test_models)
    run_test("3. Utility Functions", test_utils)
    run_test("4. Mock Data Generator", test_mock_data_generator)
    run_test("5. Sentiment Analyzer (Mock)", test_sentiment_analyzer)
    run_test("6. Stream Data Store", test_data_store)
    run_test("7. Signal Generator", test_signal_generator)
    run_test("8. Python Stream Processor", test_python_stream_processor)
    run_test("9. Backtest Engine (Data Generation)", test_backtest_engine)
    
    print("="*70)
    print(f"Test Results: {test_passed}/{test_total} tests passed")
    print("="*70)
    
    if test_passed == test_total:
        print("\n🎉 All tests passed! System is ready.")
        print("\nQuick Start Guide:")
        print("  1. Run full system: python start_all.py")
        print("  2. Run backtest: python backtest/backtest_engine.py --symbol AAPL")
        print("  3. Access dashboard at http://localhost:8501")
        return 0
    else:
        print(f"\n⚠️  {test_total - test_passed} tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
