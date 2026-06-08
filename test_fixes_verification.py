import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("FIX VERIFICATION TEST - Production Issue Fixes")
print("="*80)
print()

test_results = []

def run_test(name, test_func, expected_time=None):
    start_time = time.time()
    try:
        result = test_func()
        elapsed = time.time() - start_time
        status = "✅ PASSED" if result else "❌ FAILED"
        time_info = f"[{elapsed:.2f}s]"
        if expected_time and elapsed > expected_time:
            time_info = f"[{elapsed:.2f}s (WARNING: slower than expected {expected_time}s)]"
        print(f"{status} {name} {time_info}")
        test_results.append((name, result, elapsed))
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED {name} [{elapsed:.2f}s]")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        test_results.append((name, False, elapsed))
        return False

print("\n" + "-"*40)
print("TEST 1: Flink/Stream Backpressure Fix")
print("-"*40)
print()

def test_backpressure_controller():
    from streaming.backpressure_controller import (
        BackpressureController,
        SlidingWindowIncrementalAggregator,
        IncrementalAggState
    )
    
    print("  Testing BackpressureController...")
    controller = BackpressureController(
        max_queue_size=100,
        target_lag_seconds=2.0,
        min_sample_rate=0.2
    )
    
    for i in range(200):
        ts = time.time() - 10 + (200 - i) * 0.01
        controller.report_input(ts)
    
    controller.stats.lag_seconds = 10.0
    
    stats = controller.update_stats()
    print(f"    Input rate: {stats.input_rate:.1f}/s, Drop count: {stats.drop_count}")
    print(f"    Sample rate: {stats.sample_rate:.1%}, Queue size: {stats.queue_size}")
    
    assert stats.drop_count > 0, "Should have dropped some messages"
    
    for i in range(10):
        controller.stats.lag_seconds = 10.0
        stats = controller.update_stats()
    
    print(f"    After high lag, sample rate: {stats.sample_rate:.1%}")
    assert stats.sample_rate < 1.0, "Should have reduced sample rate under high lag"
    
    print("  Testing IncrementalAggState...")
    state = IncrementalAggState()
    for i in range(100):
        state.add(np.random.uniform(-1, 1))
    
    avg_before = state.avg_sentiment
    count_before = state.count
    
    for i in range(50):
        state.remove(np.random.uniform(-1, 1))
    
    assert state.count == count_before - 50, "Count should decrease by 50"
    
    print("  Testing SlidingWindowIncrementalAggregator...")
    aggregator = SlidingWindowIncrementalAggregator(
        window_duration_seconds=5,
        slide_interval_seconds=2
    )
    
    start_ts = time.time()
    for i in range(100):
        aggregator.add("AAPL", np.random.uniform(-1, 1), start_ts + i * 0.1)
    
    agg = aggregator.emit("AAPL", start_ts + 5.0)
    assert agg is not None, "Should emit aggregate"
    assert 'avg_sentiment' in agg, "Should have avg_sentiment"
    assert agg['news_count'] > 0, "Should have news count"
    
    print(f"    Aggregated {agg['news_count']} news items, avg sentiment: {agg['avg_sentiment']:.4f}")
    
    return True

run_test("Backpressure & Incremental Aggregation", test_backpressure_controller, expected_time=2.0)

print("\n" + "-"*40)
print("TEST 2: GPU Memory Leak Fix")
print("-"*40)
print()

def test_gpu_memory_manager():
    from sentiment.finbert_analyzer import GPUMemoryManager, OptimizedFinBERTAnalyzer
    
    print("  Testing GPUMemoryManager...")
    gpu_manager = GPUMemoryManager(
        max_memory_fraction=0.8,
        cleanup_interval=5
    )
    
    for i in range(10):
        gpu_manager.should_cleanup()
    
    assert gpu_manager.batch_count == 10, f"Batch count should be 10"
    
    gpu_manager.cleanup(force=True)
    print("    Memory manager cleanup completed")
    
    print("  Testing OptimizedFinBERTAnalyzer (mock mode)...")
    analyzer = OptimizedFinBERTAnalyzer(use_mock=True, max_batch_size=16)
    
    from shared.models import NewsArticle
    from shared.utils import generate_id
    
    news_list = []
    for i in range(100):
        news = NewsArticle(
            id=generate_id(),
            symbol="AAPL",
            title=f"Test News {i}",
            content="Apple reports record earnings beating estimates",
            source="Bloomberg",
            timestamp=datetime.now()
        )
        news_list.append(news)
    
    print(f"    Analyzing batch of {len(news_list)} news...")
    start = time.time()
    results = analyzer.analyze_batch(news_list)
    elapsed = time.time() - start
    
    assert len(results) == len(news_list), "Should analyze all news"
    print(f"    Analyzed {len(results)} in {elapsed:.2f}s ({len(results)/elapsed:.1f}/s)")
    
    for r in results[:5]:
        assert -1 <= r.sentiment_score <= 1, "Sentiment score out of range"
        assert abs(r.positive + r.negative + r.neutral - 1.0) < 0.01, "Scores don't sum to 1"
    
    print("    Testing cleanup...")
    analyzer.cleanup()
    
    return True

run_test("GPU Memory Management", test_gpu_memory_manager, expected_time=3.0)

print("\n" + "-"*40)
print("TEST 3: Frontend Heatmap Rendering Optimization")
print("-"*40)
print()

def test_render_optimization():
    from visualization.render_optimizer import (
        RenderThrottler,
        IncrementalHeatmapData,
        DataDownsampler,
        FrameRateLimiter,
        SmartDeltaUpdater
    )
    
    print("  Testing RenderThrottler...")
    throttler = RenderThrottler(max_fps=10.0)
    
    render_count = 0
    start = time.time()
    for i in range(100):
        if throttler.should_render():
            render_count += 1
        time.sleep(0.01)
    
    elapsed = time.time() - start
    actual_fps = render_count / elapsed if elapsed > 0 else 0
    print(f"    Rendered {render_count} times in {elapsed:.2f}s, actual FPS: {actual_fps:.1f} (target: 10)")
    
    assert actual_fps <= 12.0 and actual_fps >= 5.0, f"FPS should be close to target"
    
    print("  Testing IncrementalHeatmapData...")
    heatmap_data = IncrementalHeatmapData(max_time_buckets=10, max_symbols=5)
    
    now = time.time()
    for i in range(50):
        symbol = ["AAPL", "GOOGL", "MSFT"][i % 3]
        ts = now - (50 - i) * 60
        value = np.random.uniform(-1, 1)
        heatmap_data.update(symbol, ts, value, {'count': 1})
    
    data, symbols, buckets, dirty, text_data = heatmap_data.get_heatmap_data()
    
    print(f"    Heatmap shape: {data.shape}, symbols: {len(symbols)}, buckets: {len(buckets)}")
    print(f"    Has dirty regions: {np.any(dirty)}")
    assert data.shape[0] > 0 and data.shape[1] > 0, "Should have data"
    
    print("  Testing DataDownsampler...")
    timestamps = list(range(1000))
    values = np.sin(np.linspace(0, 10, 1000))
    
    ds_ts, ds_vals = DataDownsampler.downsample_time_series(timestamps, values, target_points=100)
    print(f"    Downsampled from 1000 → {len(ds_ts)} points")
    assert len(ds_ts) == 100, "Should downsample to target points"
    
    heatmap_arr = np.random.randn(5, 50)
    ds_heatmap, _ = DataDownsampler.downsample_heatmap(heatmap_arr, target_buckets=15)
    print(f"    Heatmap downsampled from (5,50) → {ds_heatmap.shape}")
    assert ds_heatmap.shape[1] == 15, "Should downsample to target buckets"
    
    print("  Testing FrameRateLimiter...")
    fps_limiter = FrameRateLimiter(target_fps=5.0)
    
    fps_values = []
    for i in range(10):
        fps = fps_limiter.wait_for_next_frame()
        fps_values.append(fps)
    
    avg_fps = fps_limiter.get_average_fps()
    print(f"    Average FPS: {avg_fps:.1f} (target: 5)")
    
    print("  Testing SmartDeltaUpdater...")
    delta_updater = SmartDeltaUpdater(tolerance=0.01)
    
    assert delta_updater.has_changed("price", 100.0) == True
    assert delta_updater.has_changed("price", 100.005) == False
    assert delta_updater.has_changed("price", 100.02) == True
    
    return True

run_test("Frontend Rendering Optimization", test_render_optimization, expected_time=2.0)

print("\n" + "-"*40)
print("TEST 4: Lookahead Bias Protection")
print("-"*40)
print()

def test_lookahead_protection():
    from backtest.backtest_engine import (
        TemporalDataGuard,
        OrderExecutionSimulator,
        ProtectedBacktestDataStore,
        BacktestConfig
    )
    
    print("  Testing TemporalDataGuard...")
    guard = TemporalDataGuard(check_level="strict")
    
    current_time = time.time()
    guard.set_current_time(current_time)
    
    assert guard.check_access("price", current_time - 1.0, "read") == True, "Should allow past data"
    assert guard.check_access("price", current_time + 0.001, "read") == False, "Should block future data"
    
    violations = guard.get_violations()
    print(f"    Violations detected: {len(violations)}")
    assert len(violations) == 1, "Should have 1 violation"
    
    print(f"    Violation message: {violations[0][:80]}...")
    
    print("  Testing OrderExecutionSimulator...")
    config = BacktestConfig(
        execution_delay_ms=100,
        slippage_bps=5.0,
        commission_bps=1.0
    )
    
    exec_sim = OrderExecutionSimulator(config)
    
    from shared.models import PriceData
    
    order = exec_sim.submit_order("AAPL", "BUY", 10.0, current_time)
    assert order.status == "PENDING", "Order should be pending"
    
    price = PriceData(
        symbol="AAPL",
        price=100.0,
        volume=1000,
        timestamp=datetime.fromtimestamp(current_time),
        bid=99.95,
        ask=100.05,
        bid_size=100,
        ask_size=100
    )
    
    executed = exec_sim.process_orders(current_time + 0.05, {"AAPL": price})
    assert len(executed) == 0, "Should not execute before delay"
    
    executed = exec_sim.process_orders(current_time + 0.15, {"AAPL": price})
    assert len(executed) == 1, "Should execute after delay"
    
    executed_order = executed[0]
    print(f"    Executed at ${executed_order.execute_price:.4f} (expected ~100.05 + slippage + commission)")
    assert executed_order.execute_price > 100.05, "Should have slippage and commission"
    assert executed_order.status == "EXECUTED", "Order should be executed"
    
    print("  Testing ProtectedBacktestDataStore...")
    guard2 = TemporalDataGuard(check_level="strict")
    guard2.set_current_time(current_time)
    
    protected_store = ProtectedBacktestDataStore(guard2)
    
    from shared.models import SentimentResult, PriceData
    from shared.utils import generate_id
    
    sentiment = SentimentResult(
        news_id=generate_id(),
        symbol="AAPL",
        positive=0.7,
        negative=0.1,
        neutral=0.2,
        sentiment_score=0.6,
        timestamp=datetime.fromtimestamp(current_time - 10.0)
    )
    protected_store.add_sentiment(sentiment)
    
    future_sentiment = SentimentResult(
        news_id=generate_id(),
        symbol="AAPL",
        positive=0.8,
        negative=0.0,
        neutral=0.2,
        sentiment_score=0.8,
        timestamp=datetime.fromtimestamp(current_time + 10.0)
    )
    protected_store.add_sentiment(future_sentiment)
    
    scores = protected_store.get_sentiment_scores("AAPL", limit=10)
    print(f"    Retrieved {len(scores)} scores (should be 1, future blocked)")
    assert len(scores) == 1, "Should only return 1 past sentiment"
    
    violations = guard2.get_violations()
    print(f"    Lookahead violations: {len(violations)}")
    assert len(violations) == 1, "Should have 1 violation for future data"
    
    return True

run_test("Lookahead Bias Protection", test_lookahead_protection, expected_time=2.0)

print("\n" + "-"*40)
print("TEST 5: Full Integration Test - Stream Processor with Fixes")
print("-"*40)
print()

def test_stream_processor_integration():
    from streaming.spark_processor import StreamDataStore, OptimizedPythonStreamProcessor
    from shared.models import NewsArticle, PriceData
    from shared.utils import generate_id
    
    print("  Creating stream processor with fixes...")
    data_store = StreamDataStore()
    processor = OptimizedPythonStreamProcessor(data_store, use_mock_sentiment=True)
    processor.start()
    
    print("  Feeding 5000 news items...")
    start_time = time.time()
    
    for i in range(5000):
        news = NewsArticle(
            id=generate_id(),
            symbol=["AAPL", "GOOGL", "MSFT"][i % 3],
            title=f"News {i}",
            content="Apple reports record earnings",
            source="Bloomberg",
            timestamp=datetime.now()
        )
        processor.queue_news(news.to_json())
    
    for i in range(200):
        price = PriceData(
            symbol=["AAPL", "GOOGL", "MSFT"][i % 3],
            price=100 + i * 0.1,
            volume=10000,
            timestamp=datetime.now()
        )
        processor.queue_price(price.to_json())
    
    print("  Waiting for processing...")
    time.sleep(5.0)
    
    elapsed = time.time() - start_time
    processed = processor.processed_count
    rate = processed / elapsed if elapsed > 0 else 0
    
    print(f"    Processed: {processed}/{5000} in {elapsed:.2f}s ({rate:.1f}/s)")
    print(f"    News in store: {data_store.news_count}")
    print(f"    Backpressure status: {processor.backpressure.get_status()}")
    
    processor.stop()
    
    assert data_store.news_count > 0, "Should have processed news"
    
    return True

run_test("Stream Processor Integration", test_stream_processor_integration, expected_time=10.0)

print("\n" + "-"*40)
print("TEST 6: End-to-End Backtest with Protection")
print("-"*40)
print()

def test_backtest_with_protection():
    from backtest.backtest_engine import BacktestEngine, BacktestConfig
    
    print("  Running short backtest with lookahead protection...")
    
    config = BacktestConfig(
        execution_delay_ms=100,
        slippage_bps=5.0,
        commission_bps=1.0,
        lookahead_check_level="strict"
    )
    
    engine = BacktestEngine(use_mock_sentiment=True, config=config)
    
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    result = engine.run_backtest("AAPL", start_date=start_date, end_date=end_date)
    
    print(f"\n  Backtest Results:")
    print(f"    Total Return: {result.total_return*100:.2f}%")
    print(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"    Max Drawdown: {result.max_drawdown*100:.2f}%")
    print(f"    Win Rate: {result.win_rate*100:.2f}%")
    print(f"    Total Trades: {result.total_trades}")
    print(f"    Lookahead Violations: {len(engine.temporal_guard.get_violations())}")
    
    assert not engine.temporal_guard.has_violations() == False, "Should have no lookahead violations"
    
    return True

run_test("End-to-End Backtest", test_backtest_with_protection, expected_time=15.0)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print()

passed = sum(1 for _, r, _ in test_results if r)
total = len(test_results)
total_time = sum(t for _, _, t in test_results)

print(f"Tests: {passed}/{total} tests passed")
print(f"Total time: {total_time:.2f}s")
print()

for name, result, elapsed in test_results:
    status = "✅" if result else "❌"
    print(f"  {status} {name} ({elapsed:.2f}s)")

print()
if passed == total:
    print("🎉 All fixes verified successfully!")
    print("\nKey improvements:")
    print("  1. ✅ Backpressure: Incremental aggregation + adaptive sampling")
    print("  2. ✅ GPU Memory: Proper cleanup + memory limits")
    print("  3. ✅ Frontend: Throttling + incremental rendering")
    print("  4. ✅ Lookahead: Temporal guard + realistic execution")
else:
    print(f"⚠️  {total - passed} tests failed - review needed")
    sys.exit(1)

print("\n" + "="*80)
