import os
import sys
import json
import time
import threading
import queue
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Deque
from collections import deque, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from shared.models import (
    NewsArticle, SentimentResult, PriceData,
    WindowSentimentAggregate, TradingSignal, OrderBook
)
from sentiment.finbert_analyzer import SentimentAnalyzer
from shared.utils import calculate_rolling_correlation
from streaming.backpressure_controller import (
    BackpressureController, SlidingWindowIncrementalAggregator
)


class StreamDataStore:
    def __init__(self):
        self.sentiment_results: Deque[SentimentResult] = deque(maxlen=10000)
        self.price_data: Dict[str, Deque[PriceData]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.window_aggregates: Dict[str, Deque[WindowSentimentAggregate]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.trading_signals: Deque[TradingSignal] = deque(maxlen=1000)
        self.order_books: Dict[str, OrderBook] = {}
        self.news_count = 0
        self._lock = threading.Lock()
    
    def add_sentiment(self, result: SentimentResult):
        with self._lock:
            self.sentiment_results.append(result)
            self.news_count += 1
    
    def add_price(self, price: PriceData):
        with self._lock:
            self.price_data[price.symbol].append(price)
    
    def add_order_book(self, order_book: OrderBook):
        with self._lock:
            self.order_books[order_book.symbol] = order_book
    
    def add_window_aggregate(self, agg: WindowSentimentAggregate):
        with self._lock:
            self.window_aggregates[agg.symbol].append(agg)
    
    def add_trading_signal(self, signal: TradingSignal):
        with self._lock:
            self.trading_signals.append(signal)
    
    def get_recent_sentiment(self, symbol: str, limit: int = 100) -> List[SentimentResult]:
        with self._lock:
            return [s for s in list(self.sentiment_results) if s.symbol == symbol][-limit:]
    
    def get_recent_prices(self, symbol: str, limit: int = 100) -> List[PriceData]:
        with self._lock:
            return list(self.price_data.get(symbol, deque()))[-limit:]
    
    def get_order_book(self, symbol: str) -> OrderBook:
        with self._lock:
            return self.order_books.get(symbol)
    
    def get_all_window_aggregates(self) -> List[WindowSentimentAggregate]:
        with self._lock:
            all_aggs = []
            for symbol_aggs in self.window_aggregates.values():
                all_aggs.extend(list(symbol_aggs))
            return sorted(all_aggs, key=lambda x: x.window_end)
    
    def get_all_signals(self, limit: int = 100) -> List[TradingSignal]:
        with self._lock:
            return list(self.trading_signals)[-limit:]
    
    def get_sentiment_scores(self, symbol: str, limit: int = 100) -> List[float]:
        recent = self.get_recent_sentiment(symbol, limit)
        return [s.sentiment_score for s in recent]
    
    def get_price_values(self, symbol: str, limit: int = 100) -> List[float]:
        recent = self.get_recent_prices(symbol, limit)
        return [p.price for p in recent]


class OptimizedPythonStreamProcessor:
    def __init__(self, data_store: StreamDataStore, use_mock_sentiment: bool = True):
        self.data_store = data_store
        self.analyzer = SentimentAnalyzer(use_mock=use_mock_sentiment)
        self.running = False
        
        self.news_queue: queue.Queue = queue.Queue(maxsize=20000)
        self.price_queue: queue.Queue = queue.Queue(maxsize=5000)
        self.orderbook_queue: queue.Queue = queue.Queue(maxsize=1000)
        
        self.processed_count = 0
        
        self.backpressure = BackpressureController(
            max_queue_size=10000,
            target_lag_seconds=5.0,
            min_sample_rate=0.1,
            max_sample_rate=1.0
        )
        
        self.window_aggregator = SlidingWindowIncrementalAggregator(
            window_duration_seconds=settings.WINDOW_DURATION,
            slide_interval_seconds=settings.SLIDE_DURATION
        )
        
        self._sentiment_buffer: Dict[str, List[SentimentResult]] = defaultdict(list)
        self._last_stats_print = time.time()
        self._gc_interval = 300
        self._last_gc = time.time()
        
        self._thread_pool = []
    
    def start(self):
        self.running = True
        
        for i in range(2):
            t = threading.Thread(target=self._process_news_worker, daemon=True, name=f"NewsWorker-{i}")
            t.start()
            self._thread_pool.append(t)
        
        t = threading.Thread(target=self._process_price_thread, daemon=True, name="PriceWorker")
        t.start()
        self._thread_pool.append(t)
        
        t = threading.Thread(target=self._window_aggregation_thread, daemon=True, name="AggWorker")
        t.start()
        self._thread_pool.append(t)
        
        t = threading.Thread(target=self._signal_generation_thread, daemon=True, name="SignalWorker")
        t.start()
        self._thread_pool.append(t)
        
        t = threading.Thread(target=self._monitoring_thread, daemon=True, name="Monitor")
        t.start()
        self._thread_pool.append(t)
        
        print("Optimized Python Stream Processor started with 2 news workers")
    
    def stop(self):
        self.running = False
        for t in self._thread_pool:
            t.join(timeout=2.0)
    
    def queue_news(self, news_json: str):
        try:
            self.news_queue.put_nowait(news_json)
        except queue.Full:
            self.backpressure.stats.drop_count += 1
    
    def queue_price(self, price_json: str):
        try:
            self.price_queue.put_nowait(price_json)
        except queue.Full:
            pass
    
    def queue_orderbook(self, orderbook_json: str):
        try:
            self.orderbook_queue.put_nowait(orderbook_json)
        except queue.Full:
            pass
    
    def _process_news_worker(self):
        batch = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                news_json = self.news_queue.get(timeout=0.05)
                
                if not self.backpressure.report_input(time.time()):
                    continue
                
                try:
                    news = NewsArticle.from_json(news_json)
                    batch.append(news)
                except Exception as e:
                    print(f"Error parsing news: {e}")
                    self.backpressure.report_processed()
                
                current_time = time.time()
                if len(batch) >= settings.BATCH_SIZE or (current_time - last_batch_time) > 0.1:
                    if batch:
                        results = self.analyzer.analyze_batch(batch)
                        
                        for result in results:
                            self.data_store.add_sentiment(result)
                            self.window_aggregator.add(
                                result.symbol,
                                result.sentiment_score,
                                result.timestamp.timestamp()
                            )
                            self._sentiment_buffer[result.symbol].append(result)
                        
                        self.backpressure.report_processed(len(batch))
                        self.processed_count += len(batch)
                        
                        batch = []
                        last_batch_time = current_time
                
            except queue.Empty:
                if batch:
                    results = self.analyzer.analyze_batch(batch)
                    for result in results:
                        self.data_store.add_sentiment(result)
                        self.window_aggregator.add(
                            result.symbol,
                            result.sentiment_score,
                            result.timestamp.timestamp()
                        )
                        self._sentiment_buffer[result.symbol].append(result)
                    
                    self.backpressure.report_processed(len(batch))
                    self.processed_count += len(batch)
                    
                    batch = []
                    last_batch_time = time.time()
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in news worker: {e}")
                time.sleep(0.1)
    
    def _process_price_thread(self):
        while self.running:
            try:
                batch_size = min(10, self.price_queue.qsize())
                for _ in range(batch_size):
                    price_json = self.price_queue.get_nowait()
                    try:
                        price = PriceData.from_json(price_json)
                        self.data_store.add_price(price)
                    except Exception as e:
                        pass
                
                batch_size = min(5, self.orderbook_queue.qsize())
                for _ in range(batch_size):
                    orderbook_json = self.orderbook_queue.get_nowait()
                    try:
                        ob_data = json.loads(orderbook_json)
                        order_book = OrderBook(
                            symbol=ob_data['symbol'],
                            bids=ob_data['bids'],
                            asks=ob_data['asks'],
                            timestamp=datetime.fromisoformat(ob_data['timestamp'])
                        )
                        self.data_store.add_order_book(order_book)
                    except Exception as e:
                        pass
                
                time.sleep(0.02)
                
            except queue.Empty:
                time.sleep(0.02)
            except Exception as e:
                print(f"Error in price processing: {e}")
                time.sleep(1)
    
    def _window_aggregation_thread(self):
        while self.running:
            try:
                current_time = time.time()
                
                for symbol in self.window_aggregator.get_all_symbols():
                    agg_data = self.window_aggregator.emit(symbol, current_time)
                    
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
                        
                        self.data_store.add_window_aggregate(agg)
                
                cutoff_time = current_time - 300
                for symbol in list(self._sentiment_buffer.keys()):
                    self._sentiment_buffer[symbol] = [
                        s for s in self._sentiment_buffer[symbol]
                        if s.timestamp.timestamp() >= cutoff_time
                    ]
                
                time.sleep(settings.SLIDE_DURATION)
                
            except Exception as e:
                print(f"Error in window aggregation: {e}")
                time.sleep(1)
    
    def _signal_generation_thread(self):
        from signal.signal_generator import TradingSignalGenerator
        
        signal_generator = TradingSignalGenerator(self.data_store)
        
        while self.running:
            try:
                for symbol in settings.SYMBOLS:
                    signal = signal_generator.generate_signal(symbol)
                    if signal:
                        self.data_store.add_trading_signal(signal)
                
                time.sleep(settings.SLIDE_DURATION)
                
            except Exception as e:
                print(f"Error in signal generation: {e}")
                time.sleep(1)
    
    def _monitoring_thread(self):
        while self.running:
            try:
                current_time = time.time()
                
                if current_time - self._last_stats_print > 10:
                    status = self.backpressure.get_status()
                    print(f"[Monitor] {status}")
                    self._last_stats_print = current_time
                
                if current_time - self._last_gc > self._gc_interval:
                    gc.collect()
                    self._last_gc = current_time
                
                time.sleep(5)
                
            except Exception as e:
                pass


PythonStreamProcessor = OptimizedPythonStreamProcessor
