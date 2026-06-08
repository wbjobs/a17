import os
import sys
import json
import time
import threading
import queue
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


class PythonStreamProcessor:
    def __init__(self, data_store: StreamDataStore, use_mock_sentiment: bool = True):
        self.data_store = data_store
        self.analyzer = SentimentAnalyzer(use_mock=use_mock_sentiment)
        self.running = False
        self.news_queue: queue.Queue = queue.Queue()
        self.price_queue: queue.Queue = queue.Queue()
        self.orderbook_queue: queue.Queue = queue.Queue()
        self.processed_count = 0
        self.last_window_time = datetime.now()
        self.sentiment_buffer: Dict[str, List[SentimentResult]] = defaultdict(list)
    
    def start(self):
        self.running = True
        threading.Thread(target=self._process_news_thread, daemon=True).start()
        threading.Thread(target=self._process_price_thread, daemon=True).start()
        threading.Thread(target=self._window_aggregation_thread, daemon=True).start()
        threading.Thread(target=self._signal_generation_thread, daemon=True).start()
        print("Python Stream Processor started")
    
    def stop(self):
        self.running = False
    
    def queue_news(self, news_json: str):
        self.news_queue.put(news_json)
    
    def queue_price(self, price_json: str):
        self.price_queue.put(price_json)
    
    def queue_orderbook(self, orderbook_json: str):
        self.orderbook_queue.put(orderbook_json)
    
    def _process_news_thread(self):
        batch = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                if not self.news_queue.empty():
                    news_json = self.news_queue.get(timeout=0.1)
                    try:
                        news = NewsArticle.from_json(news_json)
                        batch.append(news)
                    except Exception as e:
                        print(f"Error parsing news: {e}")
                
                current_time = time.time()
                if len(batch) >= settings.BATCH_SIZE or (current_time - last_batch_time) > 0.5:
                    if batch:
                        results = self.analyzer.analyze_batch(batch)
                        for result in results:
                            self.data_store.add_sentiment(result)
                            self.sentiment_buffer[result.symbol].append(result)
                        self.processed_count += len(batch)
                        
                        if self.processed_count % 1000 == 0:
                            print(f"Processed {self.processed_count} news articles")
                        
                        batch = []
                        last_batch_time = current_time
                else:
                    time.sleep(0.01)
                    
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in news processing: {e}")
                time.sleep(1)
    
    def _process_price_thread(self):
        while self.running:
            try:
                if not self.price_queue.empty():
                    price_json = self.price_queue.get(timeout=0.1)
                    try:
                        price = PriceData.from_json(price_json)
                        self.data_store.add_price(price)
                    except Exception as e:
                        print(f"Error parsing price: {e}")
                
                if not self.orderbook_queue.empty():
                    orderbook_json = self.orderbook_queue.get(timeout=0.1)
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
                        print(f"Error parsing orderbook: {e}")
                
                time.sleep(0.01)
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in price processing: {e}")
                time.sleep(1)
    
    def _window_aggregation_thread(self):
        window_duration = timedelta(seconds=settings.WINDOW_DURATION)
        slide_interval = settings.SLIDE_DURATION
        
        while self.running:
            try:
                current_time = datetime.now()
                
                for symbol in settings.SYMBOLS:
                    buffer = self.sentiment_buffer.get(symbol, [])
                    if not buffer:
                        continue
                    
                    window_end = current_time
                    window_start = window_end - window_duration
                    
                    window_results = [
                        s for s in buffer 
                        if window_start <= s.timestamp <= window_end
                    ]
                    
                    if len(window_results) > 0:
                        scores = [s.sentiment_score for s in window_results]
                        avg_sentiment = float(np.mean(scores))
                        positive_ratio = len([s for s in window_results if s.sentiment_score > 0.1]) / len(window_results)
                        negative_ratio = len([s for s in window_results if s.sentiment_score < -0.1]) / len(window_results)
                        
                        prev_scores = self.data_store.get_sentiment_scores(symbol, limit=20)
                        if len(prev_scores) > 10:
                            sentiment_momentum = avg_sentiment - float(np.mean(prev_scores[:10]))
                        else:
                            sentiment_momentum = 0.0
                        
                        agg = WindowSentimentAggregate(
                            symbol=symbol,
                            window_start=window_start,
                            window_end=window_end,
                            avg_sentiment=avg_sentiment,
                            news_count=len(window_results),
                            positive_ratio=positive_ratio,
                            negative_ratio=negative_ratio,
                            sentiment_momentum=sentiment_momentum
                        )
                        
                        self.data_store.add_window_aggregate(agg)
                    
                    cutoff_time = current_time - timedelta(minutes=5)
                    self.sentiment_buffer[symbol] = [
                        s for s in buffer if s.timestamp >= cutoff_time
                    ]
                
                time.sleep(slide_interval)
                
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


class SparkStreamProcessor:
    def __init__(self, data_store: StreamDataStore, use_mock_sentiment: bool = True):
        self.data_store = data_store
        self.use_mock_sentiment = use_mock_sentiment
        self.ssc = None
        self.running = False
    
    def start(self):
        try:
            from pyspark import SparkConf, SparkContext
            from pyspark.streaming import StreamingContext
            
            print("Starting Spark Streaming Processor...")
            
            conf = SparkConf() \
                .setMaster(settings.SPARK_MASTER) \
                .setAppName(settings.SPARK_APP_NAME) \
                .set("spark.executor.memory", "4g") \
                .set("spark.driver.memory", "2g")
            
            sc = SparkContext(conf=conf)
            sc.setLogLevel("ERROR")
            
            self.ssc = StreamingContext(sc, settings.SLIDE_DURATION)
            
            os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)
            self.ssc.checkpoint(settings.CHECKPOINT_DIR)
            
            self._setup_streaming_pipeline()
            
            self.ssc.start()
            self.running = True
            print("Spark Streaming started successfully")
            
            self.ssc.awaitTermination()
            
        except Exception as e:
            print(f"Spark Streaming not available: {e}")
            print("Falling back to Python Stream Processor")
            processor = PythonStreamProcessor(self.data_store, self.use_mock_sentiment)
            processor.start()
    
    def _setup_streaming_pipeline(self):
        from pyspark.streaming.kafka import KafkaUtils
        
        try:
            kafka_stream = KafkaUtils.createDirectStream(
                self.ssc,
                [settings.KAFKA_NEWS_TOPIC],
                {"metadata.broker.list": settings.KAFKA_BOOTSTRAP_SERVERS}
            )
            
            sentiment_stream = kafka_stream.map(
                lambda x: self._analyze_news(x[1])
            )
            
            windowed_stream = sentiment_stream.window(
                settings.WINDOW_DURATION,
                settings.SLIDE_DURATION
            )
            
            aggregated_stream = windowed_stream.map(
                lambda x: (x['symbol'], x)
            ).groupByKey().mapValues(
                lambda sentiments: self._aggregate_window(sentiments)
            )
            
            aggregated_stream.foreachRDD(
                lambda rdd: rdd.foreach(
                    lambda x: self._save_aggregate(x[1])
                )
            )
            
            signal_stream = aggregated_stream.map(
                lambda x: self._generate_signal(x[1])
            ).filter(
                lambda x: x is not None
            )
            
            signal_stream.foreachRDD(
                lambda rdd: rdd.foreach(
                    lambda signal: self._save_signal(signal)
                )
            )
            
        except Exception as e:
            print(f"Error setting up streaming pipeline: {e}")
            raise
    
    def _analyze_news(self, news_json: str) -> Dict:
        from sentiment.finbert_analyzer import analyze_news
        result_json = analyze_news(news_json, use_mock=self.use_mock_sentiment)
        return json.loads(result_json)
    
    def _aggregate_window(self, sentiments) -> Dict:
        sentiments_list = list(sentiments)
        if not sentiments_list:
            return None
        
        scores = [s['sentiment_score'] for s in sentiments_list]
        symbol = sentiments_list[0]['symbol']
        
        return {
            'symbol': symbol,
            'avg_sentiment': float(np.mean(scores)),
            'news_count': len(sentiments_list),
            'positive_ratio': len([s for s in sentiments_list if s['sentiment_score'] > 0.1]) / len(sentiments_list),
            'negative_ratio': len([s for s in sentiments_list if s['sentiment_score'] < -0.1]) / len(sentiments_list),
            'sentiment_momentum': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_aggregate(self, agg_data: Dict):
        if not agg_data:
            return
        
        try:
            agg = WindowSentimentAggregate(
                symbol=agg_data['symbol'],
                window_start=datetime.now() - timedelta(seconds=settings.WINDOW_DURATION),
                window_end=datetime.now(),
                avg_sentiment=agg_data['avg_sentiment'],
                news_count=agg_data['news_count'],
                positive_ratio=agg_data['positive_ratio'],
                negative_ratio=agg_data['negative_ratio'],
                sentiment_momentum=agg_data['sentiment_momentum']
            )
            self.data_store.add_window_aggregate(agg)
        except Exception as e:
            print(f"Error saving aggregate: {e}")
    
    def _generate_signal(self, agg_data: Dict) -> Dict:
        if not agg_data:
            return None
        
        symbol = agg_data['symbol']
        sentiment_scores = self.data_store.get_sentiment_scores(symbol, limit=settings.CORRELATION_WINDOW)
        prices = self.data_store.get_price_values(symbol, limit=settings.CORRELATION_WINDOW)
        
        correlation = calculate_rolling_correlation(
            sentiment_scores, prices, settings.CORRELATION_WINDOW
        )
        
        signal_strength = agg_data['avg_sentiment'] * (1 + abs(correlation))
        
        if signal_strength >= settings.SIGNAL_THRESHOLD_BUY:
            signal_type = "BUY"
        elif signal_strength <= settings.SIGNAL_THRESHOLD_SELL:
            signal_type = "SELL"
        else:
            return None
        
        return {
            'symbol': symbol,
            'signal': signal_type,
            'strength': abs(signal_strength),
            'sentiment_score': agg_data['avg_sentiment'],
            'price_correlation': correlation,
            'timestamp': datetime.now().isoformat(),
            'confidence': min(abs(signal_strength), 1.0),
            'reason': f"Sentiment {agg_data['avg_sentiment']:.3f}, Correlation {correlation:.3f}"
        }
    
    def _save_signal(self, signal_data: Dict):
        try:
            signal = TradingSignal(
                symbol=signal_data['symbol'],
                signal=signal_data['signal'],
                strength=signal_data['strength'],
                sentiment_score=signal_data['sentiment_score'],
                price_correlation=signal_data['price_correlation'],
                timestamp=datetime.fromisoformat(signal_data['timestamp']),
                confidence=signal_data['confidence'],
                reason=signal_data['reason']
            )
            self.data_store.add_trading_signal(signal)
        except Exception as e:
            print(f"Error saving signal: {e}")
    
    def stop(self):
        if self.ssc:
            self.ssc.stop(stopSparkContext=True, stopGraceFully=True)
        self.running = False
