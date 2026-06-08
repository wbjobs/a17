import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional
import numpy as np


@dataclass
class BackpressureStats:
    queue_size: int = 0
    processing_rate: float = 0.0
    input_rate: float = 0.0
    lag_seconds: float = 0.0
    drop_count: int = 0
    sample_rate: float = 1.0
    last_update: float = field(default_factory=time.time)


class BackpressureController:
    def __init__(
        self,
        max_queue_size: int = 10000,
        target_lag_seconds: float = 5.0,
        min_sample_rate: float = 0.1,
        max_sample_rate: float = 1.0
    ):
        self.max_queue_size = max_queue_size
        self.target_lag_seconds = target_lag_seconds
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        
        self._input_count = 0
        self._processed_count = 0
        self._last_input_count = 0
        self._last_processed_count = 0
        self._last_stats_time = time.time()
        
        self._oldest_timestamp: Optional[float] = None
        self._timestamps: Deque[float] = deque(maxlen=1000)
        
        self.stats = BackpressureStats()
        self._lock = threading.Lock()
    
    def report_input(self, timestamp: float = None) -> bool:
        with self._lock:
            self._input_count += 1
            if timestamp is not None:
                self._timestamps.append(timestamp)
                if self._oldest_timestamp is None or timestamp < self._oldest_timestamp:
                    self._oldest_timestamp = timestamp
            
            if self.stats.queue_size >= self.max_queue_size:
                self.stats.drop_count += 1
                return False
            
            if self.stats.sample_rate < 1.0:
                if np.random.random() > self.stats.sample_rate:
                    self.stats.drop_count += 1
                    return False
            
            self.stats.queue_size += 1
            return True
    
    def report_processed(self, count: int = 1):
        with self._lock:
            self._processed_count += count
            self.stats.queue_size = max(0, self.stats.queue_size - count)
    
    def update_stats(self) -> BackpressureStats:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_stats_time
            
            if elapsed > 0:
                self.stats.input_rate = (self._input_count - self._last_input_count) / elapsed
                self.stats.processing_rate = (self._processed_count - self._last_processed_count) / elapsed
            
            if self._timestamps:
                newest = self._timestamps[-1]
                oldest = self._timestamps[0]
                self.stats.lag_seconds = now - newest if newest else 0
            
            if self.stats.input_rate > 0 and self.stats.processing_rate > 0:
                processing_ratio = self.stats.processing_rate / self.stats.input_rate
                
                if self.stats.lag_seconds > self.target_lag_seconds * 2:
                    self.stats.sample_rate = max(
                        self.min_sample_rate,
                        self.stats.sample_rate * 0.8
                    )
                elif self.stats.lag_seconds > self.target_lag_seconds:
                    self.stats.sample_rate = max(
                        self.min_sample_rate,
                        self.stats.sample_rate * 0.95
                    )
                elif self.stats.lag_seconds < self.target_lag_seconds * 0.3 and processing_ratio > 1.2:
                    self.stats.sample_rate = min(
                        self.max_sample_rate,
                        self.stats.sample_rate * 1.1
                    )
            
            self._last_input_count = self._input_count
            self._last_processed_count = self._processed_count
            self._last_stats_time = now
            self.stats.last_update = now
            
            return self.stats
    
    def get_status(self) -> str:
        stats = self.update_stats()
        return (
            f"Queue: {stats.queue_size}/{self.max_queue_size} | "
            f"Input: {stats.input_rate:.1f}/s | "
            f"Process: {stats.processing_rate:.1f}/s | "
            f"Lag: {stats.lag_seconds:.1f}s | "
            f"Sample: {stats.sample_rate:.1%} | "
            f"Dropped: {stats.drop_count}"
        )


@dataclass
class IncrementalAggState:
    sum_sentiment: float = 0.0
    count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    sum_of_squares: float = 0.0
    last_sentiment: float = 0.0
    
    def add(self, sentiment_score: float):
        self.sum_sentiment += sentiment_score
        self.sum_of_squares += sentiment_score ** 2
        self.count += 1
        if sentiment_score > 0.1:
            self.positive_count += 1
        elif sentiment_score < -0.1:
            self.negative_count += 1
        self.last_sentiment = sentiment_score
    
    def remove(self, sentiment_score: float):
        self.sum_sentiment -= sentiment_score
        self.sum_of_squares -= sentiment_score ** 2
        self.count = max(0, self.count - 1)
        if sentiment_score > 0.1:
            self.positive_count = max(0, self.positive_count - 1)
        elif sentiment_score < -0.1:
            self.negative_count = max(0, self.negative_count - 1)
    
    @property
    def avg_sentiment(self) -> float:
        return self.sum_sentiment / self.count if self.count > 0 else 0.0
    
    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return (self.sum_of_squares / self.count) - (self.avg_sentiment ** 2)
    
    @property
    def positive_ratio(self) -> float:
        return self.positive_count / self.count if self.count > 0 else 0.0
    
    @property
    def negative_ratio(self) -> float:
        return self.negative_count / self.count if self.count > 0 else 0.0


class SlidingWindowIncrementalAggregator:
    def __init__(self, window_duration_seconds: int, slide_interval_seconds: int):
        self.window_duration = window_duration_seconds
        self.slide_interval = slide_interval_seconds
        
        self._states: dict[str, IncrementalAggState] = {}
        self._buffer: dict[str, Deque[tuple[float, float]]] = {}
        self._last_emit_time: dict[str, float] = {}
        
        self._lock = threading.Lock()
    
    def add(self, symbol: str, sentiment_score: float, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            if symbol not in self._buffer:
                self._buffer[symbol] = deque()
                self._states[symbol] = IncrementalAggState()
                self._last_emit_time[symbol] = timestamp
            
            self._buffer[symbol].append((timestamp, sentiment_score))
            self._states[symbol].add(sentiment_score)
            
            cutoff_time = timestamp - self.window_duration
            buffer = self._buffer[symbol]
            state = self._states[symbol]
            
            while buffer and buffer[0][0] < cutoff_time:
                old_ts, old_score = buffer.popleft()
                state.remove(old_score)
    
    def should_emit(self, symbol: str, current_time: float = None) -> bool:
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            last_emit = self._last_emit_time.get(symbol, 0)
            return (current_time - last_emit) >= self.slide_interval
    
    def emit(self, symbol: str, current_time: float = None) -> Optional[dict]:
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            if not self.should_emit(symbol, current_time):
                return None
            
            state = self._states.get(symbol)
            if state is None or state.count == 0:
                self._last_emit_time[symbol] = current_time
                return None
            
            buffer = self._buffer[symbol]
            sentiment_scores = [s for _, s in buffer]
            momentum = 0.0
            if len(sentiment_scores) > 10:
                recent = sentiment_scores[-10:]
                older = sentiment_scores[:-10]
                if older:
                    momentum = np.mean(recent) - np.mean(older)
            
            result = {
                'symbol': symbol,
                'window_start': current_time - self.window_duration,
                'window_end': current_time,
                'avg_sentiment': state.avg_sentiment,
                'news_count': state.count,
                'positive_ratio': state.positive_ratio,
                'negative_ratio': state.negative_ratio,
                'sentiment_momentum': momentum,
                'variance': state.variance
            }
            
            self._last_emit_time[symbol] = current_time
            return result
    
    def get_all_symbols(self) -> list[str]:
        with self._lock:
            return list(self._buffer.keys())
