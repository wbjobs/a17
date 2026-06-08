import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
import numpy as np
import pandas as pd


@dataclass
class RenderThrottler:
    max_fps: float = 2.0
    _last_render_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def should_render(self) -> bool:
        with self._lock:
            now = time.time()
            min_interval = 1.0 / self.max_fps
            if now - self._last_render_time >= min_interval:
                self._last_render_time = now
                return True
            return False
    
    def force_render(self):
        with self._lock:
            self._last_render_time = time.time()


class IncrementalHeatmapData:
    def __init__(self, max_time_buckets: int = 15, max_symbols: int = 10):
        self.max_time_buckets = max_time_buckets
        self.max_symbols = max_symbols
        
        self._data: Dict[str, Dict[float, Dict]] = {}
        self._time_buckets: deque = deque(maxlen=max_time_buckets)
        self._symbols: List[str] = []
        self._dirty_mask: np.ndarray = None
        self._last_hash: str = None
        
        self._lock = threading.Lock()
    
    def _get_time_bucket(self, timestamp: float, bucket_size_seconds: int = 120) -> float:
        return int(timestamp // bucket_size_seconds) * bucket_size_seconds
    
    def update(self, symbol: str, timestamp: float, value: float, metadata: Dict = None):
        with self._lock:
            bucket = self._get_time_bucket(timestamp)
            
            if symbol not in self._symbols:
                if len(self._symbols) >= self.max_symbols:
                    return False
                self._symbols.append(symbol)
                self._resize_dirty_mask()
            
            if bucket not in self._time_buckets:
                self._time_buckets.append(bucket)
                self._resize_dirty_mask()
            
            if symbol not in self._data:
                self._data[symbol] = {}
            
            old_value = self._data[symbol].get(bucket, {}).get('value', 0)
            if abs(old_value - value) > 0.01:
                self._data[symbol][bucket] = {
                    'value': value,
                    'count': metadata.get('count', 1) if metadata else 1,
                    'updated': time.time()
                }
                self._mark_dirty(symbol, bucket)
            
            return True
    
    def _resize_dirty_mask(self):
        new_shape = (len(self._symbols), len(self._time_buckets))
        if self._dirty_mask is None or self._dirty_mask.shape != new_shape:
            self._dirty_mask = np.ones(new_shape, dtype=bool)
    
    def _mark_dirty(self, symbol: str, bucket: float):
        if self._dirty_mask is None:
            return
        
        try:
            sym_idx = self._symbols.index(symbol)
            bucket_idx = list(self._time_buckets).index(bucket)
            self._dirty_mask[sym_idx, bucket_idx] = True
        except ValueError:
            pass
    
    def get_heatmap_data(self, force_full: bool = False) -> Tuple[np.ndarray, List[str], List[float], np.ndarray, List[List[str]]]:
        with self._lock:
            n_symbols = len(self._symbols)
            n_buckets = len(self._time_buckets)
            
            if n_symbols == 0 or n_buckets == 0:
                return np.zeros((0, 0)), [], [], np.zeros((0, 0), dtype=bool)
            
            data = np.zeros((n_symbols, n_buckets))
            text_data = [['' for _ in range(n_buckets)] for _ in range(n_symbols)]
            
            for i, symbol in enumerate(self._symbols):
                for j, bucket in enumerate(self._time_buckets):
                    if symbol in self._data and bucket in self._data[symbol]:
                        entry = self._data[symbol][bucket]
                        data[i, j] = entry['value']
                        count = entry.get('count', 0)
                        text_data[i][j] = f"{count} signals<br>Avg: {abs(entry['value']):.2f}"
            
            dirty = self._dirty_mask.copy() if self._dirty_mask is not None else np.ones_like(data, dtype=bool)
            
            if not force_full:
                self._dirty_mask = np.zeros_like(self._dirty_mask, dtype=bool)
            
            return data, self._symbols.copy(), list(self._time_buckets), dirty, text_data
    
    def needs_update(self) -> bool:
        with self._lock:
            return self._dirty_mask is not None and np.any(self._dirty_mask)
    
    def get_data_hash(self) -> str:
        with self._lock:
            data, _, _, _ = self.get_heatmap_data(force_full=True)
            return hash(data.tobytes())


class DataDownsampler:
    @staticmethod
    def downsample_time_series(
        timestamps: List[float],
        values: List[float],
        target_points: int = 100
    ) -> Tuple[List[float], List[float]]:
        if len(timestamps) <= target_points:
            return timestamps, values
        
        if len(timestamps) == 0:
            return [], []
        
        step = len(timestamps) / target_points
        result_ts = []
        result_vals = []
        
        for i in range(target_points):
            start_idx = int(i * step)
            end_idx = int((i + 1) * step)
            
            if start_idx >= len(timestamps):
                break
            
            end_idx = min(end_idx, len(timestamps))
            
            chunk_ts = timestamps[start_idx:end_idx]
            chunk_vals = values[start_idx:end_idx]
            
            result_ts.append(np.mean(chunk_ts))
            result_vals.append(np.mean(chunk_vals))
        
        return result_ts, result_vals
    
    @staticmethod
    def downsample_heatmap(
        data: np.ndarray,
        target_buckets: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
        if data.shape[1] <= target_buckets:
            return data, np.ones_like(data, dtype=bool)
        
        n_symbols, n_buckets = data.shape
        step = n_buckets / target_buckets
        
        result = np.zeros((n_symbols, target_buckets))
        dirty = np.ones((n_symbols, target_buckets), dtype=bool)
        
        for j in range(target_buckets):
            start = int(j * step)
            end = min(int((j + 1) * step), n_buckets)
            
            if start >= n_buckets:
                break
            
            result[:, j] = np.mean(data[:, start:end], axis=1)
        
        return result, dirty


class SmartDeltaUpdater:
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
        self._last_values: Dict[str, float] = {}
    
    def has_changed(self, key: str, new_value: float) -> bool:
        old_value = self._last_values.get(key)
        if old_value is None:
            self._last_values[key] = new_value
            return True
        
        if abs(old_value - new_value) > self.tolerance:
            self._last_values[key] = new_value
            return True
        
        return False
    
    def batch_has_changed(self, updates: Dict[str, float]) -> List[str]:
        changed = []
        for key, value in updates.items():
            if self.has_changed(key, value):
                changed.append(key)
        return changed


class FrameRateLimiter:
    def __init__(self, target_fps: float = 2.0):
        self.target_fps = target_fps
        self.min_frame_time = 1.0 / target_fps
        self._last_frame_time = 0.0
        self._frame_times: deque = deque(maxlen=30)
    
    def wait_for_next_frame(self) -> float:
        now = time.time()
        elapsed = now - self._last_frame_time
        
        if elapsed < self.min_frame_time:
            sleep_time = self.min_frame_time - elapsed
            time.sleep(sleep_time)
            now = time.time()
        
        actual_fps = 1.0 / (now - self._last_frame_time) if self._last_frame_time > 0 else 0
        self._frame_times.append(now - self._last_frame_time)
        self._last_frame_time = now
        
        return actual_fps
    
    def get_average_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        avg_frame_time = np.mean(list(self._frame_times)[1:])
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0


def optimize_plotly_figure(fig, use_webgl: bool = True, downsample: bool = True):
    try:
        for trace in fig.data:
            if hasattr(trace, 'type'):
                if trace.type == 'scatter' and use_webgl:
                    trace.type = 'scattergl'
                
                if downsample and hasattr(trace, 'x') and trace.x is not None:
                    if len(trace.x) > 500:
                        x_data = trace.x
                        y_data = trace.y
                        
                        if isinstance(x_data, (list, np.ndarray)) and isinstance(y_data, (list, np.ndarray)):
                            x_arr = np.array(x_data) if not isinstance(x_data, np.ndarray) else x_data
                            y_arr = np.array(y_data) if not isinstance(y_data, np.ndarray) else y_data
                            
                            step = max(1, len(x_arr) // 500)
                            indices = np.arange(0, len(x_arr), step)
                            
                            trace.x = x_arr[indices]
                            trace.y = y_arr[indices]
    except Exception:
        pass
    
    fig.update_layout(
        uirevision='constant',
        datarevision=int(time.time() * 1000),
    )
    
    return fig
