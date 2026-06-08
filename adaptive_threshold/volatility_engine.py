import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from shared.models import (
    PriceData, VolatilityState, AdaptiveThreshold
)


class VolatilityCalculator:
    def __init__(self, lookback_periods: int = 60,
                 min_periods: int = 20,
                 annualization_factor: int = 252):
        self.lookback_periods = lookback_periods
        self.min_periods = min_periods
        self.annualization_factor = annualization_factor
        self.price_history: Dict[str, deque] = {}
        self.return_history: Dict[str, deque] = {}
    
    def add_price(self, symbol: str, price_data: PriceData):
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback_periods)
            self.return_history[symbol] = deque(maxlen=self.lookback_periods)
        
        self.price_history[symbol].append((price_data.timestamp, price_data.price))
        
        if len(self.price_history[symbol]) >= 2:
            prev_price = self.price_history[symbol][-2][1]
            if prev_price > 0:
                ret = np.log(price_data.price / prev_price)
                self.return_history[symbol].append(ret)
    
    def calculate_volatility(self, symbol: str) -> Optional[VolatilityState]:
        if symbol not in self.return_history:
            return None
        
        returns = list(self.return_history[symbol])
        if len(returns) < self.min_periods:
            return None
        
        current_vol = np.std(returns) * np.sqrt(self.annualization_factor)
        current_vol = float(current_vol)
        
        if len(returns) >= self.lookback_periods:
            first_half = returns[:len(returns)//2]
            second_half = returns[len(returns)//2:]
            hist_vol = np.std(first_half) * np.sqrt(self.annualization_factor)
            recent_vol = np.std(second_half) * np.sqrt(self.annualization_factor)
            historical_vol = float(np.mean([hist_vol, recent_vol]))
        else:
            historical_vol = current_vol
        
        volatility_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        vix = self._calculate_vix_proxy(symbol)
        
        regime = self._classify_regime(current_vol, volatility_ratio)
        
        return VolatilityState(
            symbol=symbol,
            current_volatility=current_vol,
            historical_volatility=historical_vol,
            volatility_ratio=volatility_ratio,
            vix=vix,
            regime=regime,
            timestamp=datetime.now()
        )
    
    def _calculate_vix_proxy(self, symbol: str) -> float:
        if symbol not in self.return_history:
            return 20.0
        
        returns = list(self.return_history[symbol])
        if len(returns) < self.min_periods:
            return 20.0
        
        recent_returns = returns[-20:] if len(returns) >= 20 else returns
        vol = np.std(recent_returns) * np.sqrt(self.annualization_factor)
        return float(max(10.0, min(80.0, vol * 100)))
    
    def _classify_regime(self, current_vol: float, vol_ratio: float) -> str:
        if vol_ratio < 0.8 and current_vol < 0.2:
            return 'low_vol'
        elif 0.8 <= vol_ratio <= 1.2 and 0.15 <= current_vol <= 0.35:
            return 'normal'
        elif vol_ratio > 1.2 or current_vol > 0.35:
            return 'high_vol'
        else:
            return 'normal'
    
    def get_volatility_surface(self, symbol: str, 
                               tenors: List[int] = [1, 5, 20, 60]) -> Dict[int, float]:
        if symbol not in self.return_history:
            return {t: 0.2 for t in tenors}
        
        returns = list(self.return_history[symbol])
        surface = {}
        
        for tenor in tenors:
            if len(returns) >= max(tenor, self.min_periods):
                recent_returns = returns[-tenor:]
                vol = np.std(recent_returns) * np.sqrt(self.annualization_factor)
                surface[tenor] = float(vol)
            else:
                surface[tenor] = 0.2
        
        return surface


class AdaptiveThresholdEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.base_buy_threshold = self.config.get('base_buy_threshold', 0.6)
        self.base_sell_threshold = self.config.get('base_sell_threshold', -0.6)
        
        self.min_buy_threshold = self.config.get('min_buy_threshold', 0.3)
        self.max_buy_threshold = self.config.get('max_buy_threshold', 0.9)
        self.min_sell_threshold = self.config.get('min_sell_threshold', -0.9)
        self.max_sell_threshold = self.config.get('max_sell_threshold', -0.3)
        
        self.volatility_sensitivity = self.config.get('volatility_sensitivity', 0.5)
        self.momentum_sensitivity = self.config.get('momentum_sensitivity', 0.3)
        
        self.volatility_calculator = VolatilityCalculator(
            lookback_periods=self.config.get('vol_lookback', 60),
            min_periods=self.config.get('vol_min_periods', 20)
        )
        
        self.current_thresholds: Dict[str, AdaptiveThreshold] = {}
        self.price_momentum: Dict[str, deque] = {}
    
    def update_price(self, symbol: str, price_data: PriceData):
        self.volatility_calculator.add_price(symbol, price_data)
        
        if symbol not in self.price_momentum:
            self.price_momentum[symbol] = deque(maxlen=20)
        
        self.price_momentum[symbol].append((price_data.timestamp, price_data.price))
    
    def calculate_momentum_factor(self, symbol: str) -> float:
        if symbol not in self.price_momentum or len(self.price_momentum[symbol]) < 10:
            return 1.0
        
        prices = [p[1] for p in list(self.price_momentum[symbol])]
        
        if len(prices) < 10:
            return 1.0
        
        sma_short = np.mean(prices[-5:])
        sma_long = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
        
        if sma_long == 0:
            return 1.0
        
        momentum = (sma_short - sma_long) / sma_long
        momentum_factor = 1.0 + self.momentum_sensitivity * np.sign(momentum) * min(abs(momentum) / 0.05, 1.0)
        
        return max(0.7, min(1.3, momentum_factor))
    
    def calculate_volatility_factor(self, vol_state: VolatilityState) -> float:
        if vol_state is None:
            return 1.0
        
        vol_ratio = vol_state.volatility_ratio
        
        if vol_state.regime == 'high_vol':
            factor = 1.0 + self.volatility_sensitivity * min((vol_ratio - 1.0) * 2, 0.5)
        elif vol_state.regime == 'low_vol':
            factor = 1.0 - self.volatility_sensitivity * min((1.0 - vol_ratio) * 2, 0.3)
        else:
            factor = 1.0
        
        return max(0.8, min(1.5, factor))
    
    def get_adaptive_thresholds(self, symbol: str) -> Optional[AdaptiveThreshold]:
        vol_state = self.volatility_calculator.calculate_volatility(symbol)
        
        if vol_state is None:
            if symbol in self.current_thresholds:
                return self.current_thresholds[symbol]
            return None
        
        volatility_factor = self.calculate_volatility_factor(vol_state)
        momentum_factor = self.calculate_momentum_factor(symbol)
        
        combined_factor = volatility_factor * momentum_factor
        
        buy_threshold = self.base_buy_threshold * combined_factor
        sell_threshold = self.base_sell_threshold * combined_factor
        
        buy_threshold = max(self.min_buy_threshold, min(self.max_buy_threshold, buy_threshold))
        sell_threshold = max(self.min_sell_threshold, min(self.max_sell_threshold, sell_threshold))
        
        threshold = AdaptiveThreshold(
            symbol=symbol,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            base_buy_threshold=self.base_buy_threshold,
            base_sell_threshold=self.base_sell_threshold,
            volatility_factor=volatility_factor,
            momentum_factor=momentum_factor,
            timestamp=datetime.now()
        )
        
        self.current_thresholds[symbol] = threshold
        return threshold
    
    def check_signal(self, symbol: str, sentiment_score: float, 
                    correlation_score: float) -> Tuple[str, float, float]:
        thresholds = self.get_adaptive_thresholds(symbol)
        
        if thresholds is None:
            buy_thresh = self.base_buy_threshold
            sell_thresh = self.base_sell_threshold
        else:
            buy_thresh = thresholds.buy_threshold
            sell_thresh = thresholds.sell_threshold
        
        combined_score = 0.7 * sentiment_score + 0.3 * correlation_score
        
        if combined_score >= buy_thresh:
            signal = 'BUY'
            strength = min(1.0, (combined_score - buy_thresh) / (1.0 - buy_thresh + 1e-6))
        elif combined_score <= sell_thresh:
            signal = 'SELL'
            strength = min(1.0, (sell_thresh - combined_score) / (abs(sell_thresh) + 1e-6))
        else:
            signal = 'HOLD'
            strength = 0.0
        
        confidence = abs(combined_score)
        
        return signal, strength, confidence
    
    def reset_symbol(self, symbol: str):
        if symbol in self.current_thresholds:
            del self.current_thresholds[symbol]
        if symbol in self.price_momentum:
            self.price_momentum[symbol].clear()
    
    def get_threshold_history(self, symbol: str) -> List[Dict[str, Any]]:
        if symbol not in self.current_thresholds:
            return []
        
        threshold = self.current_thresholds[symbol]
        return [{
            'timestamp': threshold.timestamp,
            'buy_threshold': threshold.buy_threshold,
            'sell_threshold': threshold.sell_threshold,
            'volatility_factor': threshold.volatility_factor,
            'momentum_factor': threshold.momentum_factor
        }]


class RegimeBasedThresholdStrategy:
    def __init__(self):
        self.regime_configs = {
            'low_vol': {
                'buy_threshold': 0.5,
                'sell_threshold': -0.5,
                'position_size': 1.5
            },
            'normal': {
                'buy_threshold': 0.6,
                'sell_threshold': -0.6,
                'position_size': 1.0
            },
            'high_vol': {
                'buy_threshold': 0.8,
                'sell_threshold': -0.8,
                'position_size': 0.5
            },
            'extreme_vol': {
                'buy_threshold': 0.95,
                'sell_threshold': -0.95,
                'position_size': 0.25
            }
        }
    
    def get_regime(self, vol_state: VolatilityState) -> str:
        if vol_state is None:
            return 'normal'
        
        if vol_state.volatility_ratio > 2.0 or vol_state.current_volatility > 0.5:
            return 'extreme_vol'
        return vol_state.regime
    
    def get_position_size(self, vol_state: VolatilityState) -> float:
        regime = self.get_regime(vol_state)
        return self.regime_configs[regime]['position_size']
    
    def get_thresholds(self, vol_state: VolatilityState) -> Tuple[float, float]:
        regime = self.get_regime(vol_state)
        config = self.regime_configs[regime]
        return config['buy_threshold'], config['sell_threshold']
