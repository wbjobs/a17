import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from collections import deque
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from shared.models import (
    NewsArticle, SentimentResult, PriceData,
    WindowSentimentAggregate, TradingSignal, EnhancedTradingSignal,
    VolatilityState, AdaptiveThreshold, SignalExplanation
)
from adaptive_threshold.volatility_engine import AdaptiveThresholdEngine
from explanation.shap_explainer import SignalExplainer
from multimodal.multimodal_analyzer import MultimodalAnalyzer
from risk.risk_dashboard import RiskDashboardEngine, RiskLimitChecker


class EnhancedSignalGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.base_buy_threshold = self.config.get('buy_threshold', 0.6)
        self.base_sell_threshold = self.config.get('sell_threshold', -0.6)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.3)
        
        self.window_size = self.config.get('window_size', 5)
        self.slide_interval = self.config.get('slide_interval', 2)
        
        self.use_adaptive_threshold = self.config.get('use_adaptive_threshold', True)
        self.use_multimodal = self.config.get('use_multimodal', True)
        self.use_explainable = self.config.get('use_explainable', True)
        self.use_risk_limits = self.config.get('use_risk_limits', True)
        
        if self.use_adaptive_threshold:
            self.threshold_engine = AdaptiveThresholdEngine(config)
        else:
            self.threshold_engine = None
        
        if self.use_multimodal:
            self.multimodal_analyzer = MultimodalAnalyzer(config)
        else:
            self.multimodal_analyzer = None
        
        if self.use_explainable:
            self.signal_explainer = SignalExplainer()
        else:
            self.signal_explainer = None
        
        if self.use_risk_limits:
            self.risk_engine = RiskDashboardEngine(config)
            self.risk_checker = RiskLimitChecker()
        else:
            self.risk_engine = None
            self.risk_checker = None
        
        self.price_history: Dict[str, deque] = {}
        self.sentiment_history: Dict[str, deque] = {}
        self.signal_history: deque = deque(maxlen=1000)
        
        self.sentiment_window: Dict[str, List[Tuple[datetime, float]]] = {}
        self.max_window_age = 300
    
    def update_price(self, price_data: PriceData):
        symbol = price_data.symbol
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=1000)
        
        self.price_history[symbol].append((price_data.timestamp, price_data.price))
        
        if self.threshold_engine:
            self.threshold_engine.update_price(symbol, price_data)
        
        if self.risk_engine:
            self.risk_engine.update_price(symbol, price_data)
    
    def update_position(self, symbol: str, position: float, price: float):
        if self.risk_engine:
            self.risk_engine.update_position(symbol, position, price)
    
    def analyze_multimodal(self, news: NewsArticle,
                          images: Optional[List[bytes]] = None,
                          tables_html: Optional[List[str]] = None,
                          charts: Optional[List[bytes]] = None) -> Optional[Dict[str, Any]]:
        if not self.multimodal_analyzer:
            return None
        
        try:
            multimodal_result = self.multimodal_analyzer.analyze_news(
                news, images, tables_html, charts
            )
            return {
                'multimodal_score': multimodal_result.combined_score,
                'multimodal_result': multimodal_result,
                'combined_text': multimodal_result.raw_text
            }
        except Exception as e:
            logger.warning(f"Multimodal analysis failed: {e}")
            return None
    
    def calculate_sentiment_price_correlation(self, symbol: str) -> float:
        if symbol not in self.price_history or symbol not in self.sentiment_history:
            return 0.0
        
        prices = list(self.price_history[symbol])[-50:]
        sentiments = list(self.sentiment_history[symbol])[-50:]
        
        if len(prices) < 10 or len(sentiments) < 10:
            return 0.0
        
        try:
            min_len = min(len(prices), len(sentiments))
            price_series = np.array([p[1] for p in prices[-min_len:]])
            sentiment_series = np.array([s[1] for s in sentiments[-min_len:]])
            
            if len(price_series) < 5 or len(sentiment_series) < 5:
                return 0.0
            
            price_returns = np.diff(np.log(price_series))
            aligned_sentiment = sentiment_series[1:len(price_returns)+1]
            
            if len(price_returns) < 3 or len(aligned_sentiment) < 3:
                return 0.0
            
            correlation = np.corrcoef(price_returns, aligned_sentiment)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            return float(correlation)
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return 0.0
    
    def generate_signal(self, news: NewsArticle,
                       sentiment_result: SentimentResult,
                       window_aggregate: Optional[WindowSentimentAggregate] = None,
                       multimodal_data: Optional[Dict[str, Any]] = None) -> Optional[EnhancedTradingSignal]:
        
        symbol = news.symbol
        
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = deque(maxlen=1000)
        
        self.sentiment_history[symbol].append((
            sentiment_result.timestamp,
            sentiment_result.sentiment_score
        ))
        
        combined_score = sentiment_result.sentiment_score
        if multimodal_data and 'multimodal_score' in multimodal_data:
            combined_score = 0.6 * combined_score + 0.4 * multimodal_data['multimodal_score']
        
        correlation_score = self.calculate_sentiment_price_correlation(symbol)
        
        volatility_state = None
        adaptive_threshold = None
        risk_adjusted_strength = 0.0
        
        if self.threshold_engine:
            volatility_state = self.threshold_engine.volatility_calculator.calculate_volatility(symbol)
            adaptive_threshold = self.threshold_engine.get_adaptive_thresholds(symbol)
            signal, strength, confidence = self.threshold_engine.check_signal(
                symbol, combined_score, correlation_score
            )
        else:
            if combined_score >= self.base_buy_threshold and abs(correlation_score) >= self.correlation_threshold:
                signal = 'BUY'
                strength = min(1.0, (combined_score - self.base_buy_threshold) / (1.0 - self.base_buy_threshold + 1e-6))
            elif combined_score <= self.base_sell_threshold and abs(correlation_score) >= self.correlation_threshold:
                signal = 'SELL'
                strength = min(1.0, (self.base_sell_threshold - combined_score) / (abs(self.base_sell_threshold) + 1e-6))
            else:
                signal = 'HOLD'
                strength = 0.0
            
            confidence = abs(combined_score)
        
        explanation = None
        if self.use_explainable and self.signal_explainer and signal != 'HOLD':
            try:
                base_signal = TradingSignal(
                    symbol=symbol,
                    signal=signal,
                    strength=strength,
                    sentiment_score=combined_score,
                    price_correlation=correlation_score,
                    timestamp=datetime.now(),
                    confidence=confidence,
                    reason=f"Sentiment: {combined_score:.2f}, Correlation: {correlation_score:.2f}"
                )
                explanation = self.signal_explainer.explain_signal(
                    base_signal, news, sentiment_result
                )
            except Exception as e:
                logger.warning(f"Signal explanation failed: {e}")
        
        if volatility_state and adaptive_threshold:
            vol_factor = adaptive_threshold.volatility_factor
            risk_adjusted_strength = strength * min(1.0, 1.0 / max(vol_factor, 0.8))
        
        if self.use_risk_limits and signal != 'HOLD' and self.risk_engine:
            try:
                dashboard = self.risk_engine.get_dashboard()
                if dashboard.portfolio_value > 0:
                    limits_ok, violations = self.risk_checker.check_all_limits(dashboard)
                    if not limits_ok:
                        risk_reasons = [v['type'] for v in violations[:3]]
                        if strength > 0.5:
                            strength *= 0.5
                            confidence *= 0.7
                            logger.info(f"Signal {signal} scaled down due to risk limits: {risk_reasons}")
                        elif signal != 'HOLD':
                            signal = 'HOLD'
                            strength = 0.0
                            logger.info(f"Signal {signal} blocked due to risk limits: {risk_reasons}")
            except Exception as e:
                logger.warning(f"Risk limit check failed: {e}")
        
        enhanced_signal = EnhancedTradingSignal(
            symbol=symbol,
            signal=signal,
            strength=strength,
            sentiment_score=combined_score,
            price_correlation=correlation_score,
            timestamp=datetime.now(),
            confidence=confidence,
            reason=f"Sentiment: {combined_score:.2f}, Correlation: {correlation_score:.2f}",
            volatility_state=volatility_state,
            adaptive_threshold=adaptive_threshold,
            explanation=explanation,
            risk_adjusted_strength=risk_adjusted_strength
        )
        
        self.signal_history.append(enhanced_signal)
        
        return enhanced_signal
    
    def get_risk_dashboard(self) -> Optional[Dict[str, Any]]:
        if not self.risk_engine:
            return None
        
        try:
            dashboard = self.risk_engine.get_dashboard()
            alerts = self.risk_engine.get_risk_alerts()
            limits_ok, violations = self.risk_checker.check_all_limits(dashboard)
            
            return {
                'dashboard': dashboard,
                'alerts': alerts,
                'limits_ok': limits_ok,
                'violations': violations
            }
        except Exception as e:
            logger.warning(f"Failed to get risk dashboard: {e}")
            return None
    
    def get_threshold_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.threshold_engine:
            return {
                'buy_threshold': self.base_buy_threshold,
                'sell_threshold': self.base_sell_threshold,
                'adaptive': False
            }
        
        thresholds = self.threshold_engine.get_adaptive_thresholds(symbol)
        vol_state = self.threshold_engine.volatility_calculator.calculate_volatility(symbol)
        
        return {
            'current': thresholds,
            'volatility': vol_state,
            'adaptive': True
        }
    
    def get_explanation_for_signal(self, signal: TradingSignal,
                                    news: NewsArticle,
                                    sentiment_result: SentimentResult) -> Optional[SignalExplanation]:
        if not self.signal_explainer:
            return None
        
        try:
            return self.signal_explainer.explain_signal(signal, news, sentiment_result)
        except Exception as e:
            logger.warning(f"Failed to get explanation: {e}")
            return None
    
    def get_highlighted_text(self, explanation: SignalExplanation, text: str) -> str:
        if not self.signal_explainer or not explanation:
            return text
        
        return self.signal_explainer.highlight_text(
            text, explanation.shap_explanation.keyword_highlights
        )
    
    def reset(self, symbol: Optional[str] = None):
        if symbol:
            if symbol in self.price_history:
                self.price_history[symbol].clear()
            if symbol in self.sentiment_history:
                self.sentiment_history[symbol].clear()
            if self.threshold_engine:
                self.threshold_engine.reset_symbol(symbol)
        else:
            self.price_history.clear()
            self.sentiment_history.clear()
            self.signal_history.clear()
