import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from shared.models import (
    PriceData, TradingSignal, VaRResult, ExposureItem, RiskDashboard
)


class VaRCalculator:
    def __init__(self, lookback_days: int = 252,
                 confidence_levels: List[float] = None):
        self.lookback_days = lookback_days
        self.confidence_levels = confidence_levels or [0.95, 0.99, 0.999]
        self.price_history: Dict[str, deque] = {}
        self.position_history: Dict[str, deque] = {}
    
    def add_price(self, symbol: str, price_data: PriceData):
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback_days * 3)
        
        self.price_history[symbol].append((price_data.timestamp, price_data.price))
    
    def add_position(self, symbol: str, position: float, timestamp: datetime):
        if symbol not in self.position_history:
            self.position_history[symbol] = deque(maxlen=self.lookback_days * 3)
        
        self.position_history[symbol].append((timestamp, position))
    
    def calculate_returns(self, symbol: str) -> np.ndarray:
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return np.array([])
        
        prices = np.array([p[1] for p in list(self.price_history[symbol])])
        returns = np.diff(np.log(prices))
        return returns
    
    def calculate_var_historical(self, symbol: str, 
                                 position_value: float,
                                 confidence: float = 0.95) -> float:
        returns = self.calculate_returns(symbol)
        if len(returns) < 20:
            return position_value * 0.02
        
        percentile = (1 - confidence) * 100
        var_pct = -np.percentile(returns, percentile)
        
        return float(position_value * var_pct)
    
    def calculate_var_parametric(self, symbol: str,
                                  position_value: float,
                                  confidence: float = 0.95) -> float:
        returns = self.calculate_returns(symbol)
        if len(returns) < 20:
            return position_value * 0.02
        
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        z_score = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326,
            0.999: 3.090
        }.get(confidence, 1.645)
        
        var_pct = z_score * sigma - mu
        return float(position_value * var_pct)
    
    def calculate_cvar(self, symbol: str,
                       position_value: float,
                       confidence: float = 0.95) -> float:
        returns = self.calculate_returns(symbol)
        if len(returns) < 20:
            return position_value * 0.03
        
        percentile = (1 - confidence) * 100
        var_threshold = np.percentile(returns, percentile)
        
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) == 0:
            return position_value * 0.03
        
        cvar_pct = -np.mean(tail_returns)
        return float(position_value * cvar_pct)
    
    def calculate_full_var(self, portfolio_positions: Dict[str, Dict[str, float]],
                           method: str = 'parametric') -> VaRResult:
        total_value = sum(pos['value'] for pos in portfolio_positions.values())
        
        if method == 'historical':
            var_calc = self.calculate_var_historical
        else:
            var_calc = self.calculate_var_parametric
        
        var_95 = sum(var_calc(sym, pos['value'], 0.95) 
                     for sym, pos in portfolio_positions.items())
        var_99 = sum(var_calc(sym, pos['value'], 0.99)
                     for sym, pos in portfolio_positions.items())
        var_99_9 = sum(var_calc(sym, pos['value'], 0.999)
                       for sym, pos in portfolio_positions.items())
        
        cvar_95 = sum(self.calculate_cvar(sym, pos['value'], 0.95)
                      for sym, pos in portfolio_positions.items())
        cvar_99 = sum(self.calculate_cvar(sym, pos['value'], 0.99)
                      for sym, pos in portfolio_positions.items())
        
        diversification_benefit = 0.3
        if len(portfolio_positions) > 1:
            var_95 *= (1 - diversification_benefit * 0.5)
            var_99 *= (1 - diversification_benefit * 0.5)
            var_99_9 *= (1 - diversification_benefit * 0.5)
            cvar_95 *= (1 - diversification_benefit * 0.5)
            cvar_99 *= (1 - diversification_benefit * 0.5)
        
        return VaRResult(
            var_95=float(var_95),
            var_99=float(var_99),
            var_99_9=float(var_99_9),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            position_value=total_value,
            lookback_days=self.lookback_days,
            method=method
        )


class DrawdownCalculator:
    def __init__(self, lookback_periods: int = 252):
        self.lookback_periods = lookback_periods
        self.portfolio_history: deque = deque(maxlen=lookback_periods * 3)
        self.peak_value: float = 0.0
        self.max_drawdown: float = 0.0
        self.current_drawdown: float = 0.0
        self.drawdown_history: deque = deque(maxlen=lookback_periods)
    
    def update_portfolio_value(self, value: float, timestamp: datetime):
        self.portfolio_history.append((timestamp, value))
        
        if value > self.peak_value:
            self.peak_value = value
        
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - value) / self.peak_value
        else:
            self.current_drawdown = 0.0
        
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
        
        self.drawdown_history.append((timestamp, self.current_drawdown))
        
        if len(self.portfolio_history) > self.lookback_periods * 2:
            old_peak = max(v for _, v in list(self.portfolio_history)[:self.lookback_periods])
            self.peak_value = max(self.peak_value, old_peak)
    
    def get_drawdown_series(self) -> List[Tuple[datetime, float]]:
        return list(self.drawdown_history)
    
    def get_underwater_chart_data(self) -> Dict[str, List]:
        history = list(self.drawdown_history)
        return {
            'timestamps': [t for t, _ in history],
            'drawdowns': [d for _, d in history]
        }
    
    def get_max_drawdown_duration(self) -> timedelta:
        history = list(self.portfolio_history)
        if len(history) < 2:
            return timedelta(0)
        
        max_duration = timedelta(0)
        peak_idx = 0
        peak_value = history[0][1]
        
        for i, (ts, val) in enumerate(history):
            if val > peak_value:
                peak_idx = i
                peak_value = val
            else:
                duration = ts - history[peak_idx][0]
                if duration > max_duration:
                    max_duration = duration
        
        return max_duration


class ExposureAnalyzer:
    def __init__(self):
        self.sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Consumer', 'TSLA': 'Automotive', 'META': 'Technology',
            'NVDA': 'Technology', 'JPM': 'Financials', 'BAC': 'Financials',
            'WMT': 'Consumer', 'PG': 'Consumer', 'V': 'Financials',
            'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare'
        }
        
        self.beta_mapping = {
            'AAPL': 1.3, 'MSFT': 1.1, 'GOOGL': 1.05,
            'AMZN': 1.25, 'TSLA': 1.8, 'META': 1.35,
            'NVDA': 1.6, 'JPM': 1.15, 'BAC': 1.2,
            'WMT': 0.6, 'PG': 0.55, 'V': 0.95,
            'JNJ': 0.7, 'UNH': 0.9, 'PFE': 0.85
        }
    
    def get_sector(self, symbol: str) -> str:
        return self.sector_mapping.get(symbol, 'Other')
    
    def get_beta(self, symbol: str) -> float:
        return self.beta_mapping.get(symbol, 1.0)
    
    def calculate_exposure(self, positions: Dict[str, Dict[str, Any]],
                           prices: Dict[str, float]) -> Tuple[List[ExposureItem], Dict[str, float]]:
        total_value = sum(pos.get('position', 0) * prices.get(sym, 0) 
                          for sym, pos in positions.items())
        
        exposure_items = []
        sector_exposure = {}
        
        for symbol, pos_info in positions.items():
            position = pos_info.get('position', 0)
            price = prices.get(symbol, pos_info.get('price', 0))
            value = position * price
            weight = value / total_value if total_value > 0 else 0
            
            sector = self.get_sector(symbol)
            beta = self.get_beta(symbol)
            
            exposure_items.append(ExposureItem(
                symbol=symbol,
                position=position,
                value=value,
                weight=weight,
                sector=sector,
                beta=beta
            ))
            
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value
        
        sector_weights = {s: v / total_value for s, v in sector_exposure.items()}
        
        return exposure_items, sector_weights
    
    def calculate_portfolio_beta(self, exposure_items: List[ExposureItem]) -> float:
        if not exposure_items:
            return 1.0
        
        weighted_beta = sum(item.weight * item.beta for item in exposure_items)
        return float(weighted_beta)
    
    def calculate_concentration_ratio(self, exposure_items: List[ExposureItem], 
                                       top_n: int = 5) -> float:
        if not exposure_items:
            return 0.0
        
        sorted_items = sorted(exposure_items, key=lambda x: abs(x.weight), reverse=True)
        top_weights = sum(abs(item.weight) for item in sorted_items[:top_n])
        return float(top_weights)
    
    def identify_overexposed_sectors(self, sector_weights: Dict[str, float],
                                      threshold: float = 0.25) -> List[str]:
        return [sector for sector, weight in sector_weights.items() 
                if weight > threshold]


class RiskDashboardEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.var_calculator = VaRCalculator(
            lookback_days=self.config.get('var_lookback', 252)
        )
        
        self.drawdown_calculator = DrawdownCalculator(
            lookback_periods=self.config.get('dd_lookback', 252)
        )
        
        self.exposure_analyzer = ExposureAnalyzer()
        
        self.stress_test_scenarios = {
            '2008_crisis': -0.45,
            '2020_covid': -0.35,
            '2022_hawkish': -0.25,
            'dotcom_bubble': -0.50,
            'flash_crash': -0.15
        }
        
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.current_prices: Dict[str, float] = {}
        self.portfolio_value: float = 0.0
        self.risk_score: float = 0.0
    
    def update_price(self, symbol: str, price_data: PriceData):
        self.var_calculator.add_price(symbol, price_data)
        self.current_prices[symbol] = price_data.price
        
        if symbol in self.current_positions:
            self.current_positions[symbol]['price'] = price_data.price
            self._update_portfolio_value()
    
    def update_position(self, symbol: str, position: float, price: float):
        self.current_positions[symbol] = {
            'position': position,
            'price': price,
            'value': position * price
        }
        self.current_prices[symbol] = price
        self._update_portfolio_value()
    
    def _update_portfolio_value(self):
        self.portfolio_value = sum(
            pos.get('position', 0) * self.current_prices.get(sym, pos.get('price', 0))
            for sym, pos in self.current_positions.items()
        )
        
        self.drawdown_calculator.update_portfolio_value(
            self.portfolio_value, datetime.now()
        )
    
    def run_stress_test(self) -> Dict[str, float]:
        results = {}
        
        for scenario, drawdown_pct in self.stress_test_scenarios.items():
            loss = self.portfolio_value * abs(drawdown_pct)
            remaining_value = self.portfolio_value - loss
            
            portfolio_beta = self.exposure_analyzer.calculate_portfolio_beta(
                self._get_exposure_items()
            )
            
            adjusted_loss = loss * portfolio_beta
            
            results[scenario] = {
                'scenario_drawdown': drawdown_pct,
                'portfolio_beta': portfolio_beta,
                'expected_loss': float(adjusted_loss),
                'expected_loss_pct': float(adjusted_loss / self.portfolio_value * 100),
                'remaining_value': float(self.portfolio_value - adjusted_loss)
            }
        
        return results
    
    def _get_exposure_items(self) -> List[ExposureItem]:
        if not self.current_positions:
            return []
        
        exposure_items, _ = self.exposure_analyzer.calculate_exposure(
            self.current_positions, self.current_prices
        )
        return exposure_items
    
    def calculate_risk_score(self, var_result: VaRResult,
                              exposure_items: List[ExposureItem]) -> float:
        var_pct = var_result.var_95 / var_result.position_value * 100
        
        concentration = self.exposure_analyzer.calculate_concentration_ratio(exposure_items)
        portfolio_beta = self.exposure_analyzer.calculate_portfolio_beta(exposure_items)
        
        drawdown_factor = self.drawdown_calculator.current_drawdown * 100
        
        scores = []
        
        if var_pct < 2:
            scores.append(1.0)
        elif var_pct < 5:
            scores.append(0.7)
        elif var_pct < 10:
            scores.append(0.4)
        else:
            scores.append(0.1)
        
        if concentration < 0.3:
            scores.append(1.0)
        elif concentration < 0.5:
            scores.append(0.7)
        elif concentration < 0.7:
            scores.append(0.4)
        else:
            scores.append(0.1)
        
        if portfolio_beta < 0.8:
            scores.append(1.0)
        elif portfolio_beta < 1.2:
            scores.append(0.8)
        elif portfolio_beta < 1.5:
            scores.append(0.5)
        else:
            scores.append(0.2)
        
        if drawdown_factor < 5:
            scores.append(1.0)
        elif drawdown_factor < 15:
            scores.append(0.6)
        else:
            scores.append(0.2)
        
        weights = [0.35, 0.25, 0.25, 0.15]
        total_score = sum(s * w for s, w in zip(scores, weights))
        
        return float(max(0.0, min(1.0, total_score)))
    
    def get_dashboard(self, method: str = 'parametric') -> RiskDashboard:
        positions_for_var = {
            sym: {'value': pos.get('position', 0) * self.current_prices.get(sym, pos.get('price', 0))}
            for sym, pos in self.current_positions.items()
        }
        
        var_result = self.var_calculator.calculate_full_var(positions_for_var, method)
        
        exposure_items, sector_exposure = self.exposure_analyzer.calculate_exposure(
            self.current_positions, self.current_prices
        )
        
        stress_test_results = self.run_stress_test()
        
        self.risk_score = self.calculate_risk_score(var_result, exposure_items)
        
        return RiskDashboard(
            var=var_result,
            max_drawdown=self.drawdown_calculator.max_drawdown,
            current_drawdown=self.drawdown_calculator.current_drawdown,
            exposure_by_symbol=exposure_items,
            exposure_by_sector=sector_exposure,
            portfolio_value=self.portfolio_value,
            risk_score=self.risk_score,
            stress_test_result={k: v['expected_loss'] for k, v in stress_test_results.items()},
            timestamp=datetime.now()
        )
    
    def get_risk_alerts(self) -> List[Dict[str, Any]]:
        alerts = []
        
        dashboard = self.get_dashboard()
        
        if dashboard.current_drawdown > 0.1:
            alerts.append({
                'severity': 'warning',
                'type': 'drawdown',
                'message': f"Current drawdown at {dashboard.current_drawdown*100:.1f}%",
                'value': dashboard.current_drawdown,
                'threshold': 0.1
            })
        
        var_pct = dashboard.var.var_95 / dashboard.var.position_value * 100
        if var_pct > 5:
            alerts.append({
                'severity': 'warning',
                'type': 'var',
                'message': f"95% VaR at {var_pct:.1f}% of portfolio",
                'value': var_pct,
                'threshold': 5.0
            })
        
        if dashboard.risk_score < 0.3:
            alerts.append({
                'severity': 'critical',
                'type': 'risk_score',
                'message': f"Risk score critically low: {dashboard.risk_score:.2f}",
                'value': dashboard.risk_score,
                'threshold': 0.3
            })
        
        overexposed = self.exposure_analyzer.identify_overexposed_sectors(
            dashboard.exposure_by_sector
        )
        for sector in overexposed:
            alerts.append({
                'severity': 'warning',
                'type': 'concentration',
                'message': f"Overexposed to {sector}: {dashboard.exposure_by_sector[sector]*100:.1f}%",
                'value': dashboard.exposure_by_sector[sector],
                'threshold': 0.25
            })
        
        return alerts


class RiskLimitChecker:
    def __init__(self, limits: Optional[Dict[str, Any]] = None):
        self.limits = limits or {
            'max_single_position_pct': 0.15,
            'max_sector_exposure_pct': 0.30,
            'max_portfolio_var_pct': 0.05,
            'max_drawdown_pct': 0.20,
            'max_position_turnover': 5.0,
            'min_risk_score': 0.3
        }
        
        self.violations: List[Dict[str, Any]] = []
    
    def check_position_limit(self, symbol: str, position_value: float, 
                              portfolio_value: float) -> bool:
        if portfolio_value == 0:
            return True
        
        pct = position_value / portfolio_value
        if pct > self.limits['max_single_position_pct']:
            self.violations.append({
                'type': 'position_limit',
                'symbol': symbol,
                'pct': pct,
                'limit': self.limits['max_single_position_pct']
            })
            return False
        return True
    
    def check_all_limits(self, dashboard: RiskDashboard) -> Tuple[bool, List[Dict[str, Any]]]:
        self.violations = []
        portfolio_value = dashboard.portfolio_value
        
        for item in dashboard.exposure_by_symbol:
            self.check_position_limit(item.symbol, item.value, portfolio_value)
        
        for sector, pct in dashboard.exposure_by_sector.items():
            if pct > self.limits['max_sector_exposure_pct']:
                self.violations.append({
                    'type': 'sector_limit',
                    'sector': sector,
                    'pct': pct,
                    'limit': self.limits['max_sector_exposure_pct']
                })
        
        var_pct = dashboard.var.var_95 / portfolio_value if portfolio_value > 0 else 0
        if var_pct > self.limits['max_portfolio_var_pct']:
            self.violations.append({
                'type': 'var_limit',
                'var_pct': var_pct,
                'limit': self.limits['max_portfolio_var_pct']
            })
        
        if dashboard.current_drawdown > self.limits['max_drawdown_pct']:
            self.violations.append({
                'type': 'drawdown_limit',
                'drawdown': dashboard.current_drawdown,
                'limit': self.limits['max_drawdown_pct']
            })
        
        if dashboard.risk_score < self.limits['min_risk_score']:
            self.violations.append({
                'type': 'risk_score_limit',
                'risk_score': dashboard.risk_score,
                'limit': self.limits['min_risk_score']
            })
        
        return len(self.violations) == 0, self.violations
