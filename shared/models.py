from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import json


@dataclass
class NewsArticle:
    id: str
    symbol: str
    title: str
    content: str
    source: str
    timestamp: datetime
    url: Optional[str] = None

    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'NewsArticle':
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class SentimentResult:
    news_id: str
    symbol: str
    positive: float
    negative: float
    neutral: float
    sentiment_score: float
    timestamp: datetime

    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'SentimentResult':
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class PriceData:
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None

    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'PriceData':
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class OrderBookLevel:
    price: float
    size: float
    side: str


@dataclass
class OrderBook:
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime

    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)


@dataclass
class WindowSentimentAggregate:
    symbol: str
    window_start: datetime
    window_end: datetime
    avg_sentiment: float
    news_count: int
    positive_ratio: float
    negative_ratio: float
    sentiment_momentum: float

    def to_json(self) -> str:
        data = asdict(self)
        data['window_start'] = self.window_start.isoformat()
        data['window_end'] = self.window_end.isoformat()
        return json.dumps(data, ensure_ascii=False)


@dataclass
class TradingSignal:
    symbol: str
    signal: str
    strength: float
    sentiment_score: float
    price_correlation: float
    timestamp: datetime
    confidence: float
    reason: str

    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'TradingSignal':
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class BacktestResult:
    symbol: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    returns: List[float]
    signals: List[TradingSignal]

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'profit_factor': self.profit_factor
        }


@dataclass
class ImageContent:
    image_id: str
    image_type: str
    caption: str = ""
    ocr_text: str = ""
    description: str = ""
    sentiment_score: float = 0.0
    confidence: float = 0.0


@dataclass
class TableContent:
    table_id: str
    headers: List[str]
    rows: List[List[str]]
    extracted_text: str = ""
    key_insights: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0


@dataclass
class ChartData:
    chart_id: str
    chart_type: str
    title: str
    x_label: str
    y_label: str
    data_points: List[Tuple[float, float]]
    trend: str
    extracted_text: str = ""
    sentiment_score: float = 0.0


@dataclass
class MultimodalResult:
    news_id: str
    symbol: str
    images: List[ImageContent]
    tables: List[TableContent]
    charts: List[ChartData]
    combined_score: float
    timestamp: datetime
    raw_text: str = ""

    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)


@dataclass
class VolatilityState:
    symbol: str
    current_volatility: float
    historical_volatility: float
    volatility_ratio: float
    vix: float
    regime: str
    timestamp: datetime


@dataclass
class AdaptiveThreshold:
    symbol: str
    buy_threshold: float
    sell_threshold: float
    base_buy_threshold: float
    base_sell_threshold: float
    volatility_factor: float
    momentum_factor: float
    timestamp: datetime


@dataclass
class VaRResult:
    var_95: float
    var_99: float
    var_99_9: float
    cvar_95: float
    cvar_99: float
    position_value: float
    lookback_days: int
    method: str


@dataclass
class ExposureItem:
    symbol: str
    position: float
    value: float
    weight: float
    sector: str
    beta: float


@dataclass
class RiskDashboard:
    var: VaRResult
    max_drawdown: float
    current_drawdown: float
    exposure_by_symbol: List[ExposureItem]
    exposure_by_sector: Dict[str, float]
    portfolio_value: float
    risk_score: float
    stress_test_result: Dict[str, float]
    timestamp: datetime

    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)


@dataclass
class KeywordHighlight:
    word: str
    start: int
    end: int
    shap_value: float
    sentiment: str


@dataclass
class SHAPExplanation:
    news_id: str
    base_value: float
    output_value: float
    feature_importance: Dict[str, float]
    keyword_highlights: List[KeywordHighlight]
    expected_sentiment: str
    contributing_words: List[str]
    mitigating_words: List[str]


@dataclass
class SignalExplanation:
    signal_id: str
    symbol: str
    signal: str
    strength: float
    confidence: float
    shap_explanation: SHAPExplanation
    key_drivers: List[str]
    risk_factors: List[str]
    summary: str
    timestamp: datetime

    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)


@dataclass
class EnhancedTradingSignal(TradingSignal):
    volatility_state: Optional[VolatilityState] = None
    adaptive_threshold: Optional[AdaptiveThreshold] = None
    explanation: Optional[SignalExplanation] = None
    risk_adjusted_strength: float = 0.0

