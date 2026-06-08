from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
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
