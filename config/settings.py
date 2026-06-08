import os
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings:
    WEBSOCKET_HOST: str = "localhost"
    WEBSOCKET_PORT: int = 8765
    NEWS_GENERATION_RATE: int = 1200

    SPARK_MASTER: str = "local[*]"
    SPARK_APP_NAME: str = "FinancialNewsSentimentTrading"
    WINDOW_DURATION: int = 5
    SLIDE_DURATION: int = 2
    CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoint")

    FINBERT_MODEL_NAME: str = "ProsusAI/finbert"
    FINBERT_CACHE_DIR: str = os.path.join(BASE_DIR, "models", "finbert")
    BATCH_SIZE: int = 32

    SYMBOLS: List[str] = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "JPM", "BAC", "V"]
    
    SIGNAL_THRESHOLD_BUY: float = 0.6
    SIGNAL_THRESHOLD_SELL: float = -0.6
    CORRELATION_WINDOW: int = 30
    
    STREAMLIT_PORT: int = 8501
    STREAMLIT_UPDATE_INTERVAL: int = 1
    
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_NEWS_TOPIC: str = "financial_news"
    KAFKA_SIGNAL_TOPIC: str = "trading_signals"

    HISTORICAL_DATA_DIR: str = os.path.join(BASE_DIR, "data", "historical")
    BACKTEST_START_DATE: str = "2024-01-01"
    BACKTEST_END_DATE: str = "2024-12-31"
    RISK_FREE_RATE: float = 0.02

settings = Settings()
