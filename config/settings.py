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

    # Multimodal Analysis
    USE_MULTIMODAL: bool = True
    TESSERACT_CMD: str = None
    MULTIMODAL_WEIGHTS = {
        'text': 0.4,
        'image': 0.3,
        'table': 0.2,
        'chart': 0.1
    }

    # Adaptive Threshold
    USE_ADAPTIVE_THRESHOLD: bool = True
    VOLATILITY_LOOKBACK: int = 60
    VOLATILITY_MIN_PERIODS: int = 20
    BASE_BUY_THRESHOLD: float = 0.6
    BASE_SELL_THRESHOLD: float = -0.6
    MIN_BUY_THRESHOLD: float = 0.3
    MAX_BUY_THRESHOLD: float = 0.9
    MIN_SELL_THRESHOLD: float = -0.9
    MAX_SELL_THRESHOLD: float = -0.3
    VOLATILITY_SENSITIVITY: float = 0.5
    MOMENTUM_SENSITIVITY: float = 0.3

    # Risk Dashboard
    USE_RISK_LIMITS: bool = True
    VAR_LOOKBACK_DAYS: int = 252
    DRAWDOWN_LOOKBACK: int = 252
    RISK_LIMITS = {
        'max_single_position_pct': 0.15,
        'max_sector_exposure_pct': 0.30,
        'max_portfolio_var_pct': 0.05,
        'max_drawdown_pct': 0.20,
        'min_risk_score': 0.3
    }
    STRESS_TEST_SCENARIOS = {
        '2008_crisis': -0.45,
        '2020_covid': -0.35,
        '2022_hawkish': -0.25,
        'dotcom_bubble': -0.50,
        'flash_crash': -0.15
    }

    # Explainable AI
    USE_EXPLAINABLE: bool = True
    USE_TRUE_SHAP: bool = False
    SHAP_EXPLAINER_MODEL: str = None
    MAX_EXPLANATION_CACHE: int = 10000
    EXPLANATION_TTL_SECONDS: int = 3600

    # Enhanced Signal Generation
    ENHANCED_SIGNAL_CONFIG = {
        'use_adaptive_threshold': True,
        'use_multimodal': True,
        'use_explainable': True,
        'use_risk_limits': True
    }

    # Portfolio Configuration
    INITIAL_CAPITAL: float = 1000000.0
    DEFAULT_POSITION_SIZE: float = 0.1
    MAX_POSITION_SIZE: float = 0.3

settings = Settings()
