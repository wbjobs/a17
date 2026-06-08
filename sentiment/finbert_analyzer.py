import os
import sys
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from shared.models import NewsArticle, SentimentResult


class FinBERTAnalyzer:
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.model = None
        self.tokenizer = None
        self._positive_keywords = [
            'beat', 'surge', 'record', 'growth', 'profit', 'gain', 'upgrade',
            'strong', 'exceed', 'announce partnership', 'regulatory approval',
            'strategic partnership', 'innovative', 'major contract', 'demand exceeds',
            'expands', 'share buyback', 'transformative'
        ]
        self._negative_keywords = [
            'miss', 'decline', 'investigation', 'downgrade', 'delayed', 'lawsuit',
            'cuts guidance', 'recalls', 'short interest', 'restructuring',
            'quality issues', 'headwinds', 'safety concerns', 'regulatory',
            'challenging environment', 'shakeup'
        ]
        
        if not use_mock:
            self._load_model()
    
    def _load_model(self):
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            import torch
            
            print(f"Loading FinBERT model: {settings.FINBERT_MODEL_NAME}")
            os.makedirs(settings.FINBERT_CACHE_DIR, exist_ok=True)
            
            self.tokenizer = BertTokenizer.from_pretrained(
                settings.FINBERT_MODEL_NAME,
                cache_dir=settings.FINBERT_CACHE_DIR
            )
            self.model = BertForSequenceClassification.from_pretrained(
                settings.FINBERT_MODEL_NAME,
                cache_dir=settings.FINBERT_CACHE_DIR
            )
            self.model.eval()
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            print("FinBERT model loaded successfully")
        except Exception as e:
            print(f"Failed to load FinBERT model: {e}")
            print("Falling back to mock mode")
            self.use_mock = True
    
    def _analyze_mock(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        
        positive_score = 0.0
        negative_score = 0.0
        
        for keyword in self._positive_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                positive_score += 0.15
        
        for keyword in self._negative_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                negative_score += 0.15
        
        positive_score = min(positive_score + np.random.uniform(0, 0.3), 1.0)
        negative_score = min(negative_score + np.random.uniform(0, 0.3), 1.0)
        
        total = positive_score + negative_score
        if total > 0:
            neutral_score = max(0, 1.0 - positive_score - negative_score)
        else:
            positive_score = np.random.uniform(0.1, 0.3)
            negative_score = np.random.uniform(0.1, 0.3)
            neutral_score = np.random.uniform(0.4, 0.8)
        
        total = positive_score + negative_score + neutral_score
        return {
            'positive': positive_score / total,
            'negative': negative_score / total,
            'neutral': neutral_score / total
        }
    
    def _analyze_real(self, texts: List[str]) -> List[Dict[str, float]]:
        import torch
        
        results = []
        batch_size = settings.BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            for pred in predictions:
                results.append({
                    'positive': float(pred[0]),
                    'negative': float(pred[1]),
                    'neutral': float(pred[2])
                })
        
        return results
    
    def analyze(self, news: NewsArticle) -> SentimentResult:
        text = f"{news.title} {news.content}"
        
        if self.use_mock:
            scores = self._analyze_mock(text)
        else:
            scores = self._analyze_real([text])[0]
        
        sentiment_score = scores['positive'] - scores['negative']
        
        return SentimentResult(
            news_id=news.id,
            symbol=news.symbol,
            positive=scores['positive'],
            negative=scores['negative'],
            neutral=scores['neutral'],
            sentiment_score=sentiment_score,
            timestamp=datetime.now()
        )
    
    def analyze_batch(self, news_list: List[NewsArticle]) -> List[SentimentResult]:
        if not news_list:
            return []
        
        if self.use_mock:
            results = []
            for news in news_list:
                results.append(self.analyze(news))
            return results
        else:
            texts = [f"{n.title} {n.content}" for n in news_list]
            scores_list = self._analyze_real(texts)
            
            results = []
            for news, scores in zip(news_list, scores_list):
                sentiment_score = scores['positive'] - scores['negative']
                results.append(SentimentResult(
                    news_id=news.id,
                    symbol=news.symbol,
                    positive=scores['positive'],
                    negative=scores['negative'],
                    neutral=scores['neutral'],
                    sentiment_score=sentiment_score,
                    timestamp=datetime.now()
                ))
            
            return results


class SentimentAnalyzer:
    _instance = None
    
    def __new__(cls, use_mock: bool = True):
        if cls._instance is None:
            cls._instance = FinBERTAnalyzer(use_mock=use_mock)
        return cls._instance


def analyze_news(news_json: str, use_mock: bool = True) -> str:
    news = NewsArticle.from_json(news_json)
    analyzer = SentimentAnalyzer(use_mock=use_mock)
    result = analyzer.analyze(news)
    return result.to_json()


def analyze_news_batch(news_json_list: List[str], use_mock: bool = True) -> List[str]:
    news_list = [NewsArticle.from_json(j) for j in news_json_list]
    analyzer = SentimentAnalyzer(use_mock=use_mock)
    results = analyzer.analyze_batch(news_list)
    return [r.to_json() for r in results]
