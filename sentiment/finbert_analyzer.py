import os
import sys
import gc
import threading
import time
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from shared.models import NewsArticle, SentimentResult


class GPUMemoryManager:
    def __init__(self, max_memory_fraction: float = 0.8, cleanup_interval: int = 100):
        self.max_memory_fraction = max_memory_fraction
        self.cleanup_interval = cleanup_interval
        self.batch_count = 0
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._peak_memory_usage = 0
    
    def _get_gpu_memory_used(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 3)
        except:
            pass
        return 0.0
    
    def _get_gpu_memory_total(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except:
            pass
        return 0.0
    
    def should_cleanup(self) -> bool:
        self.batch_count += 1
        
        if self.batch_count % self.cleanup_interval == 0:
            return True
        
        try:
            import torch
            if torch.cuda.is_available():
                used = self._get_gpu_memory_used()
                total = self._get_gpu_memory_total()
                
                if used > self._peak_memory_usage:
                    self._peak_memory_usage = used
                
                if total > 0 and used / total > self.max_memory_fraction:
                    return True
                
                if torch.cuda.memory_allocated() > torch.cuda.max_memory_allocated() * 0.9:
                    return True
        except:
            pass
        
        return False
    
    def cleanup(self, force: bool = False):
        with self._lock:
            try:
                import torch
                if torch.cuda.is_available():
                    if force or self.should_cleanup():
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        try:
                            torch.cuda.ipc_collect()
                        except:
                            pass
                        
                        current_used = self._get_gpu_memory_used()
                        if current_used < self._peak_memory_usage * 0.5:
                            self._peak_memory_usage = current_used
                        
                        self._last_cleanup = time.time()
                        self.batch_count = 0
            except Exception as e:
                pass


class OptimizedFinBERTAnalyzer:
    def __init__(self, use_mock: bool = True, max_batch_size: int = 32):
        self.use_mock = use_mock
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_batch_size = max_batch_size
        
        self.memory_manager = GPUMemoryManager(
            max_memory_fraction=0.75,
            cleanup_interval=50
        )
        
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
        
        self._tensor_cache: Dict[str, object] = {}
        self._lock = threading.Lock()
        
        if not use_mock:
            self._load_model()
    
    def _load_model(self):
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            import torch
            
            print(f"Loading FinBERT model: {settings.FINBERT_MODEL_NAME}")
            os.makedirs(settings.FINBERT_CACHE_DIR, exist_ok=True)
            
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                print(f"GPU Memory: {total_mem:.1f} GB")
                
                torch.cuda.set_per_process_memory_fraction(0.8)
            else:
                self.device = torch.device('cpu')
                print("GPU not available, using CPU")
            
            self.tokenizer = BertTokenizer.from_pretrained(
                settings.FINBERT_MODEL_NAME,
                cache_dir=settings.FINBERT_CACHE_DIR,
                model_max_length=512
            )
            
            self.model = BertForSequenceClassification.from_pretrained(
                settings.FINBERT_MODEL_NAME,
                cache_dir=settings.FINBERT_CACHE_DIR
            )
            
            self.model.eval()
            self.model.to(self.device)
            
            for param in self.model.parameters():
                param.requires_grad = False
            
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
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
        batch_size = min(self.max_batch_size, settings.BATCH_SIZE)
        
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch = texts[batch_start:batch_end]
            
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = self.model(**inputs)
                        predictions = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                    
                    for pred in predictions:
                        results.append({
                            'positive': float(pred[0]),
                            'negative': float(pred[1]),
                            'neutral': float(pred[2])
                        })
                
                    del inputs, outputs, predictions
                    
            except Exception as e:
                print(f"Error in batch inference: {e}")
                for _ in batch:
                    results.append({
                        'positive': 0.33,
                        'negative': 0.33,
                        'neutral': 0.34
                    })
            
            finally:
                self.memory_manager.cleanup()
        
        self.memory_manager.cleanup(force=(len(texts) > 100))
        
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
    
    def cleanup(self):
        self.memory_manager.cleanup(force=True)
        
        with self._lock:
            self._tensor_cache.clear()
        
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass


class SentimentAnalyzer:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, use_mock: bool = True):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = OptimizedFinBERTAnalyzer(use_mock=use_mock)
        return cls._instance
    
    @classmethod
    def reset(cls):
        with cls._lock:
            if cls._instance:
                cls._instance.cleanup()
            cls._instance = None


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
