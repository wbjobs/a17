import numpy as np
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from collections import deque, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except (ImportError, AttributeError, Exception):
    HAS_SHAP = False
    logger.warning("SHAP not available. Using fallback explanation method.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from shared.models import (
    NewsArticle, SentimentResult, TradingSignal,
    SHAPExplanation, KeywordHighlight, SignalExplanation
)


class FinancialKeywordDictionary:
    def __init__(self):
        self.positive_keywords = [
            'increase', 'surge', 'gain', 'rise', 'grow', 'beat', 'exceed',
            'positive', 'strong', 'improve', 'record', 'high', 'boom',
            'profit', 'upside', 'outperform', 'upgrade', 'buy', 'bullish',
            'momentum', 'breakout', 'rally', 'acceleration', 'expansion',
            'guidance raised', 'backlog', 'margin expansion', 'cost cutting',
            'market share gain', 'product launch', 'partnership', 'acquisition'
        ]
        
        self.negative_keywords = [
            'decrease', 'drop', 'fall', 'decline', 'miss', 'below', 'loss',
            'negative', 'weak', 'deteriorate', 'low', 'crash', 'downside',
            'underperform', 'downgrade', 'sell', 'bearish', 'correction',
            'slowdown', 'contraction', 'guidance lowered', 'margin compression',
            'cost increase', 'market share loss', 'product delay', 'lawsuit',
            'regulatory', 'investigation', 'recall', 'bankruptcy', 'default'
        ]
        
        self.neutral_keywords = [
            'maintain', 'stable', 'steady', 'in line', 'met expectations',
            'consistent', 'flat', 'unchanged', 'neutral', 'hold', 'range bound'
        ]
        
        self.negation_words = [
            'not', 'no', 'never', 'neither', 'nor', 'barely', 'hardly',
            'scarcely', 'seldom', 'rarely', 'without', 'lack', 'failed to'
        ]
        
        self.intensifiers = [
            'very', 'extremely', 'significantly', 'substantially', 'greatly',
            'highly', 'strongly', 'dramatically', 'sharply', 'massively',
            'huge', 'enormous', 'tremendous', 'exceptional', 'record'
        ]
        
        self.diminishers = [
            'slightly', 'somewhat', 'marginally', 'moderately', 'relatively',
            'minor', 'modest', 'small', 'limited', 'partial'
        ]
    
    def get_word_sentiment(self, word: str) -> str:
        word_lower = word.lower()
        if word_lower in self.positive_keywords:
            return 'positive'
        elif word_lower in self.negative_keywords:
            return 'negative'
        elif word_lower in self.neutral_keywords:
            return 'neutral'
        return 'unknown'
    
    def calculate_word_weight(self, word: str, context: List[str] = None) -> float:
        word_lower = word.lower()
        base_weight = 1.0
        
        if context:
            for ctx_word in context:
                ctx_lower = ctx_word.lower()
                if ctx_lower in self.intensifiers:
                    base_weight *= 1.5
                elif ctx_lower in self.diminishers:
                    base_weight *= 0.7
                elif ctx_lower in self.negation_words:
                    base_weight *= -1.0
        
        return base_weight


class SHAPModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def predict(self, texts: List[str]) -> np.ndarray:
        if not HAS_TRANSFORMERS:
            return self._fallback_predict(texts)
        
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            sentiment_scores = predictions[:, 2] - predictions[:, 0]
            return sentiment_scores
        except Exception as e:
            logger.warning(f"Model prediction failed, using fallback: {e}")
            return self._fallback_predict(texts)
    
    def _fallback_predict(self, texts: List[str]) -> np.ndarray:
        kw_dict = FinancialKeywordDictionary()
        scores = []
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            pos_count = sum(1 for w in words if w in kw_dict.positive_keywords)
            neg_count = sum(1 for w in words if w in kw_dict.negative_keywords)
            total = pos_count + neg_count
            
            if total > 0:
                score = (pos_count - neg_count) / total
            else:
                score = 0.0
            
            scores.append(score)
        
        return np.array(scores)


class KeywordSHAPExplainer:
    def __init__(self, model=None, tokenizer=None):
        self.kw_dict = FinancialKeywordDictionary()
        
        if model and tokenizer and HAS_SHAP:
            self.model_wrapper = SHAPModelWrapper(model, tokenizer)
            self.explainer = shap.Explainer(self.model_wrapper.predict)
            self.use_true_shap = True
        else:
            self.use_true_shap = False
            logger.info("Using keyword-based SHAP approximation")
        
        self.cache = {}
        self.max_cache_size = 10000
    
    def explain(self, news: NewsArticle, sentiment_result: SentimentResult) -> SHAPExplanation:
        cache_key = f"{news.id}_{hash(news.content)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.use_true_shap:
            explanation = self._true_shap_explain(news, sentiment_result)
        else:
            explanation = self._keyword_based_explain(news, sentiment_result)
        
        if len(self.cache) >= self.max_cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = explanation
        
        return explanation
    
    def _true_shap_explain(self, news: NewsArticle, 
                           sentiment_result: SentimentResult) -> SHAPExplanation:
        try:
            text = news.content
            shap_values = self.explainer([text])
            
            base_value = float(shap_values.base_values[0])
            output_value = float(shap_values.values[0].sum() + base_value)
            
            feature_importance = {}
            keyword_highlights = []
            
            tokens = self._tokenize(text)
            
            for i, token in enumerate(tokens):
                if i < len(shap_values.values[0]):
                    shap_val = float(shap_values.values[0][i])
                    feature_importance[token] = shap_val
                    
                    start_idx = text.lower().find(token)
                    if start_idx >= 0:
                        sentiment = 'positive' if shap_val > 0.01 else ('negative' if shap_val < -0.01 else 'neutral')
                        keyword_highlights.append(KeywordHighlight(
                            word=token,
                            start=start_idx,
                            end=start_idx + len(token),
                            shap_value=shap_val,
                            sentiment=sentiment
                        ))
            
            contributing_words = sorted(
                [(w, v) for w, v in feature_importance.items() if v > 0.01],
                key=lambda x: x[1], reverse=True
            )[:10]
            
            mitigating_words = sorted(
                [(w, v) for w, v in feature_importance.items() if v < -0.01],
                key=lambda x: x[1]
            )[:10]
            
            expected_sentiment = self._classify_sentiment(output_value)
            
            return SHAPExplanation(
                news_id=news.id,
                base_value=base_value,
                output_value=output_value,
                feature_importance=feature_importance,
                keyword_highlights=keyword_highlights,
                expected_sentiment=expected_sentiment,
                contributing_words=[w for w, _ in contributing_words],
                mitigating_words=[w for w, _ in mitigating_words]
            )
        
        except Exception as e:
            logger.warning(f"True SHAP explanation failed, falling back: {e}")
            return self._keyword_based_explain(news, sentiment_result)
    
    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                        'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
                        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                        'will', 'would', 'could', 'should', 'may', 'might', 'shall'])
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def _keyword_based_explain(self, news: NewsArticle, 
                                sentiment_result: SentimentResult) -> SHAPExplanation:
        text = news.content
        words = self._tokenize(text)
        
        feature_importance = {}
        keyword_highlights = []
        contributing_words = []
        mitigating_words = []
        
        for i, word in enumerate(words):
            context = words[max(0, i-3):i] + words[i+1:min(len(words), i+4)]
            sentiment = self.kw_dict.get_word_sentiment(word)
            
            if sentiment != 'unknown':
                weight = self.kw_dict.calculate_word_weight(word, context)
                shap_val = weight * 0.1
                
                if sentiment == 'positive':
                    shap_val = abs(shap_val)
                    contributing_words.append(word)
                elif sentiment == 'negative':
                    shap_val = -abs(shap_val)
                    mitigating_words.append(word)
                
                feature_importance[word] = shap_val
                
                start_idx = text.lower().find(word)
                if start_idx >= 0:
                    keyword_highlights.append(KeywordHighlight(
                        word=word,
                        start=start_idx,
                        end=start_idx + len(word),
                        shap_value=shap_val,
                        sentiment=sentiment
                    ))
        
        base_value = 0.0
        output_value = sum(feature_importance.values()) + base_value
        
        expected_sentiment = self._classify_sentiment(sentiment_result.sentiment_score)
        
        return SHAPExplanation(
            news_id=news.id,
            base_value=base_value,
            output_value=output_value,
            feature_importance=feature_importance,
            keyword_highlights=keyword_highlights,
            expected_sentiment=expected_sentiment,
            contributing_words=contributing_words[:10],
            mitigating_words=mitigating_words[:10]
        )
    
    def _classify_sentiment(self, score: float) -> str:
        if score >= 0.3:
            return 'positive'
        elif score <= -0.3:
            return 'negative'
        else:
            return 'neutral'


class SignalExplainer:
    def __init__(self, shap_explainer: Optional[KeywordSHAPExplainer] = None):
        self.shap_explainer = shap_explainer or KeywordSHAPExplainer()
        
        self.signal_templates = {
            'BUY': [
                "Strong positive sentiment detected with {n_pos} contributing factors. "
                "Key drivers include {drivers}. The signal has {confidence:.0%} confidence "
                "based on {correlation:.0%} price correlation and {sentiment:.2f} sentiment score.",
                
                "Bullish signal triggered by {n_pos} positive indicators. "
                "Top drivers: {drivers}. Sentiment score of {sentiment:.2f} exceeds "
                "threshold with {correlation:.0%} price correlation. Risk-adjusted "
                "strength: {strength:.2f}.",
                
                "Buy recommendation: {n_pos} positive signals identified. "
                "Key words driving sentiment: {drivers}. Current market conditions "
                "support entry with {confidence:.0%} confidence."
            ],
            'SELL': [
                "Strong negative sentiment detected with {n_neg} risk factors. "
                "Key concerns include {risk_factors}. Caution advised with {confidence:.0%} "
                "confidence. Sentiment score: {sentiment:.2f}.",
                
                "Bearish signal triggered by {n_neg} negative indicators. "
                "Top concerns: {risk_factors}. Sentiment score of {sentiment:.2f} "
                "below threshold. Consider reducing exposure.",
                
                "Sell recommendation: {n_neg} warning signs identified. "
                "Negative drivers: {risk_factors}. Market conditions suggest "
                "exit with {confidence:.0%} confidence."
            ],
            'HOLD': [
                "Neutral sentiment detected. No strong directional signal. "
                "Positive factors: {drivers}. Negative factors: {risk_factors}. "
                "Maintain current position.",
                
                "Mixed signals: {n_pos} positive and {n_neg} negative indicators. "
                "Sentiment score {sentiment:.2f} within neutral range. "
                "Monitor for clearer direction.",
                
                "Hold recommendation: insufficient conviction for directional trade. "
                "Key watchpoints: {drivers}. Confidence: {confidence:.0%}."
            ]
        }
    
    def explain_signal(self, signal: TradingSignal, news: NewsArticle,
                       sentiment_result: SentimentResult) -> SignalExplanation:
        shap_explanation = self.shap_explainer.explain(news, sentiment_result)
        
        key_drivers = self._identify_key_drivers(shap_explanation, signal)
        risk_factors = self._identify_risk_factors(shap_explanation, signal)
        
        summary = self._generate_summary(signal, shap_explanation, 
                                          key_drivers, risk_factors)
        
        signal_id = f"sig_{news.id}_{signal.timestamp.timestamp()}"
        
        return SignalExplanation(
            signal_id=signal_id,
            symbol=signal.symbol,
            signal=signal.signal,
            strength=signal.strength,
            confidence=signal.confidence,
            shap_explanation=shap_explanation,
            key_drivers=key_drivers,
            risk_factors=risk_factors,
            summary=summary,
            timestamp=datetime.now()
        )
    
    def _identify_key_drivers(self, shap_explanation: SHAPExplanation,
                               signal: TradingSignal) -> List[str]:
        drivers = []
        
        if shap_explanation.contributing_words:
            drivers.extend(shap_explanation.contributing_words[:5])
        
        if signal.price_correlation > 0.5:
            drivers.append(f"Strong price correlation ({signal.price_correlation:.2f})")
        
        if signal.strength > 0.7:
            drivers.append("High signal strength")
        
        if signal.confidence > 0.8:
            drivers.append("High model confidence")
        
        return drivers[:8]
    
    def _identify_risk_factors(self, shap_explanation: SHAPExplanation,
                                signal: TradingSignal) -> List[str]:
        risks = []
        
        if shap_explanation.mitigating_words:
            risks.extend(shap_explanation.mitigating_words[:5])
        
        if signal.price_correlation < 0.3:
            risks.append(f"Weak price correlation ({signal.price_correlation:.2f})")
        
        if signal.confidence < 0.5:
            risks.append("Low model confidence")
        
        if abs(signal.sentiment_score) < 0.3:
            risks.append("Near-neutral sentiment")
        
        return risks[:8]
    
    def _generate_summary(self, signal: TradingSignal,
                          shap_explanation: SHAPExplanation,
                          key_drivers: List[str],
                          risk_factors: List[str]) -> str:
        templates = self.signal_templates.get(signal.signal, self.signal_templates['HOLD'])
        template = templates[hash(signal.symbol) % len(templates)]
        
        context = {
            'n_pos': len(shap_explanation.contributing_words),
            'n_neg': len(shap_explanation.mitigating_words),
            'drivers': ', '.join(key_drivers[:3]) if key_drivers else 'no strong drivers',
            'risk_factors': ', '.join(risk_factors[:3]) if risk_factors else 'no major risks',
            'confidence': signal.confidence,
            'correlation': signal.price_correlation,
            'sentiment': signal.sentiment_score,
            'strength': signal.strength
        }
        
        try:
            return template.format(**context)
        except:
            return (f"{signal.signal} signal for {signal.symbol} with {signal.strength:.2f} strength. "
                   f"Sentiment: {signal.sentiment_score:.2f}, Confidence: {signal.confidence:.2f}")
    
    def highlight_text(self, text: str, highlights: List[KeywordHighlight]) -> str:
        if not highlights:
            return text
        
        sorted_highlights = sorted(highlights, key=lambda h: h.start, reverse=True)
        
        result = text
        for hl in sorted_highlights:
            if hl.start >= len(result):
                continue
            
            color_map = {
                'positive': '#22c55e',
                'negative': '#ef4444',
                'neutral': '#6b7280'
            }
            
            color = color_map.get(hl.sentiment, '#6b7280')
            word = result[hl.start:hl.end]
            
            highlighted = f'<span style="background-color: {color}20; border-bottom: 2px solid {color}; padding: 1px 3px; border-radius: 3px;" title="SHAP: {hl.shap_value:.3f}">{word}</span>'
            
            result = result[:hl.start] + highlighted + result[hl.end:]
        
        return result
    
    def get_feature_importance_plot_data(self, shap_explanation: SHAPExplanation,
                                          top_n: int = 20) -> Dict[str, Any]:
        features = sorted(
            shap_explanation.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        return {
            'features': [f for f, _ in features],
            'shap_values': [v for _, v in features],
            'colors': ['#22c55e' if v > 0 else '#ef4444' for _, v in features],
            'base_value': shap_explanation.base_value,
            'output_value': shap_explanation.output_value
        }
    
    def get_waterfall_plot_data(self, shap_explanation: SHAPExplanation,
                                 top_n: int = 10) -> Dict[str, Any]:
        features = sorted(
            shap_explanation.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        cumulative = [shap_explanation.base_value]
        current = shap_explanation.base_value
        
        for _, val in features:
            current += val
            cumulative.append(current)
        
        return {
            'features': ['Base'] + [f for f, _ in features] + ['Final'],
            'values': [shap_explanation.base_value] + [v for _, v in features] + [cumulative[-1]],
            'cumulative': cumulative + [cumulative[-1]],
            'colors': ['#6b7280'] + ['#22c55e' if v > 0 else '#ef4444' for _, v in features] + ['#3b82f6']
        }


class ExplanationCache:
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[SignalExplanation, float]] = {}
    
    def get(self, news_id: str) -> Optional[SignalExplanation]:
        if news_id in self.cache:
            explanation, timestamp = self.cache[news_id]
            if time.time() - timestamp < self.ttl_seconds:
                return explanation
            else:
                del self.cache[news_id]
        return None
    
    def put(self, explanation: SignalExplanation):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[explanation.shap_explanation.news_id] = (explanation, time.time())
    
    def clear(self):
        self.cache.clear()


import time
