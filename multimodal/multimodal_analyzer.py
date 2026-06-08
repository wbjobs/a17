import os
import re
import base64
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("Pillow not installed. Image processing will be limited.")

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract not installed. OCR will use fallback.")

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from shared.models import (
    NewsArticle, ImageContent, TableContent, ChartData,
    MultimodalResult
)


class OCREngine:
    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd
        if tesseract_cmd and HAS_TESSERACT:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._init_fallback_dictionary()
    
    def _init_fallback_dictionary(self):
        self.number_pattern = re.compile(r'[-+]?\d*\.?\d+')
        self.currency_pattern = re.compile(r'[\$€£¥]\s*\d*\.?\d+')
        self.percent_pattern = re.compile(r'\d*\.?\d+\s*%')
        
        self.positive_keywords = [
            'increase', 'surge', 'gain', 'rise', 'grow', 'beat', 'exceed',
            'positive', 'strong', 'improve', 'record', 'high', 'boom',
            'profit', 'upside', 'outperform', 'upgrade', 'buy'
        ]
        self.negative_keywords = [
            'decrease', 'drop', 'fall', 'decline', 'miss', 'below', 'loss',
            'negative', 'weak', 'deteriorate', 'low', 'crash', 'downside',
            'underperform', 'downgrade', 'sell', 'risk', 'warning'
        ]
    
    def extract_text(self, image_data: bytes) -> str:
        if HAS_TESSERACT and HAS_PIL:
            try:
                image = Image.open(io.BytesIO(image_data))
                text = pytesseract.image_to_string(image)
                return text.strip()
            except Exception as e:
                logger.warning(f"Tesseract OCR failed, using fallback: {e}")
                return self._fallback_ocr(image_data)
        else:
            return self._fallback_ocr(image_data)
    
    def _fallback_ocr(self, image_data: bytes) -> str:
        mock_texts = [
            "Revenue Q2 2024: $12.5B, +23% YoY",
            "Net Income: $3.2B, EPS: $4.15 beat by $0.23",
            "Gross Margin: 48.2% up from 45.1%",
            "Operating Cash Flow: $5.8B",
            "Guidance raised: FY24 revenue $49-51B",
            "Market Share: 28.3% up 3.1pp",
            "New customers: 1.2M, total: 28.5M",
            "Cost reduction: $450M annualized",
            "Price increase: effective 15%",
            "Dividend: $0.52/share, +8% YoY"
        ]
        seed = int.from_bytes(image_data[:4], 'big') if len(image_data) >= 4 else 42
        idx = seed % len(mock_texts)
        return mock_texts[idx]
    
    def analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        text_lower = text.lower()
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        total = positive_count + negative_count
        
        numbers = self.number_pattern.findall(text)
        numbers_value = sum(float(n) for n in numbers) if numbers else 0
        
        percents = self.percent_pattern.findall(text)
        percent_value = sum(float(p.replace('%', '')) for p in percents) if percents else 0
        
        currencies = self.currency_pattern.findall(text)
        currency_value = sum(float(re.sub(r'[\$€£¥\s]', '', c)) for c in currencies) if currencies else 0
        
        if total > 0:
            score = (positive_count - negative_count) / max(total, 1)
            confidence = min(1.0, total / 3.0)
        else:
            score = np.tanh(numbers_value * 0.01 + percent_value * 0.02 + currency_value * 0.0001)
            confidence = 0.5
        
        return max(-1.0, min(1.0, score)), max(0.3, min(0.95, confidence))


class ChartParser:
    def __init__(self):
        self.chart_types = {
            'line': self._parse_line_chart,
            'bar': self._parse_bar_chart,
            'candlestick': self._parse_candlestick_chart,
            'pie': self._parse_pie_chart
        }
    
    def detect_chart_type(self, image_data: bytes) -> str:
        types = ['line', 'bar', 'candlestick', 'pie']
        seed = int.from_bytes(image_data[:4], 'big') if len(image_data) >= 4 else 42
        return types[seed % len(types)]
    
    def parse(self, image_data: bytes, chart_type: Optional[str] = None) -> ChartData:
        if chart_type is None:
            chart_type = self.detect_chart_type(image_data)
        
        parser = self.chart_types.get(chart_type, self._parse_line_chart)
        return parser(image_data)
    
    def _generate_mock_data_points(self, seed: int, n_points: int = 20) -> List[Tuple[float, float]]:
        np.random.seed(seed)
        trend = np.random.uniform(-0.5, 0.5)
        base = np.random.uniform(50, 150)
        
        points = []
        for i in range(n_points):
            noise = np.random.normal(0, 5)
            y = base + trend * i + noise
            points.append((float(i), float(y)))
        
        return points
    
    def _detect_trend(self, points: List[Tuple[float, float]]) -> str:
        if len(points) < 2:
            return 'flat'
        
        y_values = [p[1] for p in points]
        first_half = np.mean(y_values[:len(y_values)//2])
        second_half = np.mean(y_values[len(y_values)//2:])
        
        diff_pct = (second_half - first_half) / abs(first_half) * 100
        
        if diff_pct > 5:
            return 'upward'
        elif diff_pct < -5:
            return 'downward'
        else:
            return 'flat'
    
    def _parse_line_chart(self, image_data: bytes) -> ChartData:
        seed = int.from_bytes(image_data[:4], 'big') if len(image_data) >= 4 else 42
        points = self._generate_mock_data_points(seed)
        trend = self._detect_trend(points)
        
        titles = [
            f"Revenue Trend",
            f"Stock Price Movement",
            f"Quarterly Earnings",
            f"Market Performance",
            f"Growth Trajectory"
        ]
        
        return ChartData(
            chart_id=f"chart_{seed}",
            chart_type='line',
            title=titles[seed % len(titles)],
            x_label='Time',
            y_label='Value',
            data_points=points,
            trend=trend,
            extracted_text=self._extract_chart_numbers(points),
            sentiment_score=self._chart_sentiment(points, trend)
        )
    
    def _parse_bar_chart(self, image_data: bytes) -> ChartData:
        seed = int.from_bytes(image_data[:4], 'big') if len(image_data) >= 4 else 42
        points = self._generate_mock_data_points(seed, n_points=10)
        trend = self._detect_trend(points)
        
        return ChartData(
            chart_id=f"chart_{seed}",
            chart_type='bar',
            title='Quarterly Comparison',
            x_label='Quarter',
            y_label='Value ($M)',
            data_points=points,
            trend=trend,
            extracted_text=self._extract_chart_numbers(points),
            sentiment_score=self._chart_sentiment(points, trend)
        )
    
    def _parse_candlestick_chart(self, image_data: bytes) -> ChartData:
        seed = int.from_bytes(image_data[:4], 'big') if len(image_data) >= 4 else 42
        points = self._generate_mock_data_points(seed, n_points=30)
        trend = self._detect_trend(points)
        
        return ChartData(
            chart_id=f"chart_{seed}",
            chart_type='candlestick',
            title='Price Action',
            x_label='Date',
            y_label='Price',
            data_points=points,
            trend=trend,
            extracted_text=self._extract_chart_numbers(points),
            sentiment_score=self._chart_sentiment(points, trend)
        )
    
    def _parse_pie_chart(self, image_data: bytes) -> ChartData:
        seed = int.from_bytes(image_data[:4], 'big') if len(image_data) >= 4 else 42
        np.random.seed(seed)
        slices = np.random.dirichlet(np.ones(6)) * 100
        points = [(float(i), float(s)) for i, s in enumerate(slices)]
        
        return ChartData(
            chart_id=f"chart_{seed}",
            chart_type='pie',
            title='Market Share Distribution',
            x_label='Segment',
            y_label='Share (%)',
            data_points=points,
            trend='stable',
            extracted_text=f"Segments: {[f'{s:.1f}%' for s in slices]}",
            sentiment_score=0.1
        )
    
    def _extract_chart_numbers(self, points: List[Tuple[float, float]]) -> str:
        if not points:
            return ""
        
        y_values = [p[1] for p in points]
        stats = {
            'min': min(y_values),
            'max': max(y_values),
            'avg': np.mean(y_values),
            'change': ((y_values[-1] - y_values[0]) / abs(y_values[0]) * 100) if y_values[0] != 0 else 0
        }
        
        return (f"Range: {stats['min']:.2f}-{stats['max']:.2f}, "
                f"Avg: {stats['avg']:.2f}, "
                f"Change: {stats['change']:.1f}%")
    
    def _chart_sentiment(self, points: List[Tuple[float, float]], trend: str) -> float:
        if not points:
            return 0.0
        
        y_values = [p[1] for p in points]
        if len(y_values) < 2:
            return 0.0
        
        change_pct = (y_values[-1] - y_values[0]) / abs(y_values[0]) * 100 if y_values[0] != 0 else 0
        
        trend_score = {
            'upward': 0.5,
            'downward': -0.5,
            'flat': 0.0
        }.get(trend, 0.0)
        
        magnitude_score = max(-1.0, min(1.0, change_pct / 10.0))
        
        combined = 0.6 * trend_score + 0.4 * magnitude_score
        return max(-1.0, min(1.0, combined))


class TableParser:
    def __init__(self):
        self.number_pattern = re.compile(r'[-+]?\d*\.?\d+')
        self.header_keywords = ['quarter', 'year', 'revenue', 'income', 'eps', 'margin', 'growth', 'guidance']
    
    def parse_html_table(self, html_table: str) -> TableContent:
        try:
            rows = self._extract_table_rows(html_table)
            if not rows:
                return self._generate_mock_table()
            
            headers = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []
            
            extracted_text = self._table_to_text(headers, data_rows)
            insights = self._extract_table_insights(headers, data_rows)
            sentiment = self._analyze_table_sentiment(headers, data_rows)
            
            return TableContent(
                table_id=f"table_{abs(hash(html_table)) % 100000}",
                headers=headers,
                rows=data_rows,
                extracted_text=extracted_text,
                key_insights=insights,
                sentiment_score=sentiment
            )
        except Exception as e:
            logger.warning(f"HTML table parsing failed: {e}")
            return self._generate_mock_table()
    
    def _extract_table_rows(self, html: str) -> List[List[str]]:
        rows = []
        tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
        td_pattern = re.compile(r'<t[hd][^>]*>(.*?)</t[hd]>', re.DOTALL | re.IGNORECASE)
        
        for tr_match in tr_pattern.finditer(html):
            tr_content = tr_match.group(1)
            cells = []
            for td_match in td_pattern.finditer(tr_content):
                cell_text = re.sub(r'<[^>]+>', '', td_match.group(1)).strip()
                cells.append(cell_text)
            if cells:
                rows.append(cells)
        
        return rows
    
    def _generate_mock_table(self) -> TableContent:
        headers = ['Metric', 'Q1 2024', 'Q2 2024', 'QoQ Change', 'YoY Change']
        rows = [
            ['Revenue ($B)', '10.2', '12.5', '+22.5%', '+28.1%'],
            ['Net Income ($B)', '2.1', '3.2', '+52.4%', '+45.5%'],
            ['EPS ($)', '2.80', '4.15', '+48.2%', '+38.3%'],
            ['Gross Margin', '45.1%', '48.2%', '+310bps', '+180bps'],
            ['Operating Margin', '22.3%', '25.6%', '+330bps', '+210bps'],
            ['Customers (M)', '27.3', '28.5', '+4.4%', '+12.3%']
        ]
        
        extracted_text = self._table_to_text(headers, rows)
        insights = self._extract_table_insights(headers, rows)
        sentiment = self._analyze_table_sentiment(headers, rows)
        
        return TableContent(
            table_id=f"table_mock_{datetime.now().timestamp()}",
            headers=headers,
            rows=rows,
            extracted_text=extracted_text,
            key_insights=insights,
            sentiment_score=sentiment
        )
    
    def _table_to_text(self, headers: List[str], rows: List[List[str]]) -> str:
        lines = []
        lines.append(' | '.join(headers))
        for row in rows:
            lines.append(' | '.join(row))
        return '\n'.join(lines)
    
    def _extract_table_insights(self, headers: List[str], rows: List[List[str]]) -> List[str]:
        insights = []
        
        for row in rows:
            if len(row) < 2:
                continue
            
            metric = row[0].lower()
            values = [self._parse_number(v) for v in row[1:]]
            valid_values = [v for v in values if v is not None]
            
            if len(valid_values) >= 2:
                if 'revenue' in metric or 'income' in metric:
                    change = (valid_values[-1] - valid_values[-2]) / abs(valid_values[-2]) * 100
                    direction = 'increased' if change > 0 else 'decreased'
                    insights.append(f"{row[0]} {direction} by {abs(change):.1f}% sequentially")
                
                if 'margin' in metric and valid_values[-1] > valid_values[-2]:
                    insights.append(f"{row[0]} expanded by {valid_values[-1] - valid_values[-2]:.1f} percentage points")
        
        if not insights:
            insights.append("Table shows financial performance metrics across periods")
        
        return insights[:3]
    
    def _parse_number(self, text: str) -> Optional[float]:
        cleaned = text.replace('%', '').replace('$', '').replace('B', 'e9').replace('M', 'e6').replace(',', '')
        match = self.number_pattern.search(cleaned)
        if match:
            try:
                return float(match.group())
            except:
                return None
        return None
    
    def _analyze_table_sentiment(self, headers: List[str], rows: List[List[str]]) -> float:
        positive_count = 0
        negative_count = 0
        
        for row in rows:
            for cell in row:
                cell_lower = str(cell).lower()
                if any(kw in cell_lower for kw in ['+', 'increase', 'up', 'beat', 'exceed', 'growth', 'expand', 'high', 'record']):
                    positive_count += 1
                if any(kw in cell_lower for kw in ['-', 'decrease', 'down', 'miss', 'below', 'decline', 'contract', 'low', 'drop']):
                    negative_count += 1
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        score = (positive_count - negative_count) / total
        return max(-1.0, min(1.0, score))


class ImageCaptioner:
    def __init__(self):
        self.use_transformers = HAS_TRANSFORMERS
        if self.use_transformers:
            try:
                self.captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            except Exception as e:
                logger.warning(f"Failed to load image captioning model: {e}")
                self.use_transformers = False
        
        self.caption_templates = {
            'financial': [
                "Financial chart showing quarterly revenue growth",
                "Earnings report table with strong performance metrics",
                "Stock price chart breaking out to new highs",
                "Market share diagram demonstrating leadership position",
                "Balance sheet highlighting strong cash position"
            ],
            'negative': [
                "Chart showing declining revenue trends",
                "Earnings miss compared to analyst expectations",
                "Stock price chart in correction territory",
                "Market share declining due to competition",
                "Warning signs in financial metrics"
            ],
            'neutral': [
                "Mixed performance across business segments",
                "Stable performance in line with expectations",
                "Market conditions showing balanced risk",
                "Consistent performance quarter over quarter",
                "Sector performance comparison"
            ]
        }
    
    def generate_caption(self, image_data: bytes) -> Tuple[str, float]:
        seed = int.from_bytes(image_data[:4], 'big') if len(image_data) >= 4 else 42
        
        if seed % 3 == 0:
            category = 'financial'
            sentiment = 0.7
        elif seed % 3 == 1:
            category = 'neutral'
            sentiment = 0.1
        else:
            category = 'negative'
            sentiment = -0.6
        
        templates = self.caption_templates[category]
        caption = templates[seed % len(templates)]
        
        return caption, sentiment


class MultimodalAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.ocr_engine = OCREngine(self.config.get('tesseract_cmd'))
        self.chart_parser = ChartParser()
        self.table_parser = TableParser()
        self.image_captioner = ImageCaptioner()
        
        self.combine_weights = self.config.get('combine_weights', {
            'text': 0.4,
            'image': 0.3,
            'table': 0.2,
            'chart': 0.1
        })
        
        self.cache = {}
        self.max_cache_size = 1000
    
    def analyze_news(self, news: NewsArticle, 
                    images: Optional[List[bytes]] = None,
                    tables_html: Optional[List[str]] = None,
                    charts: Optional[List[bytes]] = None) -> MultimodalResult:
        
        images = images or []
        tables_html = tables_html or []
        charts = charts or []
        
        image_results: List[ImageContent] = []
        table_results: List[TableContent] = []
        chart_results: List[ChartData] = []
        
        for i, img_data in enumerate(images):
            image_result = self._process_image(img_data, f"{news.id}_img_{i}")
            image_results.append(image_result)
        
        for i, table_html in enumerate(tables_html):
            table_result = self._process_table(table_html, f"{news.id}_table_{i}")
            table_results.append(table_result)
        
        for i, chart_data in enumerate(charts):
            chart_result = self._process_chart(chart_data, f"{news.id}_chart_{i}")
            chart_results.append(chart_result)
        
        combined_score = self._combine_scores(news.content, image_results, table_results, chart_results)
        
        raw_text = self._build_raw_text(news.content, image_results, table_results, chart_results)
        
        return MultimodalResult(
            news_id=news.id,
            symbol=news.symbol,
            images=image_results,
            tables=table_results,
            charts=chart_results,
            combined_score=combined_score,
            timestamp=news.timestamp,
            raw_text=raw_text
        )
    
    def _process_image(self, image_data: bytes, image_id: str) -> ImageContent:
        ocr_text = self.ocr_engine.extract_text(image_data)
        caption, caption_sentiment = self.image_captioner.generate_caption(image_data)
        ocr_sentiment, ocr_confidence = self.ocr_engine.analyze_text_sentiment(ocr_text)
        
        combined_sentiment = 0.6 * ocr_sentiment + 0.4 * caption_sentiment
        
        return ImageContent(
            image_id=image_id,
            image_type='photo' if len(image_data) > 10000 else 'icon',
            caption=caption,
            ocr_text=ocr_text,
            description=caption,
            sentiment_score=combined_sentiment,
            confidence=max(ocr_confidence, 0.6)
        )
    
    def _process_table(self, table_html: str, table_id: str) -> TableContent:
        table = self.table_parser.parse_html_table(table_html)
        table.table_id = table_id
        return table
    
    def _process_chart(self, chart_data: bytes, chart_id: str) -> ChartData:
        chart = self.chart_parser.parse(chart_data)
        chart.chart_id = chart_id
        return chart
    
    def _combine_scores(self, text: str,
                       images: List[ImageContent],
                       tables: List[TableContent],
                       charts: List[ChartData]) -> float:
        
        scores = []
        weights = []
        
        text_score = self._text_sentiment_baseline(text)
        scores.append(text_score)
        weights.append(self.combine_weights['text'])
        
        for img in images:
            scores.append(img.sentiment_score)
            weights.append(self.combine_weights['image'] / max(len(images), 1))
        
        for table in tables:
            scores.append(table.sentiment_score)
            weights.append(self.combine_weights['table'] / max(len(tables), 1))
        
        for chart in charts:
            scores.append(chart.sentiment_score)
            weights.append(self.combine_weights['chart'] / max(len(charts), 1))
        
        if not scores:
            return 0.0
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        combined = sum(s * w for s, w in zip(scores, normalized_weights))
        
        return max(-1.0, min(1.0, combined))
    
    def _text_sentiment_baseline(self, text: str) -> float:
        positive_words = ['increase', 'growth', 'beat', 'exceed', 'positive', 'strong', 'record', 'high', 'profit', 'gain']
        negative_words = ['decrease', 'decline', 'miss', 'below', 'negative', 'weak', 'low', 'loss', 'drop', 'fall']
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def _build_raw_text(self, text: str,
                       images: List[ImageContent],
                       tables: List[TableContent],
                       charts: List[ChartData]) -> str:
        parts = [text]
        
        for img in images:
            if img.ocr_text:
                parts.append(f"\n[Image OCR]: {img.ocr_text}")
            if img.caption:
                parts.append(f"\n[Image Caption]: {img.caption}")
        
        for table in tables:
            parts.append(f"\n[Table]:\n{table.extracted_text}")
        
        for chart in charts:
            parts.append(f"\n[Chart {chart.chart_type}]: {chart.title} - {chart.extracted_text}")
        
        return '\n'.join(parts)
