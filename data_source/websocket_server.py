import asyncio
import websockets
import json
import random
import time
from datetime import datetime
from typing import List, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from shared.models import NewsArticle, PriceData, OrderBook
from shared.utils import generate_id, generate_symbol_price, generate_order_book


class NewsTemplatePool:
    POSITIVE_TEMPLATES = [
        "{symbol} reports record quarterly earnings, beating estimates by {pct}%",
        "{symbol} announces strategic partnership with tech giant",
        "{symbol} receives regulatory approval for new product",
        "Analysts upgrade {symbol} to strong buy with {pct}% upside",
        "{symbol} launches innovative AI-powered service",
        "{symbol} secures major contract worth ${amount}B",
        "{symbol} stock surges as demand exceeds expectations",
        "{symbol} expands into high-growth international markets",
        "{symbol} announces $10B share buyback program",
        "{symbol} CEO touts 'transformative' year ahead at earnings call"
    ]
    
    NEGATIVE_TEMPLATES = [
        "{symbol} misses earnings estimates by {pct}%",
        "{symbol} faces regulatory investigation into business practices",
        "Analysts downgrade {symbol} to sell citing valuation concerns",
        "{symbol} announces executive shakeup amid restructuring",
        "{symbol} product launch delayed due to quality issues",
        "{symbol} faces class-action lawsuit over {issue}",
        "{symbol} revenue declines {pct}% amid challenging environment",
        "{symbol} cuts full-year guidance amid macro headwinds",
        "{symbol} recalls key product over safety concerns",
        "Short interest in {symbol} surges to {pct}% of float"
    ]
    
    NEUTRAL_TEMPLATES = [
        "{symbol} appoints new CFO from industry leader",
        "{symbol} to present at investor conference next week",
        "{symbol} announces dividend of ${amount} per share",
        "{symbol} reports in-line earnings as expected",
        "{symbol} completes previously announced acquisition",
        "{symbol} expands board of directors with industry veteran",
        "{symbol} announces $500M green energy investment",
        "{symbol} provides business update at annual meeting",
        "{symbol} renews supply chain agreement through 2026",
        "{symbol} CEO to participate in fireside chat at tech summit"
    ]
    
    SOURCES = [
        "Bloomberg", "Reuters", "CNBC", "Wall Street Journal",
        "Financial Times", "MarketWatch", "Seeking Alpha",
        "Barron's", "Forbes", "Business Insider"
    ]
    
    ISSUES = [
        "accounting practices", "data privacy", "antitrust concerns",
        "product safety", "labor practices", "environmental impact"
    ]


class MockDataGenerator:
    def __init__(self):
        self.templates = NewsTemplatePool()
        self.price_cache = {sym: generate_symbol_price(sym) for sym in settings.SYMBOLS}
    
    def generate_news(self, symbol: str) -> NewsArticle:
        sentiment_choice = random.random()
        if sentiment_choice < 0.35:
            template = random.choice(self.templates.POSITIVE_TEMPLATES)
        elif sentiment_choice < 0.65:
            template = random.choice(self.templates.NEGATIVE_TEMPLATES)
        else:
            template = random.choice(self.templates.NEUTRAL_TEMPLATES)
        
        pct = random.uniform(5, 30)
        amount = random.uniform(1, 20)
        issue = random.choice(self.templates.ISSUES)
        
        content = template.format(
            symbol=symbol,
            pct=round(pct, 1),
            amount=round(amount, 2),
            issue=issue
        )
        
        return NewsArticle(
            id=generate_id(),
            symbol=symbol,
            title=content[:80] + "..." if len(content) > 80 else content,
            content=content,
            source=random.choice(self.templates.SOURCES),
            timestamp=datetime.now(),
            url=f"https://example.com/news/{generate_id()}"
        )
    
    def generate_price(self, symbol: str) -> PriceData:
        old_price = self.price_cache.get(symbol, generate_symbol_price(symbol))
        change = random.uniform(-0.005, 0.005)
        new_price = old_price * (1 + change)
        self.price_cache[symbol] = new_price
        
        return PriceData(
            symbol=symbol,
            price=round(new_price, 2),
            volume=random.uniform(10000, 1000000),
            timestamp=datetime.now(),
            bid=round(new_price * 0.9995, 2),
            ask=round(new_price * 1.0005, 2),
            bid_size=random.uniform(100, 5000),
            ask_size=random.uniform(100, 5000)
        )
    
    def generate_order_book(self, symbol: str) -> OrderBook:
        current_price = self.price_cache.get(symbol, generate_symbol_price(symbol))
        bids, asks = generate_order_book(symbol, current_price, levels=10)
        
        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )


class WebSocketNewsServer:
    def __init__(self):
        self.generator = MockDataGenerator()
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.news_count = 0
        self.start_time = time.time()
        self.running = True
    
    async def register_client(self, websocket):
        self.connected_clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.connected_clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.connected_clients)}")
    
    async def broadcast(self, message: str):
        if self.connected_clients:
            await asyncio.gather(
                *[client.send(message) for client in self.connected_clients],
                return_exceptions=True
            )
    
    async def generate_news_stream(self):
        target_rate = settings.NEWS_GENERATION_RATE
        interval = 1.0 / target_rate
        
        while self.running:
            batch_size = min(50, target_rate // 10)
            for _ in range(batch_size):
                symbol = random.choice(settings.SYMBOLS)
                news = self.generator.generate_news(symbol)
                await self.broadcast(json.dumps({
                    "type": "news",
                    "data": news.to_json()
                }))
                self.news_count += 1
            
            if self.news_count % 1000 == 0:
                elapsed = time.time() - self.start_time
                rate = self.news_count / elapsed if elapsed > 0 else 0
                print(f"Generated {self.news_count} news articles. Rate: {rate:.2f}/s")
            
            await asyncio.sleep(interval * batch_size)
    
    async def generate_price_stream(self):
        while self.running:
            for symbol in settings.SYMBOLS:
                price = self.generator.generate_price(symbol)
                await self.broadcast(json.dumps({
                    "type": "price",
                    "data": price.to_json()
                }))
                
                if random.random() < 0.1:
                    order_book = self.generator.generate_order_book(symbol)
                    await self.broadcast(json.dumps({
                        "type": "orderbook",
                        "data": order_book.to_json()
                    }))
            
            await asyncio.sleep(0.5)
    
    async def handle_client(self, websocket):
        await self.register_client(websocket)
    
    async def start(self):
        print(f"Starting WebSocket server on {settings.WEBSOCKET_HOST}:{settings.WEBSOCKET_PORT}")
        print(f"Target news generation rate: {settings.NEWS_GENERATION_RATE}/s")
        
        server = await websockets.serve(
            self.handle_client,
            settings.WEBSOCKET_HOST,
            settings.WEBSOCKET_PORT
        )
        
        await asyncio.gather(
            server.wait_closed(),
            self.generate_news_stream(),
            self.generate_price_stream()
        )


def main():
    server = WebSocketNewsServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        server.running = False


if __name__ == "__main__":
    main()
