import os
import sys
import time
import json
import threading
import websockets
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from shared.models import SentimentResult, PriceData, TradingSignal, OrderBook
from streaming.spark_processor import StreamDataStore, PythonStreamProcessor


class WebSocketClient:
    def __init__(self, data_store: StreamDataStore, processor: PythonStreamProcessor):
        self.data_store = data_store
        self.processor = processor
        self.running = False
    
    async def connect(self):
        uri = f"ws://{settings.WEBSOCKET_HOST}:{settings.WEBSOCKET_PORT}"
        print(f"Connecting to WebSocket: {uri}")
        
        while self.running:
            try:
                async with websockets.connect(uri) as websocket:
                    print("Connected to WebSocket server")
                    while self.running:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            if data['type'] == 'news':
                                self.processor.queue_news(data['data'])
                            elif data['type'] == 'price':
                                self.processor.queue_price(data['data'])
                            elif data['type'] == 'orderbook':
                                self.processor.queue_orderbook(data['data'])
                                
                        except websockets.exceptions.ConnectionClosed:
                            print("WebSocket connection closed, reconnecting...")
                            break
                        except Exception as e:
                            print(f"Error processing message: {e}")
                            break
                            
            except Exception as e:
                print(f"WebSocket connection failed: {e}, retrying in 5s...")
                await asyncio.sleep(5)
    
    def start(self):
        self.running = True
        threading.Thread(target=self._run_async, daemon=True).start()
    
    def _run_async(self):
        asyncio.run(self.connect())
    
    def stop(self):
        self.running = False


def plot_sentiment_timeseries(data_store: StreamDataStore, selected_symbols: List[str]):
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, symbol in enumerate(selected_symbols):
        window_aggs = list(data_store.window_aggregates.get(symbol, []))
        
        if window_aggs:
            timestamps = [agg.window_end for agg in window_aggs]
            sentiments = [agg.avg_sentiment for agg in window_aggs]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=sentiments,
                mode='lines+markers',
                name=symbol,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
    
    fig.add_hline(y=settings.SIGNAL_THRESHOLD_BUY, line_dash="dash", 
                  line_color="green", annotation_text="Buy Threshold")
    fig.add_hline(y=settings.SIGNAL_THRESHOLD_SELL, line_dash="dash", 
                  line_color="red", annotation_text="Sell Threshold")
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='Sentiment Score Time Series (5s Sliding Window)',
        xaxis_title='Time',
        yaxis_title='Average Sentiment Score',
        yaxis_range=[-1, 1],
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_signal_heatmap(data_store: StreamDataStore, selected_symbols: List[str]):
    all_signals = data_store.get_all_signals(limit=200)
    signals_by_symbol = {}
    
    for symbol in selected_symbols:
        symbol_signals = [s for s in all_signals if s.symbol == symbol]
        signals_by_symbol[symbol] = symbol_signals
    
    time_buckets = []
    now = datetime.now()
    for i in range(20):
        bucket_start = now - timedelta(minutes=(20 - i) * 2)
        time_buckets.append(bucket_start)
    
    heatmap_data = np.zeros((len(selected_symbols), len(time_buckets)))
    text_data = [[None for _ in range(len(time_buckets))] for _ in range(len(selected_symbols))]
    
    for i, symbol in enumerate(selected_symbols):
        for j, bucket_time in enumerate(time_buckets):
            next_bucket = bucket_time + timedelta(minutes=2)
            bucket_signals = [
                s for s in signals_by_symbol.get(symbol, [])
                if bucket_time <= s.timestamp < next_bucket
            ]
            
            if bucket_signals:
                avg_strength = np.mean([s.strength for s in bucket_signals])
                signal_types = [s.signal for s in bucket_signals]
                if 'BUY' in signal_types and 'SELL' in signal_types:
                    heatmap_data[i, j] = avg_strength if 'BUY' in signal_types else -avg_strength
                elif 'BUY' in signal_types:
                    heatmap_data[i, j] = avg_strength
                elif 'SELL' in signal_types:
                    heatmap_data[i, j] = -avg_strength
                
                text_data[i][j] = f"{len(bucket_signals)} signals<br>Avg: {avg_strength:.2f}"
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[tb.strftime("%H:%M") for tb in time_buckets],
        y=selected_symbols,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=text_data,
        hovertemplate='Symbol: %{y}<br>Time: %{x}<br>%{text}<extra></extra>',
        colorbar=dict(title='Signal Strength<br>(+Buy / -Sell)')
    ))
    
    fig.update_layout(
        title='Trading Signal Strength Heatmap',
        xaxis_title='Time Bucket (2 min intervals)',
        yaxis_title='Symbol',
        height=400
    )
    
    return fig


def plot_order_book_depth(data_store: StreamDataStore, symbol: str):
    order_book = data_store.get_order_book(symbol)
    
    if order_book is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for order book data...",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=400, title=f'Order Book Depth - {symbol}')
        return fig
    
    bids = pd.DataFrame(order_book.bids)
    asks = pd.DataFrame(order_book.asks)
    
    if len(bids) > 0:
        bids = bids.sort_values('price', ascending=False)
        bids['cum_size'] = bids['size'].cumsum()
    
    if len(asks) > 0:
        asks = asks.sort_values('price', ascending=True)
        asks['cum_size'] = asks['size'].cumsum()
    
    fig = go.Figure()
    
    if len(bids) > 0:
        fig.add_trace(go.Scatter(
            x=bids['cum_size'],
            y=bids['price'],
            mode='lines',
            fill='tozerox',
            name='Bids',
            line=dict(color='green', width=3),
            fillcolor='rgba(0, 255, 0, 0.2)'
        ))
    
    if len(asks) > 0:
        fig.add_trace(go.Scatter(
            x=asks['cum_size'],
            y=asks['price'],
            mode='lines',
            fill='tozerox',
            name='Asks',
            line=dict(color='red', width=3),
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
    
    if len(bids) > 0 and len(asks) > 0:
        best_bid = bids.iloc[0]['price']
        best_ask = asks.iloc[0]['price']
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        fig.add_hline(y=best_bid, line_dash="dot", line_color="green", 
                      annotation_text=f"Best Bid: ${best_bid:.2f}")
        fig.add_hline(y=best_ask, line_dash="dot", line_color="red", 
                      annotation_text=f"Best Ask: ${best_ask:.2f}")
        fig.add_hline(y=mid_price, line_dash="dash", line_color="blue", 
                      annotation_text=f"Mid: ${mid_price:.2f}")
    
    fig.update_layout(
        title=f'Order Book Depth - {symbol}',
        xaxis_title='Cumulative Size',
        yaxis_title='Price ($)',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_portfolio_performance(portfolio):
    fig = go.Figure()
    
    equity_curve = portfolio.equity_curve
    timestamps = [datetime.now() - timedelta(seconds=len(equity_curve) - i - 1) 
                  for i in range(len(equity_curve))]
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=equity_curve,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=3)
    ))
    
    fig.add_hline(y=portfolio.initial_cash, line_dash="dash", line_color="gray",
                  annotation_text="Initial Capital")
    
    total_return = (equity_curve[-1] - portfolio.initial_cash) / portfolio.initial_cash
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Total Return: {total_return*100:.2f}%",
        showarrow=False,
        font=dict(size=14),
        align="left"
    )
    
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Time',
        yaxis_title='Value ($)',
        height=300
    )
    
    return fig


def display_signals_table(data_store: StreamDataStore):
    signals = data_store.get_all_signals(limit=10)
    
    if not signals:
        st.info("No trading signals generated yet...")
        return
    
    signal_data = []
    for s in reversed(signals):
        signal_data.append({
            'Time': s.timestamp.strftime("%H:%M:%S"),
            'Symbol': s.symbol,
            'Signal': f"🔴 SELL" if s.signal == "SELL" else "🟢 BUY",
            'Strength': f"{s.strength:.3f}",
            'Confidence': f"{s.confidence:.1%}",
            'Sentiment': f"{s.sentiment_score:.3f}",
            'Correlation': f"{s.price_correlation:.3f}",
            'Reason': s.reason
        })
    
    df = pd.DataFrame(signal_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_statistics(data_store: StreamDataStore, portfolio):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("News Processed", f"{data_store.news_count:,}")
    
    with col2:
        all_signals = data_store.get_all_signals()
        st.metric("Signals Generated", len(all_signals))
    
    with col3:
        st.metric("Portfolio Value", f"${portfolio.total_equity():,.2f}")
    
    with col4:
        total_return = (portfolio.total_equity() - portfolio.initial_cash) / portfolio.initial_cash
        st.metric("Total Return", f"{total_return*100:.2f}%", 
                  delta=f"{total_return*100:.2f}%" if total_return != 0 else None)


def main():
    st.set_page_config(
        page_title="Financial News Sentiment Trading System",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Financial News Sentiment Trading System")
    st.markdown("Real-time sentiment analysis-driven trading signals")
    
    if 'data_store' not in st.session_state:
        st.session_state.data_store = StreamDataStore()
        st.session_state.processor = PythonStreamProcessor(st.session_state.data_store, use_mock_sentiment=True)
        st.session_state.processor.start()
        st.session_state.ws_client = WebSocketClient(st.session_state.data_store, st.session_state.processor)
        st.session_state.ws_client.start()
        st.success("Streaming system initialized!")
    
    data_store = st.session_state.data_store
    processor = st.session_state.processor
    
    from signal.signal_generator import SignalExecutor
    if 'executor' not in st.session_state:
        st.session_state.executor = SignalExecutor(data_store, initial_cash=100000.0)
    
    executor = st.session_state.executor
    
    with st.sidebar:
        st.header("Settings")
        
        selected_symbols = st.multiselect(
            "Select Symbols",
            settings.SYMBOLS,
            default=settings.SYMBOLS[:5]
        )
        
        selected_ob_symbol = st.selectbox(
            "Order Book Symbol",
            settings.SYMBOLS,
            index=0
        )
        
        st.divider()
        
        buy_threshold = st.slider("Buy Threshold", 0.0, 1.0, settings.SIGNAL_THRESHOLD_BUY, 0.05)
        sell_threshold = st.slider("Sell Threshold", -1.0, 0.0, settings.SIGNAL_THRESHOLD_SELL, 0.05)
        
        settings.SIGNAL_THRESHOLD_BUY = buy_threshold
        settings.SIGNAL_THRESHOLD_SELL = sell_threshold
        
        st.divider()
        
        if st.button("Run Backtest"):
            st.session_state.run_backtest = True
        
        st.divider()
        st.info(f"News rate: {settings.NEWS_GENERATION_RATE}/s")
        st.info(f"Window: {settings.WINDOW_DURATION}s sliding")
    
    display_statistics(data_store, executor.portfolio)
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Sentiment Analysis", 
        "🔥 Signal Heatmap", 
        "📖 Order Book",
        "💼 Portfolio"
    ])
    
    with tab1:
        fig_sentiment = plot_sentiment_timeseries(data_store, selected_symbols)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        st.subheader("Recent Trading Signals")
        display_signals_table(data_store)
    
    with tab2:
        fig_heatmap = plot_signal_heatmap(data_store, selected_symbols)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.caption("Green indicates buy signals, red indicates sell signals. Intensity shows signal strength.")
    
    with tab3:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_orderbook = plot_order_book_depth(data_store, selected_ob_symbol)
            st.plotly_chart(fig_orderbook, use_container_width=True)
        
        with col2:
            st.subheader("Market Data")
            
            for symbol in selected_symbols:
                prices = data_store.get_recent_prices(symbol, limit=5)
                if prices:
                    current = prices[-1]
                    prev = prices[0] if len(prices) > 1 else current
                    change = (current.price - prev.price) / prev.price if prev.price != 0 else 0
                    
                    st.metric(
                        f"{symbol}",
                        f"${current.price:.2f}",
                        f"{change*100:+.2f}%"
                    )
    
    with tab4:
        fig_portfolio = plot_portfolio_performance(executor.portfolio)
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Positions", len(executor.portfolio.positions))
        with col2:
            st.metric("Trades", len(executor.portfolio.trades))
        with col3:
            from shared.utils import calculate_sharpe_ratio
            sharpe = calculate_sharpe_ratio(executor.portfolio.returns, settings.RISK_FREE_RATE)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        with col4:
            from shared.utils import calculate_max_drawdown
            max_dd = calculate_max_drawdown(executor.portfolio.equity_curve)
            st.metric("Max Drawdown", f"{max_dd*100:.2f}%")
        
        if executor.portfolio.positions:
            st.subheader("Current Positions")
            pos_data = []
            for symbol, pos in executor.portfolio.positions.items():
                pos_data.append({
                    'Symbol': symbol,
                    'Entry Price': f"${pos.entry_price:.2f}",
                    'Current Price': f"${pos.current_price:.2f}",
                    'Quantity': f"{pos.quantity:.4f}",
                    'Value': f"${pos.current_price * pos.quantity:,.2f}",
                    'P&L': f"${pos.pnl:,.2f}",
                    'Return': f"{(pos.current_price - pos.entry_price)/pos.entry_price*100:+.2f}%"
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
    
    if st.session_state.get('run_backtest', False):
        st.session_state.run_backtest = False
        with st.spinner("Running backtest..."):
            from backtest.backtest_engine import BacktestEngine
            engine = BacktestEngine(use_mock_sentiment=True)
            result = engine.run_backtest("AAPL")
            
            st.subheader("📊 Backtest Results")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{result.total_return*100:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{result.max_drawdown*100:.2f}%")
            with col4:
                st.metric("Win Rate", f"{result.win_rate*100:.2f}%")
            
            st.dataframe(pd.DataFrame([result.to_dict()]), use_container_width=True, hide_index=True)
    
    executor.process_signals()
    
    time.sleep(settings.STREAMLIT_UPDATE_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
