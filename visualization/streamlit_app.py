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
from shared.models import (
    SentimentResult, PriceData, TradingSignal, OrderBook,
    RiskDashboard, SignalExplanation, AdaptiveThreshold, VolatilityState
)
from streaming.spark_processor import StreamDataStore, PythonStreamProcessor
from visualization.render_optimizer import (
    RenderThrottler, IncrementalHeatmapData, 
    DataDownsampler, SmartDeltaUpdater,
    FrameRateLimiter, optimize_plotly_figure
)
from risk.risk_dashboard import RiskDashboardEngine
from signal.enhanced_signal_generator import EnhancedSignalGenerator


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


def plot_sentiment_timeseries(data_store: StreamDataStore, selected_symbols: List[str], max_points: int = 100):
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, symbol in enumerate(selected_symbols):
        window_aggs = list(data_store.window_aggregates.get(symbol, []))
        
        if window_aggs:
            timestamps = [agg.window_end.timestamp() for agg in window_aggs]
            sentiments = [agg.avg_sentiment for agg in window_aggs]
            
            if len(timestamps) > max_points:
                timestamps, sentiments = DataDownsampler.downsample_time_series(
                    timestamps, sentiments, target_points=max_points
                )
            
            timestamps_dt = [datetime.fromtimestamp(ts) for ts in timestamps]
            
            fig.add_trace(go.Scattergl(
                x=timestamps_dt,
                y=sentiments,
                mode='lines',
                name=symbol,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"{symbol}: " + "%{y:.3f}<extra></extra>"
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision='constant'
    )
    
    return fig


def plot_signal_heatmap(
    data_store: StreamDataStore,
    selected_symbols: List[str],
    incremental_data: IncrementalHeatmapData = None,
    throttler: RenderThrottler = None
):
    if throttler and not throttler.should_render():
        return None
    
    if incremental_data is not None:
        all_signals = data_store.get_all_signals(limit=200)
        for signal in all_signals:
            if signal.symbol in selected_symbols:
                value = signal.strength if signal.signal == 'BUY' else -signal.strength
                incremental_data.update(
                    signal.symbol,
                    signal.timestamp.timestamp(),
                    value,
                    {'count': 1}
                )
        
        if not incremental_data.needs_update():
            return None
        
        heatmap_data, symbols, buckets, dirty, text_data = incremental_data.get_heatmap_data()
        
        if heatmap_data.shape[0] == 0 or heatmap_data.shape[1] == 0:
            fig = go.Figure()
            fig.add_annotation(text="Waiting for signal data...", showarrow=False, font=dict(size=16))
            fig.update_layout(height=400, title='Trading Signal Strength Heatmap')
            return fig
        
        if heatmap_data.shape[1] > 15:
            heatmap_data, _ = DataDownsampler.downsample_heatmap(heatmap_data, target_buckets=15)
            buckets = buckets[::max(1, len(buckets) // 15)]
            text_data = [row[::max(1, len(row) // 15)] for row in text_data]
    
    else:
        all_signals = data_store.get_all_signals(limit=100)
        signals_by_symbol = {}
        
        for symbol in selected_symbols:
            symbol_signals = [s for s in all_signals if s.symbol == symbol]
            signals_by_symbol[symbol] = symbol_signals
        
        time_buckets = []
        now = datetime.now()
        for i in range(15):
            bucket_start = now - timedelta(minutes=(15 - i) * 2)
            time_buckets.append(bucket_start)
        
        heatmap_data = np.zeros((len(selected_symbols), len(time_buckets)))
        text_data = [['' for _ in range(len(time_buckets))] for _ in range(len(selected_symbols))]
        buckets = time_buckets
        symbols = selected_symbols
        
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
    
    x_labels = [datetime.fromtimestamp(b).strftime("%H:%M") if isinstance(b, (int, float)) else b.strftime("%H:%M") for b in buckets]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=x_labels,
        y=symbols,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=text_data,
        hovertemplate='Symbol: %{y}<br>Time: %{x}<br>%{text}<extra></extra>',
        colorbar=dict(title='Signal Strength<br>(+Buy / -Sell)'),
        showscale=True
    ))
    
    fig.update_layout(
        title='Trading Signal Strength Heatmap',
        xaxis_title='Time Bucket (2 min intervals)',
        yaxis_title='Symbol',
        height=400,
        uirevision='constant'
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
        
        st.session_state.heatmap_throttler = RenderThrottler(max_fps=2.0)
        st.session_state.chart_throttler = RenderThrottler(max_fps=2.0)
        st.session_state.incremental_heatmap = IncrementalHeatmapData(max_time_buckets=15, max_symbols=10)
        st.session_state.delta_updater = SmartDeltaUpdater(tolerance=0.01)
        st.session_state.fps_limiter = FrameRateLimiter(target_fps=2.0)
        st.session_state._last_heatmap_fig = None
        
        st.success("Streaming system initialized!")
    
    data_store = st.session_state.data_store
    processor = st.session_state.processor
    
    heatmap_throttler = st.session_state.heatmap_throttler
    chart_throttler = st.session_state.chart_throttler
    incremental_heatmap = st.session_state.incremental_heatmap
    fps_limiter = st.session_state.fps_limiter
    
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
        fig_heatmap = plot_signal_heatmap(
            data_store, selected_symbols,
            incremental_data=incremental_heatmap,
            throttler=heatmap_throttler
        )
        
        if fig_heatmap is not None:
            st.session_state._last_heatmap_fig = fig_heatmap
            st.plotly_chart(fig_heatmap, use_container_width=True)
        elif st.session_state._last_heatmap_fig is not None:
            st.plotly_chart(st.session_state._last_heatmap_fig, use_container_width=True)
        else:
            fig = go.Figure()
            fig.add_annotation(text="Waiting for signal data...", showarrow=False, font=dict(size=16))
            fig.update_layout(height=400, title='Trading Signal Strength Heatmap')
            st.plotly_chart(fig, use_container_width=True)
        
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
    
    actual_fps = fps_limiter.wait_for_next_frame()
    
    time.sleep(max(0, settings.STREAMLIT_UPDATE_INTERVAL - fps_limiter.min_frame_time))
    st.rerun()


def plot_risk_dashboard(risk_data: Dict, throttler: Optional[RenderThrottler] = None) -> Optional[go.Figure]:
    if throttler and not throttler.should_render():
        return None
    
    if not risk_data or 'dashboard' not in risk_data:
        return None
    
    dashboard = risk_data['dashboard']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'VaR Breakdown', 'Drawdown History', 'Exposure by Sector',
            'Stress Test Losses', 'Risk Score Gauge', 'Exposure by Symbol'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'xy'}, {'type': 'pie'}],
            [{'type': 'bar'}, {'type': 'indicator'}, {'type': 'bar'}]
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    var_data = dashboard.var
    var_labels = ['VaR 95%', 'VaR 99%', 'VaR 99.9%', 'CVaR 95%', 'CVaR 99%']
    var_values = [var_data.var_95, var_data.var_99, var_data.var_99_9, var_data.cvar_95, var_data.cvar_99]
    var_colors = ['#fbbf24', '#f97316', '#ef4444', '#dc2626', '#b91c1c']
    
    fig.add_trace(
        go.Bar(x=var_labels, y=var_values, marker_color=var_colors, name='VaR'),
        row=1, col=1
    )
    
    fig.add_annotation(
        text=f"Portfolio Value: ${var_data.position_value:,.0f}",
        xref="x domain", yref="y domain",
        x=0.5, y=1.15, showarrow=False,
        row=1, col=1
    )
    
    if hasattr(dashboard, 'current_drawdown'):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=dashboard.current_drawdown * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Current Drawdown (%)"},
                delta={'reference': dashboard.max_drawdown * 100, 'relative': False},
                gauge={
                    'axis': {'range': [None, 30]},
                    'bar': {'color': "#ef4444"},
                    'steps': [
                        {'range': [0, 5], 'color': "#22c55e"},
                        {'range': [5, 15], 'color': "#f59e0b"},
                        {'range': [15, 30], 'color': "#ef4444"}
                    ]
                }
            ),
            row=1, col=2
        )
    
    if dashboard.exposure_by_sector:
        sectors = list(dashboard.exposure_by_sector.keys())
        values = [v * 100 for v in dashboard.exposure_by_sector.values()]
        fig.add_trace(
            go.Pie(labels=sectors, values=values, hole=0.4, name='Sector'),
            row=1, col=3
        )
    
    if dashboard.stress_test_result:
        scenarios = list(dashboard.stress_test_result.keys())
        losses = [abs(v) for v in dashboard.stress_test_result.values()]
        scenario_labels = [s.replace('_', ' ').title() for s in scenarios]
        
        fig.add_trace(
            go.Bar(x=scenario_labels, y=losses, marker_color='#dc2626', name='Loss'),
            row=2, col=1
        )
    
    if hasattr(dashboard, 'risk_score'):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=dashboard.risk_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score (0-100)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 30], 'color': "#ef4444"},
                        {'range': [30, 70], 'color': "#f59e0b"},
                        {'range': [70, 100], 'color': "#22c55e"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ),
            row=2, col=2
        )
    
    if dashboard.exposure_by_symbol:
        symbols = [e.symbol for e in dashboard.exposure_by_symbol[:10]]
        weights = [e.weight * 100 for e in dashboard.exposure_by_symbol[:10]]
        bar_colors = ['#3b82f6' if w > 0 else '#ef4444' for w in weights]
        
        fig.add_trace(
            go.Bar(x=symbols, y=weights, marker_color=bar_colors, name='Weight %'),
            row=2, col=3
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Risk Dashboard - Real-time Monitoring",
        title_x=0.5,
        title_font=dict(size=20)
    )
    
    return optimize_plotly_figure(fig)


def plot_threshold_evolution(threshold_history: List[AdaptiveThreshold],
                             volatility_history: List[VolatilityState],
                             throttler: Optional[RenderThrottler] = None) -> Optional[go.Figure]:
    if throttler and not throttler.should_render():
        return None
    
    if not threshold_history:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Adaptive Thresholds', 'Volatility Regime'),
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    timestamps = [t.timestamp for t in threshold_history]
    buy_thresholds = [t.buy_threshold for t in threshold_history]
    sell_thresholds = [t.sell_threshold for t in threshold_history]
    base_buy = [t.base_buy_threshold for t in threshold_history]
    base_sell = [t.base_sell_threshold for t in threshold_history]
    
    fig.add_trace(
        go.Scattergl(x=timestamps, y=buy_thresholds, mode='lines',
                     name='Buy Threshold', line=dict(color='#22c55e', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(x=timestamps, y=sell_thresholds, mode='lines',
                     name='Sell Threshold', line=dict(color='#ef4444', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(x=timestamps, y=base_buy, mode='lines',
                     name='Base Buy', line=dict(color='#22c55e', width=1, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(x=timestamps, y=base_sell, mode='lines',
                     name='Base Sell', line=dict(color='#ef4444', width=1, dash='dash')),
        row=1, col=1
    )
    
    if volatility_history:
        vol_timestamps = [v.timestamp for v in volatility_history]
        vol_values = [v.current_volatility * 100 for v in volatility_history]
        regimes = [v.regime for v in volatility_history]
        
        regime_colors = {'low_vol': '#22c55e', 'normal': '#3b82f6', 'high_vol': '#f59e0b', 'extreme_vol': '#ef4444'}
        scatter_colors = [regime_colors.get(r, '#6b7280') for r in regimes]
        
        fig.add_trace(
            go.Scattergl(x=vol_timestamps, y=vol_values, mode='lines+markers',
                         name='Volatility %', line=dict(color='#f59e0b', width=2),
                         marker=dict(color=scatter_colors, size=6)),
            row=2, col=1
        )
        
        for regime, color in regime_colors.items():
            fig.add_trace(
                go.Scattergl(x=[None], y=[None], mode='markers',
                             name=regime.replace('_', ' ').title(),
                             marker=dict(color=color, size=8),
                             showlegend=True),
                row=2, col=1
            )
    
    fig.update_layout(
        height=500,
        xaxis_title='Time',
        yaxis_title='Threshold',
        yaxis2_title='Volatility %',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        title_text="Adaptive Thresholds & Volatility Regime",
        title_x=0.5
    )
    
    return optimize_plotly_figure(fig)


def display_signal_explanation(explanation: SignalExplanation, news_content: str):
    if not explanation:
        return
    
    st.markdown(f"### 🎯 Signal: {explanation.signal}")
    st.markdown(f"**Strength:** {explanation.strength:.2f} | **Confidence:** {explanation.confidence:.1%}")
    
    st.info(f"📝 {explanation.summary}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Key Drivers")
        for driver in explanation.key_drivers[:5]:
            st.markdown(f"- {driver}")
    
    with col2:
        st.markdown("#### ⚠️ Risk Factors")
        for risk in explanation.risk_factors[:5]:
            st.markdown(f"- {risk}")
    
    st.markdown("#### 🔍 Highlighted News Text")
    shap = explanation.shap_explanation
    
    if shap.keyword_highlights:
        from explanation.shap_explainer import SignalExplainer
        explainer = SignalExplainer()
        highlighted = explainer.highlight_text(news_content, shap.keyword_highlights)
        st.markdown(f'<div style="line-height: 1.8; padding: 10px; background: #f8fafc; border-radius: 8px;">{highlighted}</div>', unsafe_allow_html=True)
    else:
        st.write(news_content)
    
    st.markdown("#### 📊 SHAP Feature Importance")
    if shap.feature_importance:
        features = sorted(shap.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        words = [f for f, _ in features]
        values = [v for _, v in features]
        colors = ['#22c55e' if v > 0 else '#ef4444' for v in values]
        
        fig = go.Figure(go.Bar(
            x=values, y=words, orientation='h',
            marker_color=colors,
            text=[f'{v:.3f}' for v in values],
            textposition='outside'
        ))
        fig.update_layout(
            height=400,
            xaxis_title='SHAP Value',
            yaxis=dict(autorange='reversed'),
            title='Top 15 Feature Contributions'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Base Value", f"{shap.base_value:.3f}")
    with col2:
        st.metric("Output Value", f"{shap.output_value:.3f}")
    with col3:
        st.metric("Expected Sentiment", shap.expected_sentiment)


def display_risk_alerts(alerts: List[Dict]):
    if not alerts:
        st.success("✅ No active risk alerts")
        return
    
    for alert in alerts:
        severity = alert.get('severity', 'info')
        alert_type = alert.get('type', 'unknown')
        message = alert.get('message', '')
        value = alert.get('value', 0)
        threshold = alert.get('threshold', 0)
        
        if severity == 'critical':
            st.error(f"🔴 CRITICAL [{alert_type.upper()}]: {message}")
        elif severity == 'warning':
            st.warning(f"🟡 WARNING [{alert_type.upper()}]: {message}")
        else:
            st.info(f"🔵 INFO [{alert_type.upper()}]: {message}")


def main():
    st.set_page_config(
        page_title="Financial News Sentiment Trading System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Financial News Sentiment Trading System")
    st.caption("Real-time multimodal sentiment analysis with adaptive thresholds, risk management, and explainable AI")
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            st.session_state.data_store = StreamDataStore()
            st.session_state.processor = PythonStreamProcessor(st.session_state.data_store)
            st.session_state.processor.start()
            
            st.session_state.ws_client = WebSocketClient(st.session_state.data_store, st.session_state.processor)
            st.session_state.ws_client.start()
            
            st.session_state.enhanced_generator = EnhancedSignalGenerator(settings.ENHANCED_SIGNAL_CONFIG)
            st.session_state.risk_engine = RiskDashboardEngine()
            
            st.session_state.heatmap_throttler = RenderThrottler(max_fps=2.0)
            st.session_state.chart_throttler = RenderThrottler(max_fps=2.0)
            st.session_state.risk_throttler = RenderThrottler(max_fps=1.0)
            st.session_state.incremental_heatmap = IncrementalHeatmapData(max_time_buckets=15, max_symbols=10)
            st.session_state.delta_updater = SmartDeltaUpdater(tolerance=0.01)
            st.session_state.fps_limiter = FrameRateLimiter(target_fps=2.0)
            st.session_state._last_heatmap_fig = None
            
            st.session_state.threshold_history = []
            st.session_state.volatility_history = []
            
            st.success("Streaming system initialized!")
            st.session_state.initialized = True
    
    data_store = st.session_state.data_store
    processor = st.session_state.processor
    enhanced_generator = st.session_state.enhanced_generator
    risk_engine = st.session_state.risk_engine
    
    heatmap_throttler = st.session_state.heatmap_throttler
    chart_throttler = st.session_state.chart_throttler
    risk_throttler = st.session_state.risk_throttler
    incremental_heatmap = st.session_state.incremental_heatmap
    fps_limiter = st.session_state.fps_limiter
    
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
        
        st.subheader("Adaptive Thresholds")
        use_adaptive = st.checkbox("Enable Adaptive Thresholds", value=settings.USE_ADAPTIVE_THRESHOLD)
        base_buy = st.slider("Base Buy Threshold", 0.0, 1.0, settings.BASE_BUY_THRESHOLD, 0.05)
        base_sell = st.slider("Base Sell Threshold", -1.0, 0.0, settings.BASE_SELL_THRESHOLD, 0.05)
        
        st.divider()
        
        st.subheader("Risk Management")
        use_risk_limits = st.checkbox("Enable Risk Limits", value=settings.USE_RISK_LIMITS)
        max_var_pct = st.slider("Max VaR %", 1.0, 10.0, settings.RISK_LIMITS['max_portfolio_var_pct'] * 100, 0.5) / 100
        max_dd_pct = st.slider("Max Drawdown %", 5.0, 40.0, settings.RISK_LIMITS['max_drawdown_pct'] * 100, 1.0) / 100
        
        st.divider()
        
        st.subheader("Portfolio")
        initial_capital = st.number_input("Initial Capital", 10000.0, 10000000.0, settings.INITIAL_CAPITAL, 10000.0)
        
        st.divider()
        
        if st.button("Run Backtest"):
            st.session_state.run_backtest = True
        
        st.divider()
        st.info(f"News rate: {settings.NEWS_GENERATION_RATE}/s")
        st.info(f"Window: {settings.WINDOW_DURATION}s sliding")
    
    display_statistics(data_store, executor.portfolio)
    
    st.divider()
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Sentiment Analysis", 
        "🔥 Signal Heatmap", 
        "📖 Order Book",
        "💼 Portfolio",
        "📈 Risk Dashboard",
        "🎛️ Adaptive Thresholds",
        "💡 Signal Explanation"
    ])
    
    with tab1:
        fig_sentiment = plot_sentiment_timeseries(data_store, selected_symbols)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        st.subheader("Recent Trading Signals")
        display_signals_table(data_store)
    
    with tab2:
        fig_heatmap = plot_signal_heatmap(
            data_store, selected_symbols,
            incremental_data=incremental_heatmap,
            throttler=heatmap_throttler
        )
        
        if fig_heatmap is not None:
            st.session_state._last_heatmap_fig = fig_heatmap
            st.plotly_chart(fig_heatmap, use_container_width=True)
        elif st.session_state._last_heatmap_fig is not None:
            st.plotly_chart(st.session_state._last_heatmap_fig, use_container_width=True)
        else:
            fig = go.Figure()
            fig.add_annotation(text="Waiting for signal data...", showarrow=False, font=dict(size=16))
            fig.update_layout(height=400, title='Trading Signal Strength Heatmap')
            st.plotly_chart(fig, use_container_width=True)
        
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
            
            for symbol, pos in executor.portfolio.positions.items():
                risk_engine.update_position(symbol, pos.quantity, pos.current_price)
    
    with tab5:
        st.subheader("Risk Dashboard")
        
        risk_data = enhanced_generator.get_risk_dashboard()
        if risk_data:
            alerts = risk_data.get('alerts', [])
            display_risk_alerts(alerts)
            
            fig_risk = plot_risk_dashboard(risk_data, throttler=risk_throttler)
            if fig_risk:
                st.plotly_chart(fig_risk, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            dashboard = risk_data['dashboard']
            var = dashboard.var
            
            with col1:
                st.metric("95% VaR", f"${var.var_95:,.0f}", f"{var.var_95/var.position_value*100:.2f}%")
            with col2:
                st.metric("99% VaR", f"${var.var_99:,.0f}", f"{var.var_99/var.position_value*100:.2f}%")
            with col3:
                st.metric("Current Drawdown", f"{dashboard.current_drawdown*100:.2f}%")
            with col4:
                st.metric("Risk Score", f"{dashboard.risk_score*100:.0f}/100")
                
                if dashboard.risk_score < 0.3:
                    st.error("High Risk")
                elif dashboard.risk_score < 0.6:
                    st.warning("Moderate Risk")
                else:
                    st.success("Low Risk")
            
            if risk_data.get('violations'):
                st.subheader("⚠️ Risk Limit Violations")
                for v in risk_data['violations']:
                    v_type = v.get('type', 'unknown')
                    if 'symbol' in v:
                        st.warning(f"{v_type.upper()}: {v['symbol']} - {v['pct']*100:.1f}% (limit: {v['limit']*100:.1f}%)")
                    elif 'sector' in v:
                        st.warning(f"{v_type.upper()}: {v['sector']} - {v['pct']*100:.1f}% (limit: {v['limit']*100:.1f}%)")
                    else:
                        st.warning(f"{v_type.upper()}: {v.get('var_pct', v.get('drawdown', v.get('risk_score', 0))*100:.1f}% (limit: {v['limit']*100:.1f}%)")
        else:
            fig = go.Figure()
            fig.add_annotation(text="Waiting for portfolio data...", showarrow=False, font=dict(size=16))
            fig.update_layout(height=600, title='Risk Dashboard')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.subheader("Adaptive Thresholds")
        
        prices = data_store.get_recent_prices(selected_symbols[0], limit=100)
        for p in prices:
            enhanced_generator.update_price(p)
            risk_engine.update_price(p)
        
        threshold_info = enhanced_generator.get_threshold_info(selected_symbols[0])
        
        col1, col2, col3 = st.columns(3)
        
        if threshold_info and threshold_info.get('adaptive'):
            thresh = threshold_info['current']
            vol = threshold_info['volatility']
            
            with col1:
                st.metric("Buy Threshold", f"{thresh.buy_threshold:.3f}", f"Base: {thresh.base_buy_threshold:.3f}")
            with col2:
                st.metric("Sell Threshold", f"{thresh.sell_threshold:.3f}", f"Base: {thresh.base_sell_threshold:.3f}")
            with col3:
                if vol:
                    st.metric("Current Volatility", f"{vol.current_volatility*100:.2f}%", 
                             f"Regime: {vol.regime.replace('_', ' ').title()}")
            
            if thresh:
                st.session_state.threshold_history.append(thresh)
                if len(st.session_state.threshold_history) > 100:
                    st.session_state.threshold_history = st.session_state.threshold_history[-100:]
            
            if vol:
                st.session_state.volatility_history.append(vol)
                if len(st.session_state.volatility_history) > 100:
                    st.session_state.volatility_history = st.session_state.volatility_history[-100:]
            
            fig_thresh = plot_threshold_evolution(
                st.session_state.threshold_history,
                st.session_state.volatility_history,
                throttler=chart_throttler
            )
            
            if fig_thresh:
                st.plotly_chart(fig_thresh, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔧 Threshold Factors")
                st.write(f"**Volatility Factor:** {thresh.volatility_factor:.3f}")
                st.write(f"**Momentum Factor:** {thresh.momentum_factor:.3f}")
                st.write(f"**Combined Factor:** {thresh.volatility_factor * thresh.momentum_factor:.3f}")
            
            with col2:
                st.markdown("#### 📊 Volatility State")
                if vol:
                    st.write(f"**Current Vol:** {vol.current_volatility*100:.2f}%")
                    st.write(f"**Historical Vol:** {vol.historical_volatility*100:.2f}%")
                    st.write(f"**Vol Ratio:** {vol.volatility_ratio:.2f}x")
                    st.write(f"**VIX Proxy:** {vol.vix:.1f}")
        else:
            st.info("Waiting for sufficient price data to calculate adaptive thresholds...")
    
    with tab7:
        st.subheader("Signal Explanation (XAI)")
        
        signals = data_store.get_recent_signals(selected_symbols[0], limit=5)
        
        if signals:
            signal_options = [
                f"{s.signal} | {s.symbol} | {s.timestamp.strftime('%H:%M:%S')} | Strength: {s.strength:.2f}"
                for s in signals
            ]
            
            selected_idx = st.selectbox("Select Signal to Explain", range(len(signal_options)),
                                       format_func=lambda i: signal_options[i])
            
            selected_signal = signals[selected_idx]
            
            news = None
            sentiment = None
            recent_news = data_store.get_recent_news(selected_signal.symbol, limit=20)
            for n in recent_news:
                recent_sentiments = data_store.get_recent_sentiments(selected_signal.symbol, limit=20)
                for s in recent_sentiments:
                    if s.news_id == n.id:
                        news = n
                        sentiment = s
                        break
                if news:
                    break
            
            if news and sentiment:
                explanation = enhanced_generator.get_explanation_for_signal(
                    selected_signal, news, sentiment
                )
                
                if explanation:
                    display_signal_explanation(explanation, news.content)
                else:
                    st.info("No explanation available for this signal")
            else:
                st.info("Waiting for news and sentiment data...")
        else:
            st.info("Waiting for trading signals...")
    
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
    
    actual_fps = fps_limiter.wait_for_next_frame()
    
    time.sleep(max(0, settings.STREAMLIT_UPDATE_INTERVAL - fps_limiter.min_frame_time))
    st.rerun()


if __name__ == "__main__":
    main()
