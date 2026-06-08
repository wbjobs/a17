import os
import sys
import time
import numpy as np
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("NEW FEATURES VERIFICATION TEST")
print("="*80)
print()

test_results = []

def run_test(name, test_func, expected_time=None):
    print(f"Testing: {name}")
    print("-"*60)
    start_time = time.time()
    
    try:
        result = test_func()
        elapsed = time.time() - start_time
        status = "✅ PASSED" if result else "❌ FAILED"
        time_info = f"[{elapsed:.2f}s]"
        if expected_time and elapsed > expected_time:
            time_info = f"[{elapsed:.2f}s (WARNING: slower than expected {expected_time}s)]"
        print(f"{status} {time_info}")
        test_results.append((name, result, elapsed))
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED [{elapsed:.2f}s]")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        test_results.append((name, False, elapsed))
        return False
    finally:
        print()

# ======================================================================
# TEST 1: Multimodal Analysis
# ======================================================================

def test_multimodal_analysis():
    print("  Testing OCR Engine...")
    from multimodal.multimodal_analyzer import OCREngine, ChartParser, TableParser, MultimodalAnalyzer
    
    ocr = OCREngine()
    test_image_data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 1000
    ocr_text = ocr.extract_text(test_image_data)
    print(f"    OCR extracted: {ocr_text[:60]}...")
    assert len(ocr_text) > 0, "OCR should return text"
    
    sentiment, confidence = ocr.analyze_text_sentiment(ocr_text)
    print(f"    Sentiment score: {sentiment:.3f}, confidence: {confidence:.3f}")
    assert -1.0 <= sentiment <= 1.0, "Sentiment should be in [-1, 1]"
    
    print("  Testing Chart Parser...")
    chart_parser = ChartParser()
    chart = chart_parser.parse(test_image_data)
    print(f"    Chart type: {chart.chart_type}, trend: {chart.trend}")
    print(f"    Data points: {len(chart.data_points)}, sentiment: {chart.sentiment_score:.3f}")
    assert len(chart.data_points) > 0, "Chart should have data points"
    assert chart.trend in ['upward', 'downward', 'flat', 'stable'], "Invalid trend"
    
    print("  Testing Table Parser...")
    table_parser = TableParser()
    html_table = """
    <table>
    <tr><th>Metric</th><th>Q1</th><th>Q2</th></tr>
    <tr><td>Revenue</td><td>100</td><td>120</td></tr>
    <tr><td>Profit</td><td>20</td><td>30</td></tr>
    </table>
    """
    table = table_parser.parse_html_table(html_table)
    print(f"    Table headers: {table.headers}")
    print(f"    Table rows: {len(table.rows)}, sentiment: {table.sentiment_score:.3f}")
    print(f"    Key insights: {table.key_insights[:2]}")
    assert len(table.rows) == 2, "Should have 2 data rows"
    assert len(table.key_insights) > 0, "Should have insights"
    
    print("  Testing Multimodal Analyzer integration...")
    from shared.models import NewsArticle
    analyzer = MultimodalAnalyzer()
    
    news = NewsArticle(
        id='test_001',
        symbol='AAPL',
        title='Strong Earnings Report',
        content='Apple reported record revenue and earnings that beat analyst expectations. Revenue increased 23% year-over-year with strong growth in all segments. Management raised guidance for the next quarter.',
        source='Bloomberg',
        timestamp=datetime.now()
    )
    
    result = analyzer.analyze_news(
        news,
        images=[test_image_data],
        tables_html=[html_table],
        charts=[test_image_data]
    )
    
    print(f"    News ID: {result.news_id}")
    print(f"    Combined score: {result.combined_score:.3f}")
    print(f"    Images analyzed: {len(result.images)}")
    print(f"    Tables analyzed: {len(result.tables)}")
    print(f"    Charts analyzed: {len(result.charts)}")
    assert -1.0 <= result.combined_score <= 1.0, "Combined score should be in [-1, 1]"
    assert len(result.images) == 1, "Should have 1 image"
    assert len(result.tables) == 1, "Should have 1 table"
    assert len(result.charts) == 1, "Should have 1 chart"
    
    return True

# ======================================================================
# TEST 2: Adaptive Threshold
# ======================================================================

def test_adaptive_threshold():
    print("  Testing Volatility Calculator...")
    from adaptive_threshold.volatility_engine import VolatilityCalculator, AdaptiveThresholdEngine
    from shared.models import PriceData
    
    vol_calc = VolatilityCalculator(lookback_periods=60, min_periods=20)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    base_price = {'AAPL': 150.0, 'GOOGL': 140.0, 'MSFT': 380.0}
    
    for i in range(100):
        for symbol in symbols:
            noise = np.random.normal(0, 0.02)
            if i > 70:
                noise *= 3
            price = base_price[symbol] * (1 + noise)
            vol_calc.add_price(symbol, PriceData(
                symbol=symbol,
                price=price,
                volume=1000000,
                timestamp=datetime.now(),
                bid=price - 0.01,
                ask=price + 0.01,
                bid_size=100,
                ask_size=100
            ))
    
    vol_state = vol_calc.calculate_volatility('AAPL')
    print(f"    Current volatility: {vol_state.current_volatility*100:.2f}%")
    print(f"    Historical volatility: {vol_state.historical_volatility*100:.2f}%")
    print(f"    Volatility ratio: {vol_state.volatility_ratio:.2f}")
    print(f"    VIX proxy: {vol_state.vix:.1f}")
    print(f"    Regime: {vol_state.regime}")
    assert vol_state is not None, "Should have volatility state"
    assert vol_state.regime in ['low_vol', 'normal', 'high_vol'], "Invalid regime"
    
    vol_surface = vol_calc.get_volatility_surface('AAPL')
    print(f"    Volatility surface: {vol_surface}")
    assert len(vol_surface) == 4, "Should have 4 tenors"
    
    print("  Testing Adaptive Threshold Engine...")
    threshold_engine = AdaptiveThresholdEngine({
        'base_buy_threshold': 0.6,
        'base_sell_threshold': -0.6
    })
    
    for i in range(100):
        for symbol in symbols:
            noise = np.random.normal(0, 0.02)
            price = base_price[symbol] * (1 + noise)
            threshold_engine.update_price(symbol, PriceData(
                symbol=symbol,
                price=price,
                volume=1000000,
                timestamp=datetime.now()
            ))
    
    thresholds = threshold_engine.get_adaptive_thresholds('AAPL')
    print(f"    Buy threshold: {thresholds.buy_threshold:.3f} (base: {thresholds.base_buy_threshold:.3f})")
    print(f"    Sell threshold: {thresholds.sell_threshold:.3f} (base: {thresholds.base_sell_threshold:.3f})")
    print(f"    Volatility factor: {thresholds.volatility_factor:.3f}")
    print(f"    Momentum factor: {thresholds.momentum_factor:.3f}")
    assert thresholds is not None, "Should have thresholds"
    assert 0.3 <= thresholds.buy_threshold <= 0.9, "Buy threshold out of range"
    assert -0.9 <= thresholds.sell_threshold <= -0.3, "Sell threshold out of range"
    
    print("  Testing signal generation with adaptive thresholds...")
    test_cases = [
        (0.8, 0.7, 'BUY'),
        (-0.8, -0.6, 'SELL'),
        (0.2, 0.1, 'HOLD'),
        (-0.2, -0.1, 'HOLD'),
    ]
    
    for sentiment, correlation, expected in test_cases:
        signal, strength, confidence = threshold_engine.check_signal('AAPL', sentiment, correlation)
        print(f"    Sentiment={sentiment:.2f}, Correlation={correlation:.2f} -> {signal} (strength={strength:.2f}, confidence={confidence:.2f})")
    
    return True

# ======================================================================
# TEST 3: Risk Dashboard
# ======================================================================

def test_risk_dashboard():
    print("  Testing VaR Calculator...")
    from risk.risk_dashboard import VaRCalculator, DrawdownCalculator, ExposureAnalyzer, RiskDashboardEngine, RiskLimitChecker
    from shared.models import PriceData
    
    var_calc = VaRCalculator(lookback_days=252)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'WMT']
    base_prices = {'AAPL': 150, 'GOOGL': 140, 'MSFT': 380, 'JPM': 145, 'WMT': 65}
    
    for i in range(300):
        for symbol in symbols:
            ret = np.random.normal(0.0005, 0.02)
            price = base_prices[symbol] * (1 + ret)
            var_calc.add_price(symbol, PriceData(
                symbol=symbol,
                price=price,
                volume=1000000,
                timestamp=datetime.now()
            ))
    
    var_95 = var_calc.calculate_var_parametric('AAPL', 100000, 0.95)
    cvar_95 = var_calc.calculate_cvar('AAPL', 100000, 0.95)
    print(f"    Parametric 95% VaR: ${var_95:,.2f} ({var_95/100000*100:.2f}%)")
    print(f"    95% CVaR: ${cvar_95:,.2f} ({cvar_95/100000*100:.2f}%)")
    assert var_95 > 0, "VaR should be positive"
    assert cvar_95 > var_95, "CVaR should be greater than VaR"
    
    portfolio = {
        'AAPL': {'value': 300000},
        'GOOGL': {'value': 250000},
        'MSFT': {'value': 200000},
        'JPM': {'value': 150000},
        'WMT': {'value': 100000}
    }
    
    full_var = var_calc.calculate_full_var(portfolio, 'parametric')
    print(f"    Portfolio 95% VaR: ${full_var.var_95:,.2f}")
    print(f"    Portfolio 99% VaR: ${full_var.var_99:,.2f}")
    print(f"    Portfolio 95% CVaR: ${full_var.cvar_95:,.2f}")
    assert full_var.position_value == 1000000, "Wrong portfolio value"
    
    print("  Testing Drawdown Calculator...")
    dd_calc = DrawdownCalculator(lookback_periods=252)
    
    portfolio_value = 1000000
    for i in range(100):
        ret = np.random.normal(0.001, 0.02)
        if i == 50:
            ret = -0.15
        portfolio_value *= (1 + ret)
        dd_calc.update_portfolio_value(portfolio_value, datetime.now())
    
    print(f"    Max drawdown: {dd_calc.max_drawdown*100:.2f}%")
    print(f"    Current drawdown: {dd_calc.current_drawdown*100:.2f}%")
    assert dd_calc.max_drawdown > dd_calc.current_drawdown, "Max should be >= current"
    
    print("  Testing Exposure Analyzer...")
    exp_analyzer = ExposureAnalyzer()
    
    positions = {
        'AAPL': {'position': 2000, 'price': 150},
        'GOOGL': {'position': 1500, 'price': 140},
        'MSFT': {'position': 500, 'price': 380},
        'JPM': {'position': 1000, 'price': 145},
        'WMT': {'position': 1500, 'price': 65}
    }
    
    prices = {sym: pos['price'] for sym, pos in positions.items()}
    
    exposure_items, sector_weights = exp_analyzer.calculate_exposure(positions, prices)
    
    for item in exposure_items[:3]:
        print(f"    {item.symbol}: ${item.value:,.0f} ({item.weight*100:.1f}%, sector: {item.sector}, beta: {item.beta})")
    
    print(f"    Sector exposure: {sector_weights}")
    assert len(exposure_items) == 5, "Should have 5 positions"
    assert abs(sum(item.weight for item in exposure_items) - 1.0) < 0.001, "Weights should sum to 1"
    
    portfolio_beta = exp_analyzer.calculate_portfolio_beta(exposure_items)
    concentration = exp_analyzer.calculate_concentration_ratio(exposure_items)
    print(f"    Portfolio beta: {portfolio_beta:.3f}")
    print(f"    Top 5 concentration: {concentration*100:.1f}%")
    
    print("  Testing Risk Dashboard Engine...")
    risk_engine = RiskDashboardEngine()
    
    for symbol, pos in positions.items():
        risk_engine.update_position(symbol, pos['position'], pos['price'])
    
    for i in range(100):
        for symbol in positions.keys():
            ret = np.random.normal(0.0005, 0.02)
            price = base_prices[symbol] * (1 + ret)
            risk_engine.update_price(symbol, PriceData(
                symbol=symbol,
                price=price,
                volume=1000000,
                timestamp=datetime.now()
            ))
    
    dashboard = risk_engine.get_dashboard()
    print(f"    Portfolio value: ${dashboard.portfolio_value:,.0f}")
    print(f"    Risk score: {dashboard.risk_score*100:.0f}/100")
    print(f"    Max drawdown: {dashboard.max_drawdown*100:.2f}%")
    print(f"    Current drawdown: {dashboard.current_drawdown*100:.2f}%")
    assert dashboard.portfolio_value > 0, "Portfolio value should be positive"
    assert 0 <= dashboard.risk_score <= 1, "Risk score should be in [0, 1]"
    
    alerts = risk_engine.get_risk_alerts()
    print(f"    Active alerts: {len(alerts)}")
    for alert in alerts[:3]:
        print(f"    - {alert['severity']}: {alert['message']}")
    
    print("  Testing Risk Limit Checker...")
    risk_checker = RiskLimitChecker()
    limits_ok, violations = risk_checker.check_all_limits(dashboard)
    
    print(f"    Limits OK: {limits_ok}")
    for v in violations:
        print(f"    Violation: {v['type']} - {v.get('symbol', v.get('sector', ''))}")
    
    return True

# ======================================================================
# TEST 4: Explainable AI
# ======================================================================

def test_explainable_ai():
    print("  Testing Financial Keyword Dictionary...")
    from explanation.shap_explainer import FinancialKeywordDictionary, KeywordSHAPExplainer, SignalExplainer
    
    kw_dict = FinancialKeywordDictionary()
    
    test_words = ['increase', 'decrease', 'stable', 'unknown', 'record', 'miss']
    for word in test_words:
        sentiment = kw_dict.get_word_sentiment(word)
        weight = kw_dict.calculate_word_weight(word, ['very'])
        print(f"    '{word}': {sentiment}, weight: {weight:.2f}")
    
    print("  Testing Keyword SHAP Explainer...")
    shap_explainer = KeywordSHAPExplainer()
    
    from shared.models import NewsArticle, SentimentResult
    
    news = NewsArticle(
        id='test_002',
        symbol='AAPL',
        title='Earnings Beat',
        content='Apple reported record revenue that beat analyst expectations. Revenue increased significantly with strong growth across all segments. Management raised forward guidance citing robust demand.',
        source='Bloomberg',
        timestamp=datetime.now()
    )
    
    sentiment = SentimentResult(
        news_id='test_002',
        symbol='AAPL',
        positive=0.8,
        negative=0.1,
        neutral=0.1,
        sentiment_score=0.7,
        timestamp=datetime.now()
    )
    
    shap_explanation = shap_explainer.explain(news, sentiment)
    
    print(f"    Base value: {shap_explanation.base_value:.3f}")
    print(f"    Output value: {shap_explanation.output_value:.3f}")
    print(f"    Expected sentiment: {shap_explanation.expected_sentiment}")
    print(f"    Contributing words: {shap_explanation.contributing_words[:5]}")
    print(f"    Mitigating words: {shap_explanation.mitigating_words[:5]}")
    print(f"    Feature importance entries: {len(shap_explanation.feature_importance)}")
    print(f"    Keyword highlights: {len(shap_explanation.keyword_highlights)}")
    assert shap_explanation is not None, "Should have SHAP explanation"
    assert len(shap_explanation.contributing_words) > 0, "Should have contributing words"
    
    print("  Testing Signal Explainer...")
    signal_explainer = SignalExplainer(shap_explainer)
    
    from shared.models import TradingSignal
    
    signal = TradingSignal(
        symbol='AAPL',
        signal='BUY',
        strength=0.75,
        sentiment_score=0.7,
        price_correlation=0.65,
        timestamp=datetime.now(),
        confidence=0.8,
        reason='Strong positive sentiment'
    )
    
    explanation = signal_explainer.explain_signal(signal, news, sentiment)
    
    print(f"    Signal: {explanation.signal}")
    print(f"    Strength: {explanation.strength:.2f}")
    print(f"    Confidence: {explanation.confidence:.2f}")
    print(f"    Summary: {explanation.summary[:100]}...")
    print(f"    Key drivers: {explanation.key_drivers[:3]}")
    print(f"    Risk factors: {explanation.risk_factors[:3]}")
    assert explanation is not None, "Should have signal explanation"
    assert explanation.signal == 'BUY', "Signal should match"
    assert len(explanation.key_drivers) > 0, "Should have key drivers"
    
    print("  Testing text highlighting...")
    highlighted = signal_explainer.highlight_text(news.content, explanation.shap_explanation.keyword_highlights)
    assert '<span' in highlighted, "Should have HTML highlighting"
    print(f"    Highlighted text: {highlighted[:150]}...")
    
    print("  Testing feature importance plot data...")
    plot_data = signal_explainer.get_feature_importance_plot_data(shap_explanation)
    print(f"    Features for plot: {len(plot_data['features'])}")
    assert len(plot_data['features']) > 0, "Should have plot features"
    
    print("  Testing waterfall plot data...")
    waterfall_data = signal_explainer.get_waterfall_plot_data(shap_explanation)
    print(f"    Waterfall features: {len(waterfall_data['features'])}")
    assert len(waterfall_data['features']) > 0, "Should have waterfall features"
    
    return True

# ======================================================================
# TEST 5: Enhanced Signal Generator Integration
# ======================================================================

def test_enhanced_signal_generator():
    print("  Testing Enhanced Signal Generator...")
    from signal.enhanced_signal_generator import EnhancedSignalGenerator
    from shared.models import NewsArticle, SentimentResult, PriceData
    
    config = {
        'buy_threshold': 0.6,
        'sell_threshold': -0.6,
        'use_adaptive_threshold': True,
        'use_multimodal': True,
        'use_explainable': True,
        'use_risk_limits': True
    }
    
    generator = EnhancedSignalGenerator(config)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    base_prices = {'AAPL': 150, 'GOOGL': 140, 'MSFT': 380}
    
    print("    Feeding price data...")
    for i in range(100):
        for symbol in symbols:
            ret = np.random.normal(0.0005, 0.02)
            price = base_prices[symbol] * (1 + ret)
            generator.update_price(PriceData(
                symbol=symbol,
                price=price,
                volume=1000000,
                timestamp=datetime.now(),
                bid=price - 0.01,
                ask=price + 0.01,
                bid_size=100,
                ask_size=100
            ))
    
    generator.update_position('AAPL', 100, 150.0)
    generator.update_position('GOOGL', 50, 140.0)
    generator.update_position('MSFT', 20, 380.0)
    
    print("    Testing signal generation...")
    news_articles = [
        ('AAPL', 'Apple reported record earnings that beat expectations. Revenue increased 23% YoY with strong growth.', 0.8),
        ('GOOGL', 'Google missed analyst expectations as ad revenue declined. Management warned of slowdown.', -0.7),
        ('MSFT', 'Microsoft showed stable performance in line with estimates. Cloud growth continues steady.', 0.2)
    ]
    
    for symbol, content, expected_score in news_articles:
        news = NewsArticle(
            id=f'news_{symbol}_{i}',
            symbol=symbol,
            title=f'News for {symbol}',
            content=content,
            source='Test',
            timestamp=datetime.now()
        )
        
        sentiment = SentimentResult(
            news_id=news.id,
            symbol=symbol,
            positive=max(0, expected_score),
            negative=max(0, -expected_score),
            neutral=1 - abs(expected_score),
            sentiment_score=expected_score,
            timestamp=datetime.now()
        )
        
        signal = generator.generate_signal(news, sentiment)
        
        if signal:
            print(f"    {symbol}: {signal.signal} (strength={signal.strength:.2f}, confidence={signal.confidence:.2f})")
            if signal.explanation:
                print(f"      Explanation: {signal.explanation.summary[:80]}...")
            if signal.volatility_state:
                print(f"      Vol regime: {signal.volatility_state.regime}, vol={signal.volatility_state.current_volatility*100:.1f}%")
            if signal.adaptive_threshold:
                print(f"      Thresholds: buy={signal.adaptive_threshold.buy_threshold:.2f}, sell={signal.adaptive_threshold.sell_threshold:.2f}")
    
    print("    Testing risk dashboard...")
    risk_data = generator.get_risk_dashboard()
    if risk_data:
        dashboard = risk_data['dashboard']
        print(f"    Portfolio value: ${dashboard.portfolio_value:,.0f}")
        print(f"    Risk score: {dashboard.risk_score*100:.0f}/100")
        print(f"    Alerts: {len(risk_data.get('alerts', []))}")
    
    print("    Testing threshold info...")
    threshold_info = generator.get_threshold_info('AAPL')
    if threshold_info and threshold_info.get('adaptive'):
        thresh = threshold_info['current']
        print(f"    Adaptive buy threshold: {thresh.buy_threshold:.3f}")
        print(f"    Adaptive sell threshold: {thresh.sell_threshold:.3f}")
    
    return True

# ======================================================================
# TEST 6: Module Import Verification
# ======================================================================

def test_imports():
    print("  Verifying all module imports...")
    
    imports = [
        ("shared.models", ["ImageContent", "TableContent", "ChartData", "MultimodalResult", 
                          "VolatilityState", "AdaptiveThreshold", "VaRResult", "RiskDashboard",
                          "SHAPExplanation", "SignalExplanation", "EnhancedTradingSignal"]),
        ("multimodal.multimodal_analyzer", ["MultimodalAnalyzer", "OCREngine", "ChartParser", "TableParser"]),
        ("adaptive_threshold.volatility_engine", ["VolatilityCalculator", "AdaptiveThresholdEngine", "RegimeBasedThresholdStrategy"]),
        ("risk.risk_dashboard", ["RiskDashboardEngine", "VaRCalculator", "DrawdownCalculator", "ExposureAnalyzer", "RiskLimitChecker"]),
        ("explanation.shap_explainer", ["KeywordSHAPExplainer", "SignalExplainer", "FinancialKeywordDictionary"]),
        ("signal.enhanced_signal_generator", ["EnhancedSignalGenerator"]),
        ("config.settings", ["settings"]),
    ]
    
    for module_name, classes in imports:
        try:
            module = __import__(module_name, fromlist=classes)
            for cls in classes:
                assert hasattr(module, cls), f"Missing {cls} in {module_name}"
            print(f"    ✅ {module_name}")
        except Exception as e:
            print(f"    ❌ {module_name}: {e}")
            return False
    
    return True

# ======================================================================
# Run all tests
# ======================================================================

print()
print("="*60)
print("RUNNING TESTS")
print("="*60)
print()

run_test("Module Imports", test_imports)
run_test("Multimodal Analysis", test_multimodal_analysis, expected_time=3.0)
run_test("Adaptive Threshold", test_adaptive_threshold, expected_time=3.0)
run_test("Risk Dashboard", test_risk_dashboard, expected_time=3.0)
run_test("Explainable AI", test_explainable_ai, expected_time=3.0)
run_test("Enhanced Signal Generator", test_enhanced_signal_generator, expected_time=5.0)

print()
print("="*60)
print("TEST SUMMARY")
print("="*60)

total = len(test_results)
passed = sum(1 for _, r, _ in test_results if r)
total_time = sum(t for _, _, t in test_results)

print(f"Tests: {passed}/{total} tests passed")
print(f"Total time: {total_time:.2f}s")
print()

for name, result, elapsed in test_results:
    status = "✅" if result else "❌"
    print(f"  {status} {name} ({elapsed:.2f}s)")

print()
if passed == total:
    print("🎉 All new features verified successfully!")
else:
    print(f"⚠️  {total - passed} test(s) failed")
    print()
    for name, result, elapsed in test_results:
        if not result:
            print(f"  ❌ {name}")

print()
print("="*80)
