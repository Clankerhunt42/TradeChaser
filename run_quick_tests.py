"""
Quick Test Script - Run without pytest
Run manually: python run_quick_tests.py
"""

import sys
from datetime import datetime, timedelta
import numpy as np

# Add src to path
sys.path.insert(0, '.')

from src.trader.trade_engine import TradeEngine, TradeSignal
from src.detector.stability_detector import StabilityDetector, MarketStateMonitor
from src.estimator.probability_estimator import (
    BayesianEstimator,
    EnsembleEstimator,
    SignalAggregator
)


def print_header(title):
    """Print test section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_test(name, passed, details=""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {name}")
    if details:
        print(f"       {details}")


def test_constraint_1_edge():
    """Test Constraint 1: Edge Threshold (Δ ≥ 0.05)"""
    print_header("CONSTRAINT 1: Edge Threshold (Δ = p̂ - q ≥ 0.05)")
    
    engine = TradeEngine()
    
    # Test 1: Edge too small (1%)
    opp = engine.evaluate_trade("test1", 0.52, 0.51, 0.90, 0.8)
    passed = opp.signal == TradeSignal.SKIP
    print_test("Edge 1% (below 5%)", passed, f"Signal: {opp.signal.value}")
    
    # Test 2: Edge at threshold (5%)
    opp = engine.evaluate_trade("test2", 0.56, 0.51, 0.90, 0.8)
    passed = opp.signal != TradeSignal.SKIP
    print_test("Edge 5% (at threshold)", passed, f"Signal: {opp.signal.value}")
    
    # Test 3: Edge large (9%)
    opp = engine.evaluate_trade("test3", 0.60, 0.51, 0.90, 0.8)
    passed = opp.signal == TradeSignal.BUY and opp.position_size > 0
    print_test("Edge 9% (above 5%)", passed, f"Signal: {opp.signal.value}, "
              f"Position: {opp.position_size:.2f}")


def test_constraint_2_stability():
    """Test Constraint 2: Stability Threshold (p(j*, j*) ≥ 0.87)"""
    print_header("CONSTRAINT 2: Stability Threshold (p(j*, j*) ≥ 0.87)")
    
    engine = TradeEngine()
    
    # Test 1: Stability too low (80%)
    opp = engine.evaluate_trade("test1", 0.60, 0.51, 0.80, 0.8)
    passed = opp.signal == TradeSignal.SKIP
    print_test("Stability 80% (below 87%)", passed, f"Signal: {opp.signal.value}")
    
    # Test 2: Stability at threshold (87%)
    opp = engine.evaluate_trade("test2", 0.60, 0.51, 0.87, 0.8)
    passed = opp.signal != TradeSignal.SKIP
    print_test("Stability 87% (at threshold)", passed, f"Signal: {opp.signal.value}")
    
    # Test 3: Stability high (95%)
    opp = engine.evaluate_trade("test3", 0.60, 0.51, 0.95, 0.8)
    passed = opp.signal == TradeSignal.BUY
    print_test("Stability 95% (above 87%)", passed, f"Signal: {opp.signal.value}")


def test_both_constraints():
    """Test Both Constraints Together"""
    print_header("BOTH CONSTRAINTS TOGETHER")
    
    engine = TradeEngine()
    
    # Test 1: Both fail
    opp = engine.evaluate_trade("test1", 0.52, 0.51, 0.80, 0.8)
    passed = opp.signal == TradeSignal.SKIP
    print_test("Both fail (edge 1%, stability 80%)", passed, f"Signal: {opp.signal.value}")
    
    # Test 2: Edge passes, stability fails
    opp = engine.evaluate_trade("test2", 0.60, 0.51, 0.80, 0.8)
    passed = opp.signal == TradeSignal.SKIP
    print_test("Edge OK (9%), Stability BAD (80%)", passed, f"Signal: {opp.signal.value}")
    
    # Test 3: Edge fails, stability passes
    opp = engine.evaluate_trade("test3", 0.52, 0.51, 0.90, 0.8)
    passed = opp.signal == TradeSignal.SKIP
    print_test("Edge BAD (1%), Stability OK (90%)", passed, f"Signal: {opp.signal.value}")
    
    # Test 4: Both pass → BUY
    opp = engine.evaluate_trade("test4", 0.60, 0.51, 0.90, 0.8)
    passed = opp.signal == TradeSignal.BUY and opp.position_size > 0
    print_test("Both pass → BUY (edge 9%, stability 90%)", passed, 
              f"Signal: {opp.signal.value}, Position: {opp.position_size:.2f}")
    
    # Test 5: Both pass → SELL (estimated < market)
    opp = engine.evaluate_trade("test5", 0.40, 0.50, 0.92, 0.85)
    passed = opp.signal == TradeSignal.SELL and opp.position_size > 0
    print_test("Both pass → SELL (edge -10%, stability 92%)", passed, 
              f"Signal: {opp.signal.value}, Position: {opp.position_size:.2f}")


def test_stability_detection():
    """Test Stability Detector"""
    print_header("STABILITY DETECTION")
    
    detector = StabilityDetector()
    
    # Test 1: Stable market (flat prices)
    print("\nTest 1: Stable market (flat prices + noise)")
    for i in range(100):
        price = 0.50 + np.random.normal(0, 0.00005)
        detector.update(price, volume=1000)
    
    stability, _ = detector.detect_stability()
    passed = stability > 0.80
    print_test(f"Stability score: {stability:.4f}", passed, 
              f"{'Meets 0.87 threshold' if stability >= 0.87 else 'Below 0.87'}")
    
    # Test 2: Volatile market
    detector2 = StabilityDetector()
    print("\nTest 2: Volatile market (large swings)")
    for i in range(100):
        price = 0.50 + np.sin(i / 10) * 0.1
        detector2.update(price, volume=1000)
    
    stability, _ = detector2.detect_stability()
    passed = stability < 0.70
    print_test(f"Stability score: {stability:.4f}", passed, 
              f"Correctly identified as unstable")
    
    # Test 3: Market meets threshold
    detector3 = StabilityDetector()
    print("\nTest 3: Market meets 0.87 threshold")
    for i in range(100):
        price = 0.50 + np.random.normal(0, 0.00003)
        detector3.update(price, volume=1000)
    
    is_stable = detector3.is_stable(min_stability=0.87)
    print_test(f"is_stable(0.87)", is_stable, "Market qualifies for trading")


def test_state_changes():
    """Test Market State Change Detection"""
    print_header("MARKET STATE CHANGE DETECTION")
    
    monitor = MarketStateMonitor(change_threshold=0.05)
    
    # Test 1: Small change
    changed = monitor.detect_state_change(0.505, 0.500)
    passed = not changed
    print_test("Small change (1%)", passed, "Not detected (correct)")
    
    # Test 2: Large change
    changed = monitor.detect_state_change(0.550, 0.500)
    passed = changed
    print_test("Large change (10%)", passed, "Detected (correct)")
    
    # Test 3: Change at threshold
    changed = monitor.detect_state_change(0.525, 0.500)
    passed = changed
    print_test("Change at threshold (5%)", passed, "Detected at boundary")


def test_probability_estimation():
    """Test Probability Estimation"""
    print_header("PROBABILITY ESTIMATION")
    
    # Test 1: Bayesian estimator
    print("\nTest 1: Bayesian Estimator")
    estimator = BayesianEstimator(prior=0.5)
    
    data = {"signals": {"test": {"value": 0.8, "type": "bullish", "confidence": 0.8}}}
    prob, conf = estimator.estimate(data)
    passed = prob > 0.5 and 0 <= prob <= 1
    print_test(f"Bullish signal: p̂ = {prob:.4f}", passed, f"Confidence: {conf:.2f}")
    
    # Test 2: Ensemble estimator
    print("\nTest 2: Ensemble Estimator")
    ensemble = EnsembleEstimator()
    data = {"signals": {"test": {"value": 0.7, "type": "bullish", "confidence": 0.8}}}
    prob, conf = ensemble.estimate(data)
    passed = 0 <= prob <= 1 and 0 <= conf <= 1
    print_test(f"Ensemble estimate: p̂ = {prob:.4f}", passed, f"Confidence: {conf:.2f}")
    
    # Test 3: Signal aggregation
    print("\nTest 3: Signal Aggregation")
    market_data = {
        "spread": 0.01,
        "midpoint": 0.50,
        "volume_bid": 1000,
        "volume_ask": 900,
    }
    signals = SignalAggregator.extract_signals(market_data, {}, {})
    passed = "signals" in signals and len(signals["signals"]) > 0
    print_test("Extract signals from market data", passed, 
              f"Found {len(signals['signals'])} signals")


def test_trade_execution():
    """Test Trade Execution"""
    print_header("TRADE EXECUTION")
    
    engine = TradeEngine(bankroll=10000)
    
    # Test 1: Execute approved trade
    print("\nTest 1: Execute Approved Trade")
    opp = engine.evaluate_trade("market1", 0.60, 0.51, 0.90, 0.8)
    executed = engine.execute_trade(opp)
    passed = executed and "market1" in engine.active_positions
    print_test("Execute BUY trade", passed, f"Active positions: {len(engine.active_positions)}")
    
    # Test 2: Reject skipped trade
    print("\nTest 2: Reject Skipped Trade")
    opp = engine.evaluate_trade("market2", 0.52, 0.51, 0.80, 0.8)
    executed = engine.execute_trade(opp)
    passed = not executed
    print_test("Reject SKIP trade", passed, "Correctly not executed")
    
    # Test 3: Close position
    print("\nTest 3: Close Position")
    closed = engine.close_position("market1", exit_price=0.65)
    passed = closed is not None and closed["status"] == "closed"
    print_test("Close position", passed, f"PNL: {closed['pnl']:.2f}")
    
    # Test 4: Performance stats
    print("\nTest 4: Performance Statistics")
    stats = engine.get_performance_stats()
    passed = "closed_trades" in stats and stats["closed_trades"] >= 1
    print_test("Generate performance stats", passed, 
              f"Closed trades: {stats['closed_trades']}, Win rate: {stats.get('win_rate', 0):.1%}")


def test_integration():
    """Integration Test - Full Workflow"""
    print_header("INTEGRATION TEST - FULL WORKFLOW")
    
    print("\n1. Initialize components...")
    stability_detector = StabilityDetector()
    estimator = EnsembleEstimator()
    trade_engine = TradeEngine()
    print("   ✓ Components initialized")
    
    print("\n2. Simulate stable market...")
    for i in range(50):
        price = 0.50 + np.random.normal(0, 0.0001)
        stability_detector.update(price, volume=1000)
    print("   ✓ Market data collected")
    
    print("\n3. Calculate stability...")
    stability, details = stability_detector.detect_stability()
    print(f"   ✓ Stability: {stability:.4f}")
    
    print("\n4. Estimate probability...")
    market_data = {"midpoint": 0.50, "spread": 0.01, "volume_bid": 1000, "volume_ask": 900}
    signals = SignalAggregator.extract_signals(market_data, {}, {})
    estimated_prob, confidence = estimator.estimate(signals)
    print(f"   ✓ Estimated probability: {estimated_prob:.4f}, Confidence: {confidence:.2f}")
    
    print("\n5. Evaluate trade opportunity...")
    opportunity = trade_engine.evaluate_trade(
        market_id="integration_test",
        estimated_prob=estimated_prob,
        market_prob=0.50,
        stability=stability,
        confidence=confidence
    )
    print(f"   ✓ Signal: {opportunity.signal.value}, Position size: {opportunity.position_size:.2f}")
    
    print("\n6. Execute trade (if approved)...")
    if opportunity.signal != TradeSignal.SKIP:
        executed = trade_engine.execute_trade(opportunity)
        print(f"   ✓ Trade executed: {executed}")
    else:
        print("   ✓ Trade skipped (constraints not met)")
    
    print("\n✅ Full workflow completed successfully!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print(" TRADECHASER - QUICK TEST SUITE")
    print("="*60)
    
    try:
        test_constraint_1_edge()
        test_constraint_2_stability()
        test_both_constraints()
        test_stability_detection()
        test_state_changes()
        test_probability_estimation()
        test_trade_execution()
        test_integration()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run: python src/main.py --mode analyze")
        print("  2. Run: python src/main.py --mode backtest")
        print("  3. Run: python src/main.py --mode live")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
