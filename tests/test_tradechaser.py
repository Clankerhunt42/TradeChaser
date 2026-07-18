"""
Complete Test Suite for TradeChaser
Tests all components: constraints, estimators, detectors, traders
"""

import pytest
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Import components to test
from src.trader.trade_engine import TradeEngine, TradeSignal
from src.detector.stability_detector import StabilityDetector, MarketStateMonitor
from src.estimator.probability_estimator import (
    BayesianEstimator,
    EnsembleEstimator,
    HistoricalEstimator,
    MLEstimator,
    SignalAggregator
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestTradeEngineConstraints:
    """Test trade engine constraint enforcement."""
    
    def setup_method(self):
        """Setup before each test."""
        self.engine = TradeEngine(bankroll=10000)
    
    def test_constraint_1_edge_too_small(self):
        """Constraint 1 FAILS: Edge < 0.05"""
        opportunity = self.engine.evaluate_trade(
            market_id="test_small_edge",
            estimated_prob=0.52,
            market_prob=0.51,  # Edge = 1% (need 5%)
            stability=0.90,    # Stability OK
            confidence=0.8
        )
        assert opportunity.signal == TradeSignal.SKIP
        assert "Edge too small" in " ".join(opportunity.reasons)
        logger.info("✓ Test passed: Edge too small → SKIP")
    
    def test_constraint_1_edge_exact_threshold(self):
        """Constraint 1 PASSES: Edge = 0.05 (exactly at threshold)"""
        opportunity = self.engine.evaluate_trade(
            market_id="test_exact_edge",
            estimated_prob=0.56,
            market_prob=0.51,  # Edge = 5% (exactly threshold)
            stability=0.90,
            confidence=0.8
        )
        assert opportunity.signal != TradeSignal.SKIP
        logger.info("✓ Test passed: Edge at threshold → ACCEPT")
    
    def test_constraint_1_edge_large(self):
        """Constraint 1 PASSES: Edge > 0.05"""
        opportunity = self.engine.evaluate_trade(
            market_id="test_large_edge",
            estimated_prob=0.60,
            market_prob=0.51,  # Edge = 9% (good)
            stability=0.90,
            confidence=0.8
        )
        assert opportunity.signal == TradeSignal.BUY
        assert "Edge sufficient" in " ".join(opportunity.reasons)
        logger.info("✓ Test passed: Large edge → BUY")
    
    def test_constraint_2_stability_too_low(self):
        """Constraint 2 FAILS: Stability < 0.87"""
        opportunity = self.engine.evaluate_trade(
            market_id="test_unstable",
            estimated_prob=0.60,
            market_prob=0.51,  # Edge OK
            stability=0.80,    # Stability = 80% (need 87%)
            confidence=0.8
        )
        assert opportunity.signal == TradeSignal.SKIP
        assert "Market unstable" in " ".join(opportunity.reasons)
        logger.info("✓ Test passed: Market unstable → SKIP")
    
    def test_constraint_2_stability_exact_threshold(self):
        """Constraint 2 PASSES: Stability = 0.87 (exactly at threshold)"""
        opportunity = self.engine.evaluate_trade(
            market_id="test_exact_stability",
            estimated_prob=0.60,
            market_prob=0.51,
            stability=0.87,    # Exactly at threshold
            confidence=0.8
        )
        assert opportunity.signal != TradeSignal.SKIP
        logger.info("✓ Test passed: Stability at threshold → ACCEPT")
    
    def test_constraint_2_stability_high(self):
        """Constraint 2 PASSES: Stability > 0.87"""
        opportunity = self.engine.evaluate_trade(
            market_id="test_stable",
            estimated_prob=0.60,
            market_prob=0.51,
            stability=0.95,    # High stability (95%)
            confidence=0.8
        )
        assert opportunity.signal == TradeSignal.BUY
        assert "Market stable" in " ".join(opportunity.reasons)
        logger.info("✓ Test passed: Stable market → BUY")
    
    def test_both_constraints_fail(self):
        """BOTH constraints FAIL → SKIP"""
        opportunity = self.engine.evaluate_trade(
            market_id="test_both_fail",
            estimated_prob=0.52,
            market_prob=0.51,  # Edge = 1% (FAIL)
            stability=0.80,    # Stability = 80% (FAIL)
            confidence=0.8
        )
        assert opportunity.signal == TradeSignal.SKIP
        assert "Edge too small" in " ".join(opportunity.reasons)
        assert "Market unstable" in " ".join(opportunity.reasons)
        logger.info("✓ Test passed: Both constraints fail → SKIP")
    
    def test_both_constraints_pass_buy(self):
        """BOTH constraints PASS → BUY (estimated > market)"""
        opportunity = self.engine.evaluate_trade(
            market_id="test_both_pass_buy",
            estimated_prob=0.60,
            market_prob=0.51,  # Edge = 9% (PASS)
            stability=0.90,    # Stability = 90% (PASS)
            confidence=0.8
        )
        assert opportunity.signal == TradeSignal.BUY
        assert opportunity.position_size > 0
        assert opportunity.edge >= 0.05
        assert opportunity.stability >= 0.87
        logger.info(f"✓ Test passed: Both pass → BUY (position size: {opportunity.position_size:.2f})")
    
    def test_both_constraints_pass_sell(self):
        """BOTH constraints PASS → SELL (estimated < market)"""
        opportunity = self.engine.evaluate_trade(
            market_id="test_both_pass_sell",
            estimated_prob=0.40,
            market_prob=0.50,  # Edge = -10% (we think market too high)
            stability=0.92,    # Stability = 92% (PASS)
            confidence=0.85
        )
        assert opportunity.signal == TradeSignal.SELL
        assert opportunity.position_size > 0
        logger.info(f"✓ Test passed: Both pass → SELL (position size: {opportunity.position_size:.2f})")
    
    def test_position_sizing(self):
        """Test position sizing is reasonable."""
        opportunity = self.engine.evaluate_trade(
            market_id="test_position",
            estimated_prob=0.65,
            market_prob=0.51,
            stability=0.95,
            confidence=0.9
        )
        # Position should be positive
        assert opportunity.position_size > 0
        # Position should not exceed bankroll
        assert opportunity.position_size <= self.engine.bankroll
        # Position should scale with edge
        logger.info(f"✓ Test passed: Position size = {opportunity.position_size:.2f} "
                   f"(bankroll: {self.engine.bankroll})")
    
    def test_edge_calculation(self):
        """Test edge calculation."""
        p_hat = 0.65
        q = 0.50
        opportunity = self.engine.evaluate_trade(
            market_id="test_edge_calc",
            estimated_prob=p_hat,
            market_prob=q,
            stability=0.90,
            confidence=0.8
        )
        assert opportunity.edge == pytest.approx(0.15, abs=0.001)
        logger.info(f"✓ Test passed: Edge calculation: {p_hat} - {q} = {opportunity.edge:.4f}")


class TestStabilityDetector:
    """Test market stability detection."""
    
    def setup_method(self):
        """Setup before each test."""
        self.detector = StabilityDetector(window_size=3600)
    
    def test_insufficient_data(self):
        """Test with insufficient data."""
        stability, details = self.detector.detect_stability()
        assert stability <= 0.5  # Should be cautious with little data
        assert details["reason"] == "insufficient_data"
        logger.info("✓ Test passed: Insufficient data handling")
    
    def test_stable_market_flat_prices(self):
        """Test stable market (prices stay flat)."""
        # Add stable prices (minimal movement)
        for i in range(100):
            price = 0.5000 + (np.random.normal(0, 0.0001))  # Tiny random walk
            self.detector.update(price, volume=1000)
        
        stability, details = self.detector.detect_stability()
        assert stability > 0.8
        logger.info(f"✓ Test passed: Stable market detection: {stability:.4f}")
    
    def test_volatile_market_wild_swings(self):
        """Test volatile market (large price swings)."""
        # Add volatile prices
        for i in range(100):
            price = 0.50 + np.sin(i / 10) * 0.1  # Large oscillations
            self.detector.update(price, volume=1000)
        
        stability, details = self.detector.detect_stability()
        assert stability < 0.7
        logger.info(f"✓ Test passed: Volatile market detection: {stability:.4f}")
    
    def test_meets_stability_threshold(self):
        """Test market meets 0.87 threshold."""
        # Create stable market
        for i in range(100):
            price = 0.50 + np.random.normal(0, 0.00005)
            self.detector.update(price, volume=1000)
        
        is_stable = self.detector.is_stable(min_stability=0.87)
        assert is_stable
        logger.info("✓ Test passed: Market meets 0.87 stability threshold")
    
    def test_fails_stability_threshold(self):
        """Test market fails 0.87 threshold."""
        # Create volatile market
        for i in range(100):
            price = 0.50 + np.sin(i / 5) * 0.05
            self.detector.update(price, volume=1000)
        
        is_stable = self.detector.is_stable(min_stability=0.87)
        assert not is_stable
        logger.info("✓ Test passed: Market fails 0.87 stability threshold")
    
    def test_volatility_score_component(self):
        """Test volatility calculation."""
        for i in range(50):
            self.detector.update(0.50, volume=1000)
        
        stability, details = self.detector.detect_stability()
        assert "volatility_score" in details
        logger.info(f"✓ Test passed: Volatility score = {details['volatility_score']:.4f}")
    
    def test_momentum_score_component(self):
        """Test momentum detection."""
        for i in range(50):
            price = 0.50 + (0.001 * i)  # Uptrend
            self.detector.update(price, volume=1000)
        
        stability, details = self.detector.detect_stability()
        assert "momentum_score" in details
        assert details["momentum_score"] > 0.1  # Should detect uptrend
        logger.info(f"✓ Test passed: Momentum score = {details['momentum_score']:.4f}")


class TestMarketStateMonitor:
    """Test market state change detection."""
    
    def setup_method(self):
        """Setup before each test."""
        self.monitor = MarketStateMonitor(change_threshold=0.05)
    
    def test_small_change_no_detection(self):
        """Test small price change not detected."""
        changed = self.monitor.detect_state_change(0.505, 0.500)
        assert not changed
        logger.info("✓ Test passed: Small change not detected")
    
    def test_large_change_detected(self):
        """Test large price change detected."""
        changed = self.monitor.detect_state_change(0.550, 0.500)  # 10% change
        assert changed
        logger.info("✓ Test passed: Large change detected")
    
    def test_state_change_at_threshold(self):
        """Test change exactly at threshold."""
        changed = self.monitor.detect_state_change(0.525, 0.500)  # Exactly 5%
        assert changed
        logger.info("✓ Test passed: Change at threshold detected")
    
    def test_recent_stability(self):
        """Test stability based on state changes."""
        stability = self.monitor.get_recent_stability(lookback_seconds=300)
        assert stability == 1.0  # No changes yet
        logger.info(f"✓ Test passed: Initial stability = {stability:.4f}")


class TestProbabilityEstimators:
    """Test probability estimation."""
    
    def test_bayesian_estimator_bullish_signal(self):
        """Test Bayesian estimator with bullish signal."""
        estimator = BayesianEstimator(prior=0.5)
        
        data = {
            "signals": {
                "bid_ask": {
                    "value": 0.8,
                    "type": "bullish",
                    "confidence": 0.8
                }
            }
        }
        
        prob, conf = estimator.estimate(data)
        assert prob > 0.5  # Should be bullish
        assert 0 <= prob <= 1
        logger.info(f"✓ Test passed: Bayesian bullish estimate = {prob:.4f}")
    
    def test_bayesian_estimator_bearish_signal(self):
        """Test Bayesian estimator with bearish signal."""
        estimator = BayesianEstimator(prior=0.5)
        
        data = {
            "signals": {
                "bid_ask": {
                    "value": 0.2,
                    "type": "bearish",
                    "confidence": 0.8
                }
            }
        }
        
        prob, conf = estimator.estimate(data)
        assert prob < 0.5  # Should be bearish
        assert 0 <= prob <= 1
        logger.info(f"✓ Test passed: Bayesian bearish estimate = {prob:.4f}")
    
    def test_ensemble_estimator_combines_models(self):
        """Test ensemble estimator combines multiple models."""
        estimator = EnsembleEstimator()
        
        data = {
            "signals": {
                "bid_ask": {"value": 0.7, "type": "bullish", "confidence": 0.8}
            },
            "historical_data": [],
            "features": []
        }
        
        prob, conf = estimator.estimate(data)
        assert 0 <= prob <= 1
        assert 0 <= conf <= 1
        logger.info(f"✓ Test passed: Ensemble estimate = {prob:.4f}, confidence = {conf:.2f}")
    
    def test_signal_aggregator_market_data(self):
        """Test signal aggregation from market data."""
        market_data = {
            "spread": 0.01,
            "midpoint": 0.50,
            "volume_bid": 1000,
            "volume_ask": 500,
        }
        
        signals = SignalAggregator.extract_signals(market_data, {}, {})
        assert "signals" in signals
        assert "bid_ask" in signals["signals"]
        assert "volume" in signals["signals"]
        logger.info("✓ Test passed: Signal aggregation from market data")


class TestTradeExecution:
    """Test trade execution."""
    
    def setup_method(self):
        """Setup before each test."""
        self.engine = TradeEngine(bankroll=10000)
    
    def test_execute_approved_trade(self):
        """Test executing an approved trade."""
        opportunity = self.engine.evaluate_trade(
            market_id="test_exec",
            estimated_prob=0.60,
            market_prob=0.51,
            stability=0.90,
            confidence=0.8
        )
        
        # Trade should be approved
        assert opportunity.signal == TradeSignal.BUY
        
        # Execute trade
        executed = self.engine.execute_trade(opportunity)
        assert executed
        assert "test_exec" in self.engine.active_positions
        logger.info("✓ Test passed: Trade executed")
    
    def test_reject_skipped_trade(self):
        """Test that skipped trades are not executed."""
        opportunity = self.engine.evaluate_trade(
            market_id="test_skip",
            estimated_prob=0.52,
            market_prob=0.51,
            stability=0.80,
            confidence=0.8
        )
        
        # Trade should be rejected
        assert opportunity.signal == TradeSignal.SKIP
        
        # Execute trade should fail
        executed = self.engine.execute_trade(opportunity)
        assert not executed
        logger.info("✓ Test passed: Skipped trade not executed")
    
    def test_close_position(self):
        """Test closing a position."""
        # Execute a trade first
        opportunity = self.engine.evaluate_trade(
            market_id="test_close",
            estimated_prob=0.60,
            market_prob=0.51,
            stability=0.90,
            confidence=0.8
        )
        self.engine.execute_trade(opportunity)
        
        # Close position
        closed = self.engine.close_position("test_close", exit_price=0.65)
        assert closed is not None
        assert closed["status"] == "closed"
        assert "pnl" in closed
        logger.info(f"✓ Test passed: Position closed with PNL = {closed['pnl']:.2f}")
    
    def test_performance_stats(self):
        """Test performance statistics calculation."""
        stats = self.engine.get_performance_stats()
        assert "total_trades" in stats
        assert "win_rate" in stats
        logger.info(f"✓ Test passed: Stats generated: {stats}")


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_trading_workflow(self):
        """Test complete trading workflow."""
        # Setup
        stability_detector = StabilityDetector()
        estimator = EnsembleEstimator()
        trade_engine = TradeEngine()
        
        # 1. Simulate market data and update stability
        for i in range(50):
            price = 0.50 + np.random.normal(0, 0.0001)
            stability_detector.update(price, volume=1000)
        
        # 2. Get stability score
        stability, _ = stability_detector.detect_stability()
        
        # 3. Estimate probability
        signals = SignalAggregator.extract_signals(
            {"midpoint": 0.50, "spread": 0.01, "volume_bid": 1000, "volume_ask": 900},
            {},
            {}
        )
        estimated_prob, confidence = estimator.estimate(signals)
        
        # 4. Evaluate trade
        market_prob = 0.50
        opportunity = trade_engine.evaluate_trade(
            market_id="integration_test",
            estimated_prob=estimated_prob,
            market_prob=market_prob,
            stability=stability,
            confidence=confidence
        )
        
        # 5. Check result
        assert opportunity is not None
        logger.info(f"✓ Integration test passed: {opportunity.signal.value}")
    
    def test_constraint_enforcement_real_scenario(self):
        """Test constraints in realistic scenario."""
        engine = TradeEngine()
        
        # Scenario 1: Good opportunity
        opp1 = engine.evaluate_trade("scenario1", 0.62, 0.51, 0.92, 0.85)
        assert opp1.signal == TradeSignal.BUY
        executed1 = engine.execute_trade(opp1)
        assert executed1
        
        # Scenario 2: Edge too small
        opp2 = engine.evaluate_trade("scenario2", 0.54, 0.51, 0.90, 0.8)
        assert opp2.signal == TradeSignal.SKIP
        executed2 = engine.execute_trade(opp2)
        assert not executed2
        
        # Scenario 3: Market unstable
        opp3 = engine.evaluate_trade("scenario3", 0.62, 0.51, 0.85, 0.85)
        assert opp3.signal == TradeSignal.SKIP
        executed3 = engine.execute_trade(opp3)
        assert not executed3
        
        logger.info("✓ Constraint enforcement real scenario test passed")


def run_all_tests():
    """Run all tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    print("="*60)
    print("TRADECHASER - COMPLETE TEST SUITE")
    print("="*60)
    run_all_tests()
