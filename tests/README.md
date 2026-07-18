# Testing

## Quick Tests (No Dependencies)

Run the quick test suite without pytest:

```bash
python run_quick_tests.py
```

This tests:
- ✓ Constraint 1: Edge threshold (Δ ≥ 0.05)
- ✓ Constraint 2: Stability threshold (p(j*, j*) ≥ 0.87)
- ✓ Both constraints combined
- ✓ Stability detection
- ✓ State change detection
- ✓ Probability estimation
- ✓ Trade execution
- ✓ Full integration workflow

## Comprehensive Tests (Pytest)

For detailed testing with pytest:

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/test_tradechaser.py -v

# Run specific test class
pytest tests/test_tradechaser.py::TestTradeEngineConstraints -v

# Run specific test
pytest tests/test_tradechaser.py::TestTradeEngineConstraints::test_constraint_1_edge_too_small -v

# Run with coverage
pip install pytest-cov
pytest tests/test_tradechaser.py --cov=src --cov-report=html
```

## Test Coverage

### Test Classes

1. **TestTradeEngineConstraints** - Trade constraint enforcement
   - Edge threshold (Δ)
   - Stability threshold (p(j*, j*))
   - Position sizing
   - Buy/Sell signals

2. **TestStabilityDetector** - Market stability detection
   - Volatility calculation
   - Momentum detection
   - Threshold enforcement
   - State changes

3. **TestMarketStateMonitor** - State change detection
   - Price change detection
   - Recent stability calculation

4. **TestProbabilityEstimators** - Probability estimation
   - Bayesian estimation
   - Ensemble models
   - Signal aggregation

5. **TestTradeExecution** - Trade lifecycle
   - Trade execution
   - Position closing
   - P&L calculation
   - Performance stats

6. **TestIntegration** - End-to-end workflows
   - Full trading workflow
   - Constraint enforcement
   - Multi-market scenarios

## Test Scenarios

### Scenario 1: Good Opportunity
- Edge: 9% (5%+ ✓)
- Stability: 90% (87%+ ✓)
- **Expected**: BUY signal with position

### Scenario 2: Edge Too Small
- Edge: 1% (5%+ ✗)
- Stability: 90% (87%+ ✓)
- **Expected**: SKIP signal

### Scenario 3: Market Unstable
- Edge: 9% (5%+ ✓)
- Stability: 80% (87%+ ✗)
- **Expected**: SKIP signal

### Scenario 4: Both Fail
- Edge: 1% (5%+ ✗)
- Stability: 80% (87%+ ✗)
- **Expected**: SKIP signal

### Scenario 5: Sell Opportunity
- Edge: -10% (estimated < market)
- Stability: 92% (87%+ ✓)
- **Expected**: SELL signal with position

## Performance Metrics

Tests verify:
- Win rate calculation
- Sharpe ratio computation
- P&L tracking
- Position sizing correctness
- Constraint enforcement accuracy

## Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

View test logs:

```bash
pytest tests/test_tradechaser.py -v -s  # -s shows print statements
```

## CI/CD Integration

For GitHub Actions, see `.github/workflows/test.yml` (to be added)
