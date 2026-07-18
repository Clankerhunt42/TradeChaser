"""
TradeChaser Main Entry Point
High-consistency prediction market trading bot.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict
import yaml
import sys

from scraper.market_scraper import DataAggregator, PolymarketScraper, ManifoldScraper
from estimator.probability_estimator import EnsembleEstimator, SignalAggregator
from detector.stability_detector import StabilityDetector, MarketStateMonitor
from trader.trade_engine import TradeEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class TradeChaser:
    """Main trading bot orchestrator."""
    
    def __init__(self, config_path: str = "config/settings.yml"):
        """Initialize TradeChaser."""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.scraper = DataAggregator()
        self.estimator = EnsembleEstimator()
        self.stability = StabilityDetector(
            window_size=self.config["stability"]["window_size"],
            volatility_threshold=self.config["stability"]["volatility_threshold"],
        )
        self.state_monitor = MarketStateMonitor()
        self.trader = TradeEngine(
            bankroll=self.config["position"]["max_position_size"] * 10,
            kelly_fraction=self.config["position"]["kelly_fraction"],
        )
        
        self.running = False
    
    @staticmethod
    def _load_config(path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {path}")
            return {}
    
    async def run_live_trading(self):
        """Run live trading mode."""
        logger.info("=" * 60)
        logger.info("TRADECHASER - LIVE TRADING MODE")
        logger.info("=" * 60)
        logger.info("Constraints:")
        logger.info(f"  • Edge threshold (Δ ≥ p̂ - q): {self.config['constraints']['edge_threshold']}")
        logger.info(f"  • Stability threshold (p(j*,j*)): {self.config['constraints']['stability_threshold']}")
        logger.info("=" * 60)
        
        self.running = True
        iteration = 0
        
        try:
            while self.running:
                iteration += 1
                logger.info(f"\n[ITERATION {iteration}] {datetime.utcnow()}")
                
                try:
                    # Fetch market data
                    logger.info("Fetching market data...")
                    market_data = await self.scraper.aggregate_market_data(
                        ["election", "crypto", "sports", "weather"]
                    )
                    
                    if market_data.empty:
                        logger.warning("No market data received")
                        await asyncio.sleep(self.config["market"]["update_frequency"])
                        continue
                    
                    # Process each market
                    for idx, row in market_data.iterrows():
                        await self._process_market(row)
                    
                    # Log performance
                    stats = self.trader.get_performance_stats()
                    if stats.get("closed_trades", 0) > 0:
                        logger.info(f"Performance: Win rate {stats['win_rate']:.2%}, "
                                   f"P&L: {stats['total_pnl']:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error in trading iteration: {e}", exc_info=True)
                
                # Wait for next update
                await asyncio.sleep(self.config["market"]["update_frequency"])
        
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
            self.running = False
    
    async def _process_market(self, market_data: Dict):
        """Process a single market."""
        market_id = market_data.get("market_id") or market_data.get("id")
        
        try:
            # Update stability detector with price
            price = market_data.get("midpoint") or market_data.get("probability", 0.5)
            self.stability.update(price, market_data.get("volume", 0))
            
            # Detect market stability
            stability_score, stability_details = self.stability.detect_stability()
            
            # Estimate probability
            signals = SignalAggregator.extract_signals(
                market_data=market_data,
                event_data={},
                sentiment_data={}
            )
            estimated_prob, confidence = self.estimator.estimate(signals)
            
            # Get market probability
            market_prob = market_data.get("midpoint") or market_data.get("probability", 0.5)
            
            # Evaluate trade
            opportunity = self.trader.evaluate_trade(
                market_id=str(market_id),
                estimated_prob=estimated_prob,
                market_prob=market_prob,
                stability=stability_score,
                confidence=confidence,
            )
            
            # Execute if approved
            if opportunity.signal.value != "skip":
                self.trader.execute_trade(opportunity)
        
        except Exception as e:
            logger.error(f"Error processing market {market_id}: {e}")
    
    async def run_backtest(self, market_data_path: str, start_date: str, end_date: str):
        """Run backtesting mode."""
        logger.info("=" * 60)
        logger.info("TRADECHASER - BACKTEST MODE")
        logger.info(f"Data: {market_data_path}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("=" * 60)
        
        # TODO: Implement backtesting
        logger.info("Backtesting not yet implemented")
    
    async def run_analysis(self):
        """Run analysis mode."""
        logger.info("=" * 60)
        logger.info("TRADECHASER - ANALYSIS MODE")
        logger.info("=" * 60)
        
        # Fetch data
        logger.info("Analyzing market conditions...")
        market_data = await self.scraper.aggregate_market_data(["all"])
        
        logger.info(f"Analyzed {len(market_data)} markets")
        
        # Generate report
        if not market_data.empty:
            logger.info("\nTop opportunities (by edge):")
            for idx, row in market_data.nlargest(5, "spread").iterrows():
                logger.info(f"  {row.get('market_id')}: Edge = {row.get('spread'):.4f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TradeChaser - Consistency over everything")
    parser.add_argument("--mode", default="live", choices=["live", "backtest", "analyze"],
                       help="Operating mode")
    parser.add_argument("--config", default="config/settings.yml",
                       help="Configuration file path")
    parser.add_argument("--data", help="Data file for backtest")
    parser.add_argument("--start-date", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Backtest end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = TradeChaser(config_path=args.config)
    
    # Run appropriate mode
    if args.mode == "live":
        asyncio.run(bot.run_live_trading())
    elif args.mode == "backtest":
        if not all([args.data, args.start_date, args.end_date]):
            logger.error("Backtest requires --data, --start-date, --end-date")
            sys.exit(1)
        asyncio.run(bot.run_backtest(args.data, args.start_date, args.end_date))
    elif args.mode == "analyze":
        asyncio.run(bot.run_analysis())


if __name__ == "__main__":
    main()
