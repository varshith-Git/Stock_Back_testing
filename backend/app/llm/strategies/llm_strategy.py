"""
LLM Smart Strategy
Event-driven intelligent trading strategy based on LLM
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.backtest_logger import BacktestLogger
from ...utils.indicators import calculate_bollinger_bands, calculate_macd
from ...utils.unrealized_pnl_tracker import UnrealizedPnLTracker
from ..analysis.enhanced_technical_analyzer import EnhancedTechnicalAnalyzer
from ..analysis.trend_analyzer import EnhancedTrendAnalyzer
from ..client import get_llm_client
from .base import (
    ParameterSpec,
    ParameterType,
    SignalType,
    StrategyConfig,
    TradingSignal,
    TradingStrategy,
)

logger = logging.getLogger(__name__)


class LLMSmartStrategy(TradingStrategy):
    """
    LLM Smart Strategy

    Workflow:
    1. Analyze stock characteristics with historical data to determine technical indicator parameters
    2. Call LLM for decision making when key events are triggered
    3. Optimize entry and exit timing by combining trend analysis
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.llm_client = get_llm_client(temperature=0.1)
        self.trend_analyzer = EnhancedTrendAnalyzer()
        self.enhanced_analyzer = EnhancedTechnicalAnalyzer()

        # Strategy parameters
        self.confidence_threshold = config.parameters.get(
            "confidence_threshold", 0.6
        )  # Reduced to 0.6 to increase execution opportunities
        self.trend_lookback = config.parameters.get("trend_lookback", 20)
        self.event_threshold = config.parameters.get("event_threshold", 0.05)

        # Strategy type selection - default to traditional
        self.strategy_type = config.parameters.get("strategy_type", "traditional")

        # Load decision principles
        self._load_strategy_prompt()
        self.max_daily_trades = config.parameters.get("max_daily_trades", 3)
        self.use_technical_filter = config.parameters.get("use_technical_filter", True)
        self.ma_short = config.parameters.get("ma_short", 10)
        self.ma_long = config.parameters.get("ma_long", 20)

        # Technical indicator default values
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.analysis_period_months = 3

        # Internal state
        self.stock_characteristics = None
        self.current_position = None
        self.decision_log = []
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.current_symbol = None  # Add current stock symbol tracking

        # LLM call statistics
        self.llm_call_count = 0
        self.llm_skipped_count = 0  # New: count of skipped LLM calls
        self.total_events_detected = 0
        self.events_filtered_out = 0

        # Progress callback function (for streaming updates)
        self.progress_callback = config.parameters.get("progress_callback", None)

        # Unrealized P&L tracker
        self.pnl_tracker = UnrealizedPnLTracker()
        self.current_position_id = None  # Current position ID

        # Risk management parameters
        self.max_loss_threshold = config.parameters.get(
            "max_loss_threshold", 0.10
        )  # 10% stop loss
        self.profit_taking_threshold = config.parameters.get(
            "profit_taking_threshold", 0.15
        )  # 15% take profit
        self.position_sizing_adjustment = config.parameters.get(
            "position_sizing_adjustment", True
        )
        self.position_size = config.parameters.get("position_size", 0.2)  # Default 20% position

        # LLM call statistics
        self.total_llm_calls = 0
        self.events_filtered_out = 0
        self.total_events_detected = 0

        # Dynamic performance tracking
        self.initial_capital = config.parameters.get("initial_capital", 100000)
        self.current_position = None  # Current holding status
        self.current_symbol = None  # Current stock symbol being traded
        self.position_entry_price = 0.0  # Entry price
        self.position_entry_date = None  # Entry date
        self.shares = 0  # Number of shares held
        self.cash = self.initial_capital  # Cash balance
        self.total_trades = 0
        self.winning_trades = 0
        self.current_portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital  # Track highest point
        self.max_drawdown = 0.0
        self.total_realized_pnl = 0.0  # Cumulative realized P&L
        self.trade_returns = []  # Record each trade's return rate (percentage)

        # Risk control related
        self._last_trend_analysis = None  # Store latest trend analysis for risk checks

        # Backtest logger initialization
        self.backtest_logger = None
        if config.parameters.get("enable_logging", True):
            log_path = config.parameters.get(
                "log_path", "backend/data/backtest_logs.db"
            )
            session_id = config.parameters.get("session_id", None)
            self.backtest_logger = BacktestLogger(log_path, session_id)
            logger.info(f"✅ Backtest logger enabled: {log_path}")

    def _load_strategy_prompt(self) -> None:
        """Load strategy decision principles"""
        try:
            # Determine current file path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_dir = os.path.join(current_dir, "prompt")

            # Use traditional strategy file
            file_path = os.path.join(prompt_dir, "traditional_strategy.md")

            # Read strategy file
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    self.strategy_prompt = f.read()
                logger.info(f"✅ Successfully loaded traditional strategy: {file_path}")
            else:
                logger.warning(f"⚠️ Strategy file does not exist: {file_path}, using default strategy")
                self.strategy_prompt = self._get_default_strategy_prompt()

        except Exception as e:
            logger.error(f"❌ Failed to load strategy file: {e}, using default strategy")
            self.strategy_prompt = self._get_default_strategy_prompt()

    def _get_default_strategy_prompt(self) -> str:
        """Get default strategy prompt"""
        return """
# Default Decision Principles

## Basic Strategy
- uptrend: Consider entering or holding positions
- downtrend: Should exit positions
- consolidation: Cautiously wait and observe

Please respond with decision in JSON format:
```json
{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "Decision rationale",
    "risk_level": "low" | "medium" | "high",
    "expected_outcome": "Expected outcome"
}
```
"""

    def _send_progress(
        self,
        day: int,
        total_days: int,
        event_type: str,
        message: str,
        extra_data: dict = None,
    ):
        """Helper method to send progress updates"""
        if self.progress_callback:
            try:
                if extra_data is not None:
                    self.progress_callback(
                        day, total_days, event_type, message, extra_data
                    )
                else:
                    # Backward compatibility: if callback doesn't support extra_data parameter, ignore it
                    import inspect

                    sig = inspect.signature(self.progress_callback)
                    if len(sig.parameters) >= 5:
                        self.progress_callback(
                            day, total_days, event_type, message, None
                        )
                    else:
                        self.progress_callback(day, total_days, event_type, message)
            except TypeError:
                # Backward compatibility: if callback doesn't support 5 parameters, use 4 parameters
                self.progress_callback(day, total_days, event_type, message)

    def set_symbol(self, symbol: str):
        """Set the current stock symbol for analysis"""
        self.current_symbol = symbol

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate LLM intelligent trading signals

        Args:
            data: DataFrame containing OHLCV data

        Returns:
            List of trading signals
        """
        signals = []

        # Check validity of input data
        if data is None:
            print("❌ Error: Input data is empty (None)")
            return signals

        if len(data) < 30:  # Reduced data requirement
            print(f"⚠️ Insufficient data: {len(data)} < 30, skipping signal generation")
            return signals

        # Initialize P&L tracker (if not already)
        if not hasattr(self, "pnl_tracker") or self.pnl_tracker is None:
            try:
                from ...utils.unrealized_pnl_tracker import UnrealizedPnLTracker

                self.pnl_tracker = UnrealizedPnLTracker()
                print(f"📊 P&L tracker initialization completed")
            except ImportError as e:
                print(f"⚠️ Unable to import P&L tracker: {e}")
                self.pnl_tracker = None

        # Step 1: Analyze stock characteristics (using early-stage data)
        self.stock_characteristics = self._analyze_stock_characteristics(data)

        # Dynamically adjust technical indicator parameters based on stock characteristics
        self._adjust_technical_parameters()

        # Calculate technical indicators
        data = self._calculate_all_indicators(data)

        # Analyze trends - add strict data length checks
        print(f"🔍 Preparing trend analysis data...")

        # Ensure sufficient data for analysis
        print(f"🔍 Data check: Total data volume = {len(data)}")
        if len(data) < 50:
            print(f"⚠️ Insufficient data for trend analysis ({len(data)} < 50), using simplified analysis")
            # Create simplified trend analysis results
            from types import SimpleNamespace

            trend_analysis = SimpleNamespace()
            trend_analysis.dominant_trend = "sideways"
            trend_analysis.complexity_score = 0.5
            trend_analysis.confidence = 0.3
        else:
            print(f"✅ Sufficient data volume ({len(data)} >= 50), starting comprehensive trend analysis")
            # Convert DataFrame to required format and validate data
            market_data_list = []
            valid_rows = 0

            for idx, row in data.iterrows():
                # Check data integrity
                close_price = row["close"]
                if pd.isna(close_price) or close_price <= 0:
                    print(f"⚠️ Skipping invalid data row: {idx}, close_price={close_price}")
                    continue

                market_data_list.append(
                    {
                        "date": idx.strftime("%Y-%m-%d")
                        if hasattr(idx, "strftime")
                        else str(idx),
                        "close": float(close_price),
                        "open": float(row["open"])
                        if "open" in row and not pd.isna(row["open"])
                        else float(close_price),
                        "high": float(row["high"])
                        if "high" in row and not pd.isna(row["high"])
                        else float(close_price),
                        "low": float(row["low"])
                        if "low" in row and not pd.isna(row["low"])
                        else float(close_price),
                        "volume": float(row["volume"])
                        if "volume" in row and not pd.isna(row["volume"])
                        else 0,
                    }
                )
                valid_rows += 1

            print(f"📊 Valid data rows: {valid_rows}/{len(data)}")

            # Check cleaned data volume again
            if len(market_data_list) < 30:
                print(
                    f"⚠️ Insufficient cleaned data volume ({len(market_data_list)} < 30), using simplified analysis"
                )
                from types import SimpleNamespace

                trend_analysis = SimpleNamespace()
                trend_analysis.dominant_trend = "sideways"
                trend_analysis.complexity_score = 0.5
                trend_analysis.confidence = 0.3
            else:
                print(
                    f"✅ Sufficient cleaned data volume ({len(market_data_list)} >= 30), calling trend analyzer..."
                )
                print(f"🔍 Starting Enhanced trend analysis...")
                try:
                    symbol = self.current_symbol or "UNKNOWN"
                    print(f"📈 Analyzing stock: {symbol}")

                    # Get current date from market data
                    current_date = None
                    if market_data_list:
                        last_data = market_data_list[-1]
                        current_date = (
                            last_data.get("date")
                            if isinstance(last_data, dict)
                            else None
                        )
                        print(f"📅 Current date: {current_date}")

                    # Ensure current_date is in string format
                    current_date_str = None
                    if current_date:
                        if hasattr(current_date, "strftime"):
                            current_date_str = current_date.strftime("%Y-%m-%d")
                        elif isinstance(current_date, str):
                            current_date_str = current_date
                        else:
                            current_date_str = str(current_date)

                    enhanced_result = self.trend_analyzer.analyze_with_llm_optimization(
                        symbol, current_date_str
                    )

                    # Extract traditional trend analysis for compatibility
                    trend_analysis = enhanced_result.original_result

                    # Store enhanced results for later use in prompts
                    self.current_enhanced_analysis = enhanced_result

                    print(f"✅ Enhanced trend analysis completed: {enhanced_result.market_phase}")
                    print(f"🎯 Reversal probability: {enhanced_result.reversal_probability:.2f}")
                    print(f"📊 Trend consistency: {enhanced_result.trend_consistency:.2f}")
                    print(f"📈 Momentum status: {enhanced_result.momentum_status}")
                    print(f"🔍 Dominant trend: {trend_analysis.dominant_trend}")

                except Exception as e:
                    print(f"❌ Enhanced trend analysis failed: {e}")
                    import traceback

                    print(f"🔍 Error details: {traceback.format_exc()}")
                    # Create backup analysis result
                    from types import SimpleNamespace

                    trend_analysis = SimpleNamespace()
                    trend_analysis.dominant_trend = "sideways"
                    trend_analysis.complexity_score = 0.5
                    trend_analysis.confidence = 0.2
                    self.current_enhanced_analysis = None

        print(f"🔄 Starting event-driven signal generation (data length: {len(data)})...")
        # Event-driven signal generation
        self._total_days = len(data)  # Set total days for other methods to use
        self._last_performance_update_day = -1  # Track last performance update day to avoid duplicates
        self._last_trend_update_day = -1  # Track last trend update day

        for i in range(30, len(data)):  # Start from 30 days instead of 100
            self._current_day_index = i  # Set current index for other methods to use

            # Periodically reanalyze global trend (every 30 days or at significant changes)
            if i % 30 == 0 and i != self._last_trend_update_day:
                print(f"🔄 Day {i}: Reanalyzing global trend...")
                try:
                    current_data_for_trend = data.iloc[: i + 1].copy()
                    market_data_list = []

                    # Reconstruct market data
                    for idx, row in current_data_for_trend.iterrows():
                        market_data_list.append(
                            {
                                "date": idx,
                                "open": row.get("open", row.get("Open", 0)),
                                "high": row.get("high", row.get("High", 0)),
                                "low": row.get("low", row.get("Low", 0)),
                                "close": row.get("close", row.get("Close", 0)),
                                "volume": row.get("volume", row.get("Volume", 0)),
                            }
                        )

                    if len(market_data_list) >= 30:
                        symbol = self.current_symbol or "UNKNOWN"
                        current_date = market_data_list[-1].get("date")

                        # Ensure current_date is in string format
                        if hasattr(current_date, "strftime"):
                            current_date_str = current_date.strftime("%Y-%m-%d")
                        elif isinstance(current_date, str):
                            current_date_str = current_date
                        else:
                            current_date_str = str(current_date)

                        enhanced_result = (
                            self.trend_analyzer.analyze_with_llm_optimization(
                                symbol, current_date_str
                            )
                        )
                        trend_analysis = enhanced_result.original_result
                        self.current_enhanced_analysis = enhanced_result

                        print(f"📊 Updated global trend: {enhanced_result.market_phase}")
                        print(
                            f"🎯 Reversal probability: {enhanced_result.reversal_probability:.2f}"
                        )
                        print(f"📈 Momentum status: {enhanced_result.momentum_status}")

                        self._last_trend_update_day = i

                except Exception as e:
                    print(f"⚠️ Global trend update failed: {e}")

            if i % 50 == 0:  # Output progress every 50 days
                progress_percentage = (i / len(data) * 100) if len(data) > 0 else 0
                progress_msg = (
                    f"📊 Processing progress: {i}/{len(data)} ({progress_percentage:.1f}%)"
                )
                print(progress_msg)

                # If there's a progress callback, send progress update
                if self.progress_callback:
                    self.progress_callback(
                        i, len(data), "processing", progress_msg, None
                    )

            # Send performance update every 10 days (including P&L status), but avoid duplicates with trade updates
            if (
                self.progress_callback
                and i % 10 == 0
                and i != self._last_performance_update_day
            ):
                current_row = data.iloc[i]
                current_price = current_row.get("close", current_row.get("Close", 0))
                if current_price > 0:
                    self._send_performance_update(i, len(data), current_price)
                    self._last_performance_update_day = i

            current_date = data.index[i]
            historical_data = data.iloc[: i + 1]

            # Safely get timestamp - handle possible integer index
            try:
                if hasattr(current_date, "date"):
                    # It's a datetime object
                    timestamp = current_date
                    current_date_obj = current_date.date()
                else:
                    # It's an integer index, create a default timestamp
                    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
                    current_date_obj = i

                if self.last_trade_date != current_date_obj:
                    self.daily_trade_count = 0
                    self.last_trade_date = current_date_obj
            except Exception:
                # If date processing fails, use default values
                timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
                current_date_obj = i
                if self.last_trade_date != current_date_obj:
                    self.daily_trade_count = 0
                    self.last_trade_date = current_date_obj

            if self.daily_trade_count >= self.max_daily_trades:
                continue

            # Detect trigger events
            events = self._detect_trigger_events(historical_data, i)
            self.total_events_detected += len(events)

            if events:
                current_price = historical_data.iloc[i]["close"]
                current_date = historical_data.index[i].strftime("%Y-%m-%d")

                # Display current P&L status
                if self.current_position:
                    position_metrics = self._calculate_position_metrics(
                        current_price, historical_data.index[i]
                    )
                    unrealized_pnl = position_metrics.get("unrealized_pnl", 0)
                    unrealized_pnl_pct = position_metrics.get("unrealized_pnl_pct", 0)
                    holding_days = position_metrics.get("holding_days", 0)

                    print(f"🎯 Day {i} events detected: {[e['event_type'] for e in events]}")
                    print(
                        f"💰 Position status: {self.shares} shares @ ${self.position_entry_price:.2f}, current price ${current_price:.2f}"
                    )
                    print(
                        f"📊 Unrealized P&L: ${unrealized_pnl:,.2f} ({unrealized_pnl_pct:+.2f}%), held for {holding_days} days"
                    )
                else:
                    print(f"🎯 Day {i} events detected: {[e['event_type'] for e in events]}")
                    print(
                        f"💵 No position: Cash ${self.cash:,.2f}, current price ${current_price:.2f}"
                    )

                # Optimization 1: Skip LLM only when no position + sideways market + no strong signals
                if (
                    not self.current_position
                    and trend_analysis.dominant_trend == "sideways"
                ):
                    # Check for strong technical signals
                    strong_signals = [
                        e
                        for e in events
                        if e["event_type"]
                        in [
                            "BB_LOWER_TOUCH",
                            "BB_UPPER_TOUCH",
                            "MACD_GOLDEN_CROSS",
                            "MACD_DEATH_CROSS",
                        ]
                    ]
                    if not strong_signals:
                        print(
                            f"⏭️ No position + sideways market + no strong signals, skipping LLM decision (Trend: {trend_analysis.dominant_trend})"
                        )
                        self.events_filtered_out += len(events)
                        continue
                    else:
                        print(
                            f"✅ Sideways market but with strong signals {[s['event_type'] for s in strong_signals]}, continuing LLM decision"
                        )

                # Optimization 2: Filter relevant events based on position status
                relevant_events = self._filter_relevant_events(
                    events, self.current_position
                )
                self.events_filtered_out += len(events) - len(relevant_events)

                if not relevant_events:
                    print(
                        f"⏭️ No relevant events (Position status: {'With position' if self.current_position else 'No position'}), skipping LLM decision"
                    )
                    continue

                # Optimization 3: Additional check - skip when holding position in uptrend without large bearish candles or other strong signals
                if self.current_position and trend_analysis.dominant_trend == "uptrend":
                    # Check for large bearish candles or other strong exit signals
                    has_large_drop = any(
                        event.get("event_type") == "LARGE_DROP"
                        for event in relevant_events
                    )
                    has_strong_exit_signal = any(
                        event.get("severity") == "high" for event in relevant_events
                    )

                    if not (has_large_drop or has_strong_exit_signal):
                        print(f"⏭️ Holding + uptrend + no large bearish candles or strong signals, skipping LLM decision")
                        continue

                print(
                    f"📋 Relevant events: {[e['event_type'] for e in relevant_events]} (Original events: {len(events)}, Filtered: {len(relevant_events)})"
                )
                print(
                    f"📈 Current trend: {trend_analysis.dominant_trend}, Position: {'With position' if self.current_position else 'No position'}"
                )

                # Decide whether to call LLM based on trend type and position status
                skip_llm = False
                skip_reason = ""

                has_position = self.current_position is not None

                if has_position:
                    # Position status: Risk management priority
                    if trend_analysis.dominant_trend == "downtrend":
                        # Position + downtrend: Need LLM to analyze stop loss/exit strategy
                        skip_llm = False
                        print(f"⚠️  Position encountering downtrend, calling LLM to analyze stop loss strategy")
                    elif trend_analysis.dominant_trend == "sideways":
                        # Position + sideways: Need LLM to find appropriate exit points
                        skip_llm = False
                        print(f"📊 Position encountering sideways, calling LLM to find best exit point")
                    elif trend_analysis.dominant_trend == "uptrend":
                        # Position + uptrend: Only need LLM judgment when encountering large bearish candles
                        has_large_drop = any(
                            event["event_type"] == "LARGE_DROP"
                            for event in relevant_events
                        )
                        if has_large_drop:
                            skip_llm = False
                            print(
                                f"⚠️  Position + uptrend + large bearish candle, calling LLM to analyze profit-taking opportunities"
                            )
                        else:
                            skip_llm = True
                            skip_reason = "Position + uptrend + no large bearish candles, continue holding"
                else:
                    # No position status: Entry timing selection
                    if trend_analysis.dominant_trend == "downtrend":
                        # No position + downtrend: Only call LLM when there are strong reversal signals
                        has_reversal_signal = any(
                            event.get("event_type")
                            in ["REVERSAL_PATTERN", "SUPPORT_BOUNCE"]
                            for event in relevant_events
                        )
                        if has_reversal_signal:
                            skip_llm = False
                            print(f"🔄 Reversal signal found in downtrend, calling LLM to analyze bottom-fishing opportunities")
                        else:
                            skip_llm = True
                            skip_reason = "No position + downtrend + no reversal signals, avoid counter-trend trading"
                    elif trend_analysis.dominant_trend == "sideways":
                        # No position + sideways: Relax conditions, increase more entry opportunities
                        has_breakout_signal = any(
                            event.get("event_type")
                            in [
                                "BREAKOUT",
                                "VOLUME_SPIKE",
                                "MOMENTUM_SHIFT",
                                "TREND_TURN_BULLISH",
                                "TREND_TURN_BEARISH",
                            ]
                            for event in relevant_events
                        )
                        has_strong_reversal = any(
                            event.get("event_type")
                            in ["BB_LOWER_TOUCH", "BB_UPPER_TOUCH"]
                            and event.get("severity") in ["high", "very_high"]
                            for event in relevant_events
                        )
                        has_macd_signal = any(
                            event.get("event_type")
                            in ["MACD_GOLDEN_CROSS", "MACD_DEATH_CROSS"]
                            for event in relevant_events
                        )
                        has_ma_signal = any(
                            event.get("event_type")
                            in ["MA_GOLDEN_CROSS", "MA_DEATH_CROSS"]
                            for event in relevant_events
                        )
                        has_multiple_signals = (
                            len(relevant_events) >= 2
                        )  # Multiple technical signals appearing simultaneously

                        # Relax conditions: Any technical signal is worth LLM analysis
                        if (
                            has_breakout_signal
                            or has_strong_reversal
                            or has_macd_signal
                            or has_ma_signal
                            or has_multiple_signals
                        ):
                            skip_llm = False
                            signal_types = [
                                event["event_type"] for event in relevant_events
                            ]
                            print(
                                f"✅ Technical signals detected in sideways market {signal_types}, calling LLM to analyze opportunities"
                            )
                        else:
                            skip_llm = True
                            skip_reason = "No position + sideways trend, waiting for clear breakout signals"
                    elif trend_analysis.dominant_trend == "uptrend":
                        # No position + uptrend: Normal LLM call to analyze entry opportunities
                        skip_llm = False
                        print(f"🚀 No position encountering uptrend, calling LLM to analyze entry opportunities")

                if skip_llm:
                    # Don't call LLM, but record events and reasons
                    self.llm_skipped_count += 1  # Increase skip counter
                    event_summary = ", ".join(
                        [e["event_type"] for e in relevant_events]
                    )
                    skip_msg = f"⏭️ {timestamp.strftime('%Y-%m-%d')} {skip_reason} (Detected events: {event_summary})"
                    print(skip_msg)

                    # Send skip progress message
                    if self.progress_callback:
                        self.progress_callback(
                            i, len(data), "llm_skipped", skip_msg, None
                        )
                    continue

                # Reanalyze trend at current time point - using Enhanced analysis
                print(f"🔍 Reanalyzing trend for {timestamp.strftime('%Y-%m-%d')}...")

                # Prioritize using Enhanced analysis, fallback to original analysis
                current_enhanced_analysis = None
                try:
                    # Use Enhanced Trend Analyzer for current time point analysis
                    symbol = self.current_symbol or "UNKNOWN"
                    current_date_str = timestamp.strftime("%Y-%m-%d")
                    current_enhanced_analysis = (
                        self.trend_analyzer.analyze_with_llm_optimization(
                            symbol, current_date_str
                        )
                    )
                    self.current_enhanced_analysis = current_enhanced_analysis
                    print(
                        f"✅ Enhanced trend analysis: {current_enhanced_analysis.market_phase}"
                    )
                    print(f"📊 Momentum status: {current_enhanced_analysis.momentum_status}")
                    print(
                        f"🎯 Reversal probability: {current_enhanced_analysis.reversal_probability:.3f}"
                    )

                    # Create compatible trend_analysis object for other code to use
                    current_trend_analysis = current_enhanced_analysis.original_result

                except Exception as e:
                    print(f"⚠️ Enhanced analysis failed, falling back to simplified analysis: {e}")
                    current_trend_analysis = self._analyze_current_trend(
                        historical_data, timestamp
                    )
                    current_enhanced_analysis = None

                self._last_trend_analysis = current_trend_analysis  # Store for risk checks

                if current_enhanced_analysis:
                    print(
                        f"📊 Enhanced trend analysis: {current_enhanced_analysis.market_phase}"
                    )
                else:
                    print(f"📊 Simplified trend analysis: {current_trend_analysis.dominant_trend}")

                # Call LLM for decision making
                self.llm_call_count += 1  # Increase counter

                # Send LLM decision start progress message
                if self.progress_callback:
                    llm_start_msg = (
                        f"🤖 {timestamp.strftime('%Y-%m-%d')} Starting LLM analysis..."
                    )
                    self.progress_callback(
                        i, len(data), "llm_decision", llm_start_msg, None
                    )

                llm_decision = self._make_llm_decision(
                    historical_data,
                    timestamp,  # Use processed timestamp
                    relevant_events,  # Use filtered events
                    current_trend_analysis,  # Pass compatible analysis result, but prompt will use enhanced
                )

                # Send LLM decision result
                if llm_decision:
                    action = llm_decision.get("action", "HOLD")
                    confidence = llm_decision.get("confidence", 0)
                    reason = llm_decision.get("reasoning", "No explanation")
                    decision_msg = f"🤖 {timestamp.strftime('%Y-%m-%d')} LLM decision: {action} (Confidence: {confidence:.2f}) - {reason}"
                    print(decision_msg)

                    if self.progress_callback:
                        self.progress_callback(
                            i, len(data), "llm_decision", decision_msg, None
                        )
                else:
                    print(f"🤖 {timestamp.strftime('%Y-%m-%d')} LLM decision: No clear recommendation")

                # Record log - daily analysis data
                if self.backtest_logger:
                    self._log_daily_analysis(
                        timestamp=timestamp,
                        historical_data=historical_data,
                        i=i,
                        events=events,
                        relevant_events=relevant_events,
                        trend_analysis=current_trend_analysis,
                        llm_decision=llm_decision,
                        comprehensive_context=getattr(
                            self, "current_comprehensive_context", None
                        ),
                    )

                if llm_decision and llm_decision.get("action") in ["BUY", "SELL"]:
                    # Check confidence threshold
                    confidence = llm_decision.get("confidence", 0)
                    if confidence >= self.confidence_threshold:
                        # Get current price
                        current_price = historical_data.iloc[-1]["close"]

                        # Use original decision
                        enhanced_decision = llm_decision.copy()

                        signal = self._create_signal_from_decision(
                            enhanced_decision,
                            timestamp,  # Use processed timestamp
                            current_price,
                        )
                        if signal:
                            signals.append(signal)
                            self.daily_trade_count += 1
                            signal_msg = f"✅ Generated trading signal: {signal.signal_type} (Confidence: {confidence:.2f} >= Threshold: {self.confidence_threshold:.2f})"
                            print(signal_msg)

                            # Record trading signal to log
                            if self.backtest_logger:
                                self._log_trading_signal(
                                    timestamp, signal, llm_decision
                                )

                            # Calculate current P&L status for frontend display
                            pnl_data = {}
                            if hasattr(self, "pnl_tracker") and self.pnl_tracker:
                                try:
                                    current_row = data.iloc[i]
                                    current_price = current_row.get(
                                        "close", current_row.get("Close", 0)
                                    )
                                    position_metrics = self._calculate_position_metrics(
                                        current_price, current_date
                                    )
                                    if position_metrics and position_metrics.get(
                                        "has_position"
                                    ):
                                        pnl_data = {
                                            "unrealized_pnl": position_metrics.get(
                                                "unrealized_pnl", 0
                                            ),
                                            "unrealized_pnl_pct": position_metrics.get(
                                                "unrealized_pnl_pct", 0
                                            ),
                                            "holding_days": position_metrics.get(
                                                "holding_days", 0
                                            ),
                                            "shares": position_metrics.get("shares", 0),
                                            "risk_level": position_metrics.get(
                                                "risk_level", "normal"
                                            ),
                                            "cash_remaining": self.cash,
                                            "total_value": self.cash
                                            + (
                                                position_metrics.get("shares", 0)
                                                * current_price
                                            ),
                                        }
                                    else:
                                        pnl_data = {
                                            "unrealized_pnl": 0,
                                            "unrealized_pnl_pct": 0,
                                            "holding_days": 0,
                                            "shares": 0,
                                            "risk_level": "normal",
                                            "cash_remaining": self.cash,
                                            "total_value": self.cash,
                                        }
                                except Exception as e:
                                    print(f"⚠️ P&L calculation failed: {e}")

                            # Send trading signal generation progress, including P&L information
                            if self.progress_callback:
                                extra_data = (
                                    {"pnl_status": pnl_data} if pnl_data else None
                                )
                                self.progress_callback(
                                    i,
                                    len(data),
                                    "signal_generated",
                                    signal_msg,
                                    extra_data,
                                )

                                # Send performance update immediately after signal generation
                                current_row = data.iloc[i]
                                current_price = current_row.get(
                                    "close", current_row.get("Close", 0)
                                )
                                if current_price > 0:
                                    self._send_performance_update(
                                        i, len(data), current_price
                                    )
                    else:
                        print(
                            f"❌ Insufficient confidence: {llm_decision.get('confidence', 0):.2f} < {self.confidence_threshold}"
                        )

        print(f"🎉 Signal generation completed! Total of {len(signals)} signals generated")

        # Output optimization statistics
        print(f"")
        print(f"📊 LLM call optimization statistics:")
        print(f"   📈 Total trading days: {len(data)} days")
        print(f"   🎯 Total detected events: {self.total_events_detected}")
        print(f"   🗑️ Filtered out events: {self.events_filtered_out}")
        print(f"   🤖 LLM actual calls: {self.llm_call_count}")
        print(f"   ⏭️  LLM skip count: {self.llm_skipped_count} (downtrend/sideways trend)")

        # Safely calculate efficiency, avoid division by zero
        data_length = len(data) if len(data) > 0 else 1
        print(f"   ⚡ Actual call efficiency: {self.llm_call_count / data_length:.3f} calls/day")

        total_potential_calls = self.llm_call_count + self.llm_skipped_count
        if total_potential_calls > 0:
            print(
                f"   🎯 Trend filtering rate: {self.llm_skipped_count / total_potential_calls:.1%}"
            )
        if self.total_events_detected > 0:
            print(
                f"   🎯 Event processing rate: {(self.total_events_detected - self.events_filtered_out) / self.total_events_detected:.1%}"
            )
        print(
            f"   💰 Cost savings: {(1 - self.llm_call_count / data_length) * 100:.1f}% (compared to daily calls)"
        )
        print(
            f"   💡 Intelligent savings: {(1 - self.llm_call_count / (self.llm_call_count + self.llm_skipped_count)) * 100:.1f}% (compared to calling for all events)"
            if total_potential_calls > 0
            else ""
        )

        return signals

    def _analyze_stock_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Deeply analyze stock characteristics (using early-stage data to intelligently determine stock behavior)

        Args:
            data: Stock data

        Returns:
            Stock characteristics analysis results
        """
        # Use data from first 3-6 months for analysis, but need at least 60 days
        analysis_days = max(60, self.analysis_period_months * 30)
        analysis_data = data.iloc[: min(analysis_days, len(data) // 2)]

        if len(analysis_data) < 30:
            analysis_data = data.iloc[:30] if len(data) >= 30 else data

        print(f"📈 Analyzing stock characteristics (using {len(analysis_data)} days of historical data)...")

        # Calculate basic statistical characteristics
        returns = analysis_data["close"].pct_change().dropna()
        prices = analysis_data["close"]

        # Ensure sufficient data for calculation
        if len(returns) < 2:
            logger.warning(
                f"Insufficient data for analysis: only {len(returns)} return values"
            )
            return None

        # 1. Volatility analysis (multi-dimensional) - correct pandas syntax
        daily_volatility = returns.std() if len(returns) > 1 else 0.0
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Calculate volatility of volatility, ensure sufficient rolling window data
        rolling_volatility = returns.rolling(10, min_periods=5).std()
        volatility_of_volatility = (
            rolling_volatility.std() if len(rolling_volatility.dropna()) > 1 else 0.0
        )

        # 2. Trend characteristic analysis
        trend_consistency = self._calculate_trend_consistency(analysis_data)
        trend_strength = self._calculate_trend_strength(analysis_data)

        # 3. Price behavior analysis - safe calculation to avoid division by zero and NaN
        price_mean = prices.mean()
        if price_mean > 0:
            price_range_ratio = (prices.max() - prices.min()) / price_mean
        else:
            price_range_ratio = 0.0

        avg_daily_return = returns.mean() if len(returns) > 0 else 0.0
        skewness = (
            returns.skew() if len(returns) > 2 else 0.0
        )  # Skewness: positive value indicates right skew (more upticks)
        kurtosis = (
            returns.kurtosis() if len(returns) > 3 else 0.0
        )  # Kurtosis: measures frequency of extreme values

        # 4. Reversal characteristics
        reversal_frequency = self._calculate_reversal_frequency(analysis_data)
        consecutive_days = self._calculate_consecutive_move_tendency(returns)

        # 5. Volume characteristics - safe calculation to avoid NaN
        volume_volatility = 0.0
        volume_price_correlation = 0.0
        if "volume" in analysis_data.columns and len(analysis_data["volume"]) > 1:
            volume_changes = analysis_data["volume"].pct_change().dropna()
            if len(volume_changes) > 1:
                volume_volatility = volume_changes.std()
                # Safely calculate correlation, ensure aligned indices
                common_index = returns.index.intersection(volume_changes.index)
                if len(common_index) > 1:
                    aligned_returns = returns.reindex(common_index)
                    aligned_volume = volume_changes.reindex(common_index)
                    volume_price_correlation = aligned_returns.corr(aligned_volume)

        # 6. Technical indicator responsiveness testing
        macd_effectiveness = self._test_macd_effectiveness(analysis_data)
        ma_crossover_effectiveness = self._test_ma_crossover_effectiveness(
            analysis_data
        )
        bb_effectiveness = self._test_bollinger_bands_effectiveness(analysis_data)

        # 7. Support resistance analysis
        support_resistance_strength = self._analyze_support_resistance(analysis_data)
        breakout_tendency = self._analyze_breakout_tendency(analysis_data)

        # 8. Stock personality classification
        stock_personality = self._classify_stock_personality(
            annualized_volatility,
            trend_consistency,
            reversal_frequency,
            macd_effectiveness,
        )

        characteristics = {
            # Volatility indicators
            "volatility": annualized_volatility,
            "daily_volatility": daily_volatility,
            "volatility_of_volatility": volatility_of_volatility,
            # Return characteristics
            "avg_daily_return": avg_daily_return,
            "annualized_return": avg_daily_return * 252,
            "sharpe_ratio": avg_daily_return / daily_volatility
            if daily_volatility > 0
            else 0,
            "skewness": skewness,
            "kurtosis": kurtosis,
            # Trend characteristics
            "trend_consistency": trend_consistency,
            "trend_strength": trend_strength,
            "reversal_frequency": reversal_frequency,
            "consecutive_move_tendency": consecutive_days,
            # Price behavior
            "price_range_ratio": price_range_ratio,
            "breakout_tendency": breakout_tendency,
            # Volume characteristics
            "volume_volatility": volume_volatility,
            "volume_price_correlation": volume_price_correlation,
            # Technical indicator responsiveness
            "macd_effectiveness": macd_effectiveness,
            "ma_crossover_effectiveness": ma_crossover_effectiveness,
            "bollinger_effectiveness": bb_effectiveness,
            # Support resistance
            "support_resistance_strength": support_resistance_strength,
            # Comprehensive classification
            "stock_personality": stock_personality,
        }

        # Output analysis results
        print(f"📊 Stock characteristics analysis completed:")
        print(f"   Annualized volatility: {annualized_volatility:.1%}")
        print(f"   Annualized return: {characteristics['annualized_return']:.1%}")
        print(f"   Sharpe ratio: {characteristics['sharpe_ratio']:.2f}")
        print(f"   Trend consistency: {trend_consistency:.2f}")
        print(f"   Reversal frequency: {reversal_frequency:.2f}")
        print(f"   Stock personality: {stock_personality}")
        print(f"   MACD effectiveness: {macd_effectiveness:.2f}")

        return characteristics

    def _adjust_technical_parameters(self):
        """Intelligently adjust technical indicator parameters based on stock characteristics"""
        if not self.stock_characteristics:
            return

        print(f"📊 Stock characteristics analysis results:")
        print(f"   Volatility: {self.stock_characteristics['volatility']:.3f}")
        print(f"   Trend consistency: {self.stock_characteristics['trend_consistency']:.3f}")
        print(f"   Reversal frequency: {self.stock_characteristics['reversal_frequency']:.3f}")
        print(
            f"   MACD effectiveness: {self.stock_characteristics.get('macd_effectiveness', 0.5):.3f}"
        )

        # Save original parameters as baseline
        original_ma_short = self.ma_short
        original_ma_long = self.ma_long
        original_macd_fast = self.macd_fast
        original_macd_slow = self.macd_slow

        # 1. Adjust MACD parameters based on trend consistency
        trend_consistency = self.stock_characteristics["trend_consistency"]
        print(f"🔍 Adjusting MACD parameters - Trend consistency: {trend_consistency:.3f}")
        if trend_consistency > 0.8:  # Extremely strong trend
            self.macd_fast = 6  # Quickly capture trend
            self.macd_slow = 18
            print(f"   Extremely strong trend -> MACD set to 6/18")
        elif trend_consistency > 0.6:  # Strong trend
            self.macd_fast = 8
            self.macd_slow = 21
            print(f"   Strong trend -> MACD set to 8/21")
        elif trend_consistency > 0.4:  # Moderate trend
            self.macd_fast = 12  # Standard setting
            self.macd_slow = 26
            print(f"   Moderate trend -> MACD set to 12/26")
        elif trend_consistency > 0.2:  # Weak trend, biased towards sideways
            self.macd_fast = 15
            self.macd_slow = 35
            print(f"   Weak trend -> MACD set to 15/35")
        else:  # Strong sideways
            self.macd_fast = 20  # Longer period, reduce false signals
            self.macd_slow = 45
            print(f"   Strong sideways -> MACD set to 20/45")

        # 3. Adjust moving average parameters based on reversal frequency
        reversal_freq = self.stock_characteristics["reversal_frequency"]
        print(f"🔍 Adjusting moving average parameters - Reversal frequency: {reversal_freq:.3f}")
        if reversal_freq > 0.15:  # High reversal frequency - volatile stock
            self.ma_short = max(5, self.ma_short - 2)  # Shorten period
            self.ma_long = max(15, self.ma_long - 5)
            print(f"   High reversal frequency -> Shorten moving average period")
        elif reversal_freq < 0.05:  # Low reversal frequency - trending stock
            self.ma_short = min(20, self.ma_short + 3)  # Extend period
            self.ma_long = min(50, self.ma_long + 10)
            print(f"   Low reversal frequency -> Extend moving average period")

        # 4. Further fine-tune based on technical indicator effectiveness
        macd_effectiveness = self.stock_characteristics.get("macd_effectiveness", 0.5)

        # If MACD effect is poor, use more conservative parameters
        if macd_effectiveness < 0.4:
            self.macd_fast = min(20, int(self.macd_fast * 1.2))
            self.macd_slow = min(50, int(self.macd_slow * 1.1))
            print(f"   Poor MACD effectiveness -> Conservative parameters {self.macd_fast}/{self.macd_slow}")

        # 5. Price range adjustment - consider stock price fluctuation range
        print(f"\n🔧 Intelligent adjustment of technical indicator parameters:")
        print(f"   MACD fast line: {original_macd_fast} → {self.macd_fast}")
        print(f"   MACD slow line: {original_macd_slow} → {self.macd_slow}")
        print(f"   Short-term moving average: {original_ma_short} → {self.ma_short}")
        print(f"   Long-term moving average: {original_ma_long} → {self.ma_long}")

        # Ensure parameter reasonableness
        self.macd_fast = max(3, min(20, self.macd_fast))
        self.macd_slow = max(10, min(50, self.macd_slow))
        self.ma_short = max(3, min(20, self.ma_short))
        self.ma_long = max(10, min(50, self.ma_long))

        # Ensure fast line < slow line
        if self.macd_fast >= self.macd_slow:
            self.macd_slow = self.macd_fast + 5
        if self.ma_short >= self.ma_long:
            self.ma_long = self.ma_short + 5

        print(f"\n✅ Final parameters (after range check):")
        print(f"   MACD: {self.macd_fast}/{self.macd_slow}")
        print(f"   Moving averages: {self.ma_short}/{self.ma_long}")

    def _calculate_position_metrics(
        self, current_price: float, current_date: pd.Timestamp = None
    ) -> Dict[str, Any]:
        """Calculate detailed metrics for current position"""
        if not self.current_position:
            return {
                "has_position": False,
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "holding_days": 0,
                "shares": 0,
                "position_value": 0.0,
                "risk_level": "normal",
            }

        # Calculate unrealized P&L
        position_value = self.shares * current_price
        unrealized_pnl = position_value - (self.shares * self.position_entry_price)
        unrealized_pnl_pct = (
            unrealized_pnl / (self.shares * self.position_entry_price) * 100
        )

        # Calculate holding days
        if self.position_entry_date and current_date:
            if isinstance(self.position_entry_date, str):
                entry_date = datetime.strptime(self.position_entry_date, "%Y-%m-%d")
            elif hasattr(self.position_entry_date, "date"):
                entry_date = self.position_entry_date
            else:
                entry_date = pd.to_datetime(self.position_entry_date)

            if hasattr(current_date, "date"):
                current_date_obj = current_date
            else:
                current_date_obj = pd.to_datetime(current_date)

            holding_days = (current_date_obj - entry_date).days
        else:
            holding_days = 0

        # Risk level assessment
        risk_level = self._assess_risk_level(unrealized_pnl_pct, holding_days)

        return {
            "has_position": True,
            "entry_price": self.position_entry_price,
            "current_price": current_price,
            "shares": self.shares,
            "position_value": position_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "holding_days": holding_days,
            "risk_level": risk_level,
            "cost_basis": self.shares * self.position_entry_price,
        }

    def _calculate_current_performance(self, current_price: float) -> Dict[str, float]:
        """Calculate current overall performance metrics"""
        # Calculate current total value
        position_value = self.shares * current_price if self.shares > 0 else 0
        total_value = self.cash + position_value

        # Calculate total return rate
        total_return = (total_value - self.initial_capital) / self.initial_capital

        # Calculate win rate
        win_rate = (
            self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        )

        # Calculate cumulative trade return rate (sum of each trade's return rate)
        cumulative_trade_return_rate = sum(self.trade_returns) / 100  # Convert to decimal form

        # Update highest point and drawdown
        if total_value > self.max_portfolio_value:
            self.max_portfolio_value = total_value

        # Calculate current drawdown
        current_drawdown = (
            (self.max_portfolio_value - total_value) / self.max_portfolio_value
            if self.max_portfolio_value > 0
            else 0
        )
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "total_value": total_value,
            "cash": self.cash,
            "position_value": position_value,
            "total_realized_pnl": self.total_realized_pnl,  # Cumulative realized P&L
            "cumulative_trade_return_rate": cumulative_trade_return_rate,  # Cumulative trade return rate
        }

    def _send_performance_update(self, day: int, total_days: int, current_price: float):
        """Send performance update message"""
        if not self.progress_callback:
            return

        performance = self._calculate_current_performance(current_price)

        # Calculate current P&L status
        pnl_status = None
        if hasattr(self, "pnl_tracker") and self.pnl_tracker and current_price > 0:
            try:
                position_metrics = self._calculate_position_metrics(current_price)
                if position_metrics["has_position"]:
                    pnl_status = {
                        "unrealized_pnl": position_metrics["unrealized_pnl"],
                        "unrealized_pnl_pct": position_metrics["unrealized_pnl_pct"],
                        "holding_days": position_metrics["holding_days"],
                        "shares": position_metrics["shares"],
                        "risk_level": position_metrics["risk_level"],
                        "cash_remaining": self.cash,
                        "total_value": self.cash + position_metrics["position_value"],
                    }
                else:
                    # Send complete P&L status even when no position
                    pnl_status = {
                        "unrealized_pnl": 0,
                        "unrealized_pnl_pct": 0,
                        "holding_days": 0,
                        "shares": 0,
                        "risk_level": "normal",
                        "cash_remaining": self.cash,
                        "total_value": self.cash,
                    }
            except Exception as e:
                print(f"⚠️ P&L calculation failed: {e}")

        # Build message matching frontend expected format
        message = f"Total return: {performance['total_return'] * 100:+.2f}% | Win rate: {performance['win_rate'] * 100:.1f}% | Max drawdown: {performance['max_drawdown'] * 100:.2f}%"

        # Also send detailed data in extra_data
        extra_data = {"performance_metrics": performance, "pnl_status": pnl_status}

        self._send_progress(day, total_days, "performance_update", message, extra_data)

    def _assess_risk_level(self, pnl_pct: float, holding_days: int) -> str:
        """Assess risk level of current position"""
        if pnl_pct <= -self.max_loss_threshold * 100:
            return "high_loss"  # High loss risk
        elif pnl_pct <= -2:
            return "moderate_loss"  # Moderate loss
        elif pnl_pct >= self.profit_taking_threshold * 100:
            return "high_profit"  # High profit
        elif pnl_pct >= 8:
            return "moderate_profit"  # Moderate profit
        elif holding_days > 30:
            return "long_hold"  # Long-term holding
        else:
            return "normal"  # Normal status

    def _generate_pnl_insights(
        self, position_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate investment insights based on unrealized P&L"""
        if not position_metrics["has_position"]:
            return {
                "pnl_signal": "neutral",
                "risk_warning": None,
                "suggested_action": "Can consider new position",
                "position_sizing_factor": 1.0,
            }

        pnl_pct = position_metrics["unrealized_pnl_pct"]
        risk_level = position_metrics["risk_level"]
        holding_days = position_metrics["holding_days"]

        insights = {
            "pnl_signal": "neutral",
            "risk_warning": None,
            "suggested_action": "Continue holding",
            "position_sizing_factor": 1.0,
        }

        # Provide suggestions based on P&L status
        if risk_level == "high_loss":
            insights.update(
                {
                    "pnl_signal": "stop_loss",
                    "risk_warning": f"Loss has reached {pnl_pct:.1f}%, consider stop loss",
                    "suggested_action": "Immediately evaluate stop loss",
                    "position_sizing_factor": 0.5,
                }
            )
        elif risk_level == "moderate_loss":
            insights.update(
                {
                    "pnl_signal": "caution",
                    "risk_warning": f"Current loss {pnl_pct:.1f}%, operate cautiously",
                    "suggested_action": "Carefully evaluate subsequent strategy",
                    "position_sizing_factor": 0.7,
                }
            )
        elif risk_level == "high_profit":
            insights.update(
                {
                    "pnl_signal": "take_profit",
                    "risk_warning": None,
                    "suggested_action": f"Profit reached {pnl_pct:.1f}%, consider taking profit",
                    "position_sizing_factor": 0.8,
                }
            )
        elif risk_level == "moderate_profit":
            insights.update(
                {
                    "pnl_signal": "bullish",
                    "risk_warning": None,
                    "suggested_action": f"Profit {pnl_pct:.1f}%, good performance",
                    "position_sizing_factor": 1.2,
                }
            )
        elif risk_level == "long_hold":
            insights.update(
                {
                    "pnl_signal": "review",
                    "risk_warning": f"Position held for {holding_days} days, recommend re-evaluation",
                    "suggested_action": "Review if position strategy needs adjustment",
                    "position_sizing_factor": 0.9,
                }
            )

        return insights

    def _update_position_state(
        self, action: str, price: float, quantity: int, date: str
    ):
        """Update position state"""
        if action == "BUY":
            if self.current_position is None:
                # New position opening
                self.current_position = "long"
                self.position_entry_price = price
                self.position_entry_date = date
                self.shares = quantity
                self.cash -= quantity * price
                print(f"📈 Position opened: {quantity} shares @ ${price:.2f}")
            else:
                # Add to position (simplified, directly average cost)
                total_cost = self.shares * self.position_entry_price + quantity * price
                self.shares += quantity
                self.position_entry_price = total_cost / self.shares
                self.cash -= quantity * price
                print(
                    f"📈 Position added: +{quantity} shares @ ${price:.2f}, average cost: ${self.position_entry_price:.2f}"
                )

        elif action == "SELL":
            if self.current_position is not None:
                # Calculate realized P&L
                sell_value = quantity * price
                cost_basis = quantity * self.position_entry_price
                realized_pnl = sell_value - cost_basis
                realized_pnl_pct = (realized_pnl / cost_basis) * 100

                self.cash += sell_value
                self.shares -= quantity

                if self.shares <= 0:
                    # Completely close position
                    print(
                        f"📉 Position closed: {quantity} shares @ ${price:.2f}, realized P&L: ${realized_pnl:,.0f} ({realized_pnl_pct:+.1f}%)"
                    )
                    self.current_position = None
                    self.position_entry_price = 0.0
                    self.position_entry_date = None
                    self.shares = 0
                else:
                    # Partially close position
                    print(
                        f"📉 Position reduced: -{quantity} shares @ ${price:.2f}, realized P&L: ${realized_pnl:,.0f} ({realized_pnl_pct:+.1f}%)"
                    )

                # Note: Trade statistics updated in _create_signal_from_decision, not duplicated here

    def calculate_position_size(self, price: float) -> int:
        """Calculate suggested position size - fixed 1000 shares"""
        return 1000

    def _calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        data = data.copy()

        # MACD
        macd_data = calculate_macd(
            data,
            short_period=self.macd_fast,
            long_period=self.macd_slow,
            signal_period=self.macd_signal,
        )
        data["macd"] = macd_data["macd"]
        data["macd_signal"] = macd_data["macd_signal"]
        data["macd_histogram"] = macd_data["macd_histogram"]

        # Bollinger Bands
        bb_data = calculate_bollinger_bands(data)
        data["bb_upper"] = bb_data["bb_upper"]
        data["bb_middle"] = bb_data["bb_middle"]
        data["bb_lower"] = bb_data["bb_lower"]

        # Moving averages
        data[f"ma_{self.ma_short}"] = data["close"].rolling(window=self.ma_short).mean()
        data[f"ma_{self.ma_long}"] = data["close"].rolling(window=self.ma_long).mean()
        data["ma_20"] = data["close"].rolling(window=20).mean()
        data["ma_50"] = data["close"].rolling(window=50).mean()

        return data

    def _analyze_current_trend(
        self, historical_data: pd.DataFrame, current_date
    ) -> Any:
        """
        Analyze trend at current time point - simplified but effective real-time trend analysis

        Args:
            historical_data: Historical data (up to current time point)
            current_date: Current date

        Returns:
            Trend analysis result at current time point
        """
        try:
            # Ensure sufficient data
            if len(historical_data) < 20:
                print(f"⚠️ Insufficient data for trend analysis ({len(historical_data)} < 20)")
                from types import SimpleNamespace

                trend_analysis = SimpleNamespace()
                trend_analysis.dominant_trend = "sideways"
                trend_analysis.complexity_score = 0.5
                trend_analysis.confidence = 0.3
                return trend_analysis

            # Use multi-timeframe analysis
            data = historical_data.copy()

            # Standardize column names (handle case issues)
            column_mapping = {}
            for col in data.columns:
                if col.lower() == "close":
                    column_mapping[col] = "close"
                elif col.lower() == "open":
                    column_mapping[col] = "open"
                elif col.lower() == "high":
                    column_mapping[col] = "high"
                elif col.lower() == "low":
                    column_mapping[col] = "low"
                elif col.lower() == "volume":
                    column_mapping[col] = "volume"

            data = data.rename(columns=column_mapping)

            # Ensure close price data exists
            if "close" not in data.columns:
                if "Close" in data.columns:
                    data["close"] = data["Close"]
                else:
                    raise ValueError("Cannot find price data (close/Close column)")

            prices = data["close"]

            # Calculate trends across multiple timeframes
            windows = [5, 10, 20]  # Short-term, medium-term, long-term
            trends = []
            trend_strengths = []

            for window in windows:
                if len(prices) >= window + 2:
                    # Use linear regression to calculate trend
                    recent_prices = prices.tail(window)
                    x = np.arange(len(recent_prices))
                    y = recent_prices.values

                    if len(y) > 1:
                        slope, _ = np.polyfit(x, y, 1)
                        correlation = np.corrcoef(x, y)[0, 1] if len(y) > 1 else 0

                        # Normalize slope
                        normalized_slope = slope / recent_prices.mean()

                        # Determine trend direction
                        if abs(normalized_slope) < 0.001:  # Almost no trend
                            trend_direction = "sideways"
                        elif normalized_slope > 0:
                            trend_direction = "uptrend"
                        else:
                            trend_direction = "downtrend"

                        trends.append(trend_direction)
                        trend_strengths.append(abs(correlation))

            # Determine dominant trend
            if not trends:
                dominant_trend = "sideways"
                confidence = 0.3
            else:
                # Count occurrences and strength of various trends
                trend_counts = {"uptrend": 0, "downtrend": 0, "sideways": 0}
                weighted_scores = {"uptrend": 0.0, "downtrend": 0.0, "sideways": 0.0}

                for trend, strength in zip(trends, trend_strengths):
                    trend_counts[trend] += 1
                    weighted_scores[trend] += strength

                # Find trend with highest weighted score
                dominant_trend = max(weighted_scores, key=weighted_scores.get)

                # Calculate confidence
                total_strength = sum(trend_strengths)
                if total_strength > 0:
                    confidence = weighted_scores[dominant_trend] / total_strength
                    confidence = min(confidence, 1.0)
                else:
                    confidence = 0.3

            # Check price momentum to confirm trend
            trend_reversal_detected = False
            reversal_strength = 0.0

            if len(prices) >= 15:
                # Detect trend conversion signals - balance upward and downward detection thresholds
                short_term_change = (
                    prices.iloc[-5:].mean() - prices.iloc[-10:-5].mean()
                ) / prices.iloc[-10:-5].mean()
                medium_term_change = (
                    prices.iloc[-10:].mean() - prices.iloc[-20:-10].mean()
                ) / prices.iloc[-20:-10].mean()

                print(
                    f"📊 Conversion signal calculation: Short-term change={short_term_change:.4f} ({short_term_change:.2%}), Medium-term change={medium_term_change:.4f} ({medium_term_change:.2%})"
                )

                # Balance upward and downward conversion detection, use same threshold
                reversal_threshold = 0.02  # Unified 2% threshold
                counter_threshold = 0.01  # Unified 1% reverse threshold

                if (
                    short_term_change > reversal_threshold
                    and medium_term_change < -counter_threshold
                ):
                    trend_reversal_detected = True
                    reversal_strength = abs(short_term_change)
                    print(
                        f"🔄 Upward conversion signal detected: Short-term change {short_term_change:.2%}, Medium-term change {medium_term_change:.2%} -> Conversion strength {reversal_strength:.2%}"
                    )
                elif (
                    short_term_change < -reversal_threshold
                    and medium_term_change > counter_threshold
                ):
                    trend_reversal_detected = True
                    reversal_strength = abs(short_term_change)
                    print(
                        f"🔄 Downward conversion signal detected: Short-term change {short_term_change:.2%}, Medium-term change {medium_term_change:.2%} -> Conversion strength {reversal_strength:.2%}"
                    )

                # Additional detection: if current trend differs from previous period trend
                if len(prices) >= 25:
                    very_recent = prices.iloc[-5:].mean()
                    recent = prices.iloc[-10:-5].mean()
                    older = prices.iloc[-15:-10].mean()
                    much_older = prices.iloc[-25:-15].mean()

                    recent_trend = (very_recent - recent) / recent
                    older_trend = (older - much_older) / much_older

                    # Use same threshold for bidirectional trend change detection
                    trend_change_threshold = 0.02  # Unified threshold
                    counter_trend_threshold = 0.015  # Unified reverse threshold

                    if (
                        recent_trend > trend_change_threshold
                        and older_trend < -counter_trend_threshold
                    ):
                        trend_reversal_detected = True
                        reversal_strength = max(reversal_strength, abs(recent_trend))
                        print(
                            f"🔄 Trend direction change detected (upward): Recent {recent_trend:.2%} vs Earlier {older_trend:.2%} -> Conversion strength {reversal_strength:.2%}"
                        )
                    elif (
                        recent_trend < -trend_change_threshold
                        and older_trend > counter_trend_threshold
                    ):
                        trend_reversal_detected = True
                        reversal_strength = max(reversal_strength, abs(recent_trend))
                        print(
                            f"🔄 Trend direction change detected (downward): Recent {recent_trend:.2%} vs Earlier {older_trend:.2%} -> Conversion strength {reversal_strength:.2%}"
                        )

            # Price momentum check - modified to not forcibly override trend, only as confirmation
            momentum_factor = 1.0
            if len(prices) >= 10:
                recent_change = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10]
                momentum_trend = "uptrend" if recent_change > 0 else "downtrend"

                # Momentum confirmation logic - changed to adjust confidence rather than forcibly change trend
                if abs(recent_change) > 0.08:  # Increase threshold to 8%, reduce misjudgment
                    if recent_change > 0 and dominant_trend == "uptrend":
                        print(f"🔄 Price momentum confirms upward trend (change: {recent_change:.2%})")
                        confidence = min(confidence + 0.15, 1.0)
                        momentum_factor = 1.2
                    elif recent_change < 0 and dominant_trend == "downtrend":
                        print(f"🔄 Price momentum confirms downward trend (change: {recent_change:.2%})")
                        confidence = min(confidence + 0.15, 1.0)
                        momentum_factor = 1.2
                    elif abs(recent_change) > 0.12:  # Only consider overriding original trend with extremely strong momentum
                        if recent_change > 0 and dominant_trend == "downtrend":
                            print(
                                f"🚨 Extremely strong upward momentum overrides downward trend (change: {recent_change:.2%})"
                            )
                            dominant_trend = "uptrend"
                            confidence = 0.7
                        elif recent_change < 0 and dominant_trend == "uptrend":
                            print(
                                f"🚨 Extremely strong downward momentum overrides upward trend (change: {recent_change:.2%})"
                            )
                            dominant_trend = "downtrend"
                            confidence = 0.7

            # Calculate complexity score
            unique_trends = len(set(trends))
            complexity_score = unique_trends / len(windows) if windows else 0.5

            # Create trend analysis result
            from types import SimpleNamespace

            trend_analysis = SimpleNamespace()
            trend_analysis.dominant_trend = dominant_trend
            trend_analysis.confidence = confidence
            trend_analysis.complexity_score = complexity_score

            # Add trend conversion information
            trend_analysis.trend_reversal_detected = trend_reversal_detected
            trend_analysis.reversal_strength = reversal_strength

            print(
                f"🎯 Real-time trend analysis: {dominant_trend} (Confidence: {confidence:.2f}, Complexity: {complexity_score:.2f})"
            )
            if trend_reversal_detected:
                print(f"⚡ Trend conversion detected: Strength {reversal_strength:.2%}")

            return trend_analysis

        except Exception as e:
            print(f"❌ Real-time trend analysis failed: {e}")
            import traceback

            traceback.print_exc()

            # Return backup result
            from types import SimpleNamespace

            trend_analysis = SimpleNamespace()
            trend_analysis.dominant_trend = "sideways"
            trend_analysis.complexity_score = 0.5
            trend_analysis.confidence = 0.2
            return trend_analysis

    def _detect_trigger_events(
        self, data: pd.DataFrame, current_index: int
    ) -> List[Dict[str, Any]]:
        """
        Detect trigger events

        Args:
            data: Historical data
            current_index: Current data index

        Returns:
            List of trigger events
        """
        events = []
        current = data.iloc[current_index]
        prev = data.iloc[current_index - 1] if current_index > 0 else current

        # MACD trigger events
        if (
            current["macd"] > current["macd_signal"]
            and prev["macd"] <= prev["macd_signal"]
        ):
            events.append(
                {
                    "event_type": "MACD_GOLDEN_CROSS",
                    "severity": "high"
                    if current["macd_histogram"] > 0.01
                    else "medium",
                    "description": "MACD golden cross signal",
                    "technical_data": {
                        "indicator": "MACD_GOLDEN_CROSS",
                        "value": None,
                        "threshold": None,
                        "strength": "high"
                        if current["macd_histogram"] > 0.01
                        else "medium",
                    },
                }
            )
        elif (
            current["macd"] < current["macd_signal"]
            and prev["macd"] >= prev["macd_signal"]
        ):
            events.append(
                {
                    "event_type": "MACD_DEATH_CROSS",
                    "severity": "high"
                    if current["macd_histogram"] < -0.01
                    else "medium",
                    "description": "MACD death cross signal",
                    "technical_data": {
                        "indicator": "MACD_DEATH_CROSS",
                        "value": None,
                        "threshold": None,
                        "strength": "high"
                        if current["macd_histogram"] < -0.01
                        else "medium",
                    },
                }
            )

        # Bollinger Bands trigger events
        if current["close"] <= current["bb_lower"] and prev["close"] > prev["bb_lower"]:
            events.append(
                {
                    "event_type": "BB_LOWER_TOUCH",
                    "severity": "high",
                    "description": "Price touches Bollinger lower band",
                    "technical_data": {
                        "indicator": "BB_LOWER_TOUCH",
                        "value": None,
                        "threshold": None,
                        "strength": "high",
                    },
                }
            )
        elif (
            current["close"] >= current["bb_upper"] and prev["close"] < prev["bb_upper"]
        ):
            events.append(
                {
                    "event_type": "BB_UPPER_TOUCH",
                    "severity": "high",
                    "description": "Price touches Bollinger upper band",
                    "technical_data": {
                        "indicator": "BB_UPPER_TOUCH",
                        "value": None,
                        "threshold": None,
                        "strength": "high",
                    },
                }
            )

        # Volume analysis events
        if len(data) >= 10:
            recent_volume = data["volume"].tail(10).mean()
            if current.get("volume", 0) > recent_volume * 2:
                events.append(
                    {
                        "event_type": "VOLUME_SPIKE",
                        "severity": "medium",
                        "description": f"Volume spike ({current.get('volume', 0) / recent_volume:.1f} times)",
                        "technical_data": {
                            "indicator": "VOLUME_SPIKE",
                            "current_volume": int(current.get("volume", 0)),
                            "avg_volume": int(recent_volume),
                            "ratio": float(current.get("volume", 0) / recent_volume),
                            "strength": "high"
                            if current.get("volume", 0) > recent_volume * 3
                            else "medium",
                        },
                    }
                )

        # Price breakout detection
        if len(data) >= 20:
            high_20 = data["high"].tail(20).max()
            low_20 = data["low"].tail(20).min()

            if current["close"] > high_20 and prev["close"] <= high_20:
                events.append(
                    {
                        "event_type": "PRICE_BREAKOUT_HIGH",
                        "severity": "high",
                        "description": f"Breaks 20-day high ({high_20:.2f})",
                        "technical_data": {
                            "indicator": "PRICE_BREAKOUT_HIGH",
                            "breakout_level": float(high_20),
                            "current_price": float(current["close"]),
                            "strength": "high",
                        },
                    }
                )
            elif current["close"] < low_20 and prev["close"] >= low_20:
                events.append(
                    {
                        "event_type": "PRICE_BREAKDOWN_LOW",
                        "severity": "high",
                        "description": f"Falls below 20-day low ({low_20:.2f})",
                        "technical_data": {
                            "indicator": "PRICE_BREAKDOWN_LOW",
                            "breakdown_level": float(low_20),
                            "current_price": float(current["close"]),
                            "strength": "high",
                        },
                    }
                )

        # Trend reversal events (using configured moving average parameters)
        ma_short_key = f"ma_{self.ma_short}"
        ma_long_key = f"ma_{self.ma_long}"

        if (
            ma_short_key in current
            and ma_long_key in current
            and ma_short_key in prev
            and ma_long_key in prev
        ):
            if (
                current[ma_short_key] > current[ma_long_key]
                and prev[ma_short_key] <= prev[ma_long_key]
            ):
                events.append(
                    {
                        "event_type": "MA_GOLDEN_CROSS",
                        "severity": "medium",
                        "description": f"Short-term MA({self.ma_short}) crosses above long-term MA({self.ma_long})",
                        "technical_data": {
                            "indicator": "MA_GOLDEN_CROSS",
                            "ma_short": float(current[ma_short_key]),
                            "ma_long": float(current[ma_long_key]),
                            "strength": "medium",
                        },
                    }
                )
            elif (
                current[ma_short_key] < current[ma_long_key]
                and prev[ma_short_key] >= prev[ma_long_key]
            ):
                events.append(
                    {
                        "event_type": "MA_DEATH_CROSS",
                        "severity": "medium",
                        "description": f"Short-term MA({self.ma_short}) crosses below long-term MA({self.ma_long})",
                        "technical_data": {
                            "indicator": "MA_DEATH_CROSS",
                            "ma_short": float(current[ma_short_key]),
                            "ma_long": float(current[ma_long_key]),
                            "strength": "medium",
                        },
                    }
                )

        # Large bearish candle detection (single day drop 8% or more)
        if prev["close"] > 0:  # Avoid division by zero error
            daily_return = (current["close"] - prev["close"]) / prev["close"]
            if daily_return <= -0.08:  # Drop 8% or more
                events.append(
                    {
                        "event_type": "LARGE_DROP",
                        "severity": "high",
                        "description": f"Large bearish candle: Single day drop {daily_return * 100:.2f}%",
                        "technical_data": {
                            "indicator": "LARGE_DROP",
                            "daily_return": float(daily_return),
                            "magnitude": float(abs(daily_return)),
                            "strength": "high",
                        },
                    }
                )
            elif daily_return >= 0.08:  # Rise 8% or more
                events.append(
                    {
                        "event_type": "LARGE_GAIN",
                        "severity": "high",
                        "description": f"Large bullish candle: Single day rise {daily_return * 100:.2f}%",
                        "technical_data": {
                            "indicator": "LARGE_GAIN",
                            "daily_return": float(daily_return),
                            "magnitude": float(daily_return),
                            "strength": "high",
                        },
                    }
                )

        # Keep original 20/50 moving average crossover detection
        if current["ma_20"] > current["ma_50"] and prev["ma_20"] <= prev["ma_50"]:
            events.append(
                {
                    "event_type": "TREND_TURN_BULLISH",
                    "severity": "medium",
                    "description": "20-day MA crosses above 50-day MA",
                    "technical_data": {
                        "indicator": "TREND_TURN_BULLISH",
                        "ma20": float(current["ma_20"]),
                        "ma50": float(current["ma_50"]),
                        "strength": "medium",
                    },
                }
            )
        elif current["ma_20"] < current["ma_50"] and prev["ma_20"] >= prev["ma_50"]:
            events.append(
                {
                    "event_type": "TREND_TURN_BEARISH",
                    "severity": "medium",
                    "description": "20-day MA crosses below 50-day MA",
                    "technical_data": {
                        "indicator": "TREND_TURN_BEARISH",
                        "ma20": float(current["ma_20"]),
                        "ma50": float(current["ma_50"]),
                        "strength": "medium",
                    },
                }
            )

        return events

    def _filter_relevant_events(
        self, events: List[Dict[str, Any]], current_position: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter relevant events based on position status to reduce unnecessary LLM calls

        Args:
            events: All detected events
            current_position: Current position status ('long' or None)

        Returns:
            List of events relevant to current position status
        """
        if not events:
            return []

        # If no position, focus on buy signal related events (relax filtering)
        if not current_position:
            buy_events = [
                "MACD_GOLDEN_CROSS",
                "RSI_OVERSOLD",
                "BB_LOWER_TOUCH",
                "PRICE_ABOVE_MA20",
                "VOLUME_SPIKE",
                "BULLISH_DIVERGENCE",
                "MA_GOLDEN_CROSS",
                "TREND_TURN_BULLISH",  # Add more buy related events
            ]
            filtered = [
                event
                for event in events
                if any(buy_event in str(event) for buy_event in buy_events)
            ]
            # If no events filtered, keep all important events to prevent missing
            if (
                not filtered and len(events) <= 3
            ):  # If not many events and none filtered, keep original events
                return events
            return filtered

        # If holding position, focus on sell signal related events (relax filtering)
        else:
            sell_events = [
                "MACD_DEATH_CROSS",
                "RSI_OVERBOUGHT",
                "BB_UPPER_TOUCH",
                "PRICE_BELOW_MA20",
                "BEARISH_DIVERGENCE",
                "VOLUME_DECLINE",
                "MA_DEATH_CROSS",
                "TREND_TURN_BEARISH",
                "LARGE_DROP",  # Add more sell related events
            ]
            filtered = [
                event
                for event in events
                if any(sell_event in str(event) for sell_event in sell_events)
            ]
            # If no events filtered, keep all important events to prevent missing
            if (
                not filtered and len(events) <= 3
            ):  # If not many events and none filtered, keep original events
                return events
            return filtered

    def set_current_symbol(self, symbol: str) -> None:
        """Set current stock symbol for trading"""
        self.current_symbol = symbol
        print(f"📊 Setting trading target: {symbol}")

    def finalize_backtest(
        self, final_price: float, final_timestamp: pd.Timestamp
    ) -> None:
        """
        Force close all positions when backtest ends

        Args:
            final_price: Closing price on last trading day
            final_timestamp: Timestamp of last trading day
        """
        if self.shares > 0 and self.current_position:
            print(f"🏁 Backtest ended, forcing position settlement...")
            print(f"💰 Position quantity: {self.shares} shares")
            print(f"📈 Settlement price: ${final_price:.2f}")

            # Calculate realized P&L
            sale_value = self.shares * final_price
            cost_basis = (
                self.shares * self.position_entry_price
                if self.position_entry_price > 0
                else 0
            )
            realized_pnl = sale_value - cost_basis
            realized_return = (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0

            print(f"💵 Settlement amount: ${sale_value:,.0f}")
            if cost_basis > 0:
                print(f"🎯 Cost basis: ${cost_basis:,.0f}")
                print(f"📊 Realized P&L: ${realized_pnl:,.0f} ({realized_return:+.2f}%)")

            # Update cumulative realized P&L and trade statistics
            if cost_basis > 0:
                self.total_realized_pnl += realized_pnl
                self.trade_returns.append(realized_return)  # Record this trade's return rate
                self.total_trades += 1
                is_winning_trade = realized_pnl > 0
                if is_winning_trade:
                    self.winning_trades += 1

                # Calculate current win rate
                current_win_rate = (
                    (self.winning_trades / self.total_trades * 100)
                    if self.total_trades > 0
                    else 0.0
                )
                print(f"💰 Cumulative realized P&L: ${self.total_realized_pnl:,.2f}")
                print(
                    f"📊 Trade statistics: {self.total_trades}th trade completed, win rate {current_win_rate:.1f}% ({self.winning_trades}/{self.total_trades})"
                )

            # Update cash balance
            self.cash += sale_value

            # Calculate overall backtest statistics
            total_return = (
                ((self.cash - self.initial_capital) / self.initial_capital * 100)
                if self.initial_capital > 0
                else 0
            )
            print(f"\n📊 === Complete Backtest Statistics ===")
            print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
            print(f"💵 Final capital: ${self.cash:,.2f}")
            print(f"📈 Total return rate: {total_return:+.2f}%")
            print(f"🎯 Cumulative realized P&L: ${self.total_realized_pnl:,.2f}")
            print(f"📊 Total trades: {self.total_trades}")
            if self.total_trades > 0:
                print(f"✅ Profitable trades: {self.winning_trades}")
                print(
                    f"📊 Overall win rate: {self.winning_trades / self.total_trades * 100:.1f}%"
                )
                print(
                    f"💰 Average P&L per trade: ${self.total_realized_pnl / self.total_trades:,.2f}"
                )

            # Clear position
            final_shares = self.shares  # Save share count for creating signal
            self.shares = 0
            self.current_position = None

            # Create settlement trade record
            final_signal = TradingSignal(
                timestamp=final_timestamp,
                signal_type=SignalType.SELL,
                price=final_price,
                confidence=1.0,
                reason="Backtest end forced settlement",
                metadata={"quantity": final_shares},  # Put quantity in metadata
            )

            # If there's P&L tracker, update final state
            if (
                hasattr(self, "pnl_tracker")
                and self.pnl_tracker
                and hasattr(self, "current_position_id")
                and self.current_position_id is not None
            ):
                try:
                    self.pnl_tracker.close_position(
                        self.current_position_id,
                        final_price,
                        final_timestamp.strftime("%Y-%m-%d"),
                    )
                    self.current_position_id = None  # Clear position ID
                    print(f"📊 P&L tracker updated final state")
                except Exception as e:
                    print(f"⚠️ Failed to update P&L tracker: {e}")

            print(f"✅ Position settlement completed, cash balance: ${self.cash:,.0f}")

        else:
            print(f"🏁 Backtest ended, no positions to settle")
            print(f"💰 Final cash balance: ${self.cash:,.0f}")

    def get_final_portfolio_value(self, final_price: float) -> float:
        """
        Calculate total investment portfolio value at end of backtest

        Args:
            final_price: Closing price on last trading day

        Returns:
            Total portfolio value (cash + position market value)
        """
        cash_value = self.cash
        position_value = self.shares * final_price if self.shares > 0 else 0
        total_value = cash_value + position_value

        print(f"📊 Final investment portfolio value:")
        print(f"   💰 Cash: ${cash_value:,.0f}")
        print(
            f"   📈 Position market value: ${position_value:,.0f} ({self.shares} shares × ${final_price:.2f})"
        )
        print(f"   🎯 Total value: ${total_value:,.0f}")

        return total_value

        # Define entry signals (focus when no position)
        entry_signals = {
            "BB_LOWER_TOUCH",  # Touches Bollinger lower band - oversold rebound
            "MACD_GOLDEN_CROSS",  # MACD golden cross - bullish signal
            "MA_GOLDEN_CROSS",  # MA golden cross - bullish signal
            "TREND_TURN_BULLISH",  # Trend turns bullish - entry signal
        }

        # Define exit signals (focus when holding position)
        exit_signals = {
            "BB_UPPER_TOUCH",  # Touches Bollinger upper band - overbought pullback
            "MACD_DEATH_CROSS",  # MACD death cross - bearish signal
            "MA_DEATH_CROSS",  # MA death cross - bearish signal
            "TREND_TURN_BEARISH",  # Trend turns bearish - exit signal
            "LARGE_DROP",  # Large bearish candle - sharp drop signal
        }

        relevant_events = []

        # Modified: Simplify logic, let all important events be considered
        # This allows LLM to simultaneously consider entry and exit opportunities
        print(f"🔍 Event filtering - considering all important technical signals")

        for event in events:
            event_type = event["event_type"]

            # Keep all important technical signals
            if event_type in entry_signals or event_type in exit_signals:
                relevant_events.append(event)
                signal_category = (
                    "Entry related" if event_type in entry_signals else "Exit related"
                )
                print(f"   ✅ {signal_category}: {event_type} - {event['description']}")
            else:
                print(f"   ❌ Non-critical signal: {event_type} - filtered out")

        return relevant_events

    def _make_llm_decision(
        self,
        data: pd.DataFrame,
        current_date: pd.Timestamp,
        events: List[Dict[str, Any]],
        trend_analysis: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Let LLM make trading decisions

        Args:
            data: Historical data
            current_date: Current date
            events: Trigger events
            trend_analysis: Trend analysis result

        Returns:
            LLM decision result
        """
        try:
            print(f"🧠 Starting LLM decision (Number of events: {len(events)})...")
            self.total_llm_calls += 1  # Increase LLM call count

            # Prepare context data
            current_data = data.iloc[-1]
            recent_data = data.tail(5)
            print(f"📊 Context data preparation completed")

            # Generate comprehensive technical analysis context
            current_date_str = current_date.strftime("%Y-%m-%d")
            comprehensive_context = (
                self.enhanced_analyzer.analyze_comprehensive_context(
                    data, current_date_str, lookback_days=10
                )
            )
            print(f"🔬 Comprehensive technical analysis completed")

            # Store comprehensive technical analysis context for log recording
            self.current_comprehensive_context = comprehensive_context

            print(f"📊 Preparing for LLM analysis...")

            # Calculate position metrics and P&L insights
            position_metrics = None
            pnl_insights = None

            if hasattr(self, "pnl_tracker") and self.pnl_tracker:
                # Use correct column name (lowercase 'close' not uppercase 'Close')
                close_price = current_data.get("close", current_data.get("Close", 0))
                position_metrics = self._calculate_position_metrics(
                    close_price, current_date
                )
                pnl_insights = self._generate_pnl_insights(position_metrics)
                print(
                    f"📈 P&L analysis completed: Position status={position_metrics.get('has_position', False)}"
                )

            # Construct LLM prompt
            prompt = self._build_decision_prompt(
                current_data,
                recent_data,
                events,
                trend_analysis,
                self.stock_characteristics,
                position_metrics,
                pnl_insights,
                comprehensive_context,  # Add comprehensive technical analysis context
            )

            # Check if prompt is None
            if prompt is None:
                print("❌ Error: LLM prompt construction failed (returned None)")
                return None

            print(f"📝 LLM prompt construction completed (Length: {len(prompt)} characters)")

            # Call LLM
            print(f"🤖 Calling LLM...")
            response = self.llm_client.invoke(prompt)

            # Check if LLM response is valid
            if response is None:
                print("❌ Error: LLM response is empty (response is None)")
                return None

            if not hasattr(response, "content") or response.content is None:
                print("❌ Error: LLM response content is empty (response.content is None)")
                return None

            print(f"📡 LLM response received (Length: {len(response.content)} characters)")

            # Parse LLM response
            decision = self._parse_llm_response(response.content)
            print(f"🔍 LLM response parsing completed: {decision}")

            # Record decision log
            self.decision_log.append(
                {
                    "date": current_date,
                    "events": events,
                    "decision": decision,
                    "reasoning": decision.get("reasoning", "") if decision else "",
                }
            )

            return decision

        except Exception as e:
            print(f"❌ LLM decision error: {e}")
            import traceback

            print(f"🔍 Error details: {traceback.format_exc()}")
            return None

    def _build_decision_prompt(
        self,
        current_data: pd.Series,
        recent_data: pd.DataFrame,
        events: List[Dict[str, Any]],
        trend_analysis: Dict[str, Any],
        stock_characteristics: Dict[str, Any],
        position_metrics: Optional[Dict[str, Any]] = None,
        pnl_insights: Optional[Dict[str, Any]] = None,
        comprehensive_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Construct LLM decision prompt"""

        prompt = f"""
You are a professional stock trading strategy analyst. Please make trading decisions based on the following information:

## Stock Characteristics Analysis
- Volatility: {stock_characteristics.get("volatility", 0):.3f}
- Trend Consistency: {stock_characteristics.get("trend_consistency", 0):.3f}
- MACD Effectiveness: {stock_characteristics.get("macd_effectiveness", 0):.3f}

## Current Market Data
- Current Price: {current_data["close"]:.2f}
- MACD: {current_data.get("macd", 0):.4f}
- MACD Signal Line: {current_data.get("macd_signal", 0):.4f}
- Bollinger Upper Band: {current_data.get("bb_upper", 0):.2f}
- Bollinger Middle Band: {current_data.get("bb_middle", 0):.2f}
- Bollinger Lower Band: {current_data.get("bb_lower", 0):.2f}
- {self.ma_short}-day MA: {current_data.get(f"ma_{self.ma_short}", 0):.2f}
- {self.ma_long}-day MA: {current_data.get(f"ma_{self.ma_long}", 0):.2f}
- 20-day MA: {current_data.get("ma_20", 0):.2f}
- 50-day MA: {current_data.get("ma_50", 0):.2f}

## Trigger Events
"""

        for event in events:
            prompt += f"- {event['event_type']}: {event['description']} (Severity: {event['severity']})\n"

        # Add comprehensive technical analysis context
        if comprehensive_context and not comprehensive_context.get("error"):
            prompt += f"""
## 📊 Comprehensive Technical Analysis

### 💰 Price Action Analysis
- Price Change: {comprehensive_context.get("price_action", {}).get("price_change_pct", 0):.2f}%
- Candlestick Pattern: {comprehensive_context.get("price_action", {}).get("candle_type", "unknown")}
- Body Ratio: {comprehensive_context.get("price_action", {}).get("body_ratio", 0):.2f}
- Volume Ratio: {comprehensive_context.get("price_action", {}).get("volume_to_avg_ratio", 1):.2f}x
- Gap: {comprehensive_context.get("price_action", {}).get("gap_pct", 0):.2f}%

### 📈 Moving Average Analysis
- MA5: ${comprehensive_context.get("moving_averages", {}).get("ma_5", 0):.2f} (Slope: {comprehensive_context.get("moving_averages", {}).get("ma_5_slope", 0):.4f})
- MA10: ${comprehensive_context.get("moving_averages", {}).get("ma_10", 0):.2f} (Slope: {comprehensive_context.get("moving_averages", {}).get("ma_10_slope", 0):.4f})
- MA20: ${comprehensive_context.get("moving_averages", {}).get("ma_20", 0):.2f} (Slope: {comprehensive_context.get("moving_averages", {}).get("ma_20_slope", 0):.4f})
- MA Alignment: {comprehensive_context.get("moving_averages", {}).get("ma_alignment", "unknown")}
- Above All MAs: {comprehensive_context.get("moving_averages", {}).get("above_all_mas", False)}

### 📊 Volume Analysis
- Current Volume: {comprehensive_context.get("volume_analysis", {}).get("current_volume", 0):,}
- Volume Ratio: {comprehensive_context.get("volume_analysis", {}).get("volume_ratio", 1):.2f}x
- Volume Trend: {comprehensive_context.get("volume_analysis", {}).get("volume_trend", 0):.2f}
- High Volume: {comprehensive_context.get("volume_analysis", {}).get("is_high_volume", False)}
- Price-Volume Confirmation: {comprehensive_context.get("volume_analysis", {}).get("volume_confirmation", False)}

### 🌊 Volatility Analysis
- ATR: {comprehensive_context.get("volatility_analysis", {}).get("atr", 0):.2f}
- Annualized Volatility: {comprehensive_context.get("volatility_analysis", {}).get("volatility_annualized", 0):.2f}%
- Volatility Percentile: {comprehensive_context.get("volatility_analysis", {}).get("volatility_percentile", 50):.1f}%
- High Volatility: {comprehensive_context.get("volatility_analysis", {}).get("is_high_volatility", False)}

### ⚡ Momentum Indicators
- RSI: {comprehensive_context.get("momentum_indicators", {}).get("rsi", 50):.2f}
- RSI Condition: {comprehensive_context.get("momentum_indicators", {}).get("rsi_condition", "neutral")}
- 5-day ROC: {comprehensive_context.get("momentum_indicators", {}).get("roc_5_day", 0):.2f}%
- 10-day ROC: {comprehensive_context.get("momentum_indicators", {}).get("roc_10_day", 0):.2f}%
- Momentum Strength: {comprehensive_context.get("momentum_indicators", {}).get("momentum_strength", "neutral")}

### 🎯 Support Resistance
- Nearest Resistance: ${comprehensive_context.get("support_resistance", {}).get("nearest_resistance", 0):.2f}
- Nearest Support: ${comprehensive_context.get("support_resistance", {}).get("nearest_support", 0):.2f}
- Distance to Resistance: {comprehensive_context.get("support_resistance", {}).get("resistance_distance_pct", 0):.2f}%
- Distance to Support: {comprehensive_context.get("support_resistance", {}).get("support_distance_pct", 0):.2f}%
- Near Key Level: {comprehensive_context.get("support_resistance", {}).get("near_resistance", False) or comprehensive_context.get("support_resistance", {}).get("near_support", False)}

### 📐 Trend Strength Analysis
- Trend Direction: {comprehensive_context.get("trend_analysis", {}).get("trend_direction", "neutral")}
- Trend Strength: {comprehensive_context.get("trend_analysis", {}).get("trend_strength", 0):.3f}
- ADX Value: {comprehensive_context.get("trend_analysis", {}).get("adx_value", 0):.2f}
- Strong Trend: {comprehensive_context.get("trend_analysis", {}).get("strong_trend", False)}

### 🏮 Market Regime
- Market Pattern: {comprehensive_context.get("market_regime", {}).get("market_regime", "unknown")}
- Pattern Description: {comprehensive_context.get("market_regime", {}).get("regime_description", "Unknown regime")}
- Trending Market: {comprehensive_context.get("market_regime", {}).get("is_trending", False)}
- High Volatility: {comprehensive_context.get("market_regime", {}).get("is_volatile", False)}

### 🎈 Bollinger Bands Analysis
- Bollinger Position: {comprehensive_context.get("bollinger_analysis", {}).get("bb_position", 0.5):.3f} (0=lower band, 1=upper band)
- Band Width: {comprehensive_context.get("bollinger_analysis", {}).get("bb_width", 0):.2f}%
- Band Squeeze: {comprehensive_context.get("bollinger_analysis", {}).get("is_squeeze", False)}
- Potential Breakout: {comprehensive_context.get("bollinger_analysis", {}).get("potential_breakout", False)}

### 📈 MACD Analysis
- MACD Line: {comprehensive_context.get("macd_analysis", {}).get("macd_line", 0):.4f}
- Signal Line: {comprehensive_context.get("macd_analysis", {}).get("signal_line", 0):.4f}
- Histogram: {comprehensive_context.get("macd_analysis", {}).get("histogram", 0):.4f}
- MACD Position: {comprehensive_context.get("macd_analysis", {}).get("macd_position", "neutral")}
- Cross Signal: {comprehensive_context.get("macd_analysis", {}).get("macd_cross", "none")}
"""

        prompt += f"""
## Trend Analysis"""

        # Use Enhanced analysis if available, otherwise fallback to original
        if (
            hasattr(self, "current_enhanced_analysis")
            and self.current_enhanced_analysis
        ):
            enhanced = self.current_enhanced_analysis
            prompt += f"""
- Dominant Trend: {enhanced.market_phase} (Enhanced analysis)
- Trend Consistency: {enhanced.trend_consistency:.3f}
- Reversal Probability: {enhanced.reversal_probability:.3f}
- Momentum Status: {enhanced.momentum_status}
- Risk Level: {enhanced.risk_level}

📊 **Trend Judgment Explanation**: 
- Using Enhanced multi-timeframe analysis, market_phase is the main trend judgment basis
- {enhanced.market_phase} represents current dominant market direction
- Trend consistency {enhanced.trend_consistency:.3f} indicates trend unification degree across multiple timeframes"""
        else:
            # Fallback to original analysis
            prompt += f"""
- Dominant Trend: {trend_analysis.dominant_trend if trend_analysis else "unknown"} (Basic analysis)
- Trend Strength: {trend_analysis.complexity_score if trend_analysis else 0:.3f}

📊 **Trend Judgment Explanation**: 
- Using basic trend analysis, dominant_trend is the main trend judgment basis"""

            # Add trend conversion information
            if (
                hasattr(trend_analysis, "trend_reversal_detected")
                and trend_analysis.trend_reversal_detected
            ):
                # Give importance rating based on conversion strength
                if trend_analysis.reversal_strength > 0.05:  # Above 5%
                    importance = "🔥 Strong Conversion Signal"
                elif trend_analysis.reversal_strength > 0.03:  # Above 3%
                    importance = "⚡ Clear Conversion Signal"
                else:
                    importance = "📊 Mild Conversion Signal"

                prompt += f"""
- {importance}: Trend conversion point detected (Strength: {trend_analysis.reversal_strength:.2%})
- 🎯 Key Timing: This is a potential trend conversion point, historically such signals often indicate important opportunities
- 💡 Strategy Reminder: Should actively consider entry when conversion signal strength ≥ 2%, should act decisively when ≥ 3%"""

        prompt += f"""

## Current Position Status
Position Status: {"Has position" if self.current_position else "No position"}"""

        # Add unrealized P&L information
        if position_metrics and position_metrics.get("has_position"):
            prompt += f"""

### 📈 Position Details
- Position Quantity: {position_metrics["shares"]:,.0f} shares
- Entry Price: ${position_metrics["entry_price"]:.2f}
- Current Price: ${position_metrics["current_price"]:.2f}
- Position Cost: ${position_metrics["cost_basis"]:,.0f}
- Current Market Value: ${position_metrics["position_value"]:,.0f}

### 💰 Unrealized P&L Analysis
- Unrealized P&L: ${position_metrics["unrealized_pnl"]:,.0f}
- Return Rate: {position_metrics["unrealized_pnl_pct"]:+.2f}%
- Holding Days: {position_metrics["holding_days"]} days
- Risk Level: {position_metrics["risk_level"]}

### 🎯 P&L Insights
- P&L Signal: {pnl_insights.get("pnl_signal", "neutral") if pnl_insights else "neutral"}
- Risk Warning: {pnl_insights.get("risk_warning", "No special risks") if pnl_insights else "No special risks"}
- Suggested Action: {pnl_insights.get("suggested_action", "Normal operation") if pnl_insights else "Normal operation"}"""
        else:
            prompt += f"""

### 📈 Position Details
- Position Status: No position
- Available Funds: ${self.cash:,.0f}
- Total Assets: ${self.cash:,.0f}

### 🎯 Investment Insights
- Suggested Action: {pnl_insights.get("suggested_action", "Can consider new position") if pnl_insights else "Can consider new position"}
- Position Suggestion: Normal position configuration"""

        # Add dynamically loaded strategy decision principles
        prompt += f"""

{self.strategy_prompt}
"""

        return prompt

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response"""
        try:
            # Try to extract JSON part
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)

            return None

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

    def _create_signal_from_decision(
        self, decision: Dict[str, Any], timestamp: pd.Timestamp, price: float
    ) -> Optional[TradingSignal]:
        """Create trading signal from LLM decision"""

        action = decision.get("action")
        if action not in ["BUY", "SELL"]:
            return None

        # Additional risk check: prevent clearly unfavorable entries
        if (
            action == "BUY"
            and hasattr(self, "_last_trend_analysis")
            and self._last_trend_analysis
        ):
            trend_analysis = self._last_trend_analysis

            # Check 1: Do not enter during strong downtrend
            if (
                trend_analysis.dominant_trend == "downtrend"
                and hasattr(trend_analysis, "trend_strength")
                and trend_analysis.trend_strength >= 0.8
            ):
                print(
                    f"🚫 Risk control: Refusing entry during strong downtrend (Trend strength: {trend_analysis.trend_strength:.3f})"
                )
                return None

            # Check 2: Do not enter during downtrend with low trend consistency
            if (
                trend_analysis.dominant_trend == "downtrend"
                and hasattr(trend_analysis, "trend_consistency")
                and trend_analysis.trend_consistency < 0.5
            ):
                print(
                    f"🚫 Risk control: Avoid entry during unclear downtrend (Consistency: {trend_analysis.trend_consistency:.3f})"
                )
                return None

            # Check 3: During downtrend with strength above 0.5, need additional confirmation
            if (
                trend_analysis.dominant_trend == "downtrend"
                and hasattr(trend_analysis, "trend_strength")
                and trend_analysis.trend_strength >= 0.5
            ):
                print(
                    f"⚠️ Risk warning: Entry during medium strength downtrend requires high caution (Trend strength: {trend_analysis.trend_strength:.3f})"
                )
                # Reduce confidence
                decision["confidence"] = min(decision.get("confidence", 0.5), 0.75)

        signal_type = SignalType.BUY if action == "BUY" else SignalType.SELL
        confidence = decision.get("confidence", 0.5)
        reasoning = decision.get("reasoning", "")

        # Update position status
        if hasattr(self, "pnl_tracker") and self.pnl_tracker:
            try:
                if action == "BUY" and not self.current_position:
                    # Fixed 1000 shares
                    shares_to_buy = 1000
                    cost = shares_to_buy * price

                    print(f"🎯 Fixed position: Buying {shares_to_buy} shares")

                    # Ensure sufficient cash
                    if cost <= self.cash:
                        # Add new position to P&L tracker
                        if self.current_symbol:
                            self.current_position_id = self.pnl_tracker.add_position(
                                self.current_symbol,
                                timestamp.strftime("%Y-%m-%d"),
                                price,
                                shares_to_buy,
                                confidence,
                            )

                        # Update internal position status
                        self.current_position = "long"
                        self.position_entry_price = price
                        self.position_entry_date = timestamp
                        self.shares = shares_to_buy
                        self.cash -= cost

                        # Use fixed stop loss ratio (5%)
                        stop_loss_price = price * 0.95

                        print(
                            f"📈 Position update: Bought {shares_to_buy} shares, price ${price:.2f}, total cost ${cost:,.0f}"
                        )
                        print(f"🛡️ Stop loss set: ${stop_loss_price:.2f} (5% stop loss)")

                        # Immediately send P&L update after trade
                        if self.progress_callback:
                            try:
                                # Calculate current index (assuming called in loop)
                                day_index = getattr(self, "_current_day_index", 0)
                                total_days = getattr(self, "_total_days", 125)
                                self._send_performance_update(
                                    day_index, total_days, price
                                )
                            except Exception as e:
                                print(f"⚠️ P&L update after buy failed: {e}")
                    else:
                        print(
                            f"⚠️ Insufficient cash, cannot buy {shares_to_buy} shares (Need ${cost:,.0f}, Have ${self.cash:,.0f})"
                        )
                        return None  # No signal generated when insufficient funds

                elif action == "SELL" and self.current_position and self.shares > 0:
                    # Sell all holdings
                    proceeds = self.shares * price

                    # Calculate and record realized P&L
                    cost_basis = self.shares * self.position_entry_price
                    realized_pnl = proceeds - cost_basis
                    realized_return = (
                        (realized_pnl / cost_basis * 100) if cost_basis > 0 else 0
                    )

                    # Update cumulative realized P&L
                    self.total_realized_pnl += realized_pnl
                    self.trade_returns.append(realized_return)  # Record this trade's return rate

                    # Update trade statistics (one complete trade: buy -> sell)
                    self.total_trades += 1
                    is_winning_trade = realized_pnl > 0
                    if is_winning_trade:
                        self.winning_trades += 1

                    # Calculate current win rate
                    current_win_rate = (
                        (self.winning_trades / self.total_trades * 100)
                        if self.total_trades > 0
                        else 0.0
                    )

                    # Reset position status
                    self.current_position = None
                    self.position_entry_price = 0.0
                    self.position_entry_date = None
                    old_shares = self.shares
                    self.shares = 0
                    self.cash += proceeds

                    # Close position in P&L tracker
                    if (
                        hasattr(self, "pnl_tracker")
                        and self.pnl_tracker
                        and hasattr(self, "current_position_id")
                        and self.current_position_id is not None
                    ):
                        try:
                            self.pnl_tracker.close_position(
                                self.current_position_id,
                                price,
                                timestamp.strftime("%Y-%m-%d"),
                            )
                            self.current_position_id = None  # Clear position ID
                        except Exception as e:
                            print(f"⚠️ P&L tracker position close failed: {e}")

                    print(f"📉 Position cleared: Sold {old_shares} shares, price ${price:.2f}")
                    print(f"💰 Sale amount: ${proceeds:,.2f}")
                    print(f"🎯 Cost basis: ${cost_basis:,.2f}")
                    print(
                        f"📊 Realized P&L: ${realized_pnl:,.2f} ({realized_return:+.2f}%) ({'✅ Profit' if is_winning_trade else '❌ Loss'})"
                    )
                    print(f"💰 Cumulative realized P&L: ${self.total_realized_pnl:,.2f}")
                    print(
                        f"📊 Trade statistics: {self.total_trades}th trade completed, win rate {current_win_rate:.1f}% ({self.winning_trades}/{self.total_trades})"
                    )
                    print(f"💵 Current cash balance: ${self.cash:,.2f}")

                    # Immediately send P&L update after trade
                    if self.progress_callback:
                        try:
                            # Calculate current index (assuming called in loop)
                            day_index = getattr(self, "_current_day_index", 0)
                            total_days = getattr(self, "_total_days", 125)
                            self._send_performance_update(day_index, total_days, price)
                        except Exception as e:
                            print(f"⚠️ P&L update after sell failed: {e}")

            except Exception as e:
                print(f"⚠️ Position status update failed: {e}")

        return TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            reason=f"LLM decision: {reasoning}",
            metadata={
                "decision": decision,
                "risk_level": decision.get("risk_level", "medium"),
                "expected_outcome": decision.get("expected_outcome", ""),
                "position_size": getattr(self, "shares", 0),
                "cash_remaining": getattr(self, "cash", 0),
            },
        )

    # Helper methods (calculating stock characteristics)
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """Calculate trend consistency"""
        try:
            returns = data["close"].pct_change().dropna()
            if len(returns) == 0:
                return 0.0

            # Calculate proportion of consecutive same-direction changes
            direction_changes = (returns > 0).astype(int).diff().abs().sum()
            if len(returns) == 0:
                return 0.0

            consistency = 1.0 - (direction_changes / len(returns))
            return max(0.0, min(1.0, consistency))

        except Exception as e:
            print(f"Trend consistency calculation error: {e}")
            return 0.0

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength (based on price movement linear regression)"""
        if len(data) < 10:
            return 0.0

        try:
            prices = data["close"].dropna()
            if len(prices) < 2:
                return 0.0

            x = np.arange(len(prices))

            # Calculate R² value of linear regression as trend strength indicator
            correlation_matrix = np.corrcoef(x, prices)
            if correlation_matrix.size == 0:
                return 0.0

            correlation = abs(correlation_matrix[0, 1])
            if np.isnan(correlation):
                return 0.0

            return correlation**2  # R² value

        except Exception as e:
            print(f"Trend strength calculation error: {e}")
            return 0.0

    def _calculate_consecutive_move_tendency(self, returns: pd.Series) -> float:
        """Calculate consecutive movement tendency (momentum characteristics)"""
        if len(returns) < 5:
            return 0.0

        consecutive_up = 0
        consecutive_down = 0
        max_consecutive_up = 0
        max_consecutive_down = 0

        for ret in returns:
            if ret > 0:
                consecutive_up += 1
                consecutive_down = 0
                max_consecutive_up = max(max_consecutive_up, consecutive_up)
            elif ret < 0:
                consecutive_down += 1
                consecutive_up = 0
                max_consecutive_down = max(max_consecutive_down, consecutive_down)
            else:
                consecutive_up = 0
                consecutive_down = 0

        return (
            (max_consecutive_up + max_consecutive_down) / len(returns)
            if len(returns) > 0
            else 0.0
        )

    def _test_ma_crossover_effectiveness(self, data: pd.DataFrame) -> float:
        """Test moving average crossover effectiveness"""
        if len(data) < 50:
            return 0.5

        # Test using 10-day and 20-day MAs
        ma_short = data["close"].rolling(10).mean()
        ma_long = data["close"].rolling(20).mean()

        successful_signals = 0
        total_signals = 0

        for i in range(21, len(data) - 5):
            # Golden cross
            if (
                ma_short.iloc[i] > ma_long.iloc[i]
                and ma_short.iloc[i - 1] <= ma_long.iloc[i - 1]
            ):
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns > 0:
                    successful_signals += 1
                total_signals += 1
            # Death cross
            elif (
                ma_short.iloc[i] < ma_long.iloc[i]
                and ma_short.iloc[i - 1] >= ma_long.iloc[i - 1]
            ):
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns < 0:
                    successful_signals += 1
                total_signals += 1

        return successful_signals / total_signals if total_signals > 0 else 0.5

    def _test_bollinger_bands_effectiveness(self, data: pd.DataFrame) -> float:
        """Test Bollinger Bands effectiveness"""
        if len(data) < 40:
            return 0.5

        # Calculate Bollinger Bands
        bb_data = calculate_bollinger_bands(data, window=20, num_std_dev=2)

        successful_signals = 0
        total_signals = 0

        for i in range(21, len(data) - 5):
            current_price = data["close"].iloc[i]
            bb_upper = bb_data["bb_upper"].iloc[i]
            bb_lower = bb_data["bb_lower"].iloc[i]

            # Touches lower band (oversold)
            if current_price <= bb_lower:
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns > 0:
                    successful_signals += 1
                total_signals += 1
            # Touches upper band (overbought)
            elif current_price >= bb_upper:
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns < 0:
                    successful_signals += 1
                total_signals += 1

        return successful_signals / total_signals if total_signals > 0 else 0.5

    def _analyze_breakout_tendency(self, data: pd.DataFrame) -> float:
        """Analyze breakout tendency"""
        if len(data) < 20:
            return 0.5

        breakouts = 0
        total_opportunities = 0

        # Use 20-day high/low points as breakout reference
        rolling_high = data["high"].rolling(20).max()
        rolling_low = data["low"].rolling(20).min()

        for i in range(20, len(data) - 1):
            if data["close"].iloc[i] > rolling_high.iloc[i - 1]:  # Upward breakout
                if data["close"].iloc[i + 1] > data["close"].iloc[i]:  # Continues rising next day
                    breakouts += 1
                total_opportunities += 1
            elif data["close"].iloc[i] < rolling_low.iloc[i - 1]:  # Downward breakout
                if data["close"].iloc[i + 1] < data["close"].iloc[i]:  # Continues falling next day
                    breakouts += 1
                total_opportunities += 1

        return breakouts / total_opportunities if total_opportunities > 0 else 0.5

    def _classify_stock_personality(
        self,
        volatility: float,
        trend_consistency: float,
        reversal_frequency: float,
        macd_effectiveness: float,
    ) -> str:
        """Classify stock personality based on characteristics analysis results"""

        if volatility > 0.4 and reversal_frequency > 0.1:
            return "High Volatility Sideways Type"
        elif volatility > 0.4 and trend_consistency > 0.6:
            return "High Volatility Trending Type"
        elif volatility < 0.2 and trend_consistency > 0.7:
            return "Stable Trending Type"
        elif volatility < 0.2 and reversal_frequency > 0.08:
            return "Low Volatility Sideways Type"
        elif trend_consistency > 0.8:
            return "Strong Trending Type"
        elif reversal_frequency > 0.12:
            return "High Frequency Reversal Type"
        elif macd_effectiveness > 0.7:
            return "Technical Indicator Sensitive Type"
        elif 0.2 <= volatility <= 0.35 and 0.4 <= trend_consistency <= 0.7:
            return "Balanced Type"
        else:
            return "Complex Mixed Type"

    def _calculate_reversal_frequency(self, data: pd.DataFrame) -> float:
        """Calculate reversal frequency"""
        if len(data) < 10:
            return 0.0

        peaks_valleys = 0
        for i in range(1, len(data) - 1):
            if (
                data["close"].iloc[i] > data["close"].iloc[i - 1]
                and data["close"].iloc[i] > data["close"].iloc[i + 1]
            ) or (
                data["close"].iloc[i] < data["close"].iloc[i - 1]
                and data["close"].iloc[i] < data["close"].iloc[i + 1]
            ):
                peaks_valleys += 1

        return peaks_valleys / len(data) if len(data) > 0 else 0.0

    def _test_macd_effectiveness(self, data: pd.DataFrame) -> float:
        """Test MACD indicator effectiveness"""
        if len(data) < 50:
            return 0.5

        macd_data = calculate_macd(data)
        macd = macd_data["macd"]
        signal = macd_data["macd_signal"]

        successful_signals = 0
        total_signals = 0

        for i in range(1, len(macd) - 5):
            if (
                macd.iloc[i] > signal.iloc[i] and macd.iloc[i - 1] <= signal.iloc[i - 1]
            ):  # Golden cross
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns > 0:
                    successful_signals += 1
                total_signals += 1
            elif (
                macd.iloc[i] < signal.iloc[i] and macd.iloc[i - 1] >= signal.iloc[i - 1]
            ):  # Death cross
                future_returns = data["close"].iloc[i + 1 : i + 6].pct_change().sum()
                if future_returns < 0:
                    successful_signals += 1
                total_signals += 1

        return successful_signals / total_signals if total_signals > 0 else 0.5

    def _calculate_gap_frequency(self, data: pd.DataFrame) -> float:
        """Calculate gap frequency"""
        if len(data) < 2:
            return 0.0

        gaps = 0
        for i in range(1, len(data)):
            gap_up = data["low"].iloc[i] > data["high"].iloc[i - 1]
            gap_down = data["high"].iloc[i] < data["low"].iloc[i - 1]
            if gap_up or gap_down:
                gaps += 1

        return gaps / len(data) if len(data) > 0 else 0.0

    def _analyze_support_resistance(self, data: pd.DataFrame) -> float:
        """Analyze support resistance strength"""
        if len(data) < 20:
            return 0.5

        try:
            # Simplified support resistance analysis
            low_min = data["low"].min()
            high_max = data["high"].max()

            if pd.isna(low_min) or pd.isna(high_max) or low_min >= high_max:
                return 0.5

            price_levels = np.linspace(low_min, high_max, 20)
            level_touches = []

            for level in price_levels:
                touches = 0
                if level != 0:  # Avoid division by zero error
                    for _, row in data.iterrows():
                        if (
                            abs(row["low"] - level) / level < 0.02
                            or abs(row["high"] - level) / level < 0.02
                        ):
                            touches += 1
                level_touches.append(touches)

            if not level_touches:
                return 0.5

            max_touches = max(level_touches)
            return min(1.0, max_touches / len(data)) if len(data) > 0 else 0.5

        except Exception as e:
            print(f"Support resistance analysis error: {e}")
            return 0.5

    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return f"""
        LLM Smart Strategy - Adaptive Parameter Optimization Version
        
        🧠 Intelligent Features:
        • Automatically analyze stock characteristics (3-6 months historical data)
        • Dynamically optimize technical indicator parameters based on volatility, trending, reversal frequency and other characteristics
        • No manual parameter tuning needed, strategy automatically adapts to different stock behaviors
        
        📊 Stock Characteristic Analysis Dimensions:
        • Volatility Analysis: Annualized volatility, volatility of volatility
        • Trend Characteristics: Trend consistency, trend strength, consecutive movement tendency
        • Reversal Characteristics: Reversal frequency, breakout tendency
        • Technical Indicator Responsiveness: MACD, moving averages, Bollinger Bands effectiveness testing
        • Comprehensive Stock Personality Classification: High volatility trending, stable trending, sideways, etc.
        
        ⚙️ Current Parameter Settings (based on configuration):
        - Confidence Threshold: {self.confidence_threshold}
        - Trend Lookback Period: {self.trend_lookback} days
        - Event Trigger Threshold: {self.event_threshold}
        - Maximum Daily Trades: {self.max_daily_trades}
        - Use Technical Filter: {"Yes" if self.use_technical_filter else "No"}
        
        🔧 Technical Indicators (dynamic optimization):
        - MACD Fast Line: {self.macd_fast} (automatically adjusted based on trending)
        - MACD Slow Line: {self.macd_slow} (automatically adjusted based on trending)
        - Short-term MA: {self.ma_short} days (automatically adjusted based on reversal frequency)
        - Long-term MA: {self.ma_long} days (automatically adjusted based on reversal frequency)
        
        🎯 Workflow:
        1. Deeply analyze stock characteristics, generate stock personality profile
        2. Intelligently optimize all technical indicator parameters based on stock personality
        3. Event-driven detection of key technical signals
        4. LLM comprehensive analysis for trading decisions
        5. Strict risk control and confidence filtering
        
        ✨ Applicable Scenarios:
        - Automatic adaptation to entire market stocks
        - Intelligent trading without manual parameter tuning
        - Suitable for stocks with different personalities
        - Medium to short-term trading strategy
        """

    @classmethod
    def get_default_config(cls) -> StrategyConfig:
        """Return default configuration"""
        return StrategyConfig(
            name="LLM Smart Strategy",
            description="Adaptive LLM trading strategy based on stock characteristic analysis, automatically optimizing technical indicator parameters",
            parameters={
                "confidence_threshold": 0.6,  # Lower threshold to let more signals pass
                "trend_lookback": 20,
                "event_threshold": 0.05,
                "max_daily_trades": 3,
                "use_technical_filter": True,
                "ma_short": 10,  # Baseline value, will be adjusted based on stock characteristics during actual use
                "ma_long": 20,  # Baseline value, will be adjusted based on stock characteristics during actual use
            },
            parameter_specs={
                "confidence_threshold": ParameterSpec(
                    name="confidence_threshold",
                    display_name="LLM Confidence Threshold",
                    description="Minimum confidence requirement for LLM decisions",
                    param_type=ParameterType.FLOAT,
                    default_value=0.6,  # Lower default value
                    min_value=0.3,
                    max_value=0.95,
                    step=0.05,
                ),
                "trend_lookback": ParameterSpec(
                    name="trend_lookback",
                    display_name="Trend Lookback Period",
                    description="Lookback days for trend analysis",
                    param_type=ParameterType.INTEGER,
                    default_value=20,
                    min_value=10,
                    max_value=50,
                    step=1,
                ),
                "event_threshold": ParameterSpec(
                    name="event_threshold",
                    display_name="Event Trigger Threshold",
                    description="Trigger sensitivity for key events",
                    param_type=ParameterType.FLOAT,
                    default_value=0.05,
                    min_value=0.01,
                    max_value=0.2,
                    step=0.01,
                ),
                "max_daily_trades": ParameterSpec(
                    name="max_daily_trades",
                    display_name="Maximum Daily Trades",
                    description="Maximum allowed trades per day",
                    param_type=ParameterType.INTEGER,
                    default_value=3,
                    min_value=1,
                    max_value=10,
                    step=1,
                ),
                "use_technical_filter": ParameterSpec(
                    name="use_technical_filter",
                    display_name="Technical Indicator Filter",
                    description="Whether to use technical indicators to filter signals",
                    param_type=ParameterType.BOOLEAN,
                    default_value=True,
                ),
                "ma_short": ParameterSpec(
                    name="ma_short",
                    display_name="Short-term MA Baseline",
                    description="Short-term moving average baseline period (will be automatically adjusted based on stock reversal frequency during actual use)",
                    param_type=ParameterType.INTEGER,
                    default_value=10,
                    min_value=5,
                    max_value=20,
                    step=1,
                ),
                "ma_long": ParameterSpec(
                    name="ma_long",
                    display_name="Long-term MA Baseline",
                    description="Long-term moving average baseline period (will be automatically adjusted based on stock reversal frequency during actual use)",
                    param_type=ParameterType.INTEGER,
                    default_value=20,
                    min_value=15,
                    max_value=50,
                    step=1,
                ),
            },
            risk_level="medium",
            market_type="all",
            strategy_type="ai_adaptive",
            category="intelligent",
        )

    def _log_daily_analysis(
        self,
        timestamp: pd.Timestamp,
        historical_data: pd.DataFrame,
        i: int,
        events: List[Dict[str, Any]],
        relevant_events: List[Dict[str, Any]],
        trend_analysis: Any,
        llm_decision: Dict[str, Any] = None,
        comprehensive_context: Dict[str, Any] = None,  # New parameter
    ):
        """
        Record daily analysis data to log

        Args:
            timestamp: Current timestamp
            historical_data: Historical data
            i: Current data index
            events: All detected events
            relevant_events: Relevant events
            trend_analysis: Trend analysis result
            llm_decision: LLM decision result
            comprehensive_context: Comprehensive technical analysis context
        """
        try:
            current_row = historical_data.iloc[i]
            current_date = timestamp.strftime("%Y-%m-%d")

            # Prepare market data
            market_data = {
                "price": float(current_row.get("close", current_row.get("Close", 0))),
                "volume": int(current_row.get("volume", current_row.get("Volume", 0))),
                "high": float(current_row.get("high", current_row.get("High", 0))),
                "low": float(current_row.get("low", current_row.get("Low", 0))),
                "open": float(current_row.get("open", current_row.get("Open", 0))),
            }

            # Calculate daily return rate
            if i > 0:
                prev_close = historical_data.iloc[i - 1].get(
                    "close",
                    historical_data.iloc[i - 1].get("Close", market_data["price"]),
                )
                market_data["daily_return"] = (
                    market_data["price"] - prev_close
                ) / prev_close
            else:
                market_data["daily_return"] = 0.0

            # Calculate volatility (using standard deviation of past 10 days)
            if i >= 10:
                recent_returns = []
                for j in range(max(0, i - 9), i + 1):
                    if j > 0:
                        curr_price = historical_data.iloc[j].get(
                            "close", historical_data.iloc[j].get("Close", 0)
                        )
                        prev_price = historical_data.iloc[j - 1].get(
                            "close",
                            historical_data.iloc[j - 1].get("Close", curr_price),
                        )
                        if prev_price > 0:
                            daily_ret = (curr_price - prev_price) / prev_price
                            recent_returns.append(daily_ret)

                if recent_returns:
                    import numpy as np

                    market_data["volatility"] = float(np.std(recent_returns))
                else:
                    market_data["volatility"] = 0.0
            else:
                market_data["volatility"] = 0.0

            # Prepare trend analysis data
            trend_data = None
            if trend_analysis:
                trend_data = {
                    "short_term": getattr(
                        trend_analysis, "short_term_trend", "neutral"
                    ),
                    "medium_term": getattr(
                        trend_analysis, "medium_term_trend", "neutral"
                    ),
                    "long_term": getattr(trend_analysis, "dominant_trend", "neutral"),
                    "trend_strength": getattr(trend_analysis, "trend_strength", 0.5),
                    "confidence": getattr(trend_analysis, "confidence", 0.5),
                }

                # Add support resistance level information
                if hasattr(trend_analysis, "support_resistance"):
                    sr = trend_analysis.support_resistance
                    trend_data["support_level"] = getattr(sr, "support", None)
                    trend_data["resistance_level"] = getattr(sr, "resistance", None)

            # Prepare event data
            triggered_events_data = []
            for event in events:
                event_data = {
                    "event_type": event.get("type", "unknown"),
                    "severity": self._determine_event_severity(event),
                    "description": event.get(
                        "description", f"{event.get('type', 'unknown')} event"
                    ),
                    "technical_data": {
                        "indicator": event.get("indicator", event.get("type")),
                        "value": event.get("value"),
                        "threshold": event.get("threshold"),
                        "strength": event.get("strength", "medium"),
                    },
                }
                triggered_events_data.append(event_data)

            # Prepare LLM decision data
            llm_decision_data = None
            if llm_decision:
                llm_decision_data = {
                    "decision_made": True,
                    "prompt_version": self.strategy_type,
                    "decision_type": llm_decision.get("action", "HOLD"),
                    "confidence": llm_decision.get("confidence", 0.0),
                    "reasoning": llm_decision.get("reasoning", ""),
                    "key_factors": llm_decision.get("factors", []),
                    "raw_response": llm_decision.get("raw_response", ""),
                }
            else:
                llm_decision_data = {
                    "decision_made": False,
                    "reason": "No significant events or filtered out",
                }

            # Prepare strategy state data
            strategy_state_data = {
                "position": "long" if self.current_position else "neutral",
                "cash": self.cash,
                "portfolio_value": self.current_portfolio_value,
                "shares": self.shares,
                "entry_price": self.position_entry_price
                if self.current_position
                else None,
                "trade_count_today": self.daily_trade_count,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
            }

            # Calculate current P&L
            if self.current_position and self.shares > 0:
                current_value = self.shares * market_data["price"]
                entry_value = self.shares * self.position_entry_price
                strategy_state_data["unrealized_pnl"] = current_value - entry_value
                strategy_state_data["unrealized_pnl_pct"] = (
                    current_value - entry_value
                ) / entry_value
            else:
                strategy_state_data["unrealized_pnl"] = 0.0
                strategy_state_data["unrealized_pnl_pct"] = 0.0

            # Record to log
            log_id = self.backtest_logger.log_daily_analysis(
                symbol=self.current_symbol or "UNKNOWN",
                date=current_date,
                market_data=market_data,
                trend_analysis=trend_data,
                comprehensive_technical_analysis=comprehensive_context,  # New parameter
                triggered_events=triggered_events_data,
                llm_decision=llm_decision_data,
                trading_signal=None,  # Will be updated separately when signal generated
                strategy_state=strategy_state_data,
            )

            # Record individual event analysis
            for event in events:
                if event.get("type"):  # Ensure event has type
                    self.backtest_logger.log_event_analysis(
                        daily_log_id=log_id,
                        event_type=event.get("type"),
                        severity=self._determine_event_severity(event),
                        market_context={
                            "price_before": market_data["price"],
                            "volume": market_data["volume"],
                            "trend": trend_data.get("short_term", "neutral")
                            if trend_data
                            else "neutral",
                        },
                        llm_response={
                            "triggered_decision": llm_decision is not None,
                            "action_taken": llm_decision.get("action", "HOLD")
                            if llm_decision
                            else "NONE",
                            "confidence": llm_decision.get("confidence", 0.0)
                            if llm_decision
                            else 0.0,
                        },
                    )

            logger.debug(f"✅ Recorded {current_date} analysis data (log_id: {log_id})")

        except Exception as e:
            logger.error(f"❌ Log recording failed: {e}")
            import traceback

            traceback.print_exc()

    def _determine_event_severity(self, event: Dict[str, Any]) -> str:
        """
        Determine event severity

        Args:
            event: Event dictionary

        Returns:
            Severity: 'high', 'medium', 'low'
        """
        event_type = event.get("type", "").lower()
        strength = event.get("strength", "medium").lower()

        # Determine severity based on event type and strength
        if strength == "high" or event_type in [
            "large_drop",
            "large_gain",
            "volume_spike",
        ]:
            return "high"
        elif strength == "low" or event_type in ["minor_support", "minor_resistance"]:
            return "low"
        else:
            return "medium"

    def _log_trading_signal(
        self,
        timestamp: pd.Timestamp,
        signal: "TradingSignal",
        llm_decision: Dict[str, Any],
    ):
        """
        Record trading signal to log

        Args:
            timestamp: Signal timestamp
            signal: Trading signal object
            llm_decision: LLM decision result
        """
        try:
            current_date = timestamp.strftime("%Y-%m-%d")

            # Find today's log record
            recent_logs = self.backtest_logger.query_logs(
                symbol=self.current_symbol,
                date_from=current_date,
                date_to=current_date,
                limit=1,
            )

            if recent_logs:
                log_id = recent_logs[0]["id"]

                # Prepare trading signal data
                signal_data = {
                    "signal_type": signal.signal_type.name,
                    "price": signal.price,
                    "quantity": signal.quantity,
                    "confidence": signal.confidence,
                    "reasoning": signal.reason,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "timestamp": timestamp.isoformat(),
                    "llm_factors": llm_decision.get("factors", []),
                    "llm_confidence": llm_decision.get("confidence", 0.0),
                }

                # Update today's record
                with sqlite3.connect(self.backtest_logger.db_path) as conn:
                    conn.execute(
                        """
                        UPDATE daily_analysis_logs 
                        SET trading_signal = ?
                        WHERE id = ?
                    """,
                        (json.dumps(signal_data), log_id),
                    )

                logger.debug(f"✅ Trading signal log updated (log_id: {log_id})")

        except Exception as e:
            logger.error(f"❌ Trading signal recording failed: {e}")

    def get_backtest_summary(self) -> Dict[str, Any]:
        """
        Get backtest summary

        Returns:
            Backtest summary data
        """
        if not self.backtest_logger:
            return {}

        return self.backtest_logger.get_session_summary()

    def export_backtest_logs(self, filepath: str = None):
        """
        Export backtest logs

        Args:
            filepath: Export file path, if not provided use default path
        """
        if not self.backtest_logger:
            logger.warning("Logger not enabled, cannot export")
            return

        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"backtest_logs_{self.current_symbol}_{timestamp}.json"

        self.backtest_logger.export_to_json(filepath)
        return filepath