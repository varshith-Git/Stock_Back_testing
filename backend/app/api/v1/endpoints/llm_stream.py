"""
LLM Strategy Streaming Backtest API - Server-Sent Events (SSE)
Provides real-time progress updates and result streaming
"""

import asyncio
import json
import os
import queue
import threading
import time
from datetime import datetime, timedelta
from typing import Generator, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ....backtesting.engine import BacktestConfig, BacktestEngine
from ....llm.strategies.base import StrategyConfig
from ....llm.strategies.llm_strategy import LLMSmartStrategy  # Switch back to original version
from ....utils.stock_data import StockService

router = APIRouter()


def safe_json_serialize(obj):
    """
    Safe JSON serialization function to handle various non-serializable objects
    """
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        # DataFrame conversion to list of dictionaries
        return obj.to_dict("records") if hasattr(obj, "to_dict") else str(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, "to_dict"):
        # If object has to_dict method, use it
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        # If custom object, try to serialize its attributes
        return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
    else:
        return str(obj)


def safe_json_dumps(obj):
    """
    Safe JSON dumps, preprocess non-serializable objects
    """

    def json_serializer(o):
        return safe_json_serialize(o)

    return json.dumps(obj, default=json_serializer, ensure_ascii=False)


@router.get("/llm-backtest-stream")
async def stream_llm_backtest(
    symbol: str = Query(..., description="Stock symbol"),
    period: str = Query("1y", description="Backtest period"),
    max_position_size: float = Query(0.3, description="Maximum position size ratio"),
    stop_loss: float = Query(0.05, description="Stop loss ratio"),
    take_profit: float = Query(0.1, description="Take profit ratio"),
):
    """
    Stream LLM strategy backtest - using Server-Sent Events
    Unlimited capital mode: Use fixed large capital, focus on pure trading P&L calculation
    """
    # Use unlimited capital mode (100 million USD)
    initial_capital = 100000000.0
    try:

        def generate_backtest_stream() -> Generator[str, None, None]:
            """Generate backtest progress stream"""

            # Create message queue for thread communication
            message_queue = queue.Queue()

            def progress_callback(
                day: int,
                total_days: int,
                event_type: str,
                message: str,
                extra_data: dict = None,
            ):
                """Progress callback function"""
                progress_data = {
                    "type": "trading_progress",
                    "day": day,
                    "total_days": total_days,
                    "progress": round((day / total_days) * 100, 1),
                    "event_type": event_type,
                    "message": message,
                }

                # If there is extra data (such as P&L information), add to progress data
                if extra_data:
                    progress_data.update(extra_data)

                message_queue.put(progress_data)

            def run_backtest():
                """Run backtest in separate thread"""
                try:
                    # 1. Get stock data
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "data_loading",
                            "message": f"Fetching {symbol} stock data...",
                        }
                    )

                    stock_service = StockService()
                    stock_data_list = stock_service.get_market_data(symbol, period)

                    if not stock_data_list or len(stock_data_list) < 30:
                        message_queue.put(
                            {"type": "error", "message": "Insufficient stock data for backtest"}
                        )
                        return

                    # Convert to DataFrame, backtest engine requires DataFrame format
                    stock_data = pd.DataFrame(stock_data_list)
                    # Set date as index and ensure column names match expectations
                    stock_data["date"] = pd.to_datetime(stock_data["date"])
                    stock_data.set_index("date", inplace=True)
                    # Rename columns to match standard format (lowercase, because strategy and engine expect lowercase column names)
                    stock_data.columns = ["open", "high", "low", "close", "volume"]

                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "data_loaded",
                            "message": f"Successfully retrieved {len(stock_data)} days of data",
                        }
                    )

                    # 2. Initialize strategy
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "strategy_init",
                            "message": "Initializing LLM strategy...",
                        }
                    )

                    # Create LLM strategy configuration - use fixed model from .env
                    strategy_config = StrategyConfig(
                        name="LLM Trading Strategy",
                        description="AI-powered trading strategy with real-time analysis",
                        parameters={
                            "initial_capital": initial_capital,
                            "max_position_size": max_position_size,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "progress_callback": progress_callback,
                            # Enable backtest logging
                            "enable_logging": True,
                            "log_path": os.path.join(
                                "backend", "data", "backtest_logs.db"
                            ),
                            "session_id": f"api_session_{symbol}_{int(time.time())}",
                        },
                    )

                    strategy = LLMSmartStrategy(strategy_config)

                    # 3. Start backtest
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "backtest_start",
                            "message": "Starting backtest execution...",
                        }
                    )

                    # Set up backtest configuration
                    backtest_config = BacktestConfig(initial_capital=initial_capital)
                    backtest_engine = BacktestEngine(backtest_config)

                    # Execute backtest
                    backtest_result = backtest_engine.run_backtest(
                        stock_data=stock_data,
                        strategy=strategy,
                        initial_cash=initial_capital,
                        symbol=symbol,
                    )

                    # 4. Send results
                    message_queue.put(
                        {
                            "type": "progress",
                            "step": "analysis",
                            "message": "Analyzing backtest results...",
                        }
                    )

                    # Convert DataFrame to JSON-serializable format
                    stock_data_json = []
                    for date, row in stock_data.iterrows():
                        stock_data_json.append(
                            {
                                "timestamp": date.strftime("%Y-%m-%d"),
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": int(row["volume"]),
                            }
                        )

                    # Extract LLM decision logs from strategy object
                    llm_decisions = []
                    if hasattr(strategy, "decision_log"):
                        for log_entry in strategy.decision_log:
                            # Safely handle decision, ensure not to call .get() on None
                            decision = log_entry.get("decision", {})
                            if decision is None:
                                decision = {}

                            llm_decisions.append(
                                {
                                    "timestamp": log_entry["date"].isoformat()
                                    if hasattr(log_entry["date"], "isoformat")
                                    else str(log_entry["date"]),
                                    "decision": decision,
                                    "reasoning": log_entry.get("reasoning", ""),
                                    "events": log_entry.get("events", []),
                                    "action": "THINK",  # LLM thinking decision, not trading signal
                                    "confidence": decision.get("confidence", 0.8),
                                    "price": 0.0,  # This will be looked up based on timestamp at frontend
                                }
                            )

                    result_data = {
                        "type": "result",
                        "data": {
                            "trades": backtest_result.get("trades", []),
                            "performance": backtest_result.get(
                                "performance_metrics", {}
                            ),
                            "stock_data": stock_data_json,
                            "signals": backtest_result.get(
                                "trading_signals", []
                            ),  # Fixed: use trading_signals instead of signals
                            "llm_decisions": llm_decisions,
                            "statistics": {
                                # Use strategy statistics, these are actual trading P&L data
                                "total_trades": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("total_trades", 0),
                                "win_rate": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("strategy_win_rate", 0)
                                * 100,  # Convert to percentage
                                "total_return": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("cumulative_trade_return_rate", 0)
                                * 100,  # Convert to percentage
                                "max_drawdown": backtest_result.get(
                                    "performance_metrics", {}
                                ).get("max_drawdown", 0),
                                "final_value": backtest_result.get(
                                    "performance_metrics", {}
                                ).get("final_value", 0),
                                "annual_return": backtest_result.get(
                                    "performance_metrics", {}
                                ).get("annual_return", 0),
                                "volatility": backtest_result.get(
                                    "performance_metrics", {}
                                ).get("volatility", 0),
                                "num_trades": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("total_trades", 0),
                                # Add strategy-specific realized P&L
                                "total_realized_pnl": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("total_realized_pnl", 0),
                                "winning_trades": backtest_result.get(
                                    "strategy_statistics", {}
                                ).get("winning_trades", 0),
                            },
                            "strategy_statistics": backtest_result.get(
                                "strategy_statistics", {}
                            ),
                        },
                    }

                    message_queue.put(result_data)
                    message_queue.put(
                        {"type": "complete", "message": "LLM strategy backtest completed!"}
                    )

                except Exception as e:
                    message_queue.put(
                        {"type": "error", "message": f"Error during backtest: {str(e)}"}
                    )
                finally:
                    message_queue.put(None)  # End signal

            # Send start signal
            yield f"data: {safe_json_dumps({'type': 'start', 'message': 'Starting LLM strategy backtest...'})}\n\n"

            # Start backtest in background thread
            backtest_thread = threading.Thread(target=run_backtest)
            backtest_thread.start()

            # Continuously read messages from queue and send
            while True:
                try:
                    message = message_queue.get(timeout=1)
                    if message is None:  # End signal
                        break
                    yield f"data: {safe_json_dumps(message)}\n\n"
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {safe_json_dumps({'type': 'heartbeat'})}\n\n"
                    continue

            # Wait for thread to complete
            backtest_thread.join()

        # Return SSE response
        return StreamingResponse(
            generate_backtest_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to start streaming backtest: {str(e)}")


@router.get("/llm-backtest-stream/status")
async def get_stream_status():
    """
    Check streaming backtest service status
    """
    return {
        "status": "ready",
        "message": "LLM streaming backtest service running normally",
        "timestamp": datetime.now().isoformat(),
    }