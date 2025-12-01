"""
Backtest Logger
Records detailed decision processes and analysis data for LLM strategies
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class BacktestLogger:
    """
    Backtest Logger

    Records detailed decision processes for LLM strategies, including:
    - Daily market analysis
    - Triggered events
    - LLM decision processes
    - Trading signals
    - Strategy states
    """

    def __init__(
        self, db_path: str = "backend/data/backtest_logs.db", session_id: str = None
    ):
        """
        Initialize the logger

        Args:
            db_path: SQLite database file path
            session_id: Backtest session ID, auto-generated if not provided
        """
        self.db_path = Path(db_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())
        self._init_database()

    def _init_database(self):
        """Initialize database structure"""
        with sqlite3.connect(self.db_path) as conn:
            # Create main table: daily analysis logs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_analysis_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    timestamp DATETIME NOT NULL,
                    
                    -- Basic market data (structured, easy to query)
                    price REAL,
                    volume INTEGER,
                    daily_return REAL,
                    volatility REAL,
                    
                    -- Trend analysis (JSON)
                    trend_analysis TEXT, -- JSON string
                    
                    -- Comprehensive technical analysis (JSON) - new
                    comprehensive_technical_analysis TEXT, -- JSON string
                    
                    -- Triggered events (JSON)
                    triggered_events TEXT, -- JSON string
                    
                    -- LLM decision (JSON)
                    llm_decision TEXT, -- JSON string
                    
                    -- Trading signal (JSON)
                    trading_signal TEXT, -- JSON string
                    
                    -- Strategy state (JSON)
                    strategy_state TEXT, -- JSON string
                    
                    -- Result evaluation (updated later)
                    actual_pnl REAL,
                    prediction_accuracy REAL,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create event analysis table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_analysis_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    daily_log_id INTEGER,
                    event_type TEXT NOT NULL,
                    severity TEXT,
                    detection_time DATETIME,
                    
                    -- Market context (JSON)
                    market_context TEXT, -- JSON string
                    
                    -- LLM response (JSON) 
                    llm_response TEXT, -- JSON string
                    
                    -- Effectiveness evaluation (JSON)
                    effectiveness TEXT, -- JSON string
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (daily_log_id) REFERENCES daily_analysis_logs (id)
                )
            """)

            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_logs_date_symbol 
                ON daily_analysis_logs (date, symbol)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_logs_session 
                ON daily_analysis_logs (session_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type 
                ON event_analysis_logs (event_type)
            """)

            conn.commit()

    def log_daily_analysis(
        self,
        symbol: str,
        date: str,
        market_data: Dict[str, Any],
        trend_analysis: Dict[str, Any] = None,
        comprehensive_technical_analysis: Dict[str, Any] = None,  # New parameter
        triggered_events: List[Dict[str, Any]] = None,
        llm_decision: Dict[str, Any] = None,
        trading_signal: Dict[str, Any] = None,
        strategy_state: Dict[str, Any] = None,
    ) -> int:
        """
        Record daily analysis data (new records overwrite old records for same stock same day)

        Args:
            symbol: Stock symbol
            date: Date (YYYY-MM-DD)
            market_data: Market data dictionary
            trend_analysis: Trend analysis result
            comprehensive_technical_analysis: Comprehensive technical analysis result
            triggered_events: Triggered events list
            llm_decision: LLM decision result
            trading_signal: Trading signal
            strategy_state: Strategy state

        Returns:
            Record ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if same record exists (same symbol + date)
            cursor.execute(
                """
                SELECT id FROM daily_analysis_logs 
                WHERE symbol = ? AND date = ?
                ORDER BY timestamp DESC
            """,
                (symbol, date),
            )

            existing_records = cursor.fetchall()

            if existing_records:
                # Delete old records and related event records
                old_ids = [record[0] for record in existing_records]
                old_ids_str = ",".join("?" * len(old_ids))

                # First delete related event analysis records
                cursor.execute(
                    f"""
                    DELETE FROM event_analysis_logs 
                    WHERE daily_log_id IN ({old_ids_str})
                """,
                    old_ids,
                )

                # Then delete daily analysis records
                cursor.execute(
                    f"""
                    DELETE FROM daily_analysis_logs 
                    WHERE id IN ({old_ids_str})
                """,
                    old_ids,
                )

                print(f"🔄 Overwriting old records for {symbol} - {date} ({len(old_ids)} records)")

            # Insert new record
            cursor.execute(
                """
                INSERT INTO daily_analysis_logs (
                    session_id, symbol, date, timestamp,
                    price, volume, daily_return, volatility,
                    trend_analysis, comprehensive_technical_analysis, triggered_events, llm_decision,
                    trading_signal, strategy_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.session_id,
                    symbol,
                    date,
                    datetime.now().isoformat(),
                    market_data.get("price"),
                    market_data.get("volume"),
                    market_data.get("daily_return"),
                    market_data.get("volatility"),
                    json.dumps(trend_analysis) if trend_analysis else None,
                    json.dumps(comprehensive_technical_analysis)
                    if comprehensive_technical_analysis
                    else None,
                    json.dumps(triggered_events) if triggered_events else None,
                    json.dumps(llm_decision) if llm_decision else None,
                    json.dumps(trading_signal) if trading_signal else None,
                    json.dumps(strategy_state) if strategy_state else None,
                ),
            )

            return cursor.lastrowid

    def log_event_analysis(
        self,
        daily_log_id: int,
        event_type: str,
        severity: str,
        market_context: Dict[str, Any] = None,
        llm_response: Dict[str, Any] = None,
        effectiveness: Dict[str, Any] = None,
    ):
        """
        Record event analysis data

        Args:
            daily_log_id: Corresponding log record ID
            event_type: Event type
            severity: Severity
            market_context: Market context
            llm_response: LLM response
            effectiveness: Effectiveness evaluation
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO event_analysis_logs (
                    session_id, daily_log_id, event_type, severity,
                    detection_time, market_context, llm_response, effectiveness
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.session_id,
                    daily_log_id,
                    event_type,
                    severity,
                    datetime.now().isoformat(),
                    json.dumps(market_context) if market_context else None,
                    json.dumps(llm_response) if llm_response else None,
                    json.dumps(effectiveness) if effectiveness else None,
                ),
            )

    def update_actual_results(
        self, log_id: int, actual_pnl: float, prediction_accuracy: float
    ):
        """
        Update actual results

        Args:
            log_id: Log record ID
            actual_pnl: Actual P&L
            prediction_accuracy: Prediction accuracy
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE daily_analysis_logs 
                SET actual_pnl = ?, prediction_accuracy = ?
                WHERE id = ?
            """,
                (actual_pnl, prediction_accuracy, log_id),
            )

    def query_logs(
        self,
        symbol: str = None,
        date_from: str = None,
        date_to: str = None,
        event_type: str = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query log records

        Args:
            symbol: Stock symbol
            date_from: Start date
            date_to: End date
            event_type: Event type
            limit: Limit return count

        Returns:
            List of log records
        """
        query = """
            SELECT d.*, GROUP_CONCAT(e.event_type) as event_types
            FROM daily_analysis_logs d
            LEFT JOIN event_analysis_logs e ON d.id = e.daily_log_id
            WHERE d.session_id = ?
        """
        params = [self.session_id]

        if symbol:
            query += " AND d.symbol = ?"
            params.append(symbol)

        if date_from:
            query += " AND d.date >= ?"
            params.append(date_from)

        if date_to:
            query += " AND d.date <= ?"
            params.append(date_to)

        query += " GROUP BY d.id ORDER BY d.date DESC, d.timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            # Convert to dictionary and parse JSON fields
            results = []
            for row in rows:
                record = dict(row)

                # Parse JSON fields
                for json_field in [
                    "trend_analysis",
                    "comprehensive_technical_analysis",
                    "triggered_events",
                    "llm_decision",
                    "trading_signal",
                    "strategy_state",
                ]:
                    if record[json_field]:
                        try:
                            record[json_field] = json.loads(record[json_field])
                        except json.JSONDecodeError:
                            record[json_field] = None

                results.append(record)

            return results

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get session summary statistics

        Returns:
            Session statistics data
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable Row factory

            # Basic statistics
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_days,
                    COUNT(DISTINCT symbol) as symbols_count,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    AVG(actual_pnl) as avg_pnl,
                    SUM(actual_pnl) as total_pnl
                FROM daily_analysis_logs 
                WHERE session_id = ?
            """,
                (self.session_id,),
            )

            row = cursor.fetchone()
            basic_stats = dict(row) if row else {}

            # LLM decision statistics
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_decisions,
                    AVG(CASE WHEN json_extract(llm_decision, '$.decision_made') = 1 
                        THEN 1 ELSE 0 END) as decision_rate,
                    AVG(CAST(json_extract(llm_decision, '$.confidence') AS REAL)) as avg_confidence
                FROM daily_analysis_logs 
                WHERE session_id = ? AND llm_decision IS NOT NULL
            """,
                (self.session_id,),
            )

            row = cursor.fetchone()
            llm_stats = dict(row) if row else {}

            # Event statistics
            cursor = conn.execute(
                """
                SELECT 
                    event_type,
                    COUNT(*) as count,
                    AVG(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_severity_rate
                FROM event_analysis_logs 
                WHERE session_id = ?
                GROUP BY event_type
                ORDER BY count DESC
            """,
                (self.session_id,),
            )

            event_stats = [dict(row) for row in cursor.fetchall()]

            return {
                "session_id": self.session_id,
                "basic_stats": basic_stats,
                "llm_stats": llm_stats,
                "event_stats": event_stats,
            }

    def export_to_json(self, filepath: str):
        """
        Export logs to JSON file

        Args:
            filepath: Output file path
        """
        logs = self.query_logs(limit=None)
        summary = self.get_session_summary()

        export_data = {"session_summary": summary, "logs": logs}

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"✅ Logs exported to: {filepath}")