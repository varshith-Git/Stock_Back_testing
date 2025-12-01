"""
Daily Decision Improvement API
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.llm.client import get_llm_client
from app.utils.backtest_logger import BacktestLogger
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


class DailyFeedbackRequest(BaseModel):
    feedback: str
    date: str  # YYYY-MM-DD
    symbol: str = None  # Optional symbol, if not provided will search all symbols


class DailyImprovementResponse(BaseModel):
    analysis: str
    suggestions: List[str]


@router.post("/daily-feedback", response_model=DailyImprovementResponse)
async def analyze_daily_decision(
    request: DailyFeedbackRequest,
    db_path: str = Query(None, description="Database path"),
) -> DailyImprovementResponse:
    """
    Analyzes daily trading decisions and provides improvement suggestions based on user feedback.
    
    Uses the same data access pattern as the working backtest_analysis API.
    """
    try:
        # 1. Use consistent path across the application
        if not db_path:
            db_path = "backend/data/backtest_logs.db"

        print(f"🔍 Analysis date: {request.date}")
        print(f"📝 User feedback: {request.feedback}")
        print(f"🗄️ Database path: {db_path}")

        if not Path(db_path).exists():
            print(f"❌ Database file does not exist: {db_path}")
            raise HTTPException(
                status_code=404, detail=f"Database not found: {db_path}"
            )

        # 2. Find all sessions that have data for the target date
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # First, get all sessions that have data for this date
        cursor.execute(
            """
            SELECT DISTINCT session_id, symbol FROM daily_analysis_logs 
            WHERE date = ?
            ORDER BY session_id DESC
        """,
            (request.date,),
        )
        date_sessions = cursor.fetchall()

        if not date_sessions:
            conn.close()
            raise HTTPException(
                status_code=404, detail=f"No trading data found for {request.date}"
            )

        print(f"📊 Sessions and stocks for this date: {date_sessions}")

        # Determine which session to use
        target_session = None
        target_symbol = None

        if request.symbol:
            # If user specified a symbol, find the session for that symbol
            for session_id, symbol in date_sessions:
                if request.symbol.upper() in symbol or symbol in request.symbol.upper():
                    target_session = session_id
                    target_symbol = symbol
                    break

            if not target_session:
                available_symbols = [symbol for _, symbol in date_sessions]
                conn.close()
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for stock {request.symbol} on {request.date}. Available stocks: {', '.join(available_symbols)}",
                )
        else:
            # If no symbol specified, prioritize NVDA, TSLA, then others
            priority_symbols = ["NVDA", "TSLA", "AAPL", "MSFT"]

            for priority_symbol in priority_symbols:
                for session_id, symbol in date_sessions:
                    if priority_symbol in symbol:
                        target_session = session_id
                        target_symbol = symbol
                        break
                if target_session:
                    break

            if not target_session:
                # Fallback to first available
                target_session, target_symbol = date_sessions[0]

        print(f"✅ Using session: {target_session}")
        print(f"🎯 Query stock: {target_symbol}")
        conn.close()

        # 3. Initialize BacktestLogger and query data
        logger = BacktestLogger(db_path, session_id=target_session)
        logs = logger.query_logs(
            symbol=target_symbol, date_from=request.date, date_to=request.date, limit=1
        )

        if not logs:
            print(
                f"❌ No data found for specified date: {request.date} (session: {target_session})"
            )
            raise HTTPException(
                status_code=404, detail=f"No trading data found for {request.date}"
            )

        daily_data = logs[0]
        print(
            f"✅ Successfully retrieved trading data: {target_symbol} - {len(daily_data.get('triggered_events', []))} technical events"
        )

        # 4. Read trading strategy content
        strategy_content = load_trading_strategy()

        # 5. Use LLM for analysis and generate improvement suggestions
        improvement_response = await generate_daily_improvement_analysis(
            request.feedback, request.date, daily_data, strategy_content
        )

        return improvement_response

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        error_msg = (
            f"Error processing feedback: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        )
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def load_trading_strategy() -> str:
    """
    Read trading strategy file content
    """
    try:
        # Construct strategy file path
        # Current file: backend/app/api/v1/endpoints/daily_feedback.py
        # Target file: backend/app/llm/strategies/prompt/traditional_strategy.md
        current_file = Path(__file__)  # daily_feedback.py
        app_dir = current_file.parent.parent.parent.parent  # Go to backend/app/
        strategy_path = (
            app_dir / "llm" / "strategies" / "prompt" / "traditional_strategy.md"
        )

        print(f"📋 Reading strategy file: {strategy_path}")

        if not strategy_path.exists():
            print(f"⚠️ Strategy file does not exist: {strategy_path}")
            print(f"🔍 Checked path: {strategy_path.absolute()}")
            return "Strategy file not found"

        with open(strategy_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"✅ Strategy file read successfully, length: {len(content)} characters")
        return content

    except Exception as e:
        print(f"❌ Strategy file read error: {e}")
        return f"Strategy file read failed: {str(e)}"


async def generate_daily_improvement_analysis(
    feedback: str, target_date: str, daily_data: Dict[str, Any], strategy_content: str
) -> DailyImprovementResponse:
    """
    Generate daily decision improvement analysis
    """
    try:
        # Prepare data summary
        triggered_events = daily_data.get("triggered_events", [])
        llm_decision = daily_data.get("llm_decision", {})
        symbol = daily_data.get("symbol", "Unknown")
        price = daily_data.get("price", 0)

        # Build LLM prompt
        context = f"""Hi! I'm your AI trading strategy discussion partner. The user has thoughts about the decision on {target_date}, let's analyze and optimize the strategy file together!

=== User's Thoughts ===
{feedback}

=== Situation of That Day ===
{target_date} - {symbol} ${price:.2f}

=== My Decision Logic at That Time ===
{llm_decision.get("decision_type", "N/A")}: {llm_decision.get("reasoning", "N/A")}

=== Current Trading Strategy File Content ===
{strategy_content}

Please analyze in detail and provide specific, actionable strategy file modification suggestions!

## My Analysis
[First explain why the current strategy made this decision, then evaluate the reasonableness of the user's suggestion, about 2-3 paragraphs]

## Strategy File Modification Suggestions
Please provide 3 specific modification suggestions, each containing:
- Modification location/section
- Specific new rule text
- Actual parameters or conditions

Format as follows:
1. [Modification Title]: [Detailed description of what specific rule to add/modify in which section of the strategy file, including complete content such as parameters, conditions, logic, at least 2-3 lines of detailed description]

2. [Modification Title]: [Detailed description of what specific rule to add/modify in which section of the strategy file, including complete content such as parameters, conditions, logic, at least 2-3 lines of detailed description]

3. [Modification Title]: [Detailed description of what specific rule to add/modify in which section of the strategy file, including complete content such as parameters, conditions, logic, at least 2-3 lines of detailed description]

## Modification Reason Explanation
[Briefly explain why these modifications are needed, and the expected improvement effects]
"""

        # Get LLM response
        llm_client = get_llm_client(temperature=0.7, max_tokens=1500)
        response = llm_client.invoke(context)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse response
        analysis_parts = parse_llm_response(response_text)

        return DailyImprovementResponse(
            analysis=analysis_parts.get("analysis", "Analysis generating..."),
            suggestions=analysis_parts.get("suggestions", ["Please try again later"]),
        )

    except Exception as e:
        print(f"❌ LLM analysis error: {e}")
        # Return fallback response
        return DailyImprovementResponse(
            analysis=f"Based on your feedback 「{feedback}」, we are analyzing the reasonableness of the decision on {target_date}.",
            suggestions=[
                "Check combination usage of technical indicators",
                "Evaluate accuracy of market trend judgment",
                "Optimize risk control parameters",
            ],
        )


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response text
    """
    try:
        print(f"🔍 Raw LLM response:\n{response_text}\n")

        parts = {"analysis": "", "suggestions": [], "strategy_review": ""}

        # Simple text segmentation parsing
        sections = response_text.split("##")
        print(f"📝 Number of sections after splitting: {len(sections)}")

        for i, section in enumerate(sections):
            section = section.strip()
            print(f"Section {i}: {section[:100]}...")

            if "My Analysis" in section:
                # Remove title and keep content
                content = section.replace("My Analysis", "", 1).strip()
                # Remove possible colon
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["analysis"] = content
                print(f"✅ Found 「My Analysis」: {content[:50]}...")
            elif "Decision Analysis" in section:  # Alternative title
                content = section.replace("Decision Analysis", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["analysis"] = content
            elif "Strategy File Modification Suggestions" in section:
                suggestions_text = section.replace("Strategy File Modification Suggestions", "", 1).strip()
                if suggestions_text.startswith(":"):
                    suggestions_text = suggestions_text[1:].strip()
                print(f"✅ Found 「Strategy File Modification Suggestions」: {suggestions_text[:100]}...")

                # More intelligent suggestion extraction - preserve complete content
                suggestions = []
                lines = suggestions_text.split("\n")
                current_suggestion = ""

                for line in lines:
                    line = line.strip()
                    # Check if it's the start of a new suggestion item
                    if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                        # If there's a previous suggestion, save it first
                        if current_suggestion:
                            suggestions.append(current_suggestion.strip())
                        # Start new suggestion, remove numbering
                        current_suggestion = line[2:].strip()
                    elif line and current_suggestion:
                        # Continue with current suggestion content
                        current_suggestion += "\n" + line
                    elif not current_suggestion and line:
                        # Handle suggestion lines without numbering
                        current_suggestion = line

                # Add the last suggestion
                if current_suggestion:
                    suggestions.append(current_suggestion.strip())

                print(f"📋 Extracted complete suggestions count: {len(suggestions)}")
                for i, suggestion in enumerate(suggestions):
                    print(f"Suggestion {i + 1}: {suggestion[:50]}...")

                parts["suggestions"] = suggestions
            elif "Some Suggestions" in section:  # Alternative title
                suggestions_text = section.replace("Some Suggestions", "", 1).strip()
                if suggestions_text.startswith(":"):
                    suggestions_text = suggestions_text[1:].strip()
                # Extract list items
                suggestions = []
                for line in suggestions_text.split("\n"):
                    line = line.strip()
                    if line and (
                        line.startswith("1.")
                        or line.startswith("2.")
                        or line.startswith("3.")
                        or line.startswith("4.")
                        or line.startswith("5.")
                    ):
                        suggestions.append(line[2:].strip())
                parts["suggestions"] = suggestions
            elif "Improvement Suggestions" in section:  # Alternative title
                suggestions_text = section.replace("Improvement Suggestions", "", 1).strip()
                if suggestions_text.startswith(":"):
                    suggestions_text = suggestions_text[1:].strip()
                suggestions = []
                for line in suggestions_text.split("\n"):
                    line = line.strip()
                    if line and (
                        line.startswith("1.")
                        or line.startswith("2.")
                        or line.startswith("3.")
                        or line.startswith("4.")
                        or line.startswith("5.")
                    ):
                        suggestions.append(line[2:].strip())
                parts["suggestions"] = suggestions
            elif "Modification Reason Explanation" in section:
                content = section.replace("Modification Reason Explanation", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["strategy_review"] = content
            elif "Strategy Optimization Ideas" in section:  # Alternative title
                content = section.replace("Strategy Optimization Ideas", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["strategy_review"] = content
            elif "Strategy Review" in section:  # Alternative title
                content = section.replace("Strategy Review", "", 1).strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                parts["strategy_review"] = content

        print(
            f"📊 Final parsing result: analysis={bool(parts['analysis'])}, suggestions={len(parts['suggestions'])}, strategy_review={bool(parts['strategy_review'])}"
        )
        return parts

    except Exception as e:
        print(f"❌ Response parsing error: {e}")
        return {
            "analysis": response_text[:200] + "...",
            "suggestions": ["Please try again later"],
            "strategy_review": "Review analysis in progress...",
        }