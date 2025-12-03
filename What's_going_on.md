# 📊 LLM Agent Trader - What's Happening Here?

## Quick Overview
AI-powered **stock trading backtesting system** that uses LLMs (GPT-4/Gemini) to make intelligent buy/sell decisions based on technical analysis.

---

## Core Flow
```
Stock Data (1yr) → Technical Analysis → LLM Decision → Trade Simulation → Performance Report
```

---

## Architecture Diagram

![Architecture Diagram](assets/architecture_diagram.png)


---

## Initial Capital & Money Management

**Starting Capital:** $1,000,000 (configurable)

```
Initial: $1,000,000
├─ Max per trade: 1,000 shares
├─ Commission: 0.14% per transaction
└─ Mode: Long-only (buy & hold, no shorting)

Daily simulation:
├─ Get signal (BUY/SELL from LLM)
├─ Execute if confidence > 60%
├─ Update: cash, positions, P&L
└─ Record daily snapshot
```

### Example Trade:
```
Day 1 - BUY: Price $100, Shares 1,000
  Cost = (1,000 × $100) + commission = $100,142.50
  Cash: $1,000,000 → $899,857.50
  Position: 1,000 shares

Day 5 - SELL: Price $110, Shares 1,000  
  Proceeds = (1,000 × $110) - commission = $109,842.75
  Cash: $899,857.50 → $1,009,700.25
  Profit: +$9,700.25
```

---

## Backtesting Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Initial Capital | $1,000,000 | Starting money |
| Max Shares/Trade | 1,000 | Position limit |
| Commission | 0.14% | Trading cost |
| Trading Mode | LONG_ONLY | No short selling |
| Confidence Threshold | 60% | Min to trade |
| Stop Loss | -10% | Risk limit |
| Take Profit | +15% | Profit target |

**Technical Indicators Used:**
- Moving Averages (5, 10, 20-day)
- RSI (14-period) → Oversold <30, Overbought >70
- MACD (12, 26, 9)
- Bollinger Bands (20-period, 2σ)
- Trend Analysis (20-day lookback)

---

## Portfolio Tracking (Daily)

Each day records:
```python
{
    "date": "2024-01-15",
    "cash": $899,857.50,           # Available funds
    "position": 1,000,              # Shares held
    "stock_price": $105.50,         # Current price
    "stock_value": $105,500,        # Shares worth
    "total_value": $1,005,357.50,   # Cash + stocks
    "cumulative_return": +0.535%,   # Overall return
    "unrealized_pnl": +$5,500,      # Open position gain
    "unrealized_pnl_pct": +5.2%     # % gain on position
}
```

---

## Performance Metrics Calculated

### Returns:
- Total Return: (Final - Initial) / Initial
- Annual Return: ((Final/Initial)^(1/years)) - 1
- Cumulative Return: Daily tracking of return %

### Risk:
- Max Drawdown: Worst peak-to-trough loss
- Volatility: Price variation (annualized)
- Sharpe Ratio: Risk-adjusted return

### Trades:
- Total Trades: # of buy + sell
- Win Rate: % of profitable trades
- Profit Factor: Gross gains / Gross losses

### Benchmark:
- Buy & Hold Return: Simple hold comparison
- Alpha: Strategy return - Buy & Hold return
- Outperformed: Yes/No vs passive strategy

---

## LLM Decision Making

**Workflow:**
```
1. Analyze technical indicators for current day
2. Detect key events (oversold, breakout, etc.)
3. Call LLM API with:
   - Price history
   - Technical analysis
   - Market conditions
   - Detected events
4. LLM returns: {action: "BUY|SELL", confidence: 0.0-1.0}
5. Execute if confidence > 60%
6. Log all decisions to SQLite database
```

**LLM Parameters:**
- Confidence Threshold: 0.6 (60%)
- Max Daily Trades: 3
- Position Size: 20% per trade
- Risk Management: Auto stop-loss & take-profit

---

## Database Logging

**Location:** `backend/data/backtest_logs.db` (SQLite)

**Logged:**
- Daily technical analysis (MA, RSI, MACD, Bollinger, etc.)
- Triggered events (oversold, breakout, volume spike)
- LLM decisions (confidence, reasoning, action)
- Trade execution details
- Portfolio snapshots

---

## Trade Execution Logic

### BUY Order:
```
1. Calculate max shares: (available_cash - commission) / price
2. Check sufficient funds
3. Execute: cash -= (shares × price) + commission
4. Record trade with timestamp & confidence
```

### SELL Order:
```
1. Check if holding position
2. Sell all: proceeds = shares × price - commission
3. Execute: cash += proceeds
4. Clear position, record P&L
```

---

## Tech Stack

- **Backend:** FastAPI (Python 3.12+)
- **Frontend:** Next.js + React + TypeScript
- **LLM:** Azure OpenAI GPT-4 OR Google Gemini
- **Data:** YFinance (stock prices)
- **Database:** SQLite (backtest logs)
- **Real-time:** Server-Sent Events (SSE streaming)

---

## File Structure

```
backend/
├── app/
│   ├── backtesting/engine.py      # Backtest simulation
│   ├── llm/strategies/
│   │   └── llm_strategy.py         # LLM decision logic
│   ├── llm/analysis/
│   │   ├── enhanced_technical_analyzer.py
│   │   └── trend_analyzer.py
│   ├── utils/
│   │   ├── indicators.py           # Technical indicators
│   │   └── backtest_logger.py      # SQLite logging
│   └── api/v1/endpoints/           # REST APIs
│
frontend/
├── src/
│   ├── components/charts/
│   │   └── SimpleTradingViewChart.tsx   # Price chart
│   ├── components/analysis/
│   │   └── BacktestResultsWithAnalysis.tsx
│   └── types/index.ts              # TypeScript types
```

---

## Key Takeaways

✅ **What it does:**
- Backtests trading strategies using AI
- Starts with $1M, executes buy/sell signals
- Tracks every penny in cash, positions, P&L
- Calculates comprehensive performance metrics

✅ **How it manages money:**
- Initial capital = $1,000,000
- Each trade limited to 1,000 shares max
- Commissions deducted from proceeds
- Cash balance updated after every trade
- Unrealized P&L calculated daily

✅ **Why LLM?**
- Analyzes complex market conditions
- Decides when technical patterns matter
- Adjusts confidence based on certainty
- Provides reasoning for every trade

✅ **Real Risk Management:**
- Stop-loss at -10%
- Take-profit at +15%
- Max 3 trades per day
- Position sizing adjustment
- Daily unrealized P&L tracking

---

**Status:** Production-ready AI trading backtesting platform with database persistence, real-time streaming, and comprehensive analytics.
