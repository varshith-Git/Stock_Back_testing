// Basic trading types
export interface TradingSignal {
  timestamp: string
  signal_type: 'BUY' | 'SELL' | 'HOLD'
  confidence: number
  price: number
  reason: string
  metadata?: Record<string, unknown>
}

// Stock data types
export interface StockData {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  // Technical indicators
  ma_5?: number
  ma_10?: number
  ma_20?: number
  rsi?: number
  macd?: number
  macd_signal?: number
  macd_histogram?: number
  // Bollinger Bands indicators
  bb_upper?: number
  bb_lower?: number
  bb_middle?: number
  // Forward compatibility
  bollinger_upper?: number
  bollinger_lower?: number
}

// LLM decision logs
export interface LLMDecisionLog {
  timestamp: string  // Timestamp
  decision: {
    action?: 'BUY' | 'SELL' | 'HOLD'
    confidence?: number
    reasoning?: string
    risk_level?: 'low' | 'medium' | 'high'
    expected_outcome?: string
  }
  reasoning: string  // Decision reasoning
  events: Array<{
    type: string
    description: string
    strength: 'low' | 'medium' | 'high'
  }>
  action: string    // Action type (e.g., "THINK")
  confidence: number
  price: number
  // Backward compatibility
  date?: string
}

// Backtest result types
export interface BacktestResult {
  trades: unknown[]
  performance: Record<string, unknown>
  stock_data: StockData[]
  signals: TradingSignal[]
  llm_decisions: LLMDecisionLog[]
  statistics: {
    total_trades: number
    win_rate: number
    total_return: number
    max_drawdown: number
    final_value?: number
    total_realized_pnl?: number
    cumulative_trade_return_rate?: number
  }
}

// Technical event types
export interface TechnicalEvent {
  type: string
  description: string
  significance: number
  impact: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL'
}

// Trend analysis types
export interface TrendAnalysis {
  primary_trend: 'BULLISH' | 'BEARISH' | 'SIDEWAYS'
  trend_strength: number
  trend_duration: number
  momentum_indicators: Record<string, unknown>
  support_resistance: {
    support_levels: number[]
    resistance_levels: number[]
  }
}

// Retrospective analysis types
export interface RetrospectiveAnalysis {
  summary: string
  decision_quality: {
    score: number
    reasoning: string
    alternatives: string[]
  }
  market_context: {
    market_conditions: string
    volatility_assessment: string
    key_factors: string[]
  }
  performance_impact: {
    immediate_impact: string
    potential_outcomes: string
    risk_assessment: string
  }
  lessons_learned: string[]
  recommendations: string[]
}

// Daily analysis response types (matching backend API)
export interface DayAnalysisResponse {
  historical_data: {
    date: string
    symbol: string
    price: number
    daily_return?: number
    volume?: number
    market_data?: {
      open?: number
      high?: number
      low?: number
      close: number
      volume?: number
    }
    trend_analysis?: {
      short_term?: string
      medium_term?: string
      long_term?: string
      trend_strength?: number
      confidence?: number
    }
    comprehensive_technical_analysis?: {
      date?: string
      price_action?: Record<string, unknown>
      moving_averages?: Record<string, unknown>
      volume_analysis?: Record<string, unknown>
      volatility_analysis?: Record<string, unknown>
      momentum_indicators?: Record<string, unknown>
      support_resistance?: Record<string, unknown>
      trend_analysis?: Record<string, unknown>
      market_regime?: Record<string, unknown>
      bollinger_analysis?: Record<string, unknown>
      macd_analysis?: Record<string, unknown>
    }
    technical_events: Array<{
      event_type: string
      severity: string
      description: string
      technical_data?: Record<string, unknown>
    }>
    llm_decision?: {
      decision_made: boolean
      decision_type?: string
      confidence?: number
      reasoning?: string
      risk_level?: string
    }
    strategy_state?: Record<string, unknown>
  }
  retrospective_analysis?: {
    llm_commentary: string
    decision_quality_score?: number
    alternative_perspective?: string
    lessons_learned?: string
  }
}