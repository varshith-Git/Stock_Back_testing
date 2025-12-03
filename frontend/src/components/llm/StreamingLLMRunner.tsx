'use client'

import React, { useState, useRef, useCallback, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Loader2, Play, Square, TrendingUp, BarChart3, Zap } from 'lucide-react'
import BacktestResultsWithAnalysis from '@/components/analysis/BacktestResultsWithAnalysis'
import { TradingSignal, LLMDecisionLog, StockData, BacktestResult as ApiBacktestResult } from '@/types'

interface StreamMessage {
  type: 'start' | 'progress' | 'trading_progress' | 'result' | 'complete' | 'error'
  message?: string
  step?: string
  day?: number
  total_days?: number
  progress?: number
  event_type?: string
  data?: {
    stock_data?: StockData[]
    performance?: Record<string, unknown>
    statistics?: Record<string, number>
    strategy_statistics?: StrategyStats
  }
  // performance_metrics may be at root level
  performance_metrics?: {
    total_return: number
    win_rate: number
    max_drawdown: number
    total_trades: number
    total_value: number
    cash: number
    position_value: number
  }
  // pnl_status may be at root level
  pnl_status?: {
    unrealized_pnl?: number
    unrealized_pnl_pct?: number
    holding_days?: number
    shares?: number
    risk_level?: string
    cash_remaining?: number
    total_value?: number
  }
  extra_data?: {
    pnl_status?: {
      unrealized_pnl?: number
      unrealized_pnl_pct?: number
      holding_days?: number
      shares?: number
      risk_level?: string
      cash_remaining?: number
      total_value?: number
    }
    performance_metrics?: {
      total_return: number
      win_rate: number
      max_drawdown: number
      total_trades: number
      total_value: number
      cash: number
      position_value: number
    }
    current_price?: number
    strategy_statistics?: StrategyStats
  }
}

interface StrategyStats {
  total_trades?: number
  strategy_win_rate?: number
  total_realized_pnl?: number
  cumulative_trade_return_rate?: number
}

interface DynamicPerformance {
  total_return: number     // Total return rate (based on total value)
  win_rate: number         // Win rate (0-1)
  max_drawdown: number     // Maximum drawdown (0-1)
  total_trades: number     // Completed trades count (meaningful)
  total_realized_pnl?: number      // Accumulated realized P&L
  cumulative_trade_return_rate?: number  // Accumulated trade return rate
  // Future additions:
  // avg_trade_return?: number    // Average trade return rate
  // profit_loss_ratio?: number   // Profit-loss ratio
  // max_single_loss?: number     // Maximum single loss
}

interface PnLStatus {
  unrealized_pnl?: number
  unrealized_pnl_pct?: number
  holding_days?: number
  shares?: number
  risk_level?: string
  cash_remaining?: number
  total_value?: number
}

type BacktestResult = ApiBacktestResult

export default function StreamingLLMRunner() {
  const [symbol, setSymbol] = useState('AAPL')
  const [period, setPeriod] = useState('1y')
  
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState('')
  const [messages, setMessages] = useState<Array<{ text: string; ts: string }>>([])
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isStarting, setIsStarting] = useState(false) // New: prevent duplicate clicks
  const [currentRunId, setCurrentRunId] = useState<string | null>(null) // New: track unique identifier for current backtest
  
  // Dynamic performance state
  const [dynamicPerformance, setDynamicPerformance] = useState<DynamicPerformance>({
    total_return: 0,
    win_rate: 0,
    max_drawdown: 0,
    total_trades: 0
  })
  
  // P&L status
  const [pnlStatus, setPnlStatus] = useState<PnLStatus | null>(null)
  
  // Real-time signal collection
  const [realTimeSignals, setRealTimeSignals] = useState<TradingSignal[]>([])
  const [realTimeLLMDecisions, setRealTimeLLMDecisions] = useState<LLMDecisionLog[]>([])
  const [realTimeStockData, setRealTimeStockData] = useState<StockData[]>([])
  
  const eventSourceRef = useRef<EventSource | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const sessionIdRef = useRef<string | null>(null) // Add session ID

  // Cleanup function - ensure EventSource is properly closed
  const cleanupEventSource = useCallback(() => {
    if (eventSourceRef.current) {
      console.log('Cleaning up EventSource connection')
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    sessionIdRef.current = null
  }, [])

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      cleanupEventSource()
    }
  }, [cleanupEventSource])

  const translateMessage = useCallback((text: string): string => {
    let out = text
    // Normalize common Chinese punctuation to English
    out = out
      .replaceAll('，', ', ')
      .replaceAll('。', '. ')
      .replaceAll('：', ': ')
      .replaceAll('（', '(')
      .replaceAll('）', ')')
    // Simple term replacements
    const termMap: Record<string, string> = {
      'LLM決策': 'LLM decision',
      '信心度': 'confidence',
      '開始 LLM 策略回測': 'Starting LLM strategy backtest',
      '開始串流回測': 'Start streaming backtest',
      '開始執行回測': 'Starting backtest execution',
      '回測完成': 'Backtest completed',
      '回測進行中': 'Backtest in progress',
      '正在啟動': 'Starting',
      '處理進度': 'Processing progress',
      '開始LLM分析': 'Starting LLM analysis',
      '分析回測結果': 'Analyzing backtest results',
      '無持倉': 'No Position',
      '持股數量': 'Shares Held',
      '死叉': 'death cross',
      '下跌趨勢': 'downtrend',
      '上升趨勢': 'uptrend',
      '看跌信號': 'bearish signal',
      '看漲信號': 'bullish signal',
      '不進場': 'no entry',
      '觀望': 'watching',
      '逆勢操作': 'counter-trend trading',
      '趨勢反轉': 'trend reversal',
      '柱狀圖': 'histogram',
      '信號線': 'signal line',
      '日均線': 'day moving average',
      '主導趨勢': 'dominant trend',
      '嚴重性': 'severity'
      ,
      // Additional terms for mixed Chinese-English messages
      '儘管': 'although',
      '尽管': 'although',
      '觸及': 'touching',
      '布林下軌': 'lower Bollinger band',
      '超賣': 'oversold',
      '信號': 'signal',
      '強烈下跌性質': 'strongly bearish nature',
      '嚴格的風險控制原則': 'strict risk control principles',
      '空倉時': 'when flat',
      '等待趨勢明確反轉': 'wait for a clear trend reversal',
      '趨勢明確反轉': 'clear trend reversal',
      '明確的': 'clear'
    }
    for (const [cn, en] of Object.entries(termMap)) {
      out = out.replaceAll(cn, en)
    }
    // Regex-based phrase tweaks
    const regexReplacements: Array<[RegExp, string]> = [
      [/LLM decision:\s*(BUY|SELL|HOLD)/g, 'LLM decision: $1'],
      [/\(confidence:\s*([0-9.]+)\)/g, '(confidence: $1)'],
      [/觸發事件為/g, 'Trigger event: '],
      [/當前主導趨勢為明確的downtrend \(downtrend\)/g, 'Current dominant trend: downtrend'],
      [/趨勢一致性為\s*([0-9.]+)/g, 'trend consistency: $1'],
      [/根據嚴格的交易原則/g, 'According to strict trading principles'],
      [/潛在反彈信號/g, 'potential rebound signals'],
      [/避免進場/g, 'avoid entry'],
      [/當前價格\(([^)]+)\)/g, 'current price ($1)'],
      [/略低於(\d+)日均線\(([^)]+)\)/g, 'slightly below $1-day moving average ($2)'],
      [/柱狀圖為負/g, 'histogram is negative'],
      [/主導趨勢為downtrend/g, 'dominant trend: downtrend'],
      [/三重技術面偏空不進場的條件/g, 'triple bearish technical no-entry condition'],
      [/綜合來看/g, 'Overall'],
      [/市場處於下跌趨勢/g, 'The market is in a downtrend'],
      [/空倉時應保持觀望/g, 'When flat, stay on the sidelines'],
      [/等待趨勢反轉或明確的上升趨勢確立後再考慮進場/g, 'Wait for a trend reversal or a clear uptrend before considering entry'],
      [/以避免逆勢操作的風險/g, 'to avoid counter-trend risk']
    ]
    for (const [re, rep] of regexReplacements) {
      out = out.replace(re, rep)
    }
    // Clean extra spaces
    out = out.replace(/\s{2,}/g, ' ').trim()
    return out
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const addMessage = useCallback((message: string) => {
    const ts = new Date().toISOString()
    setMessages(prev => [...prev, { text: translateMessage(message), ts }])
    setTimeout(scrollToBottom, 100)
  }, [translateMessage])

  const startStreaming = async () => {
    // Prevent duplicate clicks
    if (isRunning || isStarting) {
      console.log('Backtest already running, ignoring duplicate request')
      return
    }

    // Generate unique session ID and runId
    const sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const runId = `run-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    sessionIdRef.current = sessionId
    setCurrentRunId(runId)
    console.log('Starting new streaming backtest, session ID:', sessionId, 'Run ID:', runId)
    
    setIsStarting(true)
    
    // Clean up previous connection first
    cleanupEventSource()

    setIsRunning(true)
    setProgress(0)
    setCurrentStep('')
    setMessages([])
    setResult(null)
    setError(null)
    setPnlStatus(null)
    
    // Reset real-time signal data
    setRealTimeSignals([])
    setRealTimeLLMDecisions([])
    setRealTimeStockData([])

    const params = new URLSearchParams({
      symbol,
      period,
      session_id: sessionId, // Add session ID
    })

    const url = `http://localhost:8000/api/v1/llm-stream/llm-backtest-stream?${params}`
    
    try {
      console.log('Creating new EventSource:', url)
      eventSourceRef.current = new EventSource(url)
      
      eventSourceRef.current.onopen = () => {
        console.log('EventSource connection established')
        setIsStarting(false)
      }
      
      eventSourceRef.current.onmessage = (event) => {
        try {
          const data: StreamMessage = JSON.parse(event.data)
          console.log('Received streaming data:', data.type, data.event_type, data.message)
          
          // Debug: check performance_update events
          if (data.event_type === 'performance_update') {
            console.log('Performance update detail:', {
              performance_metrics: data.performance_metrics,
              extra_data: data.extra_data,
              message: data.message
            })
          }
          
          switch (data.type) {
            case 'start':
              addMessage(data.message || 'Starting backtest...')
              break
              
            case 'progress':
              setCurrentStep(translateMessage(data.message || ''))
              // Only show important progress messages, filter internal processing messages
              const progressMessage = data.message || ''
              if (!progressMessage.includes('正在獲取') && 
                  !progressMessage.includes('成功獲取') && 
                  !progressMessage.includes('初始化') &&
                  !progressMessage.includes('開始執行') &&
                  !progressMessage.includes('分析回測結果')) {
                addMessage(progressMessage)
              }
              break
              
              case 'trading_progress':
                if (data.total_days && data.day) {
                  const progressPercent = (data.day / data.total_days) * 100
                  setProgress(progressPercent)
                  
                  // Unified P&L data update - update before all event types
                  const pnlData = data.extra_data?.pnl_status || data.pnl_status
                  if (pnlData) {
                    console.log('Updating P&L status:', {
                      event_type: data.event_type,
                      holding_days: pnlData.holding_days,
                      unrealized_pnl: pnlData.unrealized_pnl,
                      shares: pnlData.shares,
                      full_data: pnlData
                    })
                    setPnlStatus(pnlData as PnLStatus)
                  }
                  
                  if (data.event_type === 'llm_decision') {
                    // Keep complete LLM decision content for subsequent optimization analysis
                    const message = data.message || ''
                    addMessage(`🤖 ${message}`)
                    
                    // Collect LLM decision data
                    if (data.extra_data) {
                      const llmDecision: LLMDecisionLog = {
                        date: new Date().toISOString(),
                        decision: {
                          confidence: 0.8,
                          reasoning: message
                        },
                        price: data.extra_data.current_price || 0,
                        timestamp: new Date().toISOString(),
                        reasoning: message,
                        events: [],
                        action: 'THINK',
                        confidence: 0.8
                      }
                      setRealTimeLLMDecisions(prev => [...prev, llmDecision])
                    }
                  } else if (data.event_type === 'signal_generated') {
                    // Optimize signal generation display
                    const message = data.message || ''
                    const signalMatch = message.match(/(BUY|SELL).*?信心度: ([\d.]+)/)
                    if (signalMatch) {
                      const signal = signalMatch[1]
                      const confidence = signalMatch[2]
                      const icon = signal === 'BUY' ? '🚀' : '📤'
                      addMessage(`${icon} Execute ${signal} signal (confidence: ${confidence})`)
                      
                      // Collect trading signal data
                      const tradingSignal: TradingSignal = {
                        timestamp: new Date().toISOString(),
                        signal_type: signal as 'BUY' | 'SELL' | 'HOLD',
                        price: data.extra_data?.current_price || 0,
                        confidence: parseFloat(confidence),
                        reason: message
                      }
                      setRealTimeSignals(prev => [...prev, tradingSignal])
                    } else {
                      addMessage(`📈 ${message}`)
                    }
                    
                    // Silently update performance data, don't repeat messages (P&L data already updated above)
                    const signalMetrics = data.extra_data?.performance_metrics || data.performance_metrics
                    const strategyStats: StrategyStats | undefined = data.data?.strategy_statistics || data.extra_data?.strategy_statistics
                    
                    if (signalMetrics) {
                      setDynamicPerformance({
                        total_return: signalMetrics.total_return || 0,
                        win_rate: (strategyStats?.strategy_win_rate ?? signalMetrics.win_rate ?? 0),
                        max_drawdown: signalMetrics.max_drawdown || 0,
                        total_trades: (strategyStats?.total_trades ?? signalMetrics.total_trades ?? 0),
                        total_realized_pnl: (strategyStats?.total_realized_pnl ?? 0),
                        cumulative_trade_return_rate: (strategyStats?.cumulative_trade_return_rate ?? 0)
                      })
                    }
                  } else if (data.event_type === 'llm_skipped') {
                    // Skip unimportant messages, reduce log noise
                    // addMessage(`⏭️ ${data.message}`)
                  } else if (data.event_type === 'entry_point') {
                    addMessage(`🚀 ${data.message}`)
                  } else if (data.event_type === 'exit_point') {
                    addMessage(`📤 ${data.message}`)
                  } else if (data.event_type === 'performance_update') {
                    // Optimize performance update logic, avoid duplicate display
                    const metrics = data.extra_data?.performance_metrics || data.performance_metrics
                    const strategyStats: StrategyStats | undefined = data.data?.strategy_statistics || data.extra_data?.strategy_statistics
                    
                    if (metrics) {
                      const newTradeCount = strategyStats?.total_trades || metrics.total_trades || 0
                      const newReturn = metrics.total_return || 0
                      const newWinRate = strategyStats?.strategy_win_rate || metrics.win_rate || 0
                      
                      const prevTradeCount = dynamicPerformance.total_trades
                      const prevReturn = dynamicPerformance.total_return
                      
                      // Only show trade completion when count actually increases
                      if (newTradeCount > prevTradeCount && newTradeCount > 0) {
                        const returnText = (newReturn * 100).toFixed(2)
                        const winRateText = (newWinRate * 100).toFixed(1)
                        addMessage(`💰 Trade completed | Total return: ${returnText}% | Win rate: ${winRateText}% | Trades: ${newTradeCount}`)
                      } else if (newTradeCount === 0 && prevTradeCount === 0 && Math.abs(newReturn - prevReturn) > 0.05) {
                        // Only show performance updates when rate meaningfully changes without trades
                        const returnText = (newReturn * 100).toFixed(2)
                        const winRateText = (newWinRate * 100).toFixed(1)
                        addMessage(`📊 Performance update | Total return: ${returnText}% | Win rate: ${winRateText}%`)
                      }
                      
                      setDynamicPerformance({
                        total_return: newReturn,
                        win_rate: newWinRate,
                        max_drawdown: metrics.max_drawdown || 0,
                        total_trades: newTradeCount,
                        total_realized_pnl: strategyStats?.total_realized_pnl || 0,
                        cumulative_trade_return_rate: strategyStats?.cumulative_trade_return_rate || 0
                      })
                    }
                    
                    // P&L status already updated above
                  } else {
                    // Filter system messages, only show important ones
                    const message = data.message || ''
                    if (!message.includes('處理進度') && 
                        !message.includes('開始LLM分析') && 
                        message.trim() !== '') {
                      addMessage(message)
                    }
                  }
                }
                break
                
              case 'result':
              setResult(data.data as BacktestResult)
              
              // Set complete stock data for charts
              if (data.data?.stock_data) {
                setRealTimeStockData(data.data.stock_data)
              }
              
              // Update final performance data, prioritize strategy statistics from statistics
              const finalStrategyStats: StrategyStats = (data.data?.strategy_statistics as StrategyStats) || {}
              const finalPerformance = (data.data?.performance as Record<string, number>) || {}
              const finalStatistics = (data.data?.statistics as Record<string, number>) || {}
              
              setDynamicPerformance({
                total_return: finalStatistics.total_return || finalPerformance.total_return || 0,
                win_rate: finalStatistics.win_rate / 100 || finalStrategyStats.strategy_win_rate || finalPerformance.win_rate || 0, // Convert percentage to decimal
                max_drawdown: finalStatistics.max_drawdown || finalPerformance.max_drawdown || 0,
                total_trades: finalStatistics.total_trades || finalStrategyStats.total_trades || 0,
                total_realized_pnl: finalStatistics.total_realized_pnl || finalStrategyStats.total_realized_pnl || 0,
                cumulative_trade_return_rate: finalStatistics.total_return / 100 || finalStrategyStats.cumulative_trade_return_rate || 0 // Use total return rate as cumulative trade return rate
              })
              
              addMessage('✅ Backtest complete, generating charts...')
              break
              
            case 'complete':
              // Only show completion messages, not potentially inaccurate summaries
              addMessage('🎉 Backtest completed! Check the cards below for accurate statistics')
              addMessage(data.message || 'All processing complete!')
              setIsRunning(false)
              cleanupEventSource()
              break
              
            case 'error':
              setError(data.message || 'Unknown error occurred')
              addMessage(`❌ Error: ${data.message}`)
              setIsRunning(false)
              cleanupEventSource()
              break
          }
        } catch (err) {
          console.error('Error parsing streaming data:', err)
        }
      }
      
      eventSourceRef.current.onerror = (event) => {
      console.error('EventSource error:', event)
      setError('Connection interrupted or server error')
      setIsRunning(false)
      setIsStarting(false)
      cleanupEventSource()
    }
      
    } catch (err) {
      console.error('Error starting streaming:', err)
      setError('Unable to start streaming backtest')
      setIsRunning(false)
      setIsStarting(false)
    }
  }

  const stopStreaming = () => {
    console.log('Manually stopping streaming')
    cleanupEventSource()
    setIsRunning(false)
    setIsStarting(false)
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold gradient-text mb-2">
          🚀 Streaming LLM Strategy Backtest
        </h1>
        <p className="text-gray-600">Watch AI trading strategy decisions in real time</p>
      </div>

      {/* Parameter Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Backtest Parameters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div suppressHydrationWarning>
              <Label htmlFor="symbol">Stock Symbol</Label>
              <Input
                id="symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="e.g., AAPL"
                disabled={isRunning}
              />
            </div>
            
            <div suppressHydrationWarning>
              <Label htmlFor="period">Backtest Period</Label>
              <Select value={period} onValueChange={setPeriod} disabled={isRunning}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="6mo">6 months</SelectItem>
                  <SelectItem value="1y">1 year</SelectItem>
                  <SelectItem value="2y">2 years</SelectItem>
                  <SelectItem value="5y">5 years</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <div className="bg-blue-50 p-4 rounded-lg mb-4">
            <div className="text-sm text-blue-800">
              <p className="font-medium">💰 Capital Mode: Unlimited Capital</p>
              <p className="text-xs mt-1">System uses unlimited capital; P&L is based on actual trade cost and does not depend on initial capital settings</p>
            </div>
          </div>
          
          <div className="flex gap-2 mt-4" suppressHydrationWarning>
            <Button 
              onClick={startStreaming} 
              disabled={isRunning || isStarting} 
              className="flex-1"
            >
              {(isRunning || isStarting) ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {isStarting ? 'Starting...' : 'Backtest in progress...'}
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Start Streaming Backtest
                </>
              )}
            </Button>
            
            {(isRunning || isStarting) && (
              <Button onClick={stopStreaming} variant="destructive">
                <Square className="mr-2 h-4 w-4" />
                Stop
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

              {/* Progress */}
      {isRunning && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Real-time Progress & Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Processing Progress</span>
                  <span>{progress.toFixed(1)}%</span>
                </div>
                <Progress value={progress} className="w-full" />
              </div>
              
              {/* Dynamic Performance Indicators */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="text-center p-2 bg-green-50 rounded">
                  <div className="text-lg font-bold text-green-600">
                    {dynamicPerformance.total_trades}
                  </div>
                  <div className="text-xs text-gray-600">Completed Trades</div>
                </div>
                <div className="text-center p-2 bg-blue-50 rounded">
                  <div className="text-lg font-bold text-blue-600">
                    {(dynamicPerformance.win_rate * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">Strategy Win Rate</div>
                </div>
                <div className="text-center p-2 bg-purple-50 rounded">
                  <div className={`text-lg font-bold ${(dynamicPerformance.total_realized_pnl ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    ${(dynamicPerformance.total_realized_pnl ?? 0).toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-600">Total Realized P&L</div>
                </div>
                <div className="text-center p-2 bg-orange-50 rounded">
                  <div className={`text-lg font-bold ${(dynamicPerformance.cumulative_trade_return_rate ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {((dynamicPerformance.cumulative_trade_return_rate ?? 0) * 100).toFixed(2)}%
                  </div>
                  <div className="text-xs text-gray-600">Cumulative Trade Return Rate</div>
                </div>
              </div>
              
              {/* P&L Status */}
              {pnlStatus && (
                <div className="border rounded-lg p-4 bg-gradient-to-r from-green-50 to-blue-50">
                  <div className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <TrendingUp className="h-4 w-4" />
                    Current Trading Status
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className={`text-xl font-bold ${(pnlStatus.unrealized_pnl ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        ${(pnlStatus.unrealized_pnl ?? 0).toFixed(2)}
                      </div>
                      <div className="text-xs text-gray-600">Unrealized P&L</div>
                    </div>
                    <div className="text-center">
                      <div className={`text-xl font-bold ${(pnlStatus.unrealized_pnl_pct ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {(pnlStatus.unrealized_pnl_pct ?? 0).toFixed(2)}%
                      </div>
                      <div className="text-xs text-gray-600">Trade Return (current)</div>
                    </div>
                    <div className="text-center">
                      <div className="text-xl font-bold text-blue-600">
                        {pnlStatus.shares ? `${(pnlStatus.shares / 1000).toFixed(1)}k shares` : 'No Position'}
                      </div>
                      <div className="text-xs text-gray-600">Shares Held</div>
                    </div>
                  </div>
                  <div className="mt-3 text-xs text-gray-500 text-center">
                    Risk Level: <span className={`font-semibold ${
                      pnlStatus.risk_level === 'high' ? 'text-red-600' : 
                      pnlStatus.risk_level === 'medium' ? 'text-yellow-600' : 'text-green-600'
                    }`}>{pnlStatus.risk_level ?? 'normal'}</span>
                  </div>
                </div>
              )}
              
              {currentStep && (
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">{currentStep}</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Real-time Log */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Real-time Decision Log
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 overflow-y-auto bg-gray-50 p-4 rounded-lg space-y-2">
            {messages.map((msg, index) => {
              // Set styles based on message type
              let messageClass = "text-sm p-3 rounded-md leading-relaxed"
              
              const message = msg.text
              if (message.includes('🤖') && (message.includes('LLM決策') || message.includes('LLM decision'))) {
                // LLM decision messages - special styling, more space to display complete content
                messageClass += " bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 text-blue-900"
              } else if (message.includes('🟢') || message.includes('🚀')) {
                // Buy-related messages
                messageClass += " bg-green-100 border-l-4 border-green-500 text-green-800"
              } else if (message.includes('🔴') || message.includes('📤')) {
                // Sell-related messages  
                messageClass += " bg-red-100 border-l-4 border-red-500 text-red-800"
              } else if (message.includes('🟡')) {
                // Hold-related messages
                messageClass += " bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800"
              } else if (message.includes('💰')) {
                // Performance update messages
                messageClass += " bg-blue-100 border-l-4 border-blue-500 text-blue-800 font-semibold"
              } else if (message.includes('✅') || message.includes('完成')) {
                // Completion messages
                messageClass += " bg-purple-100 border-l-4 border-purple-500 text-purple-800"
              } else {
                // General messages
                messageClass += " bg-white border-l-4 border-gray-300 text-gray-700"
              }
              
              return (
                <div key={index} className={messageClass}>
                  <div className="flex items-start gap-2">
                    <span className="text-xs text-gray-500 min-w-fit" suppressHydrationWarning>
                      [{new Date(msg.ts).toLocaleTimeString('en-US', { hour12: false })}]
                    </span>
                    <span className="flex-1 whitespace-pre-wrap break-words">{message}</span>
                  </div>
                </div>
              )
            })}
            <div ref={messagesEndRef} />
          </div>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Results Display */}
      {result && currentRunId && (
        <BacktestResultsWithAnalysis
          backtestResult={result}
          runId={currentRunId}
        />
      )}
    </div>
  )
}
