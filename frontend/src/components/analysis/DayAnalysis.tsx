'use client'

import React, { useState, useCallback, useEffect } from 'react'
import { Calendar, TrendingUp, FileText, BarChart3, AlertTriangle, MessageSquare, Lightbulb } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Textarea } from '@/components/ui/textarea'
import { DayAnalysisResponse, TrendAnalysis, RetrospectiveAnalysis, TechnicalEvent } from '@/types'

interface DayAnalysisProps {
  runId: string
  onDateSelect: (date: string) => void
}

interface DailyFeedbackSectionProps {
  date: string
}

interface DailyImprovementResponse {
  analysis: string
  suggestions: string[]
}

interface DayAnalysisState {
  selectedDate: string | null
  analysis: DayAnalysisResponse | null
  availableDates: string[]
  isLoading: boolean
  isLoadingDates: boolean
  error: string | null
}

/**
 * Clean markdown formatting from text
 */
const cleanMarkdown = (text: string): string => {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markdown **text**
    .replace(/\*(.*?)\*/g, '$1')     // Remove italic markdown *text*
    .replace(/`(.*?)`/g, '$1')       // Remove code markdown `text`
    .trim()
}

/**
 * Day Analysis Component - Provides detailed analysis of specific trading days
 * Allows users to select dates and view comprehensive analysis including LLM decisions
 */
function DayAnalysis({ runId, onDateSelect }: DayAnalysisProps) {
  const [state, setState] = useState<DayAnalysisState>({
    selectedDate: null,
    analysis: null,
    availableDates: [],
    isLoading: false,
    isLoadingDates: true,
    error: null
  })

  // Fetch available dates when component mounts or runId changes
  useEffect(() => {
    const fetchAvailableDates = async () => {
      setState(prev => ({ ...prev, isLoadingDates: true, error: null }))
      
      try {
        const response = await fetch(`/api/v1/backtest/available-dates/${runId}`)
        if (!response.ok) {
          throw new Error(`Failed to fetch available dates: ${response.statusText}`)
        }
        
        const data = await response.json()
        setState(prev => ({ 
          ...prev, 
          availableDates: data.dates || [],
          isLoadingDates: false 
        }))
      } catch (error) {
        setState(prev => ({ 
          ...prev, 
          error: error instanceof Error ? error.message : 'Failed to load available dates',
          isLoadingDates: false,
          availableDates: []
        }))
      }
    }

    if (runId) {
      fetchAvailableDates()
    }
  }, [runId])

  // Handle date selection and fetch analysis
  const handleDateSelect = useCallback(async (date: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null, selectedDate: date }))
    onDateSelect(date)

    try {
      const response = await fetch(`/api/v1/backtest/analysis/day/${runId}?date=${date}&include_retrospective=false`)
      if (!response.ok) {
        throw new Error(`Failed to fetch analysis: ${response.statusText}`)
      }
      
      const analysisData: DayAnalysisResponse = await response.json()
      setState(prev => ({ 
        ...prev, 
        analysis: analysisData, 
        isLoading: false 
      }))
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        isLoading: false 
      }))
    }
  }, [runId, onDateSelect])

  // Format date for display
  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString('zh-TW', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      weekday: 'long'
    })
  }

  // Get trend color based on trend type
  const getTrendColor = (trend: string): string => {
    switch (trend) {
      case 'BULLISH': return 'text-green-600'
      case 'BEARISH': return 'text-red-600'
      case 'SIDEWAYS': return 'text-gray-600'
      default: return 'text-gray-600'
    }
  }

  // Get badge variant based on impact
  const getImpactVariant = (impact: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (impact) {
      case 'POSITIVE': return 'default'
      case 'NEGATIVE': return 'destructive'
      case 'NEUTRAL': return 'secondary'
      default: return 'outline'
    }
  }

  // Translate technical event types to English
  const translateEventType = (eventType: string): string => {
    const translations: Record<string, string> = {
      // Bollinger Bands
      'BB_UPPER_TOUCH': 'Touch Upper Band',
      'BB_LOWER_TOUCH': 'Touch Lower Band',
      'BB_SQUEEZE': 'Bollinger Squeeze',
      'BB_EXPANSION': 'Bollinger Expansion',
      
      // Moving Averages
      'MA_GOLDEN_CROSS': 'Golden Cross',
      'MA_DEATH_CROSS': 'Death Cross',
      'MA_SUPPORT': 'MA Support',
      'MA_RESISTANCE': 'MA Resistance',
      
      // MACD
      'MACD_GOLDEN_CROSS': 'MACD Golden Cross',
      'MACD_DEATH_CROSS': 'MACD Death Cross',
      'MACD_DIVERGENCE': 'MACD Divergence',
      
      // RSI
      'RSI_OVERSOLD': 'RSI Oversold',
      'RSI_OVERBOUGHT': 'RSI Overbought',
      'RSI_DIVERGENCE': 'RSI Divergence',
      
      // Volume
      'VOLUME_SPIKE': 'Volume Spike',
      'VOLUME_DRY_UP': 'Volume Dry Up',
      'VOLUME_BREAKOUT': 'Volume Breakout',
      'HIGH_VOLUME': 'High Volume',
      'VOLUME_EXPLOSION': 'Explosive Volume',
      
      // Trend
      'TREND_TURN_BULLISH': 'Trend Turns Bullish',
      'TREND_TURN_BEARISH': 'Trend Turns Bearish',
      'TREND_ACCELERATION': 'Trend Acceleration',
      'TREND_WEAKNESS': 'Trend Weakness',
      
      // Momentum
      'MOMENTUM_SHIFT': 'Momentum Shift',
      'MOMENTUM_DIVERGENCE': 'Momentum Divergence',
      
      // Others
      'GAP_UP': 'Gap Up',
      'GAP_DOWN': 'Gap Down',
      'HIGH_VOLATILITY': 'High Volatility',
      'LOW_VOLATILITY': 'Low Volatility',
      
      // Unknown or default
      'unknown': 'Technical Event',
      'UNKNOWN': 'Technical Event',
      'OTHER': 'Other Technical Signal'
    }
    
    return translations[eventType] || `Technical Event: ${eventType}`
  }

  // Translate severity levels to English
  const translateSeverity = (severity: string): string => {
    const translations: Record<string, string> = {
      'high': 'High',
      'medium': 'Medium',
      'low': 'Low',
      'very_high': 'Very High',
      'very_low': 'Very Low'
    }
    
    return translations[severity] || severity
  }

  return (
    <div className="space-y-6">
      {/* Date Selection Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            📅 Trading Log Explorer
          </CardTitle>
          <CardDescription>
            Select a trading day to review the decision process and discuss insights.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {state.isLoadingDates ? (
            <div className="text-center py-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto"></div>
              <p className="mt-2 text-sm text-gray-600">Loading available dates...</p>
            </div>
          ) : state.availableDates.length === 0 ? (
            <div className="text-center py-4 text-gray-500">
              No available analysis dates found
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
              {state.availableDates.map((date: string) => (
                <Button
                  key={date}
                  variant={state.selectedDate === date ? "default" : "outline"}
                  size="sm"
                  onClick={() => handleDateSelect(date)}
                  disabled={state.isLoading}
                  className="text-xs"
                >
                  {new Date(date).toLocaleDateString('en-US', { 
                    month: 'short', 
                    day: 'numeric' 
                  })}
                </Button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error State */}
      {state.error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{state.error}</AlertDescription>
        </Alert>
      )}

      {/* Loading State */}
      {state.isLoading && (
        <div className="space-y-4">
          <div className="h-32 w-full bg-gray-200 animate-pulse rounded-lg" />
          <div className="h-48 w-full bg-gray-200 animate-pulse rounded-lg" />
          <div className="h-64 w-full bg-gray-200 animate-pulse rounded-lg" />
        </div>
      )}

      {/* Analysis Results */}
      {state.analysis && !state.isLoading && (
        <div className="space-y-6">
          {/* Technical Events */}
          {state.analysis.historical_data.technical_events.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Technical Events
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {/* Original technical events */}
                  {state.analysis.historical_data.technical_events.map((event, index) => (
                    <div key={`original-${index}`} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="font-medium">{translateEventType(event.event_type)}</div>
                        <div className="text-sm text-gray-600">{event.description}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className={event.severity === 'high' ? 'text-red-600' : 'text-yellow-600'}>
                          {translateSeverity(event.severity)}
                        </Badge>
                      </div>
                    </div>
                  ))}
                  
                  {/* No events message */}
                  {state.analysis.historical_data.technical_events.length === 0 && (
                    <div className="text-center py-8 text-gray-500">
                      No significant technical events today
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* LLM Analysis */}
          {state.analysis.historical_data.llm_decision && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded-full bg-purple-500 flex items-center justify-center">
                    <span className="text-white text-xs font-bold">AI</span>
                  </div>
                  🧠 AI Daily Reasoning
                </CardTitle>
                <CardDescription>
                  Let’s see how AI reasoned that day — does it make sense to you?
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">Decision Type</div>
                      <div className="text-lg font-bold">
                        {(() => {
                          const decisionType = state.analysis.historical_data.llm_decision.decision_type
                          const strategyState = state.analysis.historical_data.strategy_state
                          
                          // 檢查是否有持倉信息 - 根據實際數據結構
                          const shares = Number((strategyState as { shares?: number })?.shares ?? 0)
                          const position = (strategyState as { position?: 'long' | 'short' | 'flat' })?.position
                          const hasPosition = shares > 0 || position === 'long' || position === 'short'
                          
                          if (decisionType === 'BUY') {
                            return '📈 Buy'
                          } else if (decisionType === 'SELL') {
                            return '📉 Sell'
                          } else if (decisionType === 'HOLD') {
                            if (hasPosition) {
                              return '⏸️ Hold'
                            } else {
                              return '💤 No Position (Watching)'
                            }
                          } else {
                            return '⏸️ Watching'
                          }
                        })()}
                      </div>
                    </div>
                  </div>
                  
                  {state.analysis.historical_data.llm_decision.reasoning && (
                    <div>
                      <div className="text-sm text-gray-600 mb-2">Analytical Reasoning</div>
                      <div className="text-sm bg-gray-50 p-3 rounded-lg whitespace-pre-line">
                        {state.analysis.historical_data.llm_decision.reasoning}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Daily Decision Improvement */}
          {state.selectedDate && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageSquare className="h-5 w-5" />
                  💬 Strategy Discussion Room
                </CardTitle>
                <CardDescription>
                  Discuss decisions with the AI assistant, share your insights, and get strategy optimization suggestions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <DailyFeedbackSection 
                  date={state.selectedDate}
                />
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* No Date Selected State */}
      {!state.selectedDate && !state.isLoading && (
        <Card>
          <CardContent className="text-center py-12">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Calendar className="h-8 w-8 text-blue-500" />
            </div>
            <h3 className="text-lg font-medium text-gray-700 mb-2">🚀 Ready to start our strategy discussion!</h3>
            <p className="text-gray-500 max-w-md mx-auto">
              Select a trading day above; I’ll summarize what happened, then we can discuss strategy improvements 💭
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

const DailyFeedbackSection: React.FC<DailyFeedbackSectionProps> = ({ date }) => {
  const [feedback, setFeedback] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<DailyImprovementResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmitFeedback = async () => {
    if (!feedback.trim()) {
      setError('Please share your thoughts!')
      return
    }

    setIsAnalyzing(true)
    setError(null)

    try {
      const response = await fetch('http://localhost:8000/api/v1/daily/daily-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          feedback: feedback.trim(),
          date: date
        })
      })

      if (!response.ok) {
        throw new Error(`Problem occurred during discussion: ${response.status}`)
      }

      const data: DailyImprovementResponse = await response.json()
      setResult(data)
      
    } catch (err) {
      console.error('Daily feedback error:', err)
      setError(err instanceof Error ? err.message : 'An error occurred during the discussion')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    setFeedback('')
    setResult(null)
    setError(null)
  }

  return (
    <div className="space-y-4">
      {/* Interactive Header */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center">
            <span className="text-white text-sm font-bold">AI</span>
          </div>
          <h4 className="font-medium text-blue-900">Strategy Discussion Assistant</h4>
        </div>
        <p className="text-sm text-blue-700">
          I’d like to hear your view on the decision on <span className="font-semibold">{date}</span>. Let’s explore optimization directions 🤔
        </p>
      </div>

      {/* Input Section */}
      <div className="space-y-3">
        <div>
          <label className="text-sm font-medium text-gray-700 mb-2 block flex items-center gap-2">
            💬 Your thoughts...
          </label>
          <Textarea
            placeholder="Hi! Share your thoughts... e.g., ‘I think we shouldn’t have sold that day because…’ or ‘I agree with the decision, but…’"
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            rows={4}
            className="w-full border-2 border-gray-200 focus:border-blue-400 transition-colors"
            disabled={isAnalyzing}
          />
        </div>
        
        <div className="flex gap-2">
          <Button 
            onClick={handleSubmitFeedback}
            disabled={isAnalyzing || !feedback.trim()}
            className="flex items-center gap-2 bg-blue-500 hover:bg-blue-600"
          >
            <MessageSquare className="h-4 w-4" />
            {isAnalyzing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Thinking...
              </>
            ) : (
              'Start Discussion 💭'
            )}
          </Button>
          
          {result && (
            <Button 
              variant="outline" 
              onClick={handleReset}
              disabled={isAnalyzing}
              className="border-blue-300 text-blue-600 hover:bg-blue-50"
            >
              🔄 Discuss Again
            </Button>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <Alert className="bg-red-50 border-red-200">
          <AlertTriangle className="h-4 w-4 text-red-500" />
          <AlertDescription className="text-red-700">{error}</AlertDescription>
        </Alert>
      )}

      {/* Results Display - Chat Style */}
      {result && (
        <div className="space-y-4 border-t pt-4">
          {/* AI Response */}
          <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 flex items-center justify-center flex-shrink-0">
                <span className="text-white text-sm font-bold">AI</span>
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className="font-medium text-gray-900">Strategy Analysis Assistant</span>
                  <Badge variant="secondary" className="text-xs">Just now</Badge>
                </div>
                
                {/* Analysis as conversation */}
                <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-700 mb-3">
                  <div className="flex items-start gap-2">
                    <BarChart3 className="h-4 w-4 mt-0.5 text-blue-500 flex-shrink-0" />
                    <div>
                      <div className="font-medium text-gray-900 mb-1">My view:</div>
                      <div className="whitespace-pre-wrap">{cleanMarkdown(result.analysis)}</div>
                    </div>
                  </div>
                </div>

                {/* Suggestions as strategy file modifications */}
                {result.suggestions.length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm font-medium text-gray-900">
                      <FileText className="h-4 w-4 text-green-500" />
                      📝 Strategy file change suggestions (traditional_strategy.md):
                    </div>
                    {result.suggestions.map((suggestion, index) => {
                      // 分離標題和詳細內容
                      const lines = suggestion.split('\n')
                      const title = lines[0] || suggestion
                      const details = lines.slice(1).join('\n').trim()
                      
                      return (
                        <div key={index} className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 p-4 rounded-lg text-sm">
                          <div className="flex items-start gap-3">
                            <span className="text-green-600 font-bold text-xs bg-green-100 px-2 py-1 rounded-full flex-shrink-0">
                              Change {index + 1}
                            </span>
                            <div className="text-gray-700 flex-1">
                              <div className="flex items-center gap-2 mb-3">
                                <div className="font-mono text-xs text-green-800 bg-green-100 px-2 py-1 rounded">
                                  📄 traditional_strategy.md
                                </div>
                                <div className="text-xs text-green-600 font-medium">Strategy file change</div>
                              </div>
                              
                              {/* 標題 */}
                              <div className="font-semibold text-gray-800 mb-2">
                                {cleanMarkdown(title)}
                              </div>
                              
                              {/* 詳細內容 - 如果有的話 */}
                              {details && (
                                <div className="text-gray-600 text-xs leading-relaxed bg-white bg-opacity-50 p-3 rounded border-l-2 border-green-300">
                                  <div className="whitespace-pre-wrap">{cleanMarkdown(details)}</div>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}

                {/* Encourage further discussion */}
                <div className="mt-4 pt-3 border-t border-gray-100">
                  <p className="text-xs text-gray-500 italic">
                    These suggestions can be applied directly to the strategy file. Have other optimization ideas? Click &quot;Discuss Again&quot; to keep improving our trading strategy.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
export default DayAnalysis
