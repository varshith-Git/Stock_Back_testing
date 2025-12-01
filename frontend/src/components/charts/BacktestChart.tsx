'use client'

import React, { useEffect, useRef } from 'react'
import { createChart, Time } from 'lightweight-charts'
import { StockData, TradingSignal, LLMDecisionLog } from '@/types'

interface BacktestChartProps {
  stockData: StockData[]
  signals?: TradingSignal[]
  llmDecisions?: LLMDecisionLog[]
  height?: number
  showVolume?: boolean
}

export function BacktestChart({
  stockData,
  signals = [],
  llmDecisions = [],
  height = 500,
  showVolume = true,
}: BacktestChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)

  // Data validation and filtering
  const validStockData = React.useMemo(() => {
    if (!stockData || !Array.isArray(stockData)) {
      return []
    }
    
    return stockData.filter(item => {
      if (!item || typeof item !== 'object') return false
      if (!item.timestamp) return false
      if (typeof item.open !== 'number' || !isFinite(item.open)) return false
      if (typeof item.high !== 'number' || !isFinite(item.high)) return false
      if (typeof item.low !== 'number' || !isFinite(item.low)) return false
      if (typeof item.close !== 'number' || !isFinite(item.close)) return false
      if (typeof item.volume !== 'number' || !isFinite(item.volume) || item.volume < 0) return false
      
      // OHLC logical validation
      if (item.high < item.low || item.high < item.open || item.high < item.close) return false
      if (item.low > item.open || item.low > item.close) return false
      
      return true
    })
  }, [stockData])

  // Time conversion function
  const convertTimestamp = (timestamp: string): number => {
    let date: Date
    
    if (timestamp.includes('T')) {
      date = new Date(timestamp)
    } else if (timestamp.includes('-')) {
      date = new Date(timestamp + 'T00:00:00.000Z')
    } else {
      date = new Date(timestamp)
    }
    
    if (isNaN(date.getTime())) {
      console.warn('Invalid timestamp format:', timestamp)
      return Math.floor(Date.now() / 1000)
    }
    
    return Math.floor(date.getTime() / 1000)
  }

  useEffect(() => {
    if (!chartContainerRef.current || !validStockData.length) {
      return
    }

    // Debug information - check incoming data
    console.log('BacktestChart data debug:', {
      stockDataLength: validStockData.length,
      signalsLength: signals.length,
      llmDecisionsLength: llmDecisions.length,
      firstSignal: signals[0],
      firstLLMDecision: llmDecisions[0],
      stockDataSample: validStockData.slice(0, 2)
    })

    // Clear container
    chartContainerRef.current.innerHTML = ''

    // Create chart container
    const chartContainer = document.createElement('div')
    chartContainer.style.height = `${height}px`
    chartContainerRef.current.appendChild(chartContainer)

    // Create chart
    const chart = createChart(chartContainer, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        backgroundColor: '#ffffff',
        textColor: '#333',
      },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      rightPriceScale: {
        borderColor: '#cccccc',
      },
      timeScale: {
        borderColor: '#cccccc',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    // Candlestick data
    const candlestickData = validStockData.map(stock => ({
      time: convertTimestamp(stock.timestamp) as Time,
      open: stock.open,
      high: stock.high,
      low: stock.low,
      close: stock.close,
    }))

    // Add candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
    })
    candlestickSeries.setData(candlestickData)

    // Add volume
    if (showVolume) {
      const volumeData = validStockData.map(stock => ({
        time: convertTimestamp(stock.timestamp) as Time,
        value: stock.volume,
        color: stock.close >= stock.open ? '#26a69a' : '#ef5350',
      }))

      const volumeSeries = chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
      })
      
      volumeSeries.setData(volumeData)

      chart.priceScale('volume').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      })
    }

    // Merge and process all markers (trading signals + LLM decisions)
    const allMarkers: Array<{
      time: Time
      position: 'belowBar' | 'aboveBar' | 'inBar'
      color: string
      shape: 'arrowUp' | 'arrowDown' | 'circle'
      text: string
      size: number
      id?: string
    }> = []

    // 1. Add trading signal markers (BUY/SELL)
    if (signals.length > 0) {
      console.log('Processing trading signal data:', signals)
      
      const validSignals = signals.filter(signal => {
        const isValid = signal && signal.timestamp && signal.signal_type && 
               typeof signal.price === 'number' && isFinite(signal.price)
        
        if (!isValid) {
          console.warn('Invalid signal data:', signal)
        }
        return isValid
      })

      console.log(`Valid signal count: ${validSignals.length}/${signals.length}`)

      const tradingMarkers = validSignals.map(signal => {
        const position: 'belowBar' | 'aboveBar' = signal.signal_type === 'BUY' ? 'belowBar' : 'aboveBar'
        const shape: 'arrowUp' | 'arrowDown' = signal.signal_type === 'BUY' ? 'arrowUp' : 'arrowDown'
        const marker = {
          time: convertTimestamp(signal.timestamp) as Time,
          position,
          color: signal.signal_type === 'BUY' ? '#26a69a' : '#ef5350',
          shape,
          text: signal.signal_type === 'BUY' ? 'BUY' : 'SELL',
          size: 2, // Adjust arrow size (default is 1, range 0-4)
          id: `signal_${signal.timestamp}_${signal.signal_type}` // Prevent duplicates
        }
        console.log('Create trading marker:', { original: signal, marker })
        return marker
      })
      
      allMarkers.push(...tradingMarkers)
    }

    // 2. Add LLM pure thinking decision markers (excluding actual trading decisions)
    if (llmDecisions.length > 0) {
      console.log('Processing LLM decision data:', llmDecisions)
      
      const validDecisions = llmDecisions.filter(decision => {
        // Check basic data structure: requires timestamp and reasoning
        const hasBasicData = decision && decision.timestamp && decision.reasoning
        
        // LLM decisions should be action: "THINK", not actual trading signals
        const isThinkingDecision = decision.action === 'THINK'
        
        const isValid = hasBasicData && isThinkingDecision
        if (!isValid) {
          console.warn('Filtered out LLM decision:', decision, { 
            hasBasicData, 
            isThinkingDecision, 
            actualAction: decision.action 
          })
        }
        return isValid
      })

      console.log(`Valid LLM decisions: ${validDecisions.length}/${llmDecisions.length}`)

      const thinkingMarkers = validDecisions.map(decision => {
        const confidence = decision.confidence || decision.decision?.confidence || 0.5
        const alpha = Math.max(0.6, confidence)
        
        // Use timestamp field (new format) or date field (backward compatibility)
        const timeValue = decision.timestamp || decision.date || ''
        
        const marker = {
          time: convertTimestamp(timeValue) as Time,
          position: 'aboveBar' as const,  // Above the candlestick
          color: `rgba(255, 193, 7, ${alpha})`, // Yellow, adjust transparency based on confidence
          shape: 'arrowDown' as const,   // Downward arrow
          text: 'AI',  // Concise AI identifier
          size: 1.0,   // Moderate size
          id: `llm_${timeValue}_thinking`
        }
        console.log('Create LLM marker:', { original: decision, marker })
        return marker
      })
      
      allMarkers.push(...thinkingMarkers)
    }

    // 3. Set the merged markers
    if (allMarkers.length > 0) {
      try {
        // Sort markers by time
        allMarkers.sort((a, b) => (a.time as number) - (b.time as number))
        candlestickSeries.setMarkers(allMarkers)
        console.log(`✅ Successfully set ${allMarkers.length} chart markers:`, allMarkers)
      } catch (error) {
        console.error('❌ Error setting chart markers:', error)
      }
    } else {
      console.log('⚠️ No marker data to set')
      
      // If no real data, create some test markers
      if (validStockData.length > 10) {
        const testMarkers = [
          {
            time: convertTimestamp(validStockData[5].timestamp) as Time,
            position: 'belowBar' as const,
            color: '#26a69a',
            shape: 'arrowUp' as const,
            text: 'B',
            size: 2,
          },
          {
            time: convertTimestamp(validStockData[10].timestamp) as Time,
            position: 'aboveBar' as const,
            color: '#ef5350',
            shape: 'arrowDown' as const,
            text: 'S',
            size: 2,
          },
          {
            time: convertTimestamp(validStockData[7].timestamp) as Time,
            position: 'inBar' as const,
            color: 'rgba(255, 193, 7, 0.8)',
            shape: 'circle' as const,
            text: '💭',
            size: 1.2,
          }
        ]
        
        try {
          candlestickSeries.setMarkers(testMarkers)
          console.log('🧪 Set test markers to validate chart functionality')
        } catch (error) {
          console.error('❌ Failed to set test markers:', error)
        }
      }
    }

    chart.timeScale().fitContent()

    // Responsive adjustments
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [validStockData, signals, llmDecisions, height, showVolume])

  if (!validStockData.length) {
    return (
      <div className="w-full h-64 flex items-center justify-center bg-gray-50 rounded-lg">
        <p className="text-gray-500">No valid backtest data</p>
      </div>
    )
  }

  return (
    <div className="w-full">
      <div 
        ref={chartContainerRef} 
        className="w-full border rounded-lg"
        style={{ height: `${height}px` }}
      />
      
      {/* Legend */}
      <div className="flex flex-wrap justify-center mt-4 space-x-4 text-sm">
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1">
            <div className="w-2 h-4 bg-green-600"></div>
            <div className="w-2 h-4 bg-red-500"></div>
          </div>
          <span>Candlestick</span>
        </div>
        
        {showVolume && (
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-gray-400 rounded"></div>
            <span>Volume</span>
          </div>
        )}
        
        {signals.length > 0 && (
          <>
            <div className="flex items-center space-x-2">
              <span className="text-green-600 text-lg font-bold">▲</span>
              <span className="font-medium">Buy Signal</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-red-500 text-lg font-bold">▼</span>
              <span className="font-medium">Sell Signal</span>
            </div>
          </>
        )}
        
        {llmDecisions.length > 0 && (
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <span>AI Decision Points</span>
          </div>
        )}
      </div>
      
      {/* Statistics */}
      {(signals.length > 0 || llmDecisions.length > 0) && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {signals.length > 0 && (
              <>
                <div>
                  <span className="text-gray-600">Trading signals:</span>
                  <span className="ml-2 font-medium">{signals.length}</span>
                </div>
                <div>
                  <span className="text-gray-600">Buys:</span>
                  <span className="ml-2 font-medium text-green-600 text-lg">
                    ▲ {signals.filter(s => s.signal_type === 'BUY').length}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Sells:</span>
                  <span className="ml-2 font-medium text-red-500 text-lg">
                    ▼ {signals.filter(s => s.signal_type === 'SELL').length}
                  </span>
                </div>
              </>
            )}
            {llmDecisions.length > 0 && (
              <div>
                <span className="text-gray-600">AI decisions:</span>
                <span className="ml-2 font-medium">{llmDecisions.length}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
