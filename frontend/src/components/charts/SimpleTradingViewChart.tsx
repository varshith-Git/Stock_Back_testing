'use client'

import React, { useEffect, useRef } from 'react'
import { createChart, Time, IChartApi } from 'lightweight-charts'
import { StockData, TradingSignal, LLMDecisionLog } from '@/types'

interface SimpleTradingViewChartProps {
  /** Stock price data */
  stockData: StockData[]
  /** Trading signal data */
  signals?: TradingSignal[]
  /** LLM decision logs */
  llmDecisions?: LLMDecisionLog[]
  /** Chart height */
  height?: number
  /** Whether to display volume */
  showVolume?: boolean
  /** Whether to display trading signals */
  showSignals?: boolean
  /** Whether to display moving averages */
  showMA?: boolean
  /** Moving average periods */
  maPeriods?: number[]
  /** Whether to display RSI */
  showRSI?: boolean
  /** Whether to display Bollinger Bands */
  showBB?: boolean
  /** Whether to display MACD */
  showMACD?: boolean
}

/**
 * Simplified TradingView Lightweight Charts component
 * Focuses on candlestick charts to avoid complex type issues
 */
export function SimpleTradingViewChart({
  stockData,
  signals = [],
  llmDecisions: _llmDecisions = [],
  height = 400,
  showVolume = true,
  showSignals = false,
  showMA = false,
  maPeriods = [10, 20],
  showRSI = false,
  showBB = false,
  showMACD = false,
}: SimpleTradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)

  // Add data validation at component top
  const validStockData = React.useMemo(() => {
    if (!stockData || !Array.isArray(stockData)) {
      console.warn('Invalid stock data:', stockData)
      return []
    }
    
    return stockData.filter(item => {
      // Basic data validation
      if (!item || typeof item !== 'object') return false
      if (!item.timestamp) return false
      if (typeof item.open !== 'number' || isNaN(item.open) || !isFinite(item.open)) return false
      if (typeof item.high !== 'number' || isNaN(item.high) || !isFinite(item.high)) return false
      if (typeof item.low !== 'number' || isNaN(item.low) || !isFinite(item.low)) return false
      if (typeof item.close !== 'number' || isNaN(item.close) || !isFinite(item.close)) return false
      if (typeof item.volume !== 'number' || isNaN(item.volume) || !isFinite(item.volume) || item.volume < 0) return false
      
      // OHLC logical validation
      if (item.high < item.low || item.high < item.open || item.high < item.close) return false
      if (item.low > item.open || item.low > item.close) return false
      
      return true
    })
  }, [stockData])

  console.log('Original data count:', stockData?.length || 0)
  console.log('Valid data count:', validStockData.length)
  if (validStockData.length > 0) {
    console.log('Sample valid data:', validStockData[0])
  }

  // Unified time conversion function
  const convertTimestamp = (timestamp: string): number => {
    // Handle different time formats
    let date: Date
    
    if (timestamp.includes('T')) {
      // ISO format: "2024-01-15T00:00:00" or "2024-01-15T00:00:00.000Z"
      date = new Date(timestamp)
    } else if (timestamp.includes('-')) {
      // Date format: "2024-01-15"
      date = new Date(timestamp + 'T00:00:00.000Z')
    } else {
      // Other formats, try direct parsing
      date = new Date(timestamp)
    }
    
    // Ensure date is valid
    if (isNaN(date.getTime())) {
      console.warn('Invalid timestamp format:', timestamp)
      return Math.floor(Date.now() / 1000)
    }
    
    // Convert to TradingView required Unix timestamp (seconds)
    const unixTimestamp = Math.floor(date.getTime() / 1000)
    
    // Add debug info (development only)
    if (process.env.NODE_ENV === 'development') {
      console.log(`Time conversion: ${timestamp} -> ${date.toISOString()} -> ${unixTimestamp}`)
    }
    
    return unixTimestamp
  }

  useEffect(() => {
    if (!chartContainerRef.current || !validStockData.length) {
      console.warn('Chart container or data not available')
      return
    }

    console.log('Creating chart with', validStockData.length, 'data points')

    // Calculate required chart height allocation
    let mainChartHeight = height
    let subChartsCount = 0
    if (showRSI) subChartsCount++
    if (showMACD) subChartsCount++
    
    // If there are subcharts, main chart takes 70%, subcharts split remaining space
    if (subChartsCount > 0) {
      mainChartHeight = Math.floor(height * 0.7)
    }

    // Create main chart container
    const mainChartContainer = document.createElement('div')
    mainChartContainer.style.height = `${mainChartHeight}px`
    chartContainerRef.current.innerHTML = ''
    chartContainerRef.current.appendChild(mainChartContainer)

    // Create main chart
    const chart = createChart(mainChartContainer, {
      width: chartContainerRef.current.clientWidth,
      height: mainChartHeight,
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

    // Convert data format - unified time conversion and filter invalid data
    const candlestickData = validStockData.map(stock => ({
      time: convertTimestamp(stock.timestamp) as Time,
      open: stock.open,
      high: stock.high,
      low: stock.low,
      close: stock.close,
    }))

    console.log('Candlestick data sample:', candlestickData.slice(0, 2))

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

    // Add moving averages
    if (showMA && maPeriods.length > 0) {
      const colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
      maPeriods.forEach((period, index) => {
        const maKey = `ma_${period}` as keyof StockData
        const maData = validStockData
          .filter(stock => {
            const value = stock[maKey] as number
            return value !== null && value !== undefined && !isNaN(value) && isFinite(value)
          })
          .map(stock => ({
            time: convertTimestamp(stock.timestamp) as Time,
            value: stock[maKey] as number,
          }))

        console.log(`MA${period} data points:`, maData.length)

        if (maData.length > 0) {
          const maSeries = chart.addLineSeries({
            color: colors[index % colors.length],
            lineWidth: 2,
            title: `MA${period}`,
          })
          maSeries.setData(maData)
        }
      })
    }

    // Add Bollinger Bands (on main chart)
    if (showBB) {
      // Upper band
      const bbUpperData = validStockData
        .filter(stock => {
          const value = stock.bb_upper
          return value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.bb_upper!,
        }))

      console.log('BB Upper data points:', bbUpperData.length)

      if (bbUpperData.length > 0) {
        const bbUpperSeries = chart.addLineSeries({
          color: '#FF5722',
          lineWidth: 1,
          lineStyle: 2, // dashed
          title: 'BB Upper',
        })
        bbUpperSeries.setData(bbUpperData)
      }

      // Lower band
      const bbLowerData = validStockData
        .filter(stock => {
          const value = stock.bb_lower
          return value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.bb_lower!,
        }))

      if (bbLowerData.length > 0) {
        const bbLowerSeries = chart.addLineSeries({
          color: '#FF5722',
          lineWidth: 1,
          lineStyle: 2, // dashed
          title: 'BB Lower',
        })
        bbLowerSeries.setData(bbLowerData)
      }

      // Middle band
      const bbMiddleData = stockData
        .filter(stock => {
          const value = stock.bb_middle
          return stock && value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.bb_middle!,
        }))

      if (bbMiddleData.length > 0) {
        const bbMiddleSeries = chart.addLineSeries({
          color: '#FFC107',
          lineWidth: 1,
          title: 'BB Middle',
        })
        bbMiddleSeries.setData(bbMiddleData)
      }
    }

    // Add volume series (bottom of main chart)
    if (showVolume) {
      const volumeData = validStockData.map(stock => ({
        time: convertTimestamp(stock.timestamp) as Time,
        value: stock.volume,
        color: stock.close >= stock.open ? '#26a69a' : '#ef5350',
      }))

      if (volumeData.length > 0) {
        const volumeSeries = chart.addHistogramSeries({
          color: '#26a69a',
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: 'volume',
        })
        
        volumeSeries.setData(volumeData)

        // Set volume price scale (at bottom)
        chart.priceScale('volume').applyOptions({
          scaleMargins: {
            top: 0.7,
            bottom: 0,
          },
        })
      }
    }

    // Add trading signal markers
    if (showSignals && signals.length > 0) {
      const markers = signals.map(signal => ({
        time: convertTimestamp(signal.timestamp) as Time,
        position: (signal.signal_type === 'BUY' ? 'belowBar' : 'aboveBar') as 'belowBar' | 'aboveBar',
        color: signal.signal_type === 'BUY' ? '#26a69a' : '#ef5350',
        shape: (signal.signal_type === 'BUY' ? 'arrowUp' : 'arrowDown') as 'arrowUp' | 'arrowDown',
        text: signal.signal_type === 'BUY' ? 'BUY' : 'SELL',
        size: 2, // Adjust arrow size (default is 1, range 0-4)
      }))
      
      try {
        candlestickSeries.setMarkers(markers)
      } catch (error) {
        console.warn('Error setting markers:', error)
      }
    }

    // Auto-fit main chart view
    chart.timeScale().fitContent()

    // Store chart instances for cleanup
    const charts: IChartApi[] = [chart]

    // Time axis sync control - prevent infinite loops
    let isSyncing = false

    // Time axis sync - when main chart range changes, sync all subcharts
    const syncTimeRange = (timeRange: unknown, sourceChart?: IChartApi) => {
      if (!timeRange || isSyncing) return
      
      isSyncing = true
      
      charts.forEach((chartInstance) => {
        if (chartInstance && chartInstance !== sourceChart) {
          try {
            chartInstance.timeScale().setVisibleRange(timeRange as never)
          } catch (error) {
            console.warn('Time axis sync failed:', error)
          }
        }
      })
      
      // Delay resetting sync state to avoid immediate triggering
      setTimeout(() => {
        isSyncing = false
      }, 50)
    }

    // Listen for main chart time range changes
    chart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {
      syncTimeRange(timeRange, chart)
    })

    // Create RSI subchart
    let rsiChart: IChartApi | null = null
    if (showRSI) {
      const rsiChartHeight = Math.floor((height - mainChartHeight) / subChartsCount)
      const rsiChartContainer = document.createElement('div')
      rsiChartContainer.style.height = `${rsiChartHeight}px`
      rsiChartContainer.style.marginTop = '10px'
      chartContainerRef.current.appendChild(rsiChartContainer)

      rsiChart = createChart(rsiChartContainer, {
        width: chartContainerRef.current.clientWidth,
        height: rsiChartHeight,
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
          scaleMargins: {
            top: 0.1,
            bottom: 0.1,
          },
        },
        timeScale: {
          borderColor: '#cccccc',
          timeVisible: false,
          secondsVisible: false,
        },
      })

      const rsiData = stockData
        .filter(stock => {
          const value = stock.rsi
          return stock && value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.rsi!,
        }))

      if (rsiData.length > 0) {
        const rsiSeries = rsiChart.addLineSeries({
          color: '#9C27B0',
          lineWidth: 2,
          title: 'RSI',
        })
        rsiSeries.setData(rsiData)

        // Add 30 and 70 reference lines for RSI
        const rsiRef30 = rsiChart.addLineSeries({
          color: '#FF5722',
          lineWidth: 1,
          lineStyle: 2, // dashed
          title: 'RSI 30',
        })
        const rsiRef70 = rsiChart.addLineSeries({
          color: '#FF5722',
          lineWidth: 1,
          lineStyle: 2, // dashed
          title: 'RSI 70',
        })

        const ref30Data = rsiData.map(item => ({ time: item.time, value: 30 }))
        const ref70Data = rsiData.map(item => ({ time: item.time, value: 70 }))
        
        rsiRef30.setData(ref30Data)
        rsiRef70.setData(ref70Data)

        // Set RSI chart price range
        rsiChart.priceScale().applyOptions({
          scaleMargins: {
            top: 0.1,
            bottom: 0.1,
          },
        })
      }

      rsiChart.timeScale().fitContent()
      charts.push(rsiChart)

      // Add reverse time axis sync for RSI chart
      rsiChart.timeScale().subscribeVisibleTimeRangeChange((timeRange: unknown) => {
        syncTimeRange(timeRange, rsiChart!)
      })
    }

    // Create MACD subchart
    let macdChart: IChartApi | null = null
    if (showMACD) {
      const macdChartHeight = Math.floor((height - mainChartHeight) / subChartsCount)
      const macdChartContainer = document.createElement('div')
      macdChartContainer.style.height = `${macdChartHeight}px`
      macdChartContainer.style.marginTop = '10px'
      chartContainerRef.current.appendChild(macdChartContainer)

      macdChart = createChart(macdChartContainer, {
        width: chartContainerRef.current.clientWidth,
        height: macdChartHeight,
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
          scaleMargins: {
            top: 0.1,
            bottom: 0.1,
          },
        },
        timeScale: {
          borderColor: '#cccccc',
          timeVisible: true,
          secondsVisible: false,
        },
      })

      const macdData = stockData
        .filter(stock => {
          const value = stock.macd
          return stock && value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.macd!,
        }))

      const macdSignalData = stockData
        .filter(stock => {
          const value = stock.macd_signal
          return stock && value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.macd_signal!,
        }))

      const macdHistogramData = stockData
        .filter(stock => 
          stock && 
          stock.macd !== null && stock.macd !== undefined && 
          stock.macd_signal !== null && stock.macd_signal !== undefined &&
          !isNaN(stock.macd) && !isNaN(stock.macd_signal) &&
          isFinite(stock.macd) && isFinite(stock.macd_signal)
        )
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.macd! - stock.macd_signal!,
          color: (stock.macd! - stock.macd_signal!) >= 0 ? '#26a69a' : '#ef5350',
        }))

      if (macdData.length > 0) {
        // MACD line
        const macdSeries = macdChart.addLineSeries({
          color: '#2196F3',
          lineWidth: 2,
          title: 'MACD',
        })
        macdSeries.setData(macdData)

        // Signal line
        if (macdSignalData.length > 0) {
          const macdSignalSeries = macdChart.addLineSeries({
            color: '#FF9800',
            lineWidth: 2,
            title: 'Signal',
          })
          macdSignalSeries.setData(macdSignalData)
        }

        // MACD histogram
        if (macdHistogramData.length > 0) {
          const macdHistogramSeries = macdChart.addHistogramSeries({
            color: '#26a69a',
            priceFormat: {
              type: 'price',
              precision: 4,
              minMove: 0.0001,
            },
          })
          macdHistogramSeries.setData(macdHistogramData)
        }

        // Add zero-axis reference line
        const zeroLineData = macdData.map(item => ({ time: item.time, value: 0 }))
        const zeroLineSeries = macdChart.addLineSeries({
          color: '#666666',
          lineWidth: 1,
          lineStyle: 2, // dashed
          title: 'Zero Line',
        })
        zeroLineSeries.setData(zeroLineData)
      }

      macdChart.timeScale().fitContent()
      charts.push(macdChart)

      // Add reverse time axis sync for MACD chart
      macdChart.timeScale().subscribeVisibleTimeRangeChange((timeRange: unknown) => {
        syncTimeRange(timeRange, macdChart!)
      })
    }

    // Responsive adjustments - for all charts
    const handleResize = () => {
      if (chartContainerRef.current) {
        charts.forEach(chartInstance => {
          if (chartInstance) {
            chartInstance.applyOptions({
              width: chartContainerRef.current!.clientWidth,
            })
          }
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      // Clean up all chart instances
      charts.forEach(chartInstance => {
        if (chartInstance) {
          chartInstance.remove()
        }
      })
    }
  }, [stockData, validStockData, signals, showSignals, height, showMA, maPeriods, showRSI, showBB, showMACD, showVolume])

  return (
    <div className="w-full">
      <div 
        ref={chartContainerRef} 
        className="w-full border rounded-lg"
        style={{ height: `${height}px` }}
      />
      
      {/* Legend */}
      <div className="flex flex-wrap justify-center mt-4 space-x-4 text-sm">
        {/* Main chart indicators */}
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
        {showMA && maPeriods.map((period, index) => {
          const colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
          return (
            <div key={`ma-${index}-${period}`} className="flex items-center space-x-2">
              <div 
                className="w-3 h-1"
                style={{ backgroundColor: colors[index % colors.length] }}
              ></div>
              <span>MA{period}</span>
            </div>
          )
        })}
        {showBB && (
          <div className="flex items-center space-x-2">
            <div className="flex space-x-1">
              <div className="w-2 h-1 bg-red-600"></div>
              <div className="w-2 h-1 bg-yellow-500"></div>
            </div>
            <span>Bollinger Bands</span>
          </div>
        )}
        
        {/* Subchart indicators */}
        {showRSI && (
          <div className="flex items-center space-x-2">
            <div className="w-3 h-1 bg-purple-600"></div>
            <span>RSI (Subchart)</span>
          </div>
        )}
        {showMACD && (
          <div className="flex items-center space-x-2">
            <div className="flex space-x-1">
              <div className="w-2 h-1 bg-blue-500"></div>
              <div className="w-2 h-1 bg-orange-500"></div>
              <div className="w-2 h-2 bg-green-600"></div>
            </div>
            <span>MACD (Subchart)</span>
          </div>
        )}
        
        {showSignals && (
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
      </div>
    </div>
  )
}