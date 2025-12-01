#!/usr/bin/env python3
"""
Enhanced Trend Analyzer - Independent LLM Optimization
Standalone enhanced trend analyzer integrating LLM optimization concepts
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add the app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from app.utils.stock_data import StockService


# Independent data structures (backward compatible)
@dataclass
class TrendAnalysisResult:
    """Independent trend analysis result - compatible with original"""

    symbol: str
    analysis_date: str
    timeframes: Dict[str, "TimeframeTrendResult"]
    dominant_trend: str
    complexity_score: float
    trend_consistency: float
    detected_phases: List[Dict[str, Any]]
    overall_assessment: str


@dataclass
class TimeframeTrendResult:
    """Individual timeframe trend analysis result"""

    timeframe: str
    trend_strength: float  # -1 to 1
    trend_direction: str  # 'uptrend', 'downtrend', 'sideways'
    trend_label: str
    confidence: float  # 0.0 to 1.0
    volatility: float
    key_levels: Dict[str, float]


@dataclass
class EnhancedTrendResult:
    """Enhanced trend analysis result with LLM optimization"""

    # Original trend analysis
    original_result: TrendAnalysisResult

    # LLM optimized features
    market_phase: str  # 'uptrend', 'downtrend', 'consolidation'
    trend_consistency: float  # 0-1, multi-timeframe consistency
    reversal_probability: float  # 0-1, probability of trend reversal
    momentum_status: str  # 'bullish', 'bearish', 'neutral'
    risk_level: str  # 'low', 'medium', 'high'

    # Decision context for LLM
    key_observations: List[str]
    llm_recommendations: List[str]

    # Analysis metadata
    analysis_date: str
    price: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "analysis_date": self.analysis_date,
            "price": self.price,
            "market_phase": self.market_phase,
            "trend_consistency": self.trend_consistency,
            "reversal_probability": self.reversal_probability,
            "momentum_status": self.momentum_status,
            "risk_level": self.risk_level,
            "key_observations": self.key_observations,
            "llm_recommendations": self.llm_recommendations,
            "original_analysis": {
                "dominant_trend": self.original_result.dominant_trend,
                "trend_consistency": self.original_result.trend_consistency,
                "complexity_score": self.original_result.complexity_score,
                "timeframes": {
                    timeframe: {
                        "trend_direction": result.trend_direction,
                        "trend_strength": result.trend_strength,
                        "confidence": result.confidence,
                    }
                    for timeframe, result in self.original_result.timeframes.items()
                },
            },
        }


class EnhancedTrendAnalyzer:
    """
    Enhanced Trend Analyzer with LLM optimization features
    Integrates LLM optimization concepts, provides trend analysis more suitable for real-time decision making
    """

    def __init__(self):
        self.stock_service = StockService()

        # Core analysis parameters
        self.trend_threshold = 0.15  # Lower threshold for sensitive trend detection
        self.volatility_threshold_base = 0.025

        # Multi-timeframe windows for comprehensive analysis
        self.timeframe_windows = {
            "short": 5,  # Short-term: 5 days
            "medium_short": 10,  # Medium-short: 10 days
            "medium": 20,  # Medium-term: 20 days
        }

        # LLM optimization parameters
        self.sliding_windows = [
            5,
            10,
            20,
        ]  # Multi-timeframe analysis (removed 40-day lag)
        self.momentum_period = 10
        self.reversal_threshold = 0.3

        # Trend strength levels
        self.trend_strength_levels = {
            "very_strong": 0.7,
            "strong": 0.5,
            "moderate": 0.3,
            "weak": 0.15,
            "very_weak": 0.05,
        }

    def analyze_with_llm_optimization(
        self, symbol: str, target_date: Optional[str] = None
    ) -> EnhancedTrendResult:
        """
        Perform enhanced trend analysis with LLM optimization
        Execute enhanced trend analysis including LLM decision optimization functions

        Args:
            symbol: Stock symbol (e.g., '2330.TW')
            target_date: Target analysis date (format: 'YYYY-MM-DD'), None for latest

        Returns:
            EnhancedTrendResult with both original and LLM-optimized analysis
        """

        # Get market data
        data = self._get_analysis_data(symbol, target_date)
        if not data:
            raise ValueError(f"Insufficient data for {symbol}")

        # Convert to DataFrame for analysis
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Get current price and date
        current_price = df.iloc[-1]["close"]
        current_date = df.iloc[-1]["date"].strftime("%Y-%m-%d")

        # Perform independent multi-timeframe trend analysis
        original_result = self._analyze_market_phases_independent(data, symbol)

        # Perform LLM optimization analysis
        llm_features = self._perform_llm_optimization(df)

        # Generate decision context
        decision_context = self._generate_decision_context(
            original_result, llm_features, current_price
        )

        return EnhancedTrendResult(
            original_result=original_result,
            market_phase=llm_features["market_phase"],
            trend_consistency=llm_features["trend_consistency"],
            reversal_probability=llm_features["reversal_probability"],
            momentum_status=llm_features["momentum_status"],
            risk_level=llm_features["risk_level"],
            key_observations=decision_context["observations"],
            llm_recommendations=decision_context["recommendations"],
            analysis_date=current_date,
            price=current_price,
        )

    def _get_analysis_data(
        self, symbol: str, target_date: Optional[str] = None
    ) -> List[Dict]:
        """Get market data for analysis"""
        if target_date:
            # Get data up to target date with sufficient history
            end_date = datetime.strptime(target_date, "%Y-%m-%d")

            # Get extended data (2 years) to ensure sufficient history for filtering
            all_data = self.stock_service.get_market_data(symbol, "2y")
            if not all_data:
                return []

            # Filter data up to target date
            filtered_data = []
            for record in all_data:
                record_date = datetime.strptime(record["date"], "%Y-%m-%d")
                if record_date <= end_date:
                    filtered_data.append(record)

            # Return last 120 days for analysis (keeping original logic)
            return filtered_data[-120:] if len(filtered_data) >= 60 else filtered_data
        else:
            # Get latest data
            return self.stock_service.get_market_data(symbol, "6mo")

    def _analyze_market_phases_independent(
        self, market_data: List[Dict[str, Any]], symbol: str = "unknown"
    ) -> TrendAnalysisResult:
        """
        Independent multi-timeframe market analysis
        Independent multi-timeframe market analysis, replaces dependency on AdvancedTrendAnalyzer
        """

        if not market_data or len(market_data) < 20:
            return self._create_insufficient_data_result(symbol)

        # Prepare price data
        df = pd.DataFrame(market_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        prices = df["close"].values
        dates = df["date"].tolist()

        # Analyze each timeframe
        timeframe_results = {}

        for name, window in self.timeframe_windows.items():
            if len(prices) >= window:
                result = self._analyze_single_timeframe(prices, dates, window, name)
                timeframe_results[name] = result

        # Determine dominant trend and consistency
        dominant_trend = self._determine_dominant_trend(timeframe_results)
        trend_consistency = self._calculate_overall_consistency(timeframe_results)
        complexity_score = self._calculate_complexity_score(timeframe_results)

        # Detect market phases
        detected_phases = self._detect_market_phases(df)

        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(
            dominant_trend, trend_consistency, complexity_score
        )

        return TrendAnalysisResult(
            symbol=symbol,
            analysis_date=dates[-1].strftime("%Y-%m-%d") if dates else "",
            timeframes=timeframe_results,
            dominant_trend=dominant_trend,
            complexity_score=complexity_score,
            trend_consistency=trend_consistency,
            detected_phases=detected_phases,
            overall_assessment=overall_assessment,
        )

    def _analyze_single_timeframe(
        self, prices: np.ndarray, dates: List, window: int, timeframe_name: str
    ) -> TimeframeTrendResult:
        """Analyze trend for a single timeframe window"""

        if len(prices) < window:
            return self._create_neutral_timeframe_result(timeframe_name)

        # Use recent data for analysis
        recent_prices = prices[-window:]

        # Calculate trend using linear regression
        x = np.arange(len(recent_prices))
        slope, _ = np.polyfit(x, recent_prices, 1)
        correlation = (
            np.corrcoef(x, recent_prices)[0, 1] if len(recent_prices) > 1 else 0
        )

        # Normalize slope for comparison
        normalized_slope = slope / np.mean(recent_prices)

        # Determine trend direction and strength
        trend_strength = normalized_slope
        confidence = abs(correlation)

        if abs(trend_strength) < self.trend_threshold:
            trend_direction = "sideways"
            trend_label = "sideways consolidation"
        elif trend_strength > 0:
            trend_direction = "uptrend"
            if trend_strength > self.trend_strength_levels["strong"]:
                trend_label = "strong rally"
            else:
                trend_label = "moderate rise"
        else:
            trend_direction = "downtrend"
            if abs(trend_strength) > self.trend_strength_levels["strong"]:
                trend_label = "strong decline"
            else:
                trend_label = "moderate decline"

        # Calculate volatility
        volatility = np.std(recent_prices) / np.mean(recent_prices)

        # Identify key levels (simplified)
        key_levels = {
            "support": float(np.min(recent_prices)),
            "resistance": float(np.max(recent_prices)),
            "current": float(recent_prices[-1]),
        }

        return TimeframeTrendResult(
            timeframe=timeframe_name,
            trend_strength=float(trend_strength),
            trend_direction=trend_direction,
            trend_label=trend_label,
            confidence=float(confidence),
            volatility=float(volatility),
            key_levels=key_levels,
        )

    def _create_insufficient_data_result(self, symbol: str) -> TrendAnalysisResult:
        """Create result for insufficient data case"""
        return TrendAnalysisResult(
            symbol=symbol,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            timeframes={},
            dominant_trend="insufficient_data",
            complexity_score=0.0,
            trend_consistency=0.0,
            detected_phases=[],
            overall_assessment="Insufficient data for analysis",
        )

    def _create_neutral_timeframe_result(
        self, timeframe_name: str
    ) -> TimeframeTrendResult:
        """Create neutral result for insufficient timeframe data"""
        return TimeframeTrendResult(
            timeframe=timeframe_name,
            trend_strength=0.0,
            trend_direction="sideways",
            trend_label="insufficient data",
            confidence=0.0,
            volatility=0.0,
            key_levels={"support": 0.0, "resistance": 0.0, "current": 0.0},
        )

    def _determine_dominant_trend(self, timeframe_results: Dict) -> str:
        """Determine the dominant trend across all timeframes"""
        if not timeframe_results:
            return "insufficient_data"

        uptrend_weight = 0
        downtrend_weight = 0

        for result in timeframe_results.values():
            weight = result.confidence  # Use confidence as weight

            if result.trend_direction == "uptrend":
                uptrend_weight += weight
            elif result.trend_direction == "downtrend":
                downtrend_weight += weight

        total_weight = uptrend_weight + downtrend_weight

        if total_weight == 0:
            return "sideways"
        elif uptrend_weight > downtrend_weight * 1.2:  # 20% bias for trend confirmation
            return "uptrend"
        elif downtrend_weight > uptrend_weight * 1.2:
            return "downtrend"
        else:
            return "mixed"

    def _calculate_overall_consistency(self, timeframe_results: Dict) -> float:
        """Calculate trend consistency across timeframes"""
        if not timeframe_results:
            return 0.0

        directions = [result.trend_direction for result in timeframe_results.values()]
        confidences = [result.confidence for result in timeframe_results.values()]

        # Count direction agreement
        direction_counts = {}
        weighted_counts = {}

        for direction, confidence in zip(directions, confidences):
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
            weighted_counts[direction] = weighted_counts.get(direction, 0) + confidence

        if not weighted_counts:
            return 0.0

        # Calculate consistency as the dominance of the most common direction
        max_weighted_count = max(weighted_counts.values())
        total_weighted = sum(weighted_counts.values())

        return max_weighted_count / total_weighted if total_weighted > 0 else 0.0

    def _calculate_complexity_score(self, timeframe_results: Dict) -> float:
        """Calculate market complexity score"""
        if not timeframe_results:
            return 0.0

        # Complexity based on variance in trend directions and strengths
        strengths = [
            abs(result.trend_strength) for result in timeframe_results.values()
        ]
        volatilities = [result.volatility for result in timeframe_results.values()]

        # Higher variance = higher complexity
        strength_variance = np.var(strengths) if strengths else 0
        volatility_mean = np.mean(volatilities) if volatilities else 0

        # Normalize to 0-1 range
        complexity = min((strength_variance * 10 + volatility_mean * 2), 1.0)
        return float(complexity)

    def _detect_market_phases(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect and classify market phases"""
        phases = []

        if len(df) < 10:
            return phases

        # Simple phase detection based on price movements
        prices = df["close"].values
        dates = df["date"].tolist()

        # Look for significant price movements
        for i in range(10, len(prices)):
            window_prices = prices[i - 10 : i]
            current_price = prices[i]

            # Check for phase transitions
            trend_change = (current_price - window_prices[0]) / window_prices[0]

            if abs(trend_change) > 0.05:  # 5% threshold for phase detection
                phase_type = "upward_phase" if trend_change > 0 else "downward_phase"
                phases.append(
                    {
                        "start_date": dates[i - 10].strftime("%Y-%m-%d"),
                        "end_date": dates[i].strftime("%Y-%m-%d"),
                        "phase_type": phase_type,
                        "price_change": trend_change,
                    }
                )

        return phases[-5:] if len(phases) > 5 else phases  # Return last 5 phases

    def _generate_overall_assessment(
        self, dominant_trend: str, trend_consistency: float, complexity_score: float
    ) -> str:
        """Generate overall market assessment"""

        if dominant_trend == "insufficient_data":
            return "Insufficient data for effective analysis"

        consistency_desc = (
            "highly consistent"
            if trend_consistency > 0.8
            else "moderately consistent"
            if trend_consistency > 0.5
            else "clearly divergent"
        )

        complexity_desc = (
            "complex"
            if complexity_score > 0.6
            else "moderate"
            if complexity_score > 0.3
            else "simple"
        )

        trend_desc = {
            "uptrend": "upward trend",
            "downtrend": "downward trend",
            "sideways": "sideways consolidation",
            "mixed": "mixed trend",
        }.get(dominant_trend, "unknown trend")

        return f"Market shows {trend_desc}, multi-timeframes are {consistency_desc}, market structure is {complexity_desc}"

    def _perform_llm_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform LLM optimization analysis
        Execute LLM optimization analysis, avoid fixed timeframe lag issues
        """

        # 1. Multi-timeframe sliding window analysis
        sliding_trends = self._analyze_sliding_windows(df)

        # 2. Market phase detection
        market_phase = self._detect_market_phase(sliding_trends)

        # 3. Trend consistency analysis
        trend_consistency = self._calculate_trend_consistency(sliding_trends)

        # 4. Reversal probability calculation
        reversal_probability = self._calculate_reversal_probability(df, sliding_trends)

        # 5. Momentum analysis
        momentum_status = self._analyze_momentum(df)

        # 6. Risk level assessment
        risk_level = self._assess_risk_level(
            trend_consistency, reversal_probability, momentum_status
        )

        return {
            "market_phase": market_phase,
            "trend_consistency": trend_consistency,
            "reversal_probability": reversal_probability,
            "momentum_status": momentum_status,
            "risk_level": risk_level,
            "sliding_trends": sliding_trends,
        }

    def _analyze_sliding_windows(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze trends using sliding windows to avoid lag"""
        results = {}

        for window in self.sliding_windows:
            if len(df) < window + 5:
                continue

            recent_data = df.tail(window)

            # Calculate trend using linear regression
            x = np.arange(len(recent_data))
            y = recent_data["close"].values

            if len(y) > 1:
                slope, _ = np.polyfit(x, y, 1)
                correlation = np.corrcoef(x, y)[0, 1] if len(y) > 1 else 0

                # Normalize slope by price for comparison
                normalized_slope = slope / recent_data["close"].mean()

                results[f"window_{window}"] = {
                    "slope": normalized_slope,
                    "strength": abs(correlation),
                    "direction": "up" if slope > 0 else "down",
                }

        return results

    def _detect_market_phase(self, sliding_trends: Dict[str, Dict[str, float]]) -> str:
        """Detect current market phase based on multi-timeframe analysis"""
        up_count = 0
        down_count = 0
        total_strength = 0

        for window_data in sliding_trends.values():
            direction = window_data["direction"]
            strength = window_data["strength"]

            if direction == "up":
                up_count += strength
            else:
                down_count += strength

            total_strength += strength

        if total_strength == 0:
            return "consolidation"

        up_ratio = up_count / total_strength
        down_ratio = down_count / total_strength

        # Clear directional bias required
        if up_ratio > 0.65:
            return "uptrend"
        elif down_ratio > 0.65:
            return "downtrend"
        else:
            return "consolidation"

    def _calculate_trend_consistency(
        self, sliding_trends: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate trend consistency across timeframes"""
        if not sliding_trends:
            return 0.0

        directions = [data["direction"] for data in sliding_trends.values()]
        strengths = [data["strength"] for data in sliding_trends.values()]

        # Calculate weighted consistency
        up_strength = sum(s for d, s in zip(directions, strengths) if d == "up")
        down_strength = sum(s for d, s in zip(directions, strengths) if d == "down")
        total_strength = sum(strengths)

        if total_strength == 0:
            return 0.0

        # Consistency is the dominance of the stronger direction
        consistency = max(up_strength, down_strength) / total_strength
        return min(consistency, 1.0)

    def _calculate_reversal_probability(
        self, df: pd.DataFrame, sliding_trends: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate probability of trend reversal"""

        # 1. Price structure analysis
        recent_prices = df.tail(20)["close"]
        price_volatility = recent_prices.std() / recent_prices.mean()

        # 2. Trend divergence across timeframes
        short_term_trend = sliding_trends.get("window_5", {}).get(
            "direction", "neutral"
        )
        long_term_trend = sliding_trends.get("window_20", {}).get(
            "direction", "neutral"
        )

        divergence_score = 0.3 if short_term_trend != long_term_trend else 0.0

        # 3. Trend strength deterioration
        strengths = [data["strength"] for data in sliding_trends.values()]
        avg_strength = np.mean(strengths) if strengths else 0
        strength_factor = max(0, 0.8 - avg_strength)  # Higher reversal if weak trends

        # 4. Volatility factor
        volatility_factor = min(
            price_volatility * 2, 0.4
        )  # High volatility = potential reversal

        # Combine factors
        reversal_prob = divergence_score + strength_factor + volatility_factor
        return min(reversal_prob, 1.0)

    def _analyze_momentum(self, df: pd.DataFrame) -> str:
        """Analyze current momentum status"""
        if len(df) < self.momentum_period + 5:
            return "neutral"

        recent_data = df.tail(self.momentum_period)

        # Calculate price momentum
        price_change = (
            recent_data["close"].iloc[-1] - recent_data["close"].iloc[0]
        ) / recent_data["close"].iloc[0]

        # Calculate volume momentum if available
        volume_trend = 0
        if "volume" in recent_data.columns:
            volume_change = (
                recent_data["volume"].iloc[-5:].mean()
                - recent_data["volume"].iloc[:5].mean()
            ) / recent_data["volume"].iloc[:5].mean()
            volume_trend = volume_change

        # Combine price and volume momentum
        momentum_score = price_change + (volume_trend * 0.3)

        if momentum_score > 0.02:
            return "bullish"
        elif momentum_score < -0.02:
            return "bearish"
        else:
            return "neutral"

    def _assess_risk_level(
        self,
        trend_consistency: float,
        reversal_probability: float,
        momentum_status: str,
    ) -> str:
        """Assess current risk level for trading decisions"""

        # High consistency = lower risk
        consistency_risk = 1 - trend_consistency

        # High reversal probability = higher risk
        reversal_risk = reversal_probability

        # Momentum alignment risk
        momentum_risk = 0.3 if momentum_status == "neutral" else 0.0

        total_risk = (consistency_risk + reversal_risk + momentum_risk) / 3

        if total_risk < 0.3:
            return "low"
        elif total_risk < 0.6:
            return "medium"
        else:
            return "high"

    def _generate_decision_context(
        self,
        original_result: TrendAnalysisResult,
        llm_features: Dict[str, Any],
        current_price: float,
    ) -> Dict[str, List[str]]:
        """Generate decision context for LLM"""

        observations = []
        recommendations = []

        # Trend consistency observations
        if llm_features["trend_consistency"] > 0.8:
            observations.append("Multi-timeframe trends highly consistent, clear direction")
        elif llm_features["trend_consistency"] < 0.4:
            observations.append("Divergent trends across timeframes, unclear market direction")

        # Reversal signal observations
        if llm_features["reversal_probability"] > 0.5:
            signal_strength = (
                "strong" if llm_features["reversal_probability"] > 0.7 else "significant"
            )
            observations.append(f"Detected {signal_strength} reversal signals, market may be changing")

        # Momentum observations
        if llm_features["momentum_status"] != "neutral":
            momentum_desc = (
                "accelerating" if llm_features["trend_consistency"] > 0.7 else "weakening"
            )
            observations.append(
                f"Momentum showing {llm_features['momentum_status']} tendency, trend {momentum_desc}"
            )

        # Generate recommendations
        if llm_features["trend_consistency"] > 0.7:
            recommendations.append("Clear trend direction, consider trend-following operations")

        if llm_features["momentum_status"] == "bullish":
            recommendations.append("Good upward momentum, watch for long opportunities")
        elif llm_features["momentum_status"] == "bearish":
            recommendations.append("Clear downward momentum, exercise caution against risks")

        if llm_features["reversal_probability"] > 0.6:
            recommendations.append("Strong reversal signals, suggest waiting for trend confirmation before operating")

        if llm_features["risk_level"] == "high":
            recommendations.append("Current high risk level, suggest reducing positions or pausing trading")

        return {"observations": observations, "recommendations": recommendations}


def main():
    """Test Enhanced Trend Analyzer with integration example"""
    analyzer = EnhancedTrendAnalyzer()

    print("🚀 Enhanced Trend Analyzer - LLM Integration Test")
    print("=" * 60)

    # Test with latest data
    result = analyzer.analyze_with_llm_optimization("2330.TW")

    print(f"📅 Analysis Date: {result.analysis_date}")
    print(f"💰 Current Price: ${result.price:.0f}")
    print(f"📊 Market Phase: {result.market_phase}")
    print(f"🎯 Trend Consistency: {result.trend_consistency:.2f}")
    print(f"🔄 Reversal Probability: {result.reversal_probability:.2f}")
    print(f"📈 Momentum Status: {result.momentum_status}")
    print(f"⚠️  Risk Level: {result.risk_level}")

    print(f"\n🔍 Key Observations:")
    for obs in result.key_observations:
        print(f"  • {obs}")

    print(f"\n💡 LLM Recommendations:")
    for rec in result.llm_recommendations:
        print(f"  • {rec}")

    print(f"\n📊 Original Analysis Comparison:")
    print(f"  - Dominant Trend: {result.original_result.dominant_trend}")
    print(f"  - Original Consistency: {result.original_result.trend_consistency:.3f}")
    print(f"  - Complexity Score: {result.original_result.complexity_score:.3f}")

    print(f"\n✅ System Integration Complete!")
    print(f"📈 Enhanced Trend Analyzer can directly replace original AdvancedTrendAnalyzer")
    print(f"🤖 Provides LLM decision optimization functions while maintaining backward compatibility")


if __name__ == "__main__":
    main()
