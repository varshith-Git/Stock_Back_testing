
"""
Trend Analysis Benchmark Database
Used for validating and improving the accuracy of trend identification algorithms
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class TrendPhase:
    """Trend phase definition"""

    start_date: str
    end_date: str
    trend_type: str  # 'uptrend', 'downtrend', 'sideways', 'sideways_bullish', 'sideways_bearish'
    strength: str  # 'strong', 'moderate', 'weak'
    confidence: float  # 0.0 - 1.0 manually annotated confidence level
    description: str  # Descriptive explanation
    key_events: List[str] = None  # Key events or turning points


@dataclass
class BenchmarkCase:
    """Benchmark test case"""

    symbol: str
    period: str
    total_period: Tuple[str, str]  # (start_date, end_date)
    phases: List[TrendPhase]
    overall_trend: str
    complexity_level: str  # 'simple', 'moderate', 'complex'
    notes: str


class TrendBenchmarkDatabase:
    """Trend analysis benchmark database"""

    def __init__(self):
        self.benchmark_cases = self._initialize_benchmark_cases()

    def _initialize_benchmark_cases(self) -> Dict[str, BenchmarkCase]:
        """Initialize benchmark test cases"""
        cases = {}

        # TSMC 2330.TW case - complex multi-phase trend
        tsmc_2024_2025 = BenchmarkCase(
            symbol="2330.TW",
            period="Recent year (Oct 2024 - Jul 2025)",
            total_period=("2024-10-01", "2025-07-30"),
            phases=[
                TrendPhase(
                    start_date="2024-10-20",
                    end_date="2024-11-14",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.8,
                    description="Range-bound consolidation, price fluctuating within specific range, no clear directional bias",
                    key_events=["Sideways consolidation", "Awaiting breakout direction"],
                ),
                TrendPhase(
                    start_date="2025-02-20",
                    end_date="2025-04-20",
                    trend_type="downtrend",
                    strength="moderate",
                    confidence=0.85,
                    description="Clear downtrend phase with successively lower lows",
                    key_events=["Support break", "Continued decline"],
                ),
                TrendPhase(
                    start_date="2025-04-23",
                    end_date="2025-07-23",
                    trend_type="uptrend",
                    strength="moderate",
                    confidence=0.9,
                    description="Uptrend phase with progressively higher highs and lows",
                    key_events=["Resistance breakout", "Continued rally"],
                ),
            ],
            overall_trend="complex_multi_phase",
            complexity_level="complex",
            notes="Typical multi-phase trend transition case, includes complete cycle of consolidation -> decline -> rise",
        )
        cases["TSMC_2024_2025"] = tsmc_2024_2025

        # TSLA case - downtrend + two consolidation phases
        tsla_2025 = BenchmarkCase(
            symbol="TSLA",
            period="Recent six months (Jan - Jul 2025)",
            total_period=("2025-01-22", "2025-07-28"),
            phases=[
                TrendPhase(
                    start_date="2025-01-22",
                    end_date="2025-03-11",
                    trend_type="downtrend",
                    strength="strong",
                    confidence=0.9,
                    description="Clear downtrend phase",
                    key_events=["Market correction", "Profit taking"],
                ),
                TrendPhase(
                    start_date="2025-03-11",
                    end_date="2025-05-05",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.85,
                    description="First consolidation phase, price fluctuating within range",
                    key_events=["Market stabilization", "Sideways consolidation"],
                ),
                TrendPhase(
                    start_date="2025-05-28",
                    end_date="2025-07-28",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.8,
                    description="Second consolidation phase, continuing sideways movement",
                    key_events=["Continued consolidation", "Awaiting breakout"],
                ),
            ],
            overall_trend="complex_multi_phase",
            complexity_level="complex",
            notes="Complex case containing downtrend and two consolidation phases, tests range identification capability",
        )
        cases["TSLA_2025"] = tsla_2025

        # AAPL case - two-phase pattern of downtrend transitioning to consolidation
        aapl_2025 = BenchmarkCase(
            symbol="AAPL",
            period="Feb - Jul 2025",
            total_period=("2025-02-01", "2025-07-28"),
            phases=[
                TrendPhase(
                    start_date="2025-02-25",
                    end_date="2025-04-08",
                    trend_type="downtrend",
                    strength="strong",
                    confidence=0.9,
                    description="Clear downtrend phase, significant price decline from highs",
                    key_events=["Market correction", "Technical adjustment", "Decline from $246.72 to $172.19"],
                ),
                TrendPhase(
                    start_date="2025-04-09",
                    end_date="2025-07-28",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.85,
                    description="Consolidation phase, price fluctuating back and forth within range",
                    key_events=["Sideways consolidation", "12.7% range fluctuation", "Multiple support/resistance tests"],
                ),
            ],
            overall_trend="trend_to_sideways_transition",
            complexity_level="medium",
            notes="Typical trend-to-consolidation transition case, suitable for testing algorithm's ability to identify trend change points and detect range-bound movements",
        )
        cases["AAPL_2025"] = aapl_2025

        return cases

    def get_benchmark_case(self, case_id: str) -> BenchmarkCase:
        """Get specific benchmark case"""
        return self.benchmark_cases.get(case_id)

    def add_benchmark_case(self, case_id: str, benchmark_case: BenchmarkCase):
        """Add new benchmark case"""
        self.benchmark_cases[case_id] = benchmark_case

    def get_all_cases(self) -> Dict[str, BenchmarkCase]:
        """Get all benchmark cases"""
        return self.benchmark_cases

    def evaluate_algorithm_performance(
        self, case_id: str, algorithm_result: Dict
    ) -> Dict[str, Any]:
        """Evaluate algorithm performance on specific case"""
        benchmark = self.get_benchmark_case(case_id)
        if not benchmark:
            return {"error": f"Benchmark case not found: {case_id}"}

        evaluation = {
            "case_id": case_id,
            "benchmark_phases": len(benchmark.phases),
            "detected_phases": algorithm_result.get("detected_phases", 0),
            "overall_accuracy": 0.0,
            "phase_accuracy": [],
            "missed_transitions": [],
            "false_positives": [],
        }

        # Can implement specific evaluation logic here
        # Compare algorithm-detected trend phases with manually annotated benchmarks

        return evaluation


# Global instance
trend_benchmark_db = TrendBenchmarkDatabase()


def get_benchmark_database() -> TrendBenchmarkDatabase:
    """Get benchmark database instance"""
    return trend_benchmark_db


def create_test_case_from_data(
    symbol: str, market_data: List[Dict], manual_phases: List[Dict]
) -> BenchmarkCase:
    """Create test case from market data and manual annotations"""
    phases = []
    for phase_data in manual_phases:
        phase = TrendPhase(
            start_date=phase_data["start_date"],
            end_date=phase_data["end_date"],
            trend_type=phase_data["trend_type"],
            strength=phase_data.get("strength", "moderate"),
            confidence=phase_data.get("confidence", 0.8),
            description=phase_data.get("description", ""),
            key_events=phase_data.get("key_events", []),
        )
        phases.append(phase)

    # Determine overall complexity
    complexity = "simple"
    if len(phases) > 2:
        complexity = "moderate"
    if len(phases) > 3 or any(p.trend_type.startswith("sideways") for p in phases):
        complexity = "complex"

    return BenchmarkCase(
        symbol=symbol,
        period=f"{phases[0].start_date} to {phases[-1].end_date}",
        total_period=(phases[0].start_date, phases[-1].end_date),
        phases=phases,
        overall_trend="multi_phase" if len(phases) > 1 else phases[0].trend_type,
        complexity_level=complexity,
        notes=f"Automatically generated test case, contains {len(phases)} trend phases",
    )