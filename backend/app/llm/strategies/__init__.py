"""
LLM Strategies Module
LLM-based Trading Strategies - Domain Separation Architecture

This module implements refactored LLM trading strategies using a domain separation architecture that breaks down the original monolithic class into multiple specialized modules:

Core Modules:
- LLMSmartStrategy: Main strategy class (refactored version)
- LLMDecisionEngine: LLM decision engine
- RiskManager: Risk manager
- PositionManager: Position manager
- StockCharacteristicsAnalyzer: Stock characteristics analyzer
- TradingEventDetector: Trading event detector
- PerformanceTracker: Performance tracker

Refactoring Comparison:
- Before refactoring: Single class (~2900 lines, 30+ methods)
- After refactoring: 8 specialized modules (~500 lines/module, clear responsibilities)
"""

from .base import (
    ParameterSpec,
    ParameterType,
    SignalType,
    StrategyConfig,
    TradingSignal,
    TradingStrategy,
    get_available_strategies,
)

# New refactored modules (temporarily commented out because data_types has been removed)
# from .data_types import (
#     # Decision related
#     DecisionContext,
#     LLMDecision,
#     PerformanceMetrics,
#     PnLInsights,
#     PositionMetrics,
#     StockCharacteristics,
#     # Strategy state
#     StrategyState,
#     TechnicalParameters,
#     # Core data types
#     TradingEvent,
#     TradingSignalRequest,
# )
# from .llm_decision_engine import LLMDecisionEngine
# from .llm_smart_strategy import LLMSmartStrategy
# Original strategy (backward compatibility)
from .llm_strategy import LLMSmartStrategy as LLMStrategyLegacy

# from .performance_tracker import PerformanceTracker
# from .position_manager import PositionManager
# from .risk_manager import RiskManager
# from .stock_characteristics_analyzer import StockCharacteristicsAnalyzer
# from .trading_event_detector import TradingEventDetector

__all__ = [
    # Base Classes
    "ParameterSpec",
    "ParameterType",
    "SignalType",
    "StrategyConfig",
    "TradingSignal",
    "TradingStrategy",
    "get_available_strategies",
    # Original strategy (backward compatibility)
    "LLMStrategyLegacy",
    # New main strategy class (temporarily commented, because module has been removed)
    # "LLMSmartStrategy",
    # Core modules (temporarily commented, because modules have been removed)
    # "LLMDecisionEngine",
    # "RiskManager",
    # "PositionManager",
    # "StockCharacteristicsAnalyzer",
    # "TradingEventDetector",
    # "PerformanceTracker",
    # Data types (temporarily commented, because module has been removed)
    # "TradingEvent",
    # "StockCharacteristics",
    # "TechnicalParameters",
    # "PositionMetrics",
    # "PnLInsights",
    # "PerformanceMetrics",
    # "DecisionContext",
    # "LLMDecision",
    # "TradingSignalRequest",
    # "StrategyState",
    # Helper functions
    "print_architecture_info",
    "get_module_info",
]

# Version information
__version__ = "2.0.0"
__author__ = "LLM Agent Trader Team"
__description__ = "Refactored LLM Trading Strategy with Domain Separation Architecture"

# Architecture description
ARCHITECTURE_INFO = """
Domain Separation Architecture (Domain Separation Architecture):

📊 LLMSmartStrategy (Main Controller)
├── 🤖 LLMDecisionEngine (LLM Decision Engine)
│   ├── Prompt construction
│   ├── LLM invocation
│   └── Response parsing
├── ⚡ RiskManager (Risk Manager)  
│   ├── Risk assessment
│   ├── P&L insights
│   └── Decision validation
├── 💼 PositionManager (Position Manager)
│   ├── Position tracking
│   ├── Trade execution
│   └── P&L calculation
├── 📈 StockCharacteristicsAnalyzer (Stock Characteristics Analyzer)
│   ├── Volatility analysis
│   ├── Trend consistency
│   └── MACD effectiveness
├── 🔍 TradingEventDetector (Trading Event Detector)
│   ├── MACD signals
│   ├── Moving average crossovers
│   ├── Bollinger Band breakouts
│   └── Price breakouts
└── 📊 PerformanceTracker (Performance Tracker)
    ├── Trade records
    ├── Performance calculation
    └── Report generation

📋 data_types (Shared Data Structures)
├── DTOs and data classes
├── Type definitions
└── Interface standards
"""


def print_architecture_info():
    """Print architecture information"""
    print(ARCHITECTURE_INFO)


def get_module_info():
    """Get module information"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "modules": len(__all__),
        "architecture": "Domain Separation Architecture",
    }