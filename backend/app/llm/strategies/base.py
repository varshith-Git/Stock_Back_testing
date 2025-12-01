"""
Trading Strategy Base Framework
Provides standardized strategy interfaces, specifically designed for LLM strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class SignalType(Enum):
    """Trading signal types"""

    BUY = 1  # Buy signal (open long position)
    SELL = -1  # Sell signal (close long position or open short position, depending on trading_mode)
    HOLD = 0  # Hold signal

    # Clear signal types for LLM long/short selection
    LONG_OPEN = 11  # Explicitly open long position
    LONG_CLOSE = -11  # Explicitly close long position
    SHORT_OPEN = 12  # Explicitly open short position
    SHORT_CLOSE = -12  # Explicitly close short position


class ParameterType(Enum):
    """Parameter type enumeration"""

    INTEGER = "integer"  # Integer
    FLOAT = "float"  # Floating point number
    BOOLEAN = "boolean"  # Boolean
    SELECT = "select"  # Selection option
    RANGE = "range"  # Range value


@dataclass
class ParameterSpec:
    """
    Parameter specification definition
    Used for frontend dynamic generation of parameter adjustment interface
    """

    name: str  # Parameter name
    display_name: str  # Display name
    description: str  # Parameter description
    param_type: ParameterType  # Parameter type
    default_value: Any = None  # Default value
    min_value: Optional[float] = None  # Minimum value (numeric types)
    max_value: Optional[float] = None  # Maximum value (numeric types)
    step: Optional[float] = None  # Step size (numeric types)
    options: Optional[List[Any]] = None  # Option list (selection types)
    required: bool = True  # Whether required


@dataclass
class StrategyConfig:
    """Strategy configuration"""

    name: str  # Strategy name
    description: str  # Strategy description
    parameters: Dict[str, Any] = field(default_factory=dict)  # Strategy parameters


@dataclass
class TradingSignal:
    """
    Trading signal
    Contains specific trade execution information
    """

    signal_type: SignalType  # Signal type
    timestamp: pd.Timestamp  # Signal timestamp
    price: float  # Suggested price
    quantity: int = 0  # Trade quantity (0 indicates proportional)
    confidence: float = 0.0  # Confidence level 0-1
    reason: str = ""  # Trade reason
    stop_loss: Optional[float] = None  # Stop loss price
    take_profit: Optional[float] = None  # Take profit price
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional information

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "signal_type": self.signal_type.name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "price": self.price,
            "quantity": self.quantity,
            "confidence": self.confidence,
            "reason": self.reason,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": self.metadata,
        }


class TradingStrategy(ABC):
    """
    Abstract base class for trading strategies
    Simplified version specifically designed for LLM strategies
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.description = config.description
        self.parameters = config.parameters

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals

        Args:
            data: Market data DataFrame

        Returns:
            List of trading signals
        """
        pass

    def get_parameter_specs(self) -> List[ParameterSpec]:
        """
        Get strategy parameter specifications
        Subclasses should override this method to provide parameter definitions
        """
        return []

    def update_parameters(self, parameters: Dict[str, Any]):
        """Update strategy parameters"""
        self.parameters.update(parameters)

    def get_info(self) -> Dict[str, Any]:
        """Get basic strategy information"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "parameter_specs": [spec.__dict__ for spec in self.get_parameter_specs()],
        }


def get_available_strategies() -> Dict[str, Dict[str, Any]]:
    """
    Get available strategy list
    Now only includes LLM strategies (using original version)
    """
    from .llm_strategy import LLMSmartStrategy  # Switch back to original version

    strategies = {
        "llm_smart": {
            "name": "LLM Smart Strategy",
            "description": "Intelligent trading strategy based on large language models",
            "class": LLMSmartStrategy,
            "category": "AI/LLM",
            "parameters": [
                {
                    "name": "confidence_threshold",
                    "display_name": "Confidence Threshold",
                    "description": "Minimum confidence requirement for executing trades",
                    "type": "float",
                    "default": 0.6,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                },
                {
                    "name": "max_daily_trades",
                    "display_name": "Maximum Daily Trades",
                    "description": "Limit daily trading frequency",
                    "type": "integer",
                    "default": 3,
                    "min": 1,
                    "max": 10,
                },
                {
                    "name": "max_loss_threshold",
                    "display_name": "Maximum Loss Threshold",
                    "description": "Maximum loss ratio to trigger stop loss",
                    "type": "float",
                    "default": 0.05,
                    "min": 0.01,
                    "max": 0.2,
                    "step": 0.01,
                },
            ],
        }
    }

    return strategies