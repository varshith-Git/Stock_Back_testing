"""
Unrealized P&L Tracker
For calculating and displaying unrealized profit and loss of current positions
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


class UnrealizedPnLTracker:
    """Unrealized P&L Tracker"""

    def __init__(self):
        self.positions = []  # Position records

    def add_position(
        self,
        symbol: str,
        entry_date: str,
        entry_price: float,
        quantity: int = 1,
        signal_confidence: float = 1.0,
    ):
        """Add position record"""
        position = {
            "symbol": symbol,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "quantity": quantity,
            "signal_confidence": signal_confidence,
            "entry_timestamp": datetime.now(),
        }
        self.positions.append(position)
        return len(self.positions) - 1  # Return position ID

    def calculate_unrealized_pnl(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Calculate unrealized P&L for all positions"""
        results = []

        for i, position in enumerate(self.positions):
            symbol = position["symbol"]
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            entry_price = position["entry_price"]
            quantity = position["quantity"]

            # Calculate P&L
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_return = (current_price - entry_price) / entry_price
            unrealized_pnl_percent = unrealized_return * 100

            result = {
                "position_id": i,
                "symbol": symbol,
                "entry_date": position["entry_date"],
                "entry_price": entry_price,
                "current_price": current_price,
                "quantity": quantity,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_return": unrealized_return,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "signal_confidence": position["signal_confidence"],
                "holding_days": (datetime.now() - position["entry_timestamp"]).days,
            }
            results.append(result)

        return results

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get portfolio unrealized P&L summary"""
        pnl_results = self.calculate_unrealized_pnl(current_prices)

        if not pnl_results:
            return {"error": "No position records"}

        total_unrealized_pnl = sum(r["unrealized_pnl"] for r in pnl_results)
        total_investment = sum(r["entry_price"] * r["quantity"] for r in pnl_results)
        portfolio_return = (
            total_unrealized_pnl / total_investment if total_investment > 0 else 0
        )

        # Weighted average unrealized P&L (weighted by confidence)
        weighted_return = 0
        total_weight = sum(r["signal_confidence"] for r in pnl_results)
        if total_weight > 0:
            weighted_return = (
                sum(
                    r["unrealized_pnl_percent"] * r["signal_confidence"]
                    for r in pnl_results
                )
                / total_weight
            )

        return {
            "total_positions": len(pnl_results),
            "total_investment": total_investment,
            "total_unrealized_pnl": total_unrealized_pnl,
            "portfolio_return_percent": portfolio_return * 100,
            "weighted_avg_return_percent": weighted_return,
            "best_position": max(
                pnl_results, key=lambda x: x["unrealized_pnl_percent"]
            ),
            "worst_position": min(
                pnl_results, key=lambda x: x["unrealized_pnl_percent"]
            ),
            "positions": pnl_results,
        }

    def close_position(self, position_id: int, exit_price: float, exit_date: str):
        """Close position (remove position record)"""
        if 0 <= position_id < len(self.positions):
            position = self.positions.pop(position_id)

            # Calculate realized P&L
            realized_pnl = (exit_price - position["entry_price"]) * position["quantity"]
            realized_return = (exit_price - position["entry_price"]) / position[
                "entry_price"
            ]

            return {
                "symbol": position["symbol"],
                "entry_date": position["entry_date"],
                "exit_date": exit_date,
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "quantity": position["quantity"],
                "realized_pnl": realized_pnl,
                "realized_return_percent": realized_return * 100,
                "holding_days": (
                    datetime.strptime(exit_date, "%Y-%m-%d")
                    - datetime.strptime(position["entry_date"], "%Y-%m-%d")
                ).days,
            }
        return None


# Usage example
def demo_unrealized_pnl():
    """Demonstrate unrealized P&L functionality"""
    tracker = UnrealizedPnLTracker()

    # Simulate adding several positions
    tracker.add_position("2330.TW", "2024-12-01", 980.0, 100, 0.75)
    tracker.add_position("2330.TW", "2025-01-15", 1050.0, 50, 0.68)
    tracker.add_position("TSLA", "2024-11-20", 350.0, 10, 0.82)

    # Simulate current prices
    current_prices = {"2330.TW": 1080.0, "TSLA": 380.0}

    # Calculate unrealized P&L
    summary = tracker.get_portfolio_summary(current_prices)

    print("📊 Portfolio Unrealized P&L Summary:")
    print(f"  Total Positions: {summary['total_positions']}")
    print(f"  Total Investment: ${summary['total_investment']:,.2f}")
    print(f"  Total Unrealized P&L: ${summary['total_unrealized_pnl']:+,.2f}")
    print(f"  Portfolio Return Rate: {summary['portfolio_return_percent']:+.2f}%")
    print(f"  Weighted Average Return Rate: {summary['weighted_avg_return_percent']:+.2f}%")

    print(f"\n📈 Individual Positions:")
    for pos in summary["positions"]:
        print(
            f"  {pos['symbol']}: Entry@${pos['entry_price']:.2f}, "
            f"Current@${pos['current_price']:.2f}, "
            f"Unrealized P&L: {pos['unrealized_pnl_percent']:+.2f}%"
        )


if __name__ == "__main__":
    demo_unrealized_pnl()