"""
Stock data service.
Handles stock data retrieval from yfinance, caching, and database operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class StockService:
    """Stock Data Service"""

    def __init__(self):
        self.cache = {}

    def get_market_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Get stock market data

        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start_date: Start date (YYYY-MM-DD format), takes priority over period
            end_date: End date (YYYY-MM-DD format), used together with start_date

        Returns:
            List of market data
        """
        try:
            # Clean date format - remove timezone info
            if start_date:
                start_date = self._clean_date_string(start_date)
            if end_date:
                end_date = self._clean_date_string(end_date)

            # Create cache key
            if start_date and end_date:
                cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
                date_range_mode = True
            else:
                cache_key = f"{symbol}_{period}_{interval}"
                date_range_mode = False

            # Check cache
            if cache_key in self.cache:
                cached_data, cache_time = self.cache[cache_key]
                # Use cache if it's less than 1 hour old
                if datetime.now() - cache_time < timedelta(hours=1):
                    logger.info(f"Using cached data for {symbol}")
                    return cached_data

            # Get data
            if date_range_mode:
                logger.info(
                    f"Fetching market data for {symbol} from {start_date} to {end_date}"
                )
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                logger.info(f"Fetching market data for {symbol} with period {period}")
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return []

            # Convert to list of dictionaries
            market_data = []
            for date, row in hist.iterrows():
                market_data.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"])
                        if not pd.isna(row["Volume"])
                        else 0,
                    }
                )

            # Cache data
            self.cache[cache_key] = (market_data, datetime.now())

            logger.info(
                f"Successfully fetched {len(market_data)} data points for {symbol}"
            )
            return market_data

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return []

    def _clean_date_string(self, date_str: str) -> str:
        """
        Clean date string, remove timezone info and time part

        Args:
            date_str: Original date string, may contain timezone info

        Returns:
            Cleaned date string (YYYY-MM-DD format)
        """
        if not date_str:
            return date_str

        # Remove timezone info (T00:00:00+08:00, T00:00:00Z etc.)
        if "T" in date_str:
            date_str = date_str.split("T")[0]

        # Ensure format is YYYY-MM-DD
        try:
            from datetime import datetime

            # Try to parse the date and reformat it
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            # If parsing fails, return the original string
            logger.warning(f"Could not parse date string: {date_str}")
            return date_str

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get stock basic information

        Args:
            symbol: Stock code

        Returns:
            Stock information dictionary
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "beta": info.get("beta", 1.0),
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {"symbol": symbol, "name": symbol}



stock_service = StockService()
