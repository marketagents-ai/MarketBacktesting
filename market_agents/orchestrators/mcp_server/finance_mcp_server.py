# finance_mcp_server.py

from datetime import datetime
from mcp.server.fastmcp import FastMCP
import yfinance as yf
import pandas as pd

# Create an MCP server instance with the name "FinanceDataServer"
mcp = FastMCP("FinanceDataServer")

@mcp.tool()
def get_current_stock_price(symbol: str) -> float:
    """
    Get the current stock price for a given symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        # Try to fetch the regular market price; fall back to currentPrice
        price = stock.info.get("regularMarketPrice") or stock.info.get("currentPrice")
        return price if price is not None else 0.0
    except Exception as e:
        return 0.0

@mcp.tool()
def get_stock_fundamentals(symbol: str) -> dict:
    """
    Get fundamental data for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        fundamentals = {
            'symbol': symbol,
            'company_name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('forwardPE', None),
            'pb_ratio': info.get('priceToBook', None),
            'dividend_yield': info.get('dividendYield', None),
            'eps': info.get('trailingEps', None),
            'beta': info.get('beta', None),
            '52_week_high': info.get('fiftyTwoWeekHigh', None),
            '52_week_low': info.get('fiftyTwoWeekLow', None)
        }
        return fundamentals
    except Exception as e:
        return {}

@mcp.tool()
def get_financial_statements(symbol: str) -> dict:
    """
    Get financial statements for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        # Convert DataFrame to dictionary (if not empty)
        return financials.to_dict() if not financials.empty else {}
    except Exception as e:
        return {}

@mcp.tool()
def get_key_financial_ratios(symbol: str) -> dict:
    """
    Get key financial ratios for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        return stock.info  # Contains various ratios among other data
    except Exception as e:
        return {}

@mcp.tool()
def get_analyst_recommendations(symbol: str) -> dict:
    """
    Get analyst recommendations for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        recs = stock.recommendations
        return recs.to_json() if recs is not None else {}
    except Exception as e:
        return {}

@mcp.tool()
def get_dividend_data(symbol: str) -> dict:
    """
    Get dividend data for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        dividends = stock.dividends
        return dividends.to_json() if not dividends.empty else {}
    except Exception as e:
        return {}

@mcp.tool()
def get_company_news(symbol: str) -> dict:
    """
    Get company news for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        return news if news is not None else {}
    except Exception as e:
        return {}

@mcp.tool()
def get_technical_indicators(symbol: str) -> dict:
    """
    Get technical indicators for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        history = stock.history(period="max")
        return history.to_json() if not history.empty else {}
    except Exception as e:
        return {}

@mcp.tool()
def get_company_profile(symbol: str) -> dict:
    """
    Get company profile and overview for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        return stock.info
    except Exception as e:
        return {}
    
@mcp.tool()
def get_historical_prices(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Get historical daily price data for a given symbol and date range.
    Returns OHLCV data.
    """
    try:
        stock = yf.Ticker(symbol)
        history = stock.history(start=start_date, end=end_date, interval='1d')
        return history.to_dict(orient='index')
    except Exception as e:
        return {}

@mcp.tool()
def get_historical_fundamentals(symbol: str, date: str) -> dict:
    """
    Get fundamental data for a given symbol as of a specific date.
    """
    try:
        stock = yf.Ticker(symbol)
        # Note: yfinance might not provide historical fundamentals
        # We'll return current data as fallback
        return stock.info
    except Exception as e:
        return {}

@mcp.tool()
def get_historical_financials(symbol: str, date: str) -> dict:
    """
    Get financial statements data as of a specific date.
    """
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        # Filter to get data up to the specified date
        filtered = financials.loc[:date] if not financials.empty else pd.DataFrame()
        return filtered.to_dict() if not filtered.empty else {}
    except Exception as e:
        return {}

@mcp.tool()
def get_historical_news(symbol: str, start_date: str, end_date: str) -> list:
    """
    Get news articles for a given symbol within a date range.
    """
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        # Filter news by date range
        if news:
            filtered_news = [
                n for n in news 
                if start_date <= datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d') <= end_date
            ]
            return filtered_news
        return []
    except Exception as e:
        return []

if __name__ == "__main__":
    mcp.run()