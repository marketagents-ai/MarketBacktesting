from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Type
from market_agents.orchestrators.config import EnvironmentConfig
from market_agents.environments.mechanisms.mcp_server import MCPServerEnvironmentConfig, MCPServerMechanism
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from market_agents.environments.environment import (
    LocalEnvironmentStep,
    Mechanism,
    LocalAction,
    GlobalAction,
    LocalObservation,
    GlobalObservation,
    EnvironmentStep,
    MultiAgentEnvironment,
    ActionSpace,
    ObservationSpace
)

class BacktestingEnvironmentConfig(EnvironmentConfig):
    """Configuration for backtesting environment"""
    name: str = Field(default="backtesting", description="Name of the backtesting environment")
    mechanism: str = Field(default="backtesting", description="Type of mechanism")
    tickers: List[str] = Field(..., description="List of ticker symbols to backtest")
    start_date: str = Field(..., description="Start date for backtesting in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date for backtesting in YYYY-MM-DD format")
    initial_capital: float = Field(..., description="Initial capital for the portfolio")
    margin_requirement: float = Field(default=0.0, description="Margin requirement for short positions")
    form_cohorts: bool = Field(default=False, description="Whether to organize agents into cohorts")
    group_size: int = Field(default=4, description="Size of agent groups when using cohorts")
    task_prompt: str = Field(default="", description="Initial task prompt")
    mcp_server_module: str = Field(default="market_agents.orchestrators.mcp_server.finance_mcp_server", 
                                 description="Module path to MCP server for data fetching")
    
    model_config = {
        "extra": "allow"
    }

# Trade action schema
class TradeAction(BaseModel):
    """Schema for trade actions"""
    ticker: str = Field(..., description="The ticker symbol to trade")
    action: str = Field(..., description="Trade action: buy, sell, short, or cover")
    quantity: float = Field(..., description="Number of shares to trade")
    reason: str = Field(default="", description="Reasoning behind the trade")

class BacktestAction(LocalAction):
    """Local action for a single agent in backtesting"""
    action: TradeAction

    @classmethod
    def sample(cls, agent_id: str) -> 'BacktestAction':
        """Sample a random trade action"""
        return cls(
            agent_id=agent_id,
            action=TradeAction(
                ticker="SAMPLE",
                action="hold",
                quantity=0,
                reason="Sample trade"
            )
        )

class PortfolioState(BaseModel):
    """Current portfolio state"""
    cash: float
    positions: Dict[str, Dict[str, Any]]
    margin_used: float
    realized_gains: Dict[str, Dict[str, float]]

class MarketData(BaseModel):
    """Market data for current timestamp"""
    date: str
    prices: Dict[str, float]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    news: List[Dict[str, Any]] = Field(default_factory=list)

class BacktestObservation(BaseModel):
    """Observation containing market data and portfolio state"""
    market_data: MarketData
    portfolio: PortfolioState
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

class BacktestLocalObservation(LocalObservation):
    """Local observation for a single agent"""
    observation: BacktestObservation

class BacktestGlobalObservation(GlobalObservation):
    """Global observation containing all agent observations"""
    observations: Dict[str, BacktestLocalObservation]
    all_actions_this_round: Optional[Dict[str, Any]] = None
    
    @property
    def global_obs(self) -> Optional[Any]:
        return {
            'market_data': next(iter(self.observations.values())).observation.market_data.dict(),
            'all_actions_this_round': self.all_actions_this_round
        }
    
class BacktestDataProvider:
    """Helper class to fetch data using MCP server tools"""
    def __init__(self, mcp_server_module: str = "market_agents.orchestrators.mcp_server.finance_mcp_server"):
        config = MCPServerEnvironmentConfig(
            name="finance_data",
            mcp_server_module=mcp_server_module,
            mcp_server_class="mcp"
        )
        
        # Import the MCP server module
        import importlib
        spec = importlib.util.find_spec(config.mcp_server_module)
        if spec is None:
            raise ImportError(f"Could not find module {config.mcp_server_module}")
        server_path = spec.origin
        
        # Initialize mechanism
        self.mechanism = MCPServerMechanism(server_path=server_path)
        
    async def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> dict:
        """Get historical price data using MCP server"""
        return await self.mechanism.execute_tool(
            "get_historical_prices",
            {"symbol": symbol, "start_date": start_date, "end_date": end_date}
        )
    
    async def get_historical_fundamentals(self, symbol: str, date: str) -> dict:
        """Get fundamental data for a specific date"""
        return await self.mechanism.execute_tool(
            "get_historical_fundamentals",
            {"symbol": symbol, "date": date}
        )
    
    async def get_historical_financials(self, symbol: str, date: str) -> dict:
        """Get financial data for a specific date"""
        return await self.mechanism.execute_tool(
            "get_historical_financials",
            {"symbol": symbol, "date": date}
        )

    async def get_historical_news(self, symbol: str, start_date: str, end_date: str) -> list:
        """Get news within a date range"""
        return await self.mechanism.execute_tool(
            "get_historical_news",
            {"symbol": symbol, "start_date": start_date, "end_date": end_date}
        )

    def convert_prices_to_df(self, prices: dict) -> pd.DataFrame:
        """Convert price dictionary to DataFrame"""
        if not prices:
            return pd.DataFrame()
        
        # Handle the nested response structure from MCP server
        if isinstance(prices, dict) and 'content' in prices and isinstance(prices['content'], list):
            try:
                # Extract the actual price data from the text field
                import json
                print(f"DEBUG: Extracting price data from nested structure")
                text_data = prices['content'][0]['text']
                prices = json.loads(text_data)
                print(f"DEBUG: Successfully extracted price data")
            except Exception as e:
                print(f"ERROR: Failed to extract price data from nested structure: {e}")
                return pd.DataFrame()
        
        # Convert the dictionary to DataFrame
        try:
            df = pd.DataFrame.from_dict(prices, orient='index')
            
            # Remove timezone info when parsing dates
            df.index = pd.to_datetime(df.index, format='ISO8601', errors='coerce')
            
            # Drop any invalid parsed dates
            df = df.loc[df.index.notnull()]
            
            df.sort_index(inplace=True)
            print(f"DEBUG: Created DataFrame with {len(df)} rows")
            return df
        except Exception as e:
            print(f"ERROR: Failed to convert prices to DataFrame: {e}")
            return pd.DataFrame()

class BacktestingMechanism(Mechanism):
    """Mechanism that manages backtesting simulation with integrated trade execution"""
    sequential: bool = Field(default=False)
    tickers: List[str] = Field(default_factory=list)
    dates: List[datetime] = Field(default_factory=list)
    current_index: int = Field(default=0)
    portfolio: Dict[str, Any] = Field(default_factory=dict)
    market_data: Dict[str, Any] = Field(default_factory=dict)
    portfolio_values: List[Dict[str, Any]] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    form_cohorts: bool = Field(default=False)
    cohorts: Dict[str, List[Any]] = Field(default_factory=dict)
    data_provider: Any = Field(default=None)
    
    # Rename fields to remove leading underscores
    price_data: Dict[str, pd.DataFrame] = Field(default_factory=dict)
    fundamental_data: Dict[str, Dict] = Field(default_factory=dict)
    financial_data: Dict[str, Dict] = Field(default_factory=dict)
    news_data: Dict[str, List] = Field(default_factory=dict)

    data_fetched: bool = Field(default=False)

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        margin_requirement: float = 0.0,
        form_cohorts: bool = False,
        group_size: int = 4,
        **kwargs
    ):
        """Initialize the backtesting mechanism"""
        super().__init__(
            form_cohorts=form_cohorts,
            **kwargs
        )
        
        self.tickers = tickers
        self.dates = pd.date_range(start_date, end_date, freq='B')
        self.data_provider = BacktestDataProvider()

        self.market_data = {
            "date": self.dates[0].strftime("%Y-%m-%d") if len(self.dates) > 0 else None,
            "prices": {ticker: 0.0 for ticker in tickers},  # Default zero prices until updated
            "metrics": {},
            "news": []
        }
        
        # Initialize portfolio with support for long/short positions
        self.portfolio = {
            "cash": initial_capital,
            "margin_used": 0.0,
            "margin_requirement": margin_requirement,
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0
                } for ticker in tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,
                    "short": 0.0,
                } for ticker in tickers
            }
        }

        # Initialize portfolio values with initial capital
        if len(self.dates) > 0:
            self.portfolio_values = [{
                "Date": self.dates[0],
                "Portfolio Value": initial_capital
            }]
        
        # Pre-fetch data
        self.prefetch_data()

    async def prefetch_data(self):
        """Pre-fetch all data needed for the backtest period"""
        print("\nPre-fetching data for the entire backtest period...")

        start_date_str = self.dates[0].strftime("%Y-%m-%d")
        end_date_str = self.dates[-1].strftime("%Y-%m-%d")

        for ticker in self.tickers:
            try:
                # Get historical prices for the entire period
                prices = await self.data_provider.get_historical_prices(
                    ticker, start_date_str, end_date_str
                )
                if prices:
                    self.price_data[ticker] = self.data_provider.convert_prices_to_df(prices)

                # Get fundamentals as of the start date
                fundamentals = await self.data_provider.get_historical_fundamentals(
                    ticker, start_date_str
                )
                if fundamentals:
                    self.fundamental_data[ticker] = fundamentals

                # Get financial data
                financials = await self.data_provider.get_historical_financials(
                    ticker, end_date_str
                )
                if financials:
                    self.financial_data[ticker] = financials

                # Get historical news
                news = await self.data_provider.get_historical_news(
                    ticker, start_date_str, end_date_str
                )
                if news:
                    self.news_data[ticker] = news

            except Exception as e:
                print(f"Error prefetching data for {ticker}: {e}")
                continue

        print("Data pre-fetch complete.")
        
        # Update market_data for the first date after all data is fetched
        if len(self.dates) > 0:
            first_date = self.dates[0].strftime("%Y-%m-%d")
            self._use_prefetched_market_data(first_date)
            print(f"Initial market data prepared for {first_date}")

    async def form_agent_cohorts(self, agents: List[Any]) -> None:
        """Form trading teams if cohorts enabled"""
        if not self.form_cohorts:
            return

        self.cohorts.clear()
        current_cohort = []
        cohort_count = 1

        for agent in agents:
            current_cohort.append(agent)
            if len(current_cohort) >= self.group_size:
                self.cohorts[f"trading_team_{cohort_count}"] = current_cohort
                current_cohort = []
                cohort_count += 1

        if current_cohort:
            self.cohorts[f"trading_team_{cohort_count}"] = current_cohort

    def execute_trade(self, ticker: str, action: str, quantity: float, current_price: float) -> int:
        """Execute trades with support for both long and short positions"""
        if quantity <= 0:
            return 0

        quantity = int(quantity)  # force integer shares
        position = self.portfolio["positions"][ticker]

        if action == "buy":
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                old_shares = position["long"]
                old_cost_basis = position["long_cost_basis"]
                new_shares = quantity
                total_shares = old_shares + new_shares

                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_shares
                    total_new_cost = cost
                    position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["long"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                max_quantity = int(self.portfolio["cash"] / current_price)
                if max_quantity > 0:
                    cost = max_quantity * current_price
                    old_shares = position["long"]
                    old_cost_basis = position["long_cost_basis"]
                    total_shares = old_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_shares
                        total_new_cost = cost
                        position["long_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["long"] += max_quantity
                    self.portfolio["cash"] -= cost
                    return max_quantity
                return 0

        elif action == "sell":
            quantity = min(quantity, position["long"])
            if quantity > 0:
                avg_cost_per_share = position["long_cost_basis"] if position["long"] > 0 else 0
                realized_gain = (current_price - avg_cost_per_share) * quantity
                self.portfolio["realized_gains"][ticker]["long"] += realized_gain

                position["long"] -= quantity
                self.portfolio["cash"] += quantity * current_price

                if position["long"] == 0:
                    position["long_cost_basis"] = 0.0

                return quantity

        elif action == "short":
            proceeds = current_price * quantity
            margin_required = proceeds * self.portfolio["margin_requirement"]
            if margin_required <= self.portfolio["cash"]:
                old_short_shares = position["short"]
                old_cost_basis = position["short_cost_basis"]
                new_shares = quantity
                total_shares = old_short_shares + new_shares

                if total_shares > 0:
                    total_old_cost = old_cost_basis * old_short_shares
                    total_new_cost = current_price * new_shares
                    position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                position["short"] += quantity
                position["short_margin_used"] += margin_required
                self.portfolio["margin_used"] += margin_required
                self.portfolio["cash"] += proceeds
                self.portfolio["cash"] -= margin_required
                return quantity
            else:
                margin_ratio = self.portfolio["margin_requirement"]
                if margin_ratio > 0:
                    max_quantity = int(self.portfolio["cash"] / (current_price * margin_ratio))
                else:
                    max_quantity = 0

                if max_quantity > 0:
                    proceeds = current_price * max_quantity
                    margin_required = proceeds * margin_ratio

                    old_short_shares = position["short"]
                    old_cost_basis = position["short_cost_basis"]
                    total_shares = old_short_shares + max_quantity

                    if total_shares > 0:
                        total_old_cost = old_cost_basis * old_short_shares
                        total_new_cost = current_price * max_quantity
                        position["short_cost_basis"] = (total_old_cost + total_new_cost) / total_shares

                    position["short"] += max_quantity
                    position["short_margin_used"] += margin_required
                    self.portfolio["margin_used"] += margin_required
                    self.portfolio["cash"] += proceeds
                    self.portfolio["cash"] -= margin_required
                    return max_quantity
                return 0

        elif action == "cover":
            quantity = min(quantity, position["short"])
            if quantity > 0:
                cover_cost = quantity * current_price
                avg_short_price = position["short_cost_basis"] if position["short"] > 0 else 0
                realized_gain = (avg_short_price - current_price) * quantity

                portion = quantity / position["short"] if position["short"] > 0 else 1.0
                margin_to_release = portion * position["short_margin_used"]

                position["short"] -= quantity
                position["short_margin_used"] -= margin_to_release
                self.portfolio["margin_used"] -= margin_to_release
                self.portfolio["cash"] += margin_to_release
                self.portfolio["cash"] -= cover_cost
                self.portfolio["realized_gains"][ticker]["short"] += realized_gain

                if position["short"] == 0:
                    position["short_cost_basis"] = 0.0
                    position["short_margin_used"] = 0.0

                return quantity

        return 0

    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash and positions"""
        total_value = self.portfolio["cash"]

        for ticker in self.tickers:
            position = self.portfolio["positions"][ticker]
            price = current_prices[ticker]

            # Long position value
            long_value = position["long"] * price
            total_value += long_value

            # Short position unrealized PnL
            if position["short"] > 0:
                total_value += position["short"] * (position["short_cost_basis"] - price)

        return total_value

    def step(
        self, 
        action: Union[LocalAction, GlobalAction],
        cohort_id: Optional[str] = None
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Execute one step of the backtesting simulation"""
        if self.current_index >= len(self.dates):
            return self._create_step(done=True, cohort_id=cohort_id)

        # Get current date and prices
        current_date = self.dates[self.current_index]
        current_date_str = current_date.strftime("%Y-%m-%d")
        
        # Use prefetched data for the current date
        self._use_prefetched_market_data(current_date_str)

        # Use provided cohort_id or default
        effective_cohort = cohort_id if cohort_id else "default"

        # Process trades
        if isinstance(action, LocalAction):
            action_dict = action.action
            try:
                # Check if action is nested inside the action dictionary
                if isinstance(action_dict, dict) and 'action' in action_dict and isinstance(action_dict['action'], dict):
                    # Extract the inner action dictionary
                    trade_info = action_dict['action']
                    
                    # Validate the trade information
                    validated_trade = TradeAction(
                        ticker=trade_info.get('ticker', ''),
                        action=trade_info.get('action', 'hold'),
                        quantity=float(trade_info.get('quantity', 0)),
                        reason=trade_info.get('reason', '')
                    )
                elif isinstance(action_dict, dict) and 'Action' in action_dict and isinstance(action_dict['Action'], dict):
                    # Handle the rich text format from log output
                    inner_action = action_dict['Action']
                    validated_trade = TradeAction(
                        ticker=inner_action.get('Ticker', ''),
                        action=inner_action.get('Action', 'hold'),
                        quantity=float(inner_action.get('Quantity', 0)),
                        reason=inner_action.get('Reason', '')
                    )
                else:
                    # Direct validation of the action dictionary
                    validated_trade = TradeAction.model_validate(action_dict)
                
                # Execute the trade
                if validated_trade.ticker in self.market_data["prices"]:
                    price = self.market_data["prices"][validated_trade.ticker]
                    self.execute_trade(
                        validated_trade.ticker, 
                        validated_trade.action, 
                        validated_trade.quantity, 
                        price
                    )
                else:
                    print(f"Warning: No price data for {validated_trade.ticker} on {current_date_str}")
                    
            except Exception as e:
                print(f"Error validating trade action: {e}")
                print(f"Debug - Action structure: {action_dict}")
                
            step = self._create_local_step(action.agent_id, done=False)
            
        else:  # GlobalAction
            for agent_id, local_action in action.actions.items():
                action_dict = local_action.action
                
                try:
                    # Check if action is nested inside the action dictionary
                    if isinstance(action_dict, dict) and 'action' in action_dict and isinstance(action_dict['action'], dict):
                        # Extract the inner action dictionary
                        trade_info = action_dict['action']
                        
                        # Validate the trade information
                        validated_trade = TradeAction(
                            ticker=trade_info.get('ticker', ''),
                            action=trade_info.get('action', 'hold'),
                            quantity=float(trade_info.get('quantity', 0)),
                            reason=trade_info.get('reason', '')
                        )
                    elif isinstance(action_dict, dict) and 'Action' in action_dict and isinstance(action_dict['Action'], dict):
                        # Handle the rich text format from log output
                        inner_action = action_dict['Action']
                        validated_trade = TradeAction(
                            ticker=inner_action.get('Ticker', ''),
                            action=inner_action.get('Action', 'hold'),
                            quantity=float(inner_action.get('Quantity', 0)),
                            reason=inner_action.get('Reason', '')
                        )
                    else:
                        # Direct validation of the action dictionary
                        validated_trade = TradeAction.model_validate(action_dict)
                    
                    # Execute the trade
                    if validated_trade.ticker in self.market_data["prices"]:
                        price = self.market_data["prices"][validated_trade.ticker]
                        self.execute_trade(
                            validated_trade.ticker, 
                            validated_trade.action, 
                            validated_trade.quantity, 
                            price
                        )
                    else:
                        print(f"Warning: No price data for {validated_trade.ticker} on {current_date_str}")
                        
                except Exception as e:
                    print(f"Error validating trade action for agent {agent_id}: {e}")
                    print(f"Debug - Action structure for {agent_id}: {action_dict}")
                    
            step = self._create_step(done=False, cohort_id=cohort_id)

        # Update portfolio value history
        total_value = self.calculate_portfolio_value(self.market_data["prices"])
        self.portfolio_values.append({
            "Date": current_date,
            "Portfolio Value": total_value
        })

        # Update performance metrics
        self._update_performance_metrics()

        # Move to next date
        self.current_index += 1
        return step

    def _use_prefetched_market_data(self, current_date: str):
        """Use the prefetched data for the current date"""
        prices = {}
        metrics = {}
        news = []
        
        for ticker in self.tickers:
            try:
                # Get price from cached data
                if ticker in self.price_data:
                    price_df = self.price_data[ticker]
                    
                    # Handle different date formats
                    try:
                        # Try exact string match
                        if current_date in price_df.index:
                            # Convert NumPy scalar to native Python float
                            prices[ticker] = float(price_df.loc[current_date, 'Close'])
                        else:
                            # Try datetime index
                            date_obj = pd.to_datetime(current_date)
                            if date_obj in price_df.index:
                                # Convert NumPy scalar to native Python float
                                prices[ticker] = float(price_df.loc[date_obj, 'Close'])
                            else:
                                # Try next business day if needed
                                next_date = price_df.index[price_df.index > date_obj]
                                if not next_date.empty:
                                    # Convert NumPy scalar to native Python float
                                    prices[ticker] = float(price_df.iloc[price_df.index.get_indexer([next_date[0]], method='nearest')[0]]['Close'])
                    except Exception as e:
                        # If date not found, log warning
                        print(f"Error finding price for {ticker} on {current_date}: {e}")
                        
                # Get metrics from cached data
                if ticker in self.fundamental_data:
                    # Convert any NumPy values to Python native types
                    metrics[ticker] = self._convert_numpy_to_python(self.fundamental_data[ticker])
                    
                # Get news from cached data
                if ticker in self.news_data:
                    relevant_news = [
                        n for n in self.news_data[ticker] 
                        if current_date in n.get('date', '')
                    ]
                    news.extend(relevant_news)
                    
            except Exception as e:
                print(f"Error updating market data for {ticker}: {e}")
                continue

        self.market_data = {
            "date": current_date,
            "prices": prices,
            "metrics": metrics,
            "news": news
        }

    def _convert_numpy_to_python(self, obj):
        """Helper method to recursively convert NumPy types to Python native types"""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python(item) for item in obj]
        else:
            return obj

    def _update_performance_metrics(self):
        """Update performance metrics using daily returns."""
        if len(self.portfolio_values) > 2:
            values_df = pd.DataFrame(self.portfolio_values).set_index("Date")
            values_df["Daily Return"] = values_df["Portfolio Value"].pct_change()
            clean_returns = values_df["Daily Return"].dropna()

            if len(clean_returns) < 2:
                return  # not enough data points

            # Assumes 252 trading days/year
            daily_risk_free_rate = 0.0434 / 252  # Using 4.34% annual risk-free rate
            excess_returns = clean_returns - daily_risk_free_rate
            mean_excess_return = excess_returns.mean()
            std_excess_return = excess_returns.std()

            # Sharpe ratio
            if std_excess_return > 1e-12:
                self.performance_metrics["sharpe_ratio"] = np.sqrt(252) * (mean_excess_return / std_excess_return)
            else:
                self.performance_metrics["sharpe_ratio"] = 0.0

            # Sortino ratio
            negative_returns = excess_returns[excess_returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                if downside_std > 1e-12:
                    self.performance_metrics["sortino_ratio"] = np.sqrt(252) * (mean_excess_return / downside_std)
                else:
                    self.performance_metrics["sortino_ratio"] = float('inf') if mean_excess_return > 0 else 0
            else:
                self.performance_metrics["sortino_ratio"] = float('inf') if mean_excess_return > 0 else 0

            # Maximum drawdown
            rolling_max = values_df["Portfolio Value"].cummax()
            drawdown = (values_df["Portfolio Value"] - rolling_max) / rolling_max
            
            if len(drawdown) > 0:
                min_drawdown = drawdown.min()
                # Store as a negative percentage
                self.performance_metrics["max_drawdown"] = min_drawdown * 100
                
                # Store the date of max drawdown for reference
                if min_drawdown < 0:
                    self.performance_metrics["max_drawdown_date"] = drawdown.idxmin().strftime('%Y-%m-%d')
                else:
                    self.performance_metrics["max_drawdown_date"] = None
            else:
                self.performance_metrics["max_drawdown"] = 0.0
                self.performance_metrics["max_drawdown_date"] = None

    def _create_observation(self) -> BacktestObservation:
        """Create observation from current state"""
        return BacktestObservation(
            market_data=MarketData(**self.market_data),
            portfolio=PortfolioState(**self.portfolio),
            performance_metrics=self.performance_metrics
        )

    def _create_local_step(self, agent_id: str, done: bool) -> LocalEnvironmentStep:
        """Create a local step for single agent"""
        obs = self._create_observation()
        return LocalEnvironmentStep(
            observation=BacktestLocalObservation(
                agent_id=agent_id,
                observation=obs
            ),
            done=done,
            info={"date": self.market_data["date"]}
        )

    def _create_step(self, done: bool, cohort_id: Optional[str] = None) -> EnvironmentStep:
        """Create a global step"""
        obs = self._create_observation()
        observations = {}
        
        # Get relevant agents based on cohort
        if cohort_id and cohort_id in self.cohorts:
            agents = self.cohorts[cohort_id]
            
            # Create observation for each agent in the cohort
            for agent in agents:
                observations[agent.id] = BacktestLocalObservation(
                    agent_id=agent.id,
                    observation=obs
                )
        else:
            # If no cohort information or agents are available directly,
            pass

        return EnvironmentStep(
            global_observation=BacktestGlobalObservation(
                observations=observations,
                all_actions_this_round=None
            ),
            done=done,
            info={"date": self.market_data["date"]}
        )

    def get_global_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the current global state with JSON-serializable values"""
        # Create the state dictionary
        state = {
            "current_date": self.dates[self.current_index].strftime("%Y-%m-%d") if self.current_index < len(self.dates) else None,
            "portfolio": self._convert_to_serializable(self.portfolio),
            "market_data": self._convert_to_serializable(self.market_data),
            "portfolio_values": self._convert_to_serializable(self.portfolio_values[-5:]),
            "performance_metrics": self._convert_to_serializable(self.performance_metrics)
        }
        
        if self.form_cohorts:
            # Add cohort-specific information
            if agent_id:
                cohort_id = next(
                    (cid for cid, agents in self.cohorts.items() 
                    if any(a.id == agent_id for a in agents)),
                    None
                )
                if cohort_id:
                    state["cohort_id"] = cohort_id
                    state["cohort_agents"] = [a.id for a in self.cohorts[cohort_id]]
            else:
                state["cohorts"] = {
                    cid: [a.id for a in agents] 
                    for cid, agents in self.cohorts.items()
                }
        
        return state

    def _convert_to_serializable(self, obj):
        """Convert all values to JSON-serializable types"""
        import pandas as pd
        import numpy as np
        
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj

    def reset(self) -> None:
        """Reset the mechanism state"""
        self.current_index = 0
        self.portfolio_values = [{
            "Date": self.dates[0],
            "Portfolio Value": self.portfolio["cash"]
        }] if len(self.dates) > 0 else []
        self.performance_metrics = {}
        self.market_data = {}
        self.cohorts.clear()

    async def initialize(self):
        """Initialize the mechanism by prefetching data"""
        if not hasattr(self, 'data_fetched') or not self.data_fetched:
            print("\nInitializing backtesting mechanism - prefetching data...")
            await self.prefetch_data()
            self.data_fetched = True
            print("Data prefetch complete")


class BacktestActionSpace(ActionSpace):
    """Action space for backtesting environment"""
    allowed_actions: List[Type[LocalAction]] = [BacktestAction]

    def sample(self, agent_id: str) -> BacktestAction:
        return BacktestAction.sample(agent_id)


class BacktestObservationSpace(ObservationSpace):
    """Observation space for backtesting environment"""
    allowed_observations: List[Type[LocalObservation]] = [BacktestLocalObservation]


class BacktestingEnvironment(MultiAgentEnvironment):
    """Multi-agent environment for backtesting simulation"""
    name: str = Field(default="backtesting")
    action_space: BacktestActionSpace = Field(default_factory=BacktestActionSpace)
    observation_space: BacktestObservationSpace = Field(default_factory=BacktestObservationSpace)
    mechanism: BacktestingMechanism = Field(default_factory=BacktestingMechanism)

    def __init__(self, **config):
        """Initialize environment with config parameters."""
        try:
            # Parse and validate config
            env_config = BacktestingEnvironmentConfig(**config)
            
            # Initialize mechanism with config parameters
            mechanism = BacktestingMechanism(
                tickers=env_config.tickers,
                start_date=env_config.start_date,
                end_date=env_config.end_date,
                initial_capital=env_config.initial_capital,
                margin_requirement=env_config.margin_requirement,
                form_cohorts=env_config.form_cohorts,
                group_size=env_config.group_size
            )

            # Initialize parent class with processed config
            super().__init__(
                name=env_config.name,
                action_space=BacktestActionSpace(),
                observation_space=BacktestObservationSpace(),
                mechanism=mechanism
            )
            
            # Form cohorts during initialization if enabled
            if env_config.form_cohorts and hasattr(self, 'agents'):
                self.mechanism.form_agent_cohorts(self.agents)
                
        except Exception as e:
            raise ValueError(f"Failed to initialize BacktestingEnvironment: {e}")

    def get_global_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the environment's global state"""
        return self.mechanism.get_global_state(agent_id)

    def reset(self) -> GlobalObservation:
        """Reset the environment"""
        self.mechanism.reset()
        return GlobalObservation(observations={})
    
    async def initialize(self):
        """Initialize the environment and prefetch data"""
        print("\nInitializing backtesting environment...")
        await self.mechanism.initialize()
        print("Backtesting environment initialization complete")