# MarketBacktesting/market_agents/verbal_rl/backtest_reward_function.py

from typing import Dict, Any, Optional
from market_agents.verbal_rl.rl_models import BaseRewardFunction
import logging

logger = logging.getLogger(__name__)

class BacktestRewardFunction(BaseRewardFunction):
    """Reward function that integrates backtesting performance metrics"""
    
    def compute(
        self,
        environment_reward: Optional[float],
        reflection_data: Dict[str, Any],
        economic_value: Optional[float] = None,
        portfolio_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute reward for backtesting agents, with emphasis on portfolio performance
        
        Args:
            environment_reward: Base reward from environment
            reflection_data: Agent self-evaluation
            economic_value: Value from economic agent
            portfolio_data: Portfolio performance metrics
        """
        # Extract self-reward component
        self_reward = self._extract_self_reward(reflection_data)
        
        # Calculate economic performance from portfolio data
        portfolio_reward = 0.0
        
        if portfolio_data:
            # Get daily return if available
            if "daily_return" in portfolio_data:
                daily_return = portfolio_data["daily_return"]
                # Scale return to appropriate reward range (-1 to 1)
                portfolio_reward = min(max(daily_return * 10, -1.0), 1.0)
            
            # If not, try to calculate from portfolio values
            elif "current_value" in portfolio_data and "previous_value" in portfolio_data:
                prev_value = portfolio_data["previous_value"]
                current_value = portfolio_data["current_value"]
                
                if prev_value > 0:
                    daily_return = (current_value - prev_value) / prev_value
                    # Scale return to appropriate reward range (-1 to 1)
                    portfolio_reward = min(max(daily_return * 10, -1.0), 1.0)
        
        # Combine components with weights
        composite = (
            (environment_reward or 0.0) * self.environment_weight +
            self_reward * self.self_eval_weight +
            portfolio_reward * self.economic_weight
        )
        
        # Log the reward calculation
        logger.info(f"Reward calculation: env={environment_reward or 0.0}, self={self_reward}, portfolio={portfolio_reward}, total={composite}")
        
        # Return full reward details
        return {
            "total_reward": round(composite, 4),
            "components": {
                "environment": environment_reward or 0.0,
                "self_eval": self_reward,
                "portfolio": portfolio_reward
            },
            "weights": self.dict(include={'environment_weight', 'self_eval_weight', 'economic_weight'}),
            "type": self.__class__.__name__
        }