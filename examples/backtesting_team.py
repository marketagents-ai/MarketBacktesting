from market_agents.verbal_rl.reward_functions.backtest_reward_function import BacktestRewardFunction
from market_agents.agents.market_agent import MarketAgent
from market_agents.orchestrators.market_agent_team import MarketAgentTeam
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import load_config_from_yaml
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.agents.personas.persona import Persona
from minference.lite.models import LLMConfig, ResponseFormat
from typing import Dict, Any, List
from datetime import datetime


async def create_backtesting_hedge_fund_team(
    tickers: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 10_000_000.0
) -> MarketAgentTeam:
    """Create a hierarchical hedge fund team with backtesting capabilities."""
    
    # Load storage config
    storage_config = load_config_from_yaml("market_agents/memory/storage_config.yaml")
    storage_utils = AgentStorageAPIUtils(config=storage_config)

    # Create custom reward function with higher economic weight
    backtest_reward_fn = BacktestRewardFunction(
        environment_weight=0.2,
        self_eval_weight=0.3,
        economic_weight=0.5
    )

    # Create Portfolio Manager Persona
    portfolio_manager_persona = Persona(
        name="Sarah Chen",
        role="Portfolio Manager",
        persona="Experienced investment professional with strong risk management background",
        objectives=[
            "Analyze team insights and make final investment decisions",
            "Manage portfolio risk and allocation",
            "Coordinate team analysis efforts"
        ],
        trader_type=["Expert", "Moderate", "Rational"],
        communication_style="Direct",
        routines=[
            "Review team analyses",
            "Make portfolio decisions",
            "Monitor risk metrics"
        ],
        skills=[
            "Portfolio management",
            "Risk assessment",
            "Team leadership"
        ]
    )

    # Create Portfolio Manager
    portfolio_manager = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="portfolio_manager",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.3,
            use_cache=True
        ),
        persona=portfolio_manager_persona,
        econ_agent=EconomicAgent(
            generate_wallet=True,
            initial_holdings={"USDC": initial_capital}
        ),
        reward_function=backtest_reward_fn
    )

    # Create Fundamental Analyst Persona
    fundamental_analyst_persona = Persona(
        name="Michael Wong",
        role="Fundamental Analysis Specialist",
        persona="Detail-oriented financial analyst focused on company fundamentals",
        objectives=[
            "Analyze financial statements and metrics",
            "Evaluate business models and competitive positions",
            "Assess company valuations and fair value estimates"
        ],
        trader_type=["Expert", "Conservative", "Rational"],
        communication_style="Formal",
        routines=[
            "Review financial statements",
            "Conduct company research",
            "Build valuation models"
        ],
        skills=[
            "Financial analysis",
            "Valuation modeling",
            "Industry research"
        ]
    )

    # Create Fundamental Analyst
    fundamental_analyst = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="fundamental_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.2,
            use_cache=True
        ),
        persona=fundamental_analyst_persona,
        reward_function=backtest_reward_fn
    )

    # Create Technical Analyst Persona
    technical_analyst_persona = Persona(
        name="Alex Rodriguez",
        role="Technical Analysis Specialist",
        persona="Experienced technical trader focused on price patterns",
        objectives=[
            "Analyze price trends and patterns",
            "Identify key support/resistance levels",
            "Generate trading signals based on technical indicators"
        ],
        trader_type=["Expert", "Aggressive", "Impulsive"],
        communication_style="Direct",
        routines=[
            "Monitor price charts",
            "Update technical indicators",
            "Track market momentum"
        ],
        skills=[
            "Technical analysis",
            "Pattern recognition",
            "Momentum trading"
        ]
    )

    # Create Technical Analyst
    technical_analyst = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="technical_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.2,
            use_cache=True
        ),
        persona=technical_analyst_persona,
        reward_function=backtest_reward_fn
    )

    # Create Macro Analyst Persona
    macro_analyst_persona = Persona(
        name="Emma Thompson",
        role="Macro Research Specialist",
        persona="Global macro analyst focused on economic trends",
        objectives=[
            "Monitor global economic indicators",
            "Analyze central bank policies and implications",
            "Assess geopolitical risks and market impacts"
        ],
        trader_type=["Expert", "Moderate", "Rational"],
        communication_style="Formal",
        routines=[
            "Review economic data",
            "Monitor policy changes",
            "Analyze global trends"
        ],
        skills=[
            "Economic analysis",
            "Policy research",
            "Geopolitical assessment"
        ]
    )

    # Create Macro Analyst
    macro_analyst = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="macro_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.2,
            use_cache=True
        ),
        persona=macro_analyst_persona,
        reward_function=backtest_reward_fn
    )

    # Create Risk Analyst Persona
    risk_analyst_persona = Persona(
        name="David Kumar",
        role="Risk Management Specialist",
        persona="Risk-focused analyst specializing in portfolio risk assessment",
        objectives=[
            "Monitor portfolio risk metrics",
            "Analyze position sizing and leverage",
            "Assess market volatility and correlation risks"
        ],
        trader_type=["Expert", "Conservative", "Rational"],
        communication_style="Direct",
        routines=[
            "Calculate risk metrics",
            "Monitor exposures",
            "Update risk models"
        ],
        skills=[
            "Risk modeling",
            "Portfolio analysis",
            "Quantitative methods"
        ]
    )

    # Create Risk Analyst
    risk_analyst = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="risk_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            response_format=ResponseFormat.tool,
            temperature=0.2,
            use_cache=True
        ),
        persona=risk_analyst_persona,
        reward_function=backtest_reward_fn
    )
    
    # Load storage config
    storage_config = load_config_from_yaml("market_agents/memory/storage_config.yaml")
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Define backtesting environment configuration
    backtesting_env = {
        "name": "backtesting",
        "mechanism": "backtesting",
        "assets": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "margin_requirement": 0.5,
        "form_cohorts": False,
        "sub_rounds": 2,
        "group_size": 5,
        "task_prompt": "",
        "mcp_server_module": "market_agents.orchestrators.mcp_server.finance_mcp_server",
    }

    # Create the hedge fund team with backtesting environment
    hedge_fund_team = MarketAgentTeam(
        name="Quantum Backtesting Fund",
        manager=portfolio_manager,
        agents=[
            fundamental_analyst,
            technical_analyst,
            macro_analyst,
            risk_analyst
        ],
        mode="hierarchical",
        use_group_chat=False,
        shared_context={
            "investment_coverage": {
                "focus_areas": ["Technology", "AI/ML", "Cloud Computing", "Digital Platforms", "Semiconductors"],
                "investment_thesis": "Focus on market-leading tech companies with strong moats, sustainable growth, and exposure to AI transformation",
                "selection_criteria": "Companies with robust cash flows, high R&D investment, and dominant market positions",
                "strategic_approach": "Identify companies benefiting from digital transformation and AI adoption trends"
            },
            "risk_management_strategy": {
                "position_sizing": "Dynamic position sizing based on backtesting results and risk metrics",
                "sector_diversification": "Maintain sector exposure limits validated through historical performance",
                "downside_protection": "Use stop-loss levels determined by historical volatility and drawdown analysis"
            },
            "portfolio_strategy": {
                "diversification_approach": "Optimize portfolio weights using historical correlation data",
                "capital_efficiency": "Leverage historical volatility for position sizing",
                "rebalancing_discipline": "Implement systematic rebalancing based on backtested performance"
            },
            "backtesting_parameters": {
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
                "universe": tickers,
                "performance_metrics": [
                    "Sharpe Ratio",
                    "Maximum Drawdown",
                    "Win Rate",
                    "Profit Factor"
                ]
            }
        },
        environments=[backtesting_env]
    )

    return hedge_fund_team

async def run_backtest_analysis(team: MarketAgentTeam, current_date: str):
    """Run a backtesting-based investment analysis using the hedge fund team."""
    
    task = f"""
    Conduct a comprehensive backtesting analysis for the current portfolio as of {current_date}.
    
    Required Analysis Components:
    1. Historical Performance Analysis
       - Analyze historical price patterns and trends
       - Calculate key performance metrics
       - Evaluate trading signals' effectiveness
    
    2. Portfolio Analysis
       - Review current positions and weights
       - Analyze historical correlations
       - Calculate portfolio-level metrics
    
    3. Risk Analysis
       - Calculate historical VaR and drawdowns
       - Analyze position-level risk metrics
       - Evaluate stop-loss effectiveness
    
    4. Trade Recommendations
       - Generate trade signals based on historical patterns
       - Propose position sizing based on risk metrics
       - Identify optimal entry/exit points
    
    Collaboration Guidelines:
    - Each analyst should focus on their specialty area using historical data
    - Use backtesting results to validate strategies
    - Consider both individual position and portfolio-level impacts
    - Share findings through the team structure
    
    The Portfolio Manager will synthesize all analyses and backtesting results
    to make final trading decisions and portfolio adjustments.
    """
    
    result = await team.execute(task)
    return result

async def main():
    # Define backtest parameters
    tickers = ["NVDA"]
    start_date = "2023-05-01"
    end_date = "2023-07-01"
    initial_capital = 10_000_000.0
    
    # Create the backtesting hedge fund team
    team = await create_backtesting_hedge_fund_team(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    # Run backtest analysis for a specific date
    current_date = "2023-06-01"
    result = await run_backtest_analysis(team, current_date)
    
    # Print results
    print("\nBacktesting Analysis Results:")
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())