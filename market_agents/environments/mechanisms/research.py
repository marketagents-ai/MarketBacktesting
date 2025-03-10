# research.py

from datetime import datetime
from importlib import import_module
import json
import random
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field

from market_agents.environments.config import EnvironmentConfig
from market_agents.environments.environment import (
    EnvironmentHistory,
    Mechanism,
    LocalAction,
    GlobalAction,
    LocalObservation,
    GlobalObservation,
    EnvironmentStep,
    ActionSpace,
    ObservationSpace,
    MultiAgentEnvironment,
    LocalEnvironmentStep,
)
import logging

logger = logging.getLogger(__name__)


class ResearchAction(LocalAction):
    action: BaseModel

    @classmethod
    def sample(cls, agent_id: str, summary_model: Type[BaseModel]) -> 'ResearchAction':
        demo_instance = summary_model.construct()
        return cls(agent_id=agent_id, action=demo_instance)


class ResearchGlobalAction(GlobalAction):
    """Global container of local ResearchActions for each agent."""
    actions: Dict[str, ResearchAction]


class ResearchObservation(BaseModel):
    """Individual observation containing research summary data"""
    current_topic: str = ""
    own_summary: Optional[BaseModel] = None
    aggregator_notes: str = ""

    def dict(self, *args, **kwargs):
        """Custom dict method to handle BaseModel serialization"""
        d = super().dict(*args, **kwargs)
        if self.own_summary:
            d['own_summary'] = self.own_summary.dict()
        return d


class ResearchLocalObservation(LocalObservation):
    """Local observation for a specific agent"""
    agent_id: str
    observation: ResearchObservation

    def dict(self, *args, **kwargs):
        """Custom dict method to handle nested observation"""
        d = super().dict(*args, **kwargs)
        if self.observation:
            d['observation'] = self.observation.dict()
        return d


class ResearchGlobalObservation(GlobalObservation):
    """Global observation containing all agent observations"""
    observations: Dict[str, ResearchLocalObservation]
    all_actions_this_round: Optional[Dict[str, Any]] = None
    final_all_summaries: Optional[List[Dict[str, Any]]] = None
    current_topic: str = ""
    aggregator_notes: str = ""

    def dict(self, *args, **kwargs):
        """Custom dict method to handle nested observations"""
        d = super().dict(*args, **kwargs)
        if self.observations:
            d['observations'] = {
                k: v.dict() for k, v in self.observations.items()
            }
        return d

    @property
    def global_obs(self) -> Optional[Any]:
        """Get the global observation for all agents."""
        return {
            'all_actions_this_round': self.all_actions_this_round,
            'final_all_summaries': self.final_all_summaries,
            'aggregator_notes': self.aggregator_notes
        }
    
class ResearchActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [ResearchAction]
    summary_model: Type[BaseModel]

    def get_action_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for whichever summary_model is used.
        """
        return self.summary_model.model_json_schema()

    def sample(self, agent_id: str) -> LocalAction:
        """
        For debugging or random testing; create a sample instance of the
        user-defined summary model.
        """
        return ResearchAction.sample(agent_id, self.summary_model)


class ResearchObservationSpace(ObservationSpace):
    """
    Observations revolve around local/global research data.
    """
    allowed_observations: List[Type[LocalObservation]] = [ResearchLocalObservation]


class ResearchMechanism(Mechanism):
    """Mechanism that manages research rounds and agent summaries."""
    sequential: bool = Field(
        default=False,
        description="Whether the mechanism is sequential (one agent at a time)."
    )
    max_rounds: int = Field(default=3, description="Number of research steps allowed.")
    current_round: int = Field(default=0, description="Current step or round.")
    current_topic: str = Field(default="", description="Current research topic")
    round_summaries: List[Dict[str, Any]] = Field(default_factory=list, description="History of all round summaries")
    last_step: Optional[EnvironmentStep] = Field(default=None, description="Last environment step")
    summary_model: Type[BaseModel] = Field(default=BaseModel, description="Model for validating research summaries")

    def step(self, action: Union[ResearchAction, ResearchGlobalAction, Dict[str, Any]]) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Process agent actions and return the next state with observations."""
        logger.debug(f"ResearchMechanism step: {action}")
        self.current_round += 1
        done = (self.current_round >= self.max_rounds)

        if self.sequential:
            if isinstance(action, dict):
                try:
                    action = ResearchAction.parse_obj(action)
                except Exception as e:
                    logger.error(f"Failed to parse dict into ResearchAction: {e}")
                    raise
            if not isinstance(action, ResearchAction):
                raise TypeError(f"Expected ResearchAction, got {type(action).__name__}")

            # Store in round_summaries
            round_actions = {action.agent_id: action.action.dict()}
            self.round_summaries.append(round_actions)

            # Build local observation
            obs = ResearchObservation(
                current_topic=self.current_topic,
                own_summary=action.action,
                aggregator_notes=f"Round {self.current_round}, single-agent action."
            )
            local_obs = ResearchLocalObservation(
                agent_id=action.agent_id,
                observation=obs
            )

            local_step = LocalEnvironmentStep(
                observation=local_obs,
                reward=0.0,
                done=done,
                info={
                    "round": self.current_round,
                    "note": "Sequential step",
                    "agent_rewards": {action.agent_id: 1.0},
                    "current_topic": self.current_topic
                }
            )
            self.last_step = local_step
            return local_step

        else:
            if isinstance(action, GlobalAction):
                research_actions = {}
                for agent_id, local_action in action.actions.items():
                    action_content = local_action.action
                    if not isinstance(action_content, BaseModel):
                        action_content = self.summary_model.model_validate(action_content)
                    
                    research_actions[agent_id] = ResearchAction(
                        agent_id=agent_id,
                        action=action_content
                    )
                action = ResearchGlobalAction(actions=research_actions)
            elif isinstance(action, dict):
                try:
                    action = ResearchGlobalAction.parse_obj(action)
                except Exception as e:
                    logger.error(f"Failed to parse dict into ResearchGlobalAction: {e}")
                    raise
            elif not isinstance(action, ResearchGlobalAction):
                raise TypeError(f"Expected ResearchGlobalAction or GlobalAction, got {type(action).__name__}")

            # Rest of batch mode remains the same
            round_actions = {
                agent_id: local_action.action.dict()
                for agent_id, local_action in action.actions.items()
            }
            self.round_summaries.append(round_actions)

            # Build local observations for each agent
            local_observations: Dict[str, ResearchLocalObservation] = {}
            agent_rewards: Dict[str, float] = {}
            
            for agent_id, local_action in action.actions.items():
                local_obs = ResearchLocalObservation(
                    agent_id=agent_id,
                    observation=ResearchObservation(
                        current_topic=self.current_topic,
                        own_summary=local_action.action,
                        aggregator_notes=f"Round {self.current_round}, agent {agent_id}"
                    )
                )
                local_observations[agent_id] = local_obs
                agent_rewards[agent_id] = 1.0

            # Create global observation
            global_obs = ResearchGlobalObservation(
                observations=local_observations,
                all_actions_this_round=round_actions,  # Current round summaries
                current_topic=self.current_topic,
                aggregator_notes=f"End of round {self.current_round}"
            )

            # Add historical data only on final round
            if done:
                global_obs.final_all_summaries = [
                    {"round_index": i + 1, "summaries": summaries}
                    for i, summaries in enumerate(self.round_summaries)
                ]
                global_obs.aggregator_notes += " (Final) returning all collected summaries."

            env_step = EnvironmentStep(
                global_observation=global_obs,
                done=done,
                info={
                    "round": self.current_round,
                    "current_topic": self.current_topic,
                    "agent_rewards": agent_rewards
                }
            )
            self.last_step = env_step
            return env_step

    def get_global_state(self) -> Dict[str, Any]:
        """Return the mechanism's overall state, including the current topic and summaries."""
        return {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "round_summaries": self.round_summaries,
            "round_summaries_count": len(self.round_summaries),
            "last_step": self.last_step.dict() if self.last_step else None,
            "current_topic": self.current_topic
        }

    def reset(self) -> None:
        """Reset mechanism for a new run."""
        self.current_round = 0
        self.round_summaries.clear()
        self.last_step = None
        logger.info("ResearchMechanism reset complete.")

class ResearchEnvironmentConfig(EnvironmentConfig):
    """Configuration for research environment orchestration"""
    name: str = Field(
        default="research",
        description="Name of the research environment"
    )
    api_url: str = Field(
        default="http://localhost:8003",
        description="API endpoint for research environment"
    )
    sub_rounds: int = Field(
        default=2,
        description="Number of sub-rounds within each main round"
    )
    initial_topic: str = Field(
        default="Market Analysis",
        description="Initial research topic"
    )
    group_size: int = Field(
        default=4,
        description="Number of agents in research group"
    )
    schema_model: str = Field(
        default="LiteraryAnalysis",
        description="Name of Pydantic model defining research output schema"
    )

    model_config = {"extra": "ignore"}

class ResearchEnvironment(MultiAgentEnvironment):
    """
    Multi-agent environment that orchestrates a research session.
    It references a ResearchMechanism that collects agent actions.
    """
    name: str = Field(default="research", description="Name of the environment")
    action_space: ResearchActionSpace = Field(
        default_factory=lambda: ResearchActionSpace(summary_model=BaseModel),
        description="Defines the Pydantic model for agent's research summaries"
    )
    observation_space: ResearchObservationSpace = Field(
        default_factory=ResearchObservationSpace,
        description="Observation space"
    )
    mechanism: ResearchMechanism = Field(default_factory=ResearchMechanism)
    initial_topic: Optional[str] = Field(default=None, description="Initial research topic")

    def __init__(self, **config):
        """Initialize environment with config parameters."""
        try:
            # Parse and validate config
            env_config = ResearchEnvironmentConfig(**config)
            
            # Get the schema model
            summary_model = self._get_schema_model(env_config.schema_model)
            
            # Initialize action space with the schema model
            action_space = ResearchActionSpace(summary_model=summary_model)
            
            # Initialize mechanism with relevant config
            mechanism = ResearchMechanism(
                max_rounds=env_config.sub_rounds,
                initial_topic=env_config.initial_topic,
                summary_model=summary_model
            )

            # Initialize parent class with processed config
            super().__init__(
                name=env_config.name,
                action_space=action_space,
                observation_space=ResearchObservationSpace(),
                mechanism=mechanism,
                initial_topic=env_config.initial_topic
            )
            self._global_state: Dict[str, Any] = {}
            
        except Exception as e:
            raise ValueError(f"Failed to initialize ResearchEnvironment: {e}")

    def _get_schema_model(self, schema_name: str) -> Type[BaseModel]:
        """Dynamically import and return the schema model class."""
        try:
            # Update the import path to match your project structure
            schemas_module = import_module('market_agents.orchestrators.research_schemas')
            
            if not hasattr(schemas_module, schema_name):
                raise ValueError(f"Schema model '{schema_name}' not found in research_schemas")
                
            model_class = getattr(schemas_module, schema_name)
            
            if not issubclass(model_class, BaseModel):
                raise ValueError(f"Schema model {schema_name} must be a Pydantic model")
                
            return model_class
        except ImportError as e:
            raise ValueError(f"Could not import research_schemas module: {e}")
        except Exception as e:
            raise ValueError(f"Could not load schema model '{schema_name}': {e}")

    def get_global_state(self) -> Any:
        """Expose both mechanism state and our internal state."""
        mechanism_state = self.mechanism.get_global_state()
        return {
            **self._global_state,
            **mechanism_state,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "current_topic": self.mechanism.current_topic
        }

    def reset(self) -> GlobalObservation:
        """Override reset to handle our own state management"""
        self.current_step = 0
        self._global_state = {}
        self.history = EnvironmentHistory()
        self.mechanism.reset()
        
        if self.initial_topic:
            self.mechanism.current_topic = self.initial_topic
            
        return GlobalObservation(observations={})