"""ARC-AGI-3 Benchmarking Framework"""

from .schemas import (
    # ARC-AGI-3 Game Schemas
    GameAction,
    GameState,
    GameResult,
    # Provider Schemas
    Cost,
    Usage,
    ModelConfig,
    Attempt,
    # Adapter compatibility schemas
    ARCTaskOutput,
    ARCPair,
)
from .agent import MultimodalAgent
from .game_client import GameClient

__version__ = "0.1.0"

__all__ = [
    # Game components
    "MultimodalAgent",
    "GameClient",
    # Schemas
    "GameAction",
    "GameState", 
    "GameResult",
    "Cost",
    "Usage",
    "ModelConfig",
    "Attempt",
    "ARCTaskOutput",
    "ARCPair",
]

