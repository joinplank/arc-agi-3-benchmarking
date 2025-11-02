"""
Main CLI for running ARC-AGI-3 benchmarks on single games.

Usage:
    python main.py --game_id ls20-016295f7601e --config gpt-4o-2024-11-20
"""
import sys
import os
import argparse
import logging
from typing import Optional
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arcagi3.agent import MultimodalAgent
from arcagi3.game_client import GameClient
from arcagi3.utils import read_models_config, save_result
from arcagi3.schemas import GameResult


load_dotenv()
logger = logging.getLogger(__name__)


class ARC3Tester:
    """Main tester class for running ARC-AGI-3 games"""
    
    def __init__(
        self,
        config: str,
        save_results_dir: Optional[str] = None,
        overwrite_results: bool = False,
        max_actions: int = 40,
        retry_attempts: int = 3,
        api_retries: int = 3,
        num_plays: int = 1,
    ):
        """
        Initialize the tester.
        
        Args:
            config: Model configuration name from models.yml
            save_results_dir: Directory to save results (None to skip saving)
            overwrite_results: Whether to overwrite existing results
            max_actions: Maximum actions per game
            retry_attempts: Number of retry attempts for API failures
            api_retries: Number of retry attempts for ARC-AGI-3 API calls
            num_plays: Number of times to play the game (continues session with memory)
        """
        self.config = config
        self.model_config = read_models_config(config)
        self.save_results_dir = save_results_dir
        self.overwrite_results = overwrite_results
        self.max_actions = max_actions
        self.retry_attempts = retry_attempts
        self.num_plays = num_plays
        
        # Initialize game client
        self.game_client = GameClient(max_retries=api_retries)
        
        logger.info(f"Initialized ARC3Tester with config: {config}")
        logger.info(f"Model: {self.model_config.model_name}, Provider: {self.model_config.provider}")
    
    def play_game(self, game_id: str, card_id: Optional[str] = None) -> GameResult:
        """
        Play a single game.
        
        Args:
            game_id: Game identifier
            card_id: Optional scorecard ID (generated if not provided)
            
        Returns:
            GameResult with complete game information
        """
        # Note: Results are saved with unique timestamps, so multiple runs are allowed
        # Each run creates a new file: {game_id}_{config}_{timestamp}.json
        
        scorecard_response = self.game_client.open_scorecard([game_id], card_id=card_id)
        card_id = scorecard_response.get("card_id", card_id)
        
        
        try:
            # Create agent
            agent = MultimodalAgent(
                config=self.config,
                game_client=self.game_client,
                card_id=card_id,
                max_actions=self.max_actions,
                retry_attempts=self.retry_attempts,
                num_plays=self.num_plays,
            )
            
            # Play game
            result = agent.play_game(game_id)
            
            # Save result if directory provided
            if self.save_results_dir:
                result_file = save_result(self.save_results_dir, result)
                logger.info(f"Saved result to {result_file}")
            
            return result
            
        finally:
            # Close scorecard (may already be closed by API)
            try:
                self.game_client.close_scorecard(card_id)
                logger.info(f"Closed scorecard {card_id}")
            except Exception as e:
                # Scorecard may have been auto-closed or already closed
                logger.debug(f"Could not close scorecard (may already be closed): {e}")


def main_cli(cli_args: Optional[list] = None):
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI-3 benchmark on a single game"
    )
    parser.add_argument(
        "--game_id",
        type=str,
        required=True,
        help="Game ID to play (e.g., 'ls20-016295f7601e')"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model configuration name from models.yml"
    )
    parser.add_argument(
        "--save_results_dir",
        type=str,
        default=None,
        help="Directory to save results (default: results/<config>)"
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Overwrite existing result files"
    )
    parser.add_argument(
        "--max_actions",
        type=int,
        default=40,
        help="Maximum actions per game (default: 40)"
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=3,
        help="Number of retry attempts for API failures (default: 3)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts for ARC-AGI-3 API calls (default: 3)"
    )
    parser.add_argument(
        "--num_plays",
        type=int,
        default=1,
        help="Number of times to play the game (continues session with memory on subsequent plays) (default: 1)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level for app, WARNING for libraries)"
    )
    
    args = parser.parse_args(cli_args)
    
    # Configure logging
    if args.verbose:
        # Verbose mode: Show DEBUG for our code, WARNING+ for libraries
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set library loggers to WARNING
        library_loggers = [
            'openai', 'httpx', 'httpcore', 'urllib3', 'requests',
            'anthropic', 'google', 'pydantic'
        ]
        for lib_logger in library_loggers:
            logging.getLogger(lib_logger).setLevel(logging.WARNING)
        
        # Keep our application loggers at DEBUG
        logging.getLogger('arcagi3').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
        
        logger.info("Verbose mode enabled")
    else:
        # Normal mode: Use the specified log level
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Set default save directory if not provided
    if not args.save_results_dir:
        args.save_results_dir = f"results/{args.config}"
    
    # Create tester
    tester = ARC3Tester(
        config=args.config,
        save_results_dir=args.save_results_dir,
        overwrite_results=args.overwrite_results,
        max_actions=args.max_actions,
        retry_attempts=args.retry_attempts,
        api_retries=args.retries,
        num_plays=args.num_plays,
    )
    
    # Play game
    result = tester.play_game(args.game_id)
    
    if result:
        print(f"\n{'='*60}")
        print(f"Game Result: {result.game_id}")
        print(f"{'='*60}")
        print(f"Final Score: {result.final_score}")
        print(f"Final State: {result.final_state}")
        print(f"Actions Taken: {result.actions_taken}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Total Cost: ${result.total_cost.total_cost:.4f}")
        print(f"Total Tokens: {result.usage.total_tokens}")
        print(f"\nView your scorecard online: {result.scorecard_url}")
        print(f"{'='*60}\n")
    

if __name__ == "__main__":
    main_cli()

