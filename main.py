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
from arcagi3.utils import read_models_config, save_result, generate_scorecard_tags
from arcagi3.schemas import GameResult
from arcagi3.checkpoint import CheckpointManager


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
        use_vision: bool = True,
        checkpoint_frequency: int = 1,
        close_on_exit: bool = False,
        memory_word_limit: Optional[int] = None,
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
            use_vision: Whether to use vision (images) or text-only mode
            checkpoint_frequency: Save checkpoint every N actions (default: 1, 0 to disable)
            close_on_exit: Close scorecard on exit even if not won (prevents checkpoint resume)
            memory_word_limit: Maximum number of words allowed in memory scratchpad (overrides config/default)
        """
        self.config = config
        self.model_config = read_models_config(config)
        self.save_results_dir = save_results_dir
        self.overwrite_results = overwrite_results
        self.max_actions = max_actions
        self.retry_attempts = retry_attempts
        self.num_plays = num_plays
        self.use_vision = use_vision
        self.checkpoint_frequency = checkpoint_frequency
        self.close_on_exit = close_on_exit
        
        # Determine memory limit: CLI > Config > Default (500)
        if memory_word_limit is not None:
            self.memory_word_limit = memory_word_limit
        else:
            # Check if defined in model config kwargs
            self.memory_word_limit = self.model_config.kwargs.get("memory_word_limit", 500)
            
        # Initialize game client
        self.game_client = GameClient(max_retries=api_retries)
        
        logger.info(f"Initialized ARC3Tester with config: {config}")
        logger.info(f"Model: {self.model_config.model_name}, Provider: {self.model_config.provider}")
    
    def play_game(self, game_id: str, card_id: Optional[str] = None, resume_from_checkpoint: bool = False) -> GameResult:
        """
        Play a single game.

        Args:
            game_id: Game identifier
            card_id: Optional scorecard ID (generated if not provided, or loaded from checkpoint)
            resume_from_checkpoint: If True, resume from existing checkpoint

        Returns:
            GameResult with complete game information
        """
        # If resuming from checkpoint, try to load the card_id and game_id
        checkpoint_card_id = None  # Track original checkpoint card_id
        if resume_from_checkpoint and card_id:
            checkpoint_mgr = CheckpointManager(card_id)
            if checkpoint_mgr.checkpoint_exists():
                checkpoint_info = CheckpointManager.get_checkpoint_info(card_id)
                if checkpoint_info:
                    game_id = checkpoint_info.get("game_id", game_id)
                    checkpoint_card_id = card_id  # Preserve original checkpoint card_id
                    logger.info(f"Resuming from checkpoint: card_id={card_id}, game_id={game_id}")
            else:
                logger.warning(f"No checkpoint found for card_id={card_id}, starting fresh")
                resume_from_checkpoint = False

        # Note: Results are saved with unique timestamps, so multiple runs are allowed
        # Each run creates a new file: {game_id}_{config}_{timestamp}.json

        # Check if scorecard still exists on server when resuming
        if resume_from_checkpoint and card_id:
            try:
                # Try to get the scorecard to verify it still exists
                self.game_client.get_scorecard(card_id)
                logger.info(f"Verified existing scorecard: {card_id}")
            except Exception as e:
                logger.warning(f"Scorecard {card_id} no longer exists on server: {e}")
                logger.info("Opening new scorecard for checkpoint recovery...")
                # Open a new scorecard with the same card_id (API will reject, so we get a new one)
                tags = generate_scorecard_tags(self.model_config)
                scorecard_response = self.game_client.open_scorecard([game_id], card_id=None, tags=tags)
                old_card_id = card_id
                card_id = scorecard_response.get("card_id", card_id)
                logger.info(f"Created new scorecard: {card_id} (old: {old_card_id})")
                logger.info(f"Checkpoint will continue using original card_id: {checkpoint_card_id}")
                # Note: We keep resume_from_checkpoint=True to restore agent state,
                # but the game will need to reset since the old session is gone
        else:
            # Generate tags from model config for scorecard tracking
            tags = generate_scorecard_tags(self.model_config)
            scorecard_response = self.game_client.open_scorecard([game_id], card_id=card_id, tags=tags)
            card_id = scorecard_response.get("card_id", card_id)
        
        
        try:
            from arcagi3.utils import load_hints, find_hints_file
            
            hints_file = find_hints_file()
            hint_found = False
            if hints_file:
                hints = load_hints(hints_file, game_id=game_id)
                hint_found = game_id in hints
            
            if hint_found:
                logger.info(f"✓ Hint found for game {game_id}")
            else:
                logger.debug(f"⊘ No hint found for game {game_id}")
            
            # Create agent
            # Use checkpoint_card_id for checkpoint management if resuming, otherwise use card_id
            agent = MultimodalAgent(
                config=self.config,
                game_client=self.game_client,
                card_id=card_id,
                max_actions=self.max_actions,
                retry_attempts=self.retry_attempts,
                num_plays=self.num_plays,
                use_vision=self.use_vision,
                checkpoint_frequency=self.checkpoint_frequency,
                checkpoint_card_id=checkpoint_card_id,
                memory_word_limit=self.memory_word_limit,
            )

            # Play game (with checkpoint support)
            result = agent.play_game(game_id, resume_from_checkpoint=resume_from_checkpoint)
            
            # Save result if directory provided
            if self.save_results_dir:
                result_file = save_result(self.save_results_dir, result)
                logger.info(f"Saved result to {result_file}")
            
            # Determine the actual checkpoint card_id for logging
            actual_checkpoint_id = checkpoint_card_id if checkpoint_card_id else card_id

            # Only close scorecard on WIN (checkpoint is deleted anyway)
            # or if user explicitly requested close_on_exit
            # Otherwise, keep scorecard open for checkpoint resume
            if result.final_state == "WIN" or self.close_on_exit:
                try:
                    self.game_client.close_scorecard(card_id)
                    logger.info(f"Closed scorecard {card_id}")
                except Exception as e:
                    logger.debug(f"Could not close scorecard: {e}")
            else:
                logger.info(f"Scorecard {card_id} left open for potential checkpoint resume")
                logger.info(f"Checkpoint saved at: .checkpoint/{actual_checkpoint_id}")

            return result

        except KeyboardInterrupt:
            # Determine the actual checkpoint card_id for logging
            actual_checkpoint_id = checkpoint_card_id if checkpoint_card_id else card_id
            logger.info("Interrupted by user (Ctrl-C)")
            logger.info(f"Checkpoint saved at: .checkpoint/{actual_checkpoint_id}")
            logger.info(f"Resume with: python main.py --checkpoint {actual_checkpoint_id}")
            raise
        except Exception as e:
            # Determine the actual checkpoint card_id for logging
            actual_checkpoint_id = checkpoint_card_id if checkpoint_card_id else card_id
            logger.error(f"Error during game execution: {e}")
            logger.info(f"Checkpoint may be available at: .checkpoint/{actual_checkpoint_id}")
            raise


def main_cli(cli_args: Optional[list] = None):
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI-3 benchmark on a single game"
    )

    # Checkpoint options
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint",
        type=str,
        metavar="CARD_ID",
        help="Resume from existing checkpoint using the specified scorecard ID"
    )
    checkpoint_group.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List all available checkpoints and exit"
    )
    checkpoint_group.add_argument(
        "--close-scorecard",
        type=str,
        metavar="CARD_ID",
        help="Close a scorecard by ID and exit"
    )

    parser.add_argument(
        "--game_id",
        type=str,
        help="Game ID to play (e.g., 'ls20-016295f7601e'). Not required when using --checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Model configuration name from models.yml. Not required when using --checkpoint."
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
        "--use_vision",
        action="store_true",
        help="Use vision to play the game (default: True)"
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=1,
        help="Save checkpoint every N actions (default: 1, 0 to disable periodic checkpoints)"
    )
    parser.add_argument(
        "--close-on-exit",
        action="store_true",
        help="Close scorecard on exit even if game not won (prevents checkpoint resume)"
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
    parser.add_argument(
        "--memory-limit",
        type=int,
        help="Maximum number of words allowed in memory scratchpad (overrides model config)"
    )
    
    args = parser.parse_args(cli_args)

    # Handle --list-checkpoints
    if args.list_checkpoints:
        checkpoints = CheckpointManager.list_checkpoints()
        if checkpoints:
            print("\nAvailable Checkpoints:")
            print("=" * 80)
            for card_id in checkpoints:
                info = CheckpointManager.get_checkpoint_info(card_id)
                if info:
                    print(f"  Card ID: {card_id}")
                    print(f"    Game: {info.get('game_id', 'N/A')}")
                    print(f"    Config: {info.get('config', 'N/A')}")
                    print(f"    Actions: {info.get('action_counter', 0)}")
                    print(f"    Play: {info.get('current_play', 1)}/{info.get('num_plays', 1)}")
                    print(f"    Timestamp: {info.get('checkpoint_timestamp', 'N/A')}")
                    print()
            print("=" * 80)
            print(f"Total: {len(checkpoints)} checkpoint(s)\n")
        else:
            print("No checkpoints found.\n")
        return

    # Handle --close-scorecard
    if args.close_scorecard:
        card_id = args.close_scorecard
        print(f"\nClosing scorecard: {card_id}")
        try:
            game_client = GameClient()
            response = game_client.close_scorecard(card_id)
            print(f"✓ Successfully closed scorecard {card_id}")
            print(f"Response: {response}")

            # Optionally delete local checkpoint
            checkpoint_mgr = CheckpointManager(card_id)
            if checkpoint_mgr.checkpoint_exists():
                print(f"\nLocal checkpoint still exists at: .checkpoint/{card_id}")
                print(f"To delete it, run: rm -rf .checkpoint/{card_id}")
        except Exception as e:
            print(f"✗ Failed to close scorecard: {e}")
            import traceback
            traceback.print_exc()
        return

    # Validate arguments
    if args.checkpoint:
        # When resuming from checkpoint, config and game_id are optional (loaded from checkpoint)
        checkpoint_info = CheckpointManager.get_checkpoint_info(args.checkpoint)
        if not checkpoint_info:
            print(f"Error: Checkpoint '{args.checkpoint}' not found.")
            print("Use --list-checkpoints to see available checkpoints.")
            return

        # Use checkpoint values if not provided
        if not args.config:
            args.config = checkpoint_info.get("config")
            print(f"Using config from checkpoint: {args.config}")
        if not args.game_id:
            args.game_id = checkpoint_info.get("game_id")
            print(f"Using game_id from checkpoint: {args.game_id}")
    else:
        # When not using checkpoint, both are required
        if not args.game_id or not args.config:
            parser.error("--game_id and --config are required unless using --checkpoint")

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
        use_vision=args.use_vision,
        checkpoint_frequency=args.checkpoint_frequency,
        close_on_exit=args.close_on_exit,
        memory_word_limit=args.memory_limit,
    )

    # Play game (with checkpoint support)
    card_id = args.checkpoint if args.checkpoint else None
    resume_from_checkpoint = bool(args.checkpoint)
    result = tester.play_game(
        args.game_id,
        card_id=card_id,
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    if result:
        logger.info(f"\n{'='*60}")
        logger.info(f"Game Result: {result.game_id}")
        logger.info(f"{'='*60}")
        logger.info(f"Final Score: {result.final_score}")
        logger.info(f"Final State: {result.final_state}")
        logger.info(f"Actions Taken: {result.actions_taken}")
        logger.info(f"Duration: {result.duration_seconds:.2f}s")
        logger.info(f"Total Cost: ${result.total_cost.total_cost:.4f}")
        logger.info(f"Total Tokens: {result.usage.total_tokens}")
        logger.info(f"\nView your scorecard online: {result.scorecard_url}")
        logger.info(f"{'='*60}\n")
    

if __name__ == "__main__":
    main_cli()

