
import logging
from arcagi3.checkpoint import CheckpointManager
from arcagi3.game_client import GameClient
from typing import List, Optional
from arcagi3.arc3tester import ARC3Tester

logger = logging.getLogger(__name__)

# ============================================================================
# CLI Arguments
# ============================================================================

def configure_args(parser):
    # Configuration
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
        "--retry_attempts",
        type=int,
        default=3,
        help="Number of retry attempts for API failures (default: 3)"
    )

    # Display
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Display game frames in the terminal"
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

def configure_cli_args(parser):
    # Game selection (mutually exclusive)
    game_group = parser.add_mutually_exclusive_group(required=False)
    game_group.add_argument(
        "--games",
        type=str,
        help="Comma-separated list of game IDs"
    )
    game_group.add_argument(
        "--all-games",
        action="store_true",
        help="Run all available games from the API"
    )
    game_group.add_argument(
        "--list-games",
        action="store_true",
        help="List all available games and exit"
    )

def configure_main_args(parser):
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
        "--num_concurrent_runs",
        type=int,
        default=1,
        help="Number of parallel game runs to execute (default: 1)"
    )

# ============================================================================
# CLI Configurers
# ============================================================================

def validate_args(args, parser):
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

def configure_logging(args):
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

# ============================================================================
# CLI Handlers
# ============================================================================

def list_available_games(game_client: GameClient) -> List[dict]:
    """List all available games from the API"""
    try:
        games = game_client.list_games()
        return games
    except Exception as e:
        logger.error(f"Failed to list games: {e}")
        return []

def handle_list_games(game_client: GameClient):
    games = list_available_games(game_client)
    if games:
        logger.info("\nAvailable Games:")
        logger.info("=" * 60)
        for game in games:
            logger.info(f"  {game['game_id']:<30} {game['title']}")
        logger.info("=" * 60)
        logger.info(f"Total: {len(games)} games\n")
    else:
        logger.warning("No games available or failed to fetch games.")

def handle_list_checkpoints():
    checkpoints = CheckpointManager.list_checkpoints()
    if checkpoints:
        logger.info("\nAvailable Checkpoints:")
        logger.info("=" * 80)
        for card_id in checkpoints:
            info = CheckpointManager.get_checkpoint_info(card_id)
            if info:
                logger.info(f"  Card ID: {card_id}")
                logger.info(f"    Game: {info.get('game_id', 'N/A')}")
                logger.info(f"    Config: {info.get('config', 'N/A')}")
                logger.info(f"    Actions: {info.get('action_counter', 0)}")
                logger.info(f"    Play: {info.get('current_play', 1)}/{info.get('num_plays', 1)}")
                logger.info(f"    Timestamp: {info.get('checkpoint_timestamp', 'N/A')}")
                logger.info("")
        logger.info("=" * 80)
        logger.info(f"Total: {len(checkpoints)} checkpoint(s)\n")
    else:
        logger.info("No checkpoints found.\n")

def handle_close_scorecard(args):
    card_id = args.close_scorecard
    logger.info(f"\nClosing scorecard: {card_id}")
    try:
        game_client = GameClient()
        response = game_client.close_scorecard(card_id)
        logger.info(f"✓ Successfully closed scorecard {card_id}")
        logger.info(f"Response: {response}")

        # Optionally delete local checkpoint
        checkpoint_mgr = CheckpointManager(card_id)
        if checkpoint_mgr.checkpoint_exists():
            logger.info(f"\nLocal checkpoint still exists at: .checkpoint/{card_id}")
            logger.info(f"To delete it, run: rm -rf .checkpoint/{card_id}")
    except Exception as e:
        logger.error(f"✗ Failed to close scorecard: {e}", exc_info=True)

def print_result(result):
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


def run_batch_games(
    game_ids: List[str],
    config: str,
    save_results_dir: Optional[str] = None,
    overwrite_results: bool = False,
    max_actions: int = 40,
    retry_attempts: int = 3,
    show_images: bool = False,
    memory_word_limit: Optional[int] = None,
    checkpoint_frequency: int = 1,
    close_on_exit: bool = False,
    use_vision: bool = True,
):
    """
    Run multiple games sequentially.
    
    Args:
        game_ids: List of game IDs to run
        config: Model configuration name
        save_results_dir: Directory to save results
        overwrite_results: Whether to overwrite existing results
        max_actions: Maximum actions per game
        retry_attempts: Number of retry attempts
        show_images: Whether to display game frames in the terminal
        memory_word_limit: Maximum number of words allowed in memory scratchpad
        checkpoint_frequency: Save checkpoint every N actions (default: 1, 0 to disable periodic checkpoints)
        close_on_exit: Close scorecard on exit even if game not won
        use_vision: Use vision to play the game
    """
    logger.info(f"Running {len(game_ids)} games with config {config}")
    
    # Create tester
    tester = ARC3Tester(
        config=config,
        save_results_dir=save_results_dir,
        overwrite_results=overwrite_results,
        max_actions=max_actions,
        retry_attempts=retry_attempts,
        show_images=show_images,
        memory_word_limit=memory_word_limit,
        checkpoint_frequency=checkpoint_frequency,
        close_on_exit=close_on_exit,
        use_vision=use_vision,
    )
    
    # Track results
    results = []
    successes = 0
    failures = 0
    skipped = 0

    for i, game_id in enumerate(game_ids, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Game {i}/{len(game_ids)}: {game_id}")
        logger.info(f"{'='*60}")
        
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
        
        try:
            result = tester.play_game(game_id)
            if result:
                results.append(result)
                successes += 1
                logger.info(
                    f"✓ Completed: {result.final_state}, "
                    f"Score: {result.final_score}, "
                    f"Cost: ${result.total_cost.total_cost:.4f}"
                )
            else:
                logger.info(f"⊘ Skipped (result already exists)")
        except Exception as e:
            failures += 1
            logger.error(f"✗ Failed: {e}", exc_info=True)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Batch Run Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total Games: {len(game_ids)}")
    logger.info(f"Successes: {successes}")
    logger.info(f"Failures: {failures}")
    logger.info(f"Skipped: {skipped}")
    
    if results:
        total_cost = sum(r.total_cost.total_cost for r in results)
        total_actions = sum(r.actions_taken for r in results)
        total_duration = sum(r.duration_seconds for r in results)
        
        logger.info(f"\nAggregates:")
        logger.info(f"  Total Cost: ${total_cost:.4f}")
        logger.info(f"  Total Actions: {total_actions}")
        logger.info(f"  Total Duration: {total_duration:.2f}s")
        logger.info(f"  Avg Cost per Game: ${total_cost/len(results):.4f}")
        logger.info(f"  Avg Actions per Game: {total_actions/len(results):.1f}")
    
    logger.info(f"{'='*60}\n")