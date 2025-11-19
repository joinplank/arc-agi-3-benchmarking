"""
Batch CLI for running multiple ARC-AGI-3 games.

Usage:
    # List available games
    python -m arcagi3.cli --list-games
    
    # Run specific games
    python -m arcagi3.cli \
        --games "ls20-016295f7601e,ft09-16726c5b26ff" \
        --config gpt-4o-2024-11-20
    
    # Run all available games
    python -m arcagi3.cli \
        --all-games \
        --config claude-sonnet-4-5-20250929
"""
import sys
import os
import argparse
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main import ARC3Tester
from arcagi3.game_client import GameClient


load_dotenv()
logger = logging.getLogger(__name__)


def list_available_games(game_client: GameClient) -> List[dict]:
    """List all available games from the API"""
    try:
        games = game_client.list_games()
        return games
    except Exception as e:
        logger.error(f"Failed to list games: {e}")
        return []


def run_batch_games(
    game_ids: List[str],
    config: str,
    save_results_dir: Optional[str] = None,
    overwrite_results: bool = False,
    max_actions: int = 40,
    retry_attempts: int = 3,
    memory_word_limit: Optional[int] = None,
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
    """
    logger.info(f"Running {len(game_ids)} games with config {config}")
    
    # Create tester
    tester = ARC3Tester(
        config=config,
        save_results_dir=save_results_dir,
        overwrite_results=overwrite_results,
        max_actions=max_actions,
        retry_attempts=retry_attempts,
        memory_word_limit=memory_word_limit,
    )
    
    # Track results
    results = []
    successes = 0
    failures = 0
    
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
    logger.info(f"Skipped: {len(game_ids) - successes - failures}")
    
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


def main_cli(cli_args: Optional[list] = None):
    """Main CLI entry point for batch operations"""
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI-3 benchmarks on multiple games"
    )
    
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
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Model configuration name from models.yml (required unless --list-games)"
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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--verbose",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        help="Maximum number of words allowed in memory scratchpad (overrides model config)"
    )
    
    args = parser.parse_args(cli_args)
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        library_loggers = [
            'openai', 'httpx', 'httpcore', 'urllib3', 'requests',
            'anthropic', 'google', 'pydantic'
        ]
        for lib_logger in library_loggers:
            logging.getLogger(lib_logger).setLevel(logging.WARNING)
        logging.getLogger('arcagi3').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Initialize game client
    game_client = GameClient()
    
    # Handle --list-games
    if args.list_games:
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
        return
    
    # Require config for running games
    if not args.config:
        parser.error("--config is required unless using --list-games")
    
    # Determine which games to run
    game_ids = []
    
    if args.all_games:
        logger.info("Fetching all available games...")
        games = list_available_games(game_client)
        game_ids = [g['game_id'] for g in games]
        if not game_ids:
            logger.error("No games available")
            return
        logger.info(f"Found {len(game_ids)} games")
    elif args.games:
        game_ids = [g.strip() for g in args.games.split(',') if g.strip()]
        if not game_ids:
            parser.error("No valid game IDs provided in --games")
    else:
        parser.error("Must specify --games, --all-games, or --list-games")
    
    # Set default save directory
    if not args.save_results_dir:
        args.save_results_dir = f"results/{args.config}"
    
    # Run batch
    run_batch_games(
        game_ids=game_ids,
        config=args.config,
        save_results_dir=args.save_results_dir,
        overwrite_results=args.overwrite_results,
        max_actions=args.max_actions,
        retry_attempts=args.retry_attempts,
        memory_word_limit=args.memory_limit,
    )


if __name__ == "__main__":
    main_cli()

