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
from arcagi3.checkpoint import CheckpointManager


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
    checkpoint_card_id: Optional[str] = None,
    checkpoint_frequency: int = 1,
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
        checkpoint_card_id: If provided, resume from this checkpoint
        checkpoint_frequency: Save checkpoint every N actions
    """
    logger.info(f"Running {len(game_ids)} games with config {config}")
    
    # Create tester
    tester = ARC3Tester(
        config=config,
        save_results_dir=save_results_dir,
        overwrite_results=overwrite_results,
        max_actions=max_actions,
        retry_attempts=retry_attempts,
        checkpoint_frequency=checkpoint_frequency,
    )
    
    # Track results
    results = []
    successes = 0
    failures = 0
    
    # If checkpoint provided, only run that game
    if checkpoint_card_id:
        checkpoint_mgr = CheckpointManager(checkpoint_card_id)
        if not checkpoint_mgr.checkpoint_exists():
            logger.error(f"Checkpoint not found: {checkpoint_card_id}")
            return
        
        checkpoint_info = CheckpointManager.get_checkpoint_info(checkpoint_card_id)
        game_id = checkpoint_info.get("game_id")
        logger.info(f"Resuming from checkpoint: {checkpoint_card_id}, game: {game_id}")
        
        try:
            result = tester.play_game(
                game_id,
                card_id=checkpoint_card_id,
                resume_from_checkpoint=True
            )
            if result:
                results.append(result)
                successes += 1
                logger.info(
                    f"✓ Completed: {result.final_state}, "
                    f"Score: {result.final_score}, "
                    f"Cost: ${result.total_cost.total_cost:.4f}"
                )
        except Exception as e:
            failures += 1
            logger.error(f"✗ Failed: {e}", exc_info=True)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Checkpoint Resume Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Success: {successes > 0}")
        if results:
            logger.info(f"Final Cost: ${results[0].total_cost.total_cost:.4f}")
        logger.info(f"{'='*60}\n")
        return
    
    # Otherwise, run all games normally
    for i, game_id in enumerate(game_ids, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Game {i}/{len(game_ids)}: {game_id}")
        logger.info(f"{'='*60}")
        
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
        "--checkpoint-frequency",
        type=int,
        default=1,
        help="Save checkpoint every N actions (default: 1, 0 to disable periodic checkpoints)"
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
        help="Enable verbose output"
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
            print("\nAvailable Games:")
            print("=" * 60)
            for game in games:
                print(f"  {game['game_id']:<30} {game['title']}")
            print("=" * 60)
            print(f"Total: {len(games)} games\n")
        else:
            print("No games available or failed to fetch games.")
        return
    
    # Handle checkpoint mode
    if args.checkpoint:
        checkpoint_info = CheckpointManager.get_checkpoint_info(args.checkpoint)
        if not checkpoint_info:
            print(f"Error: Checkpoint '{args.checkpoint}' not found.")
            print("Use --list-checkpoints to see available checkpoints.")
            return
        
        # Use checkpoint config if not provided
        if not args.config:
            args.config = checkpoint_info.get("config")
            print(f"Using config from checkpoint: {args.config}")
        
        # Set default save directory
        if not args.save_results_dir:
            args.save_results_dir = f"results/{args.config}"
        
        # Run with checkpoint
        run_batch_games(
            game_ids=[],  # Not needed for checkpoint resume
            config=args.config,
            save_results_dir=args.save_results_dir,
            overwrite_results=args.overwrite_results,
            max_actions=args.max_actions,
            retry_attempts=args.retry_attempts,
            checkpoint_card_id=args.checkpoint,
            checkpoint_frequency=args.checkpoint_frequency,
        )
        return
    
    # Require config for running games
    if not args.config:
        parser.error("--config is required unless using --list-games, --list-checkpoints, or --checkpoint")
    
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
        parser.error("Must specify --games, --all-games, --checkpoint, --list-games, or --list-checkpoints")
    
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
        checkpoint_frequency=args.checkpoint_frequency,
    )


if __name__ == "__main__":
    main_cli()

