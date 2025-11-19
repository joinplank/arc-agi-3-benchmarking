"""
Main CLI for running ARC-AGI-3 benchmarks on single games.

Usage:
    python main.py --game_id ls20-016295f7601e --config gpt-4o-2024-11-20
"""
import sys
import os
import argparse
import logging
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from arcagi3.agent import MultimodalAgent
from arcagi3.game_client import GameClient
from arcagi3.utils import read_models_config, save_result, generate_scorecard_tags
from arcagi3.schemas import GameResult

from arcagi3.utils.cli import configure_logging, validate_args, handle_list_checkpoints, handle_close_scorecard, configure_args, configure_main_args, print_result
from arcagi3.arc3tester import ARC3Tester

load_dotenv()
logger = logging.getLogger(__name__)

def _run_single_game(args):
    """Run a single game (sequential execution)"""
    # Create tester
    tester = ARC3Tester(
        config=args.config,
        save_results_dir=args.save_results_dir,
        overwrite_results=args.overwrite_results,
        max_actions=args.max_actions,
        retry_attempts=args.retry_attempts,
        api_retries=args.retries,
        num_plays=args.num_plays,
        show_images=args.show_images,
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
        print_result(result)

def _run_single_game_worker(args, run_number: int) -> Optional[GameResult]:
    """Worker function for concurrent game execution"""
    try:
        # Create a separate tester instance for each run
        tester = ARC3Tester(
            config=args.config,
            save_results_dir=args.save_results_dir,
            overwrite_results=args.overwrite_results,
            max_actions=args.max_actions,
            retry_attempts=args.retry_attempts,
            api_retries=args.retries,
            num_plays=args.num_plays,
            show_images=args.show_images,
            use_vision=args.use_vision,
            checkpoint_frequency=args.checkpoint_frequency,
            close_on_exit=args.close_on_exit,
            memory_word_limit=args.memory_limit,
        )
        
        logger.info(f"Starting concurrent run {run_number}/{args.num_concurrent_runs} for game {args.game_id}")
        result = tester.play_game(args.game_id)
        logger.info(f"Completed concurrent run {run_number}/{args.num_concurrent_runs}")
        return result
    except Exception as e:
        logger.error(f"Error in concurrent run {run_number}: {e}", exc_info=True)
        return None

def _run_concurrent_games(args):
    """Run multiple games concurrently"""
    logger.info(f"Running {args.num_concurrent_runs} concurrent runs for game {args.game_id}")
    
    results: List[Optional[GameResult]] = []
    
    with ThreadPoolExecutor(max_workers=args.num_concurrent_runs) as executor:
        # Submit all tasks
        future_to_run = {
            executor.submit(_run_single_game_worker, args, i+1): i+1 
            for i in range(args.num_concurrent_runs)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_run):
            run_number = future_to_run[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Concurrent run {run_number} raised exception: {e}", exc_info=True)
                results.append(None)
    
    # Filter out None results
    successful_results = [r for r in results if r is not None]
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Concurrent Runs Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total Runs: {args.num_concurrent_runs}")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(results) - len(successful_results)}")
    
    if successful_results:
        # Print individual results
        for i, result in enumerate(successful_results, 1):
            logger.info(f"\n--- Run {i} ---")
            print_result(result)
        
        # Aggregate statistics
        total_cost = sum(r.total_cost.total_cost for r in successful_results)
        total_tokens = sum(r.usage.total_tokens for r in successful_results)
        total_duration = sum(r.duration_seconds for r in successful_results)
        wins = sum(1 for r in successful_results if r.final_state == "WIN")
        avg_score = sum(r.final_score for r in successful_results) / len(successful_results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Aggregate Statistics")
        logger.info(f"{'='*60}")
        logger.info(f"Wins: {wins}/{len(successful_results)} ({wins/len(successful_results)*100:.1f}%)")
        logger.info(f"Average Score: {avg_score:.2f}")
        logger.info(f"Total Cost: ${total_cost:.4f}")
        logger.info(f"Total Tokens: {total_tokens:,}")
        logger.info(f"Total Duration: {total_duration:.2f}s")
        logger.info(f"Average Cost per Run: ${total_cost/len(successful_results):.4f}")
        logger.info(f"{'='*60}\n")
    else:
        logger.warning("No successful runs to summarize")

def main_cli(cli_args: Optional[list] = None):
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI-3 benchmark on a single game"
    )
    # Configure arguments
    configure_args(parser)
    configure_main_args(parser)
    
    # Parse arguments
    args = parser.parse_args(cli_args)

    # Configure logging
    configure_logging(args)

    # Handle --list-checkpoints
    if args.list_checkpoints:
        handle_list_checkpoints()
        return

    # Handle --close-scorecard
    if args.close_scorecard:
        handle_close_scorecard(args)
        return

    # Validate arguments
    validate_args(args, parser)
   
    # Validate concurrent runs with checkpoints
    if args.num_concurrent_runs > 1 and args.checkpoint:
        parser.error("--num_concurrent_runs > 1 is not supported with --checkpoint. Use sequential runs for checkpoint resumption.")
    
    # Set default save directory if not provided
    if not args.save_results_dir:
        args.save_results_dir = f"results/{args.config}"
    
    # Handle concurrent runs
    if args.num_concurrent_runs > 1:
        _run_concurrent_games(args)
    else:
        _run_single_game(args)
    

if __name__ == "__main__":
    main_cli()

