"""
Main CLI for running ARC-AGI-3 benchmarks on single games.

Usage:
    python main.py --game_id ls20-016295f7601e --config gpt-4o-2024-11-20
"""
import sys
import os
import argparse
import logging
import threading
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
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

# Thread-local storage for tracking which run we're in
_run_context = threading.local()

@contextmanager
def setup_per_run_logging(args, run_number: int, log_dir: str):
    """
    Context manager to set up per-run logging to a separate file.
    Uses thread-local storage to identify which run's logs should go to which file.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{run_number:02d}_{timestamp}.log")
    
    # Store run number in thread-local storage
    _run_context.run_number = run_number
    _run_context.log_file = log_file
    
    # Create file handler for this run
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG if args.verbose else getattr(logging, args.log_level.upper()))
    
    # Use the same format as the main logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Filter to only log arcagi3-related messages for this specific run
    class RunFilter(logging.Filter):
        def __init__(self, target_run_number):
            super().__init__()
            self.target_run_number = target_run_number
        
        def filter(self, record):
            # Only log if this is from the correct run's thread
            current_run = getattr(_run_context, 'run_number', None)
            if current_run != self.target_run_number:
                return False
            # Only log messages from arcagi3 modules or __main__ (prevents library noise)
            return record.name.startswith('arcagi3') or record.name == '__main__'
    
    file_handler.addFilter(RunFilter(run_number))
    
    # Add handler to root logger so it captures all arcagi3 logs
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Create a run-specific logger for explicit run messages
    run_logger = logging.getLogger(f"arcagi3.run_{run_number}")
    run_logger.setLevel(logging.DEBUG if args.verbose else getattr(logging, args.log_level.upper()))
    
    try:
        yield run_logger, log_file
    finally:
        # Clean up: remove handler and close file
        root_logger.removeHandler(file_handler)
        file_handler.close()
        # Clear thread-local storage
        if hasattr(_run_context, 'run_number'):
            delattr(_run_context, 'run_number')
        if hasattr(_run_context, 'log_file'):
            delattr(_run_context, 'log_file')

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

def _run_single_game_worker(args, run_number: int, log_dir: str) -> Tuple[Optional[GameResult], Optional[str]]:
    """
    Worker function for concurrent game execution with per-run logging.
    
    Returns:
        Tuple of (result, log_file_path)
    """
    log_file = None
    try:
        # Set up per-run logging
        with setup_per_run_logging(args, run_number, log_dir) as (run_logger, log_file_path):
            log_file = log_file_path
            
            run_logger.info(f"{'='*60}")
            run_logger.info(f"Concurrent Run {run_number}/{args.num_concurrent_runs}")
            run_logger.info(f"Game ID: {args.game_id}")
            run_logger.info(f"Config: {args.config}")
            run_logger.info(f"{'='*60}")
            
            # Also log to console (main logger) for progress tracking
            logger.info(f"[Run {run_number}/{args.num_concurrent_runs}] Starting...")
            
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
            
            run_logger.info(f"Starting game execution...")
            result = tester.play_game(args.game_id)
            
            if result:
                run_logger.info(f"{'='*60}")
                run_logger.info(f"Run {run_number} Completed Successfully")
                run_logger.info(f"Final State: {result.final_state}")
                run_logger.info(f"Final Score: {result.final_score}")
                run_logger.info(f"Actions Taken: {result.actions_taken}")
                run_logger.info(f"Duration: {result.duration_seconds:.2f}s")
                run_logger.info(f"Total Cost: ${result.total_cost.total_cost:.4f}")
                if result.scorecard_url:
                    run_logger.info(f"Scorecard URL: {result.scorecard_url}")
                run_logger.info(f"{'='*60}")
                
                # Log success to console
                logger.info(f"[Run {run_number}/{args.num_concurrent_runs}] ✓ Completed: {result.final_state}, Score: {result.final_score}")
            else:
                run_logger.warning(f"Run {run_number} completed but no result returned")
                logger.warning(f"[Run {run_number}/{args.num_concurrent_runs}] ⊘ No result returned")
            
            return result, log_file
    except Exception as e:
        # Log error to both console and file (if file logger was set up)
        logger.error(f"[Run {run_number}/{args.num_concurrent_runs}] ✗ Failed: {type(e).__name__}: {e}")
        if log_file and os.path.exists(log_file):
            # Try to log to file if it exists
            try:
                with open(log_file, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"ERROR in Run {run_number}\n")
                    f.write(f"{type(e).__name__}: {e}\n")
                    import traceback
                    f.write(traceback.format_exc())
                    f.write(f"{'='*60}\n")
            except Exception:
                pass  # If we can't write to log file, that's okay
        return None, log_file

def _run_concurrent_games(args):
    """Run multiple games concurrently with separate log files for each run"""
    # Create log directory for this concurrent run session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", args.config, f"concurrent_{args.game_id}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {args.num_concurrent_runs} concurrent runs for game {args.game_id}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"{'='*60}\n")
    
    results: List[Optional[GameResult]] = []
    log_files: List[Optional[str]] = []
    run_status: List[Tuple[int, str]] = []  # (run_number, status)
    
    with ThreadPoolExecutor(max_workers=args.num_concurrent_runs) as executor:
        # Submit all tasks
        future_to_run = {
            executor.submit(_run_single_game_worker, args, i+1, log_dir): i+1 
            for i in range(args.num_concurrent_runs)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_run):
            run_number = future_to_run[future]
            try:
                result, log_file = future.result()
                results.append(result)
                log_files.append(log_file)
                
                if result:
                    status = f"✓ Success ({result.final_state}, Score: {result.final_score})"
                else:
                    status = "⊘ No result"
                run_status.append((run_number, status))
            except Exception as e:
                logger.error(f"[Run {run_number}/{args.num_concurrent_runs}] ✗ Exception: {type(e).__name__}")
                results.append(None)
                log_files.append(None)
                run_status.append((run_number, f"✗ Failed: {type(e).__name__}"))
    
    # Filter out None results
    successful_results = [r for r in results if r is not None]
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Concurrent Runs Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total Runs: {args.num_concurrent_runs}")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(results) - len(successful_results)}")
    logger.info(f"\nLog files saved to: {log_dir}")
    logger.info(f"\nRun Status:")
    for run_num, status in sorted(run_status):
        log_file = log_files[run_num - 1] if run_num <= len(log_files) else None
        log_name = os.path.basename(log_file) if log_file else "N/A"
        logger.info(f"  Run {run_num:2d}: {status} | Log: {log_name}")
    
    if successful_results:
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
        logger.info(f"{'='*60}")
        
        # Print individual results summary (detailed logs are in files)
        logger.info(f"\nIndividual Results (detailed logs in files above):")
        for i, result in enumerate(successful_results, 1):
            logger.info(f"  Run {i}: {result.final_state} | Score: {result.final_score} | "
                       f"Cost: ${result.total_cost.total_cost:.4f} | "
                       f"Actions: {result.actions_taken}")
    else:
        logger.warning("No successful runs to summarize")
    
    logger.info(f"\nDetailed logs available in: {log_dir}\n")

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

