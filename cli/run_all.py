"""
Parallel execution orchestrator for ARC-AGI-3 benchmarking.

Usage:
    python cli/run_all.py \
        --game_list_file games.txt \
        --model_configs "gpt-4o-2024-11-20,claude_opus" \
        --max_actions 40
"""
import asyncio
import os
import sys
import argparse
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timezone

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.insert(0, project_root)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from main import ARC3Tester
from arcagi3.utils import (
    read_models_config,
    read_provider_rate_limits,
    save_result_in_timestamped_structure,
    generate_execution_map,
    generate_summary,
    AsyncRequestRateLimiter,
)

logger = logging.getLogger(__name__)

# Default values
DEFAULT_RATE_LIMIT_RATE = 400
DEFAULT_RATE_LIMIT_PERIOD = 60

# Default model configurations to test if not provided via CLI
DEFAULT_MODEL_CONFIGS_TO_TEST: List[str] = [
    "gpt-4o-mini-2024-07-18",
]

# --- Globals for Orchestrator ---
PROVIDER_RATE_LIMITERS: Dict[str, AsyncRequestRateLimiter] = {}
MODEL_CONFIG_CACHE: Dict[str, Any] = {}


def get_model_config(config_name: str):
    """Get model config from cache or load it"""
    if config_name not in MODEL_CONFIG_CACHE:
        MODEL_CONFIG_CACHE[config_name] = read_models_config(config_name)
    return MODEL_CONFIG_CACHE[config_name]


def get_or_create_rate_limiter(provider_name: str, all_provider_limits: Dict) -> AsyncRequestRateLimiter:
    """Get or create rate limiter for a provider"""
    if provider_name not in PROVIDER_RATE_LIMITERS:
        if provider_name not in all_provider_limits:
            logger.warning(
                f"No rate limit configuration found for provider '{provider_name}' in provider_config.yml. "
                f"Using default ({DEFAULT_RATE_LIMIT_RATE} req/{DEFAULT_RATE_LIMIT_PERIOD}s)."
            )
            default_config_rate = DEFAULT_RATE_LIMIT_RATE
            default_config_period = DEFAULT_RATE_LIMIT_PERIOD
            actual_rate_for_limiter = default_config_rate / default_config_period
            actual_capacity_for_limiter = max(1.0, actual_rate_for_limiter)
        else:
            limits = all_provider_limits[provider_name]
            config_rate = limits['rate']
            config_period = limits['period']
            if config_period <= 0:
                actual_rate_for_limiter = float('inf')
                actual_capacity_for_limiter = float('inf')
                logger.warning(
                    f"Provider '{provider_name}' has period <= 0 in config. Treating as unconstrained."
                )
            else:
                calculated_rps = config_rate / config_period
                actual_rate_for_limiter = calculated_rps
                actual_capacity_for_limiter = max(1.0, calculated_rps)
        logger.info(
            f"Initializing rate limiter for provider '{provider_name}' with "
            f"rate={actual_rate_for_limiter:.2f} req/s, capacity={actual_capacity_for_limiter:.2f}."
        )
        PROVIDER_RATE_LIMITERS[provider_name] = AsyncRequestRateLimiter(
            rate=actual_rate_for_limiter, capacity=actual_capacity_for_limiter
        )
    return PROVIDER_RATE_LIMITERS[provider_name]


async def run_single_game_wrapper(
    config_name: str,
    game_id: str,
    limiter: AsyncRequestRateLimiter,
    timestamp_dir: str,
    overwrite_results: bool,
    max_actions: int,
    retry_attempts: int,
) -> bool:
    """
    Wrapper to run a single game with rate limiting and retry logic.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.debug(f"[Orchestrator] Queuing game: {game_id}, config: {config_name}")

    def _synchronous_game_execution():
        """Synchronous game execution (to be run in thread pool)"""
        logger.debug(f"[Thread-{game_id}-{config_name}] Spawning ARC3Tester (Executing attempt)...")
        tester = ARC3Tester(
            config=config_name,
            save_results_dir=None,  # We'll save manually using timestamped structure
            overwrite_results=overwrite_results,
            max_actions=max_actions,
            retry_attempts=retry_attempts,
        )
        logger.debug(f"[Thread-{game_id}-{config_name}] Starting play_game...")
        
        # Play the game
        result = tester.play_game(game_id)
        
        # Save result in timestamped structure
        if result:
            result_file = save_result_in_timestamped_structure(timestamp_dir, result)
            logger.info(f"[Thread-{game_id}-{config_name}] Result saved to {result_file}")
        
        logger.debug(f"[Thread-{game_id}-{config_name}] Game execution completed successfully.")
        return result is not None

    try:
        async with limiter:
            logger.debug(
                f"[Orchestrator] Rate limiter acquired for: {config_name}. "
                f"Executing game {game_id}..."
            )
            success = await asyncio.to_thread(_synchronous_game_execution)
        
        if success:
            logger.info(f"✓ {config_name} / {game_id}")
        else:
            logger.warning(f"✗ {config_name} / {game_id} - No result")
        
        return success
    except Exception as e:
        logger.error(
            f"[Orchestrator] Failed to process: {config_name} / {game_id}. "
            f"Error: {type(e).__name__} - {e}",
            exc_info=True,
        )
        return False


async def main(
    game_list_file: Optional[str],
    model_configs_to_test: List[str],
    results_root: str,
    overwrite_results: bool,
    max_actions: int,
    retry_attempts: int,
) -> int:
    """
    Main orchestrator function.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    start_time = time.perf_counter()
    logger.info("Starting ARC-AGI-3 Test Orchestrator...")
    logger.info(f"Testing with model configurations: {model_configs_to_test}")

    # Create timestamped directory
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamp_dir = os.path.join(results_root, timestamp_str)
    os.makedirs(timestamp_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {timestamp_dir}")

    # Load game IDs
    game_ids: List[str] = []
    try:
        if game_list_file:
            logger.info(f"Using game list file: {game_list_file}")
            with open(game_list_file, 'r') as f:
                game_ids = [line.strip() for line in f if line.strip()]
            if not game_ids:
                logger.error(f"No game IDs found in {game_list_file}. Exiting.")
                return 1
            logger.info(f"Loaded {len(game_ids)} game IDs from {game_list_file}.")
        else:
            logger.error("No game list file provided. Use --game_list_file or provide game IDs.")
            return 1

    except FileNotFoundError:
        logger.error(f"Game list file not found: {game_list_file}. Exiting.")
        return 1
    except Exception as e:
        logger.error(f"Error loading games: {e}", exc_info=True)
        return 1

    # Create all job combinations
    all_jobs_to_run: List[Tuple[str, str]] = []
    for config_name in model_configs_to_test:
        for game_id in game_ids:
            all_jobs_to_run.append((config_name, game_id))

    if not all_jobs_to_run:
        logger.warning("No jobs to run (check model_configs_to_test and game list). Exiting.")
        return 1

    logger.info(f"Total jobs to process: {len(all_jobs_to_run)}")

    # Load provider rate limits
    try:
        all_provider_limits = read_provider_rate_limits()
        logger.info(
            f"Loaded rate limits from provider_config.yml for providers: {list(all_provider_limits.keys())}"
        )
    except FileNotFoundError:
        logger.warning(
            f"provider_config.yml not found. Using default rate limits "
            f"({DEFAULT_RATE_LIMIT_RATE} req/{DEFAULT_RATE_LIMIT_PERIOD}s per provider)."
        )
        all_provider_limits = {}
    except Exception as e:
        logger.warning(f"Error reading or parsing provider_config.yml: {e}. Using default rate limits.")
        all_provider_limits = {}

    # Create async tasks
    async_tasks_to_execute = []
    for config_name, game_id in all_jobs_to_run:
        try:
            model_config_obj = get_model_config(config_name)
            provider_name = model_config_obj.provider
            limiter = get_or_create_rate_limiter(provider_name, all_provider_limits)
            async_tasks_to_execute.append(
                run_single_game_wrapper(
                    config_name,
                    game_id,
                    limiter,
                    timestamp_dir,
                    overwrite_results,
                    max_actions,
                    retry_attempts,
                )
            )
        except ValueError as e:
            logger.error(
                f"Skipping config '{config_name}' for game '{game_id}' due to model config error: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error setting up task for '{config_name}', '{game_id}': {e}",
                exc_info=True,
            )

    if not async_tasks_to_execute:
        logger.warning("No tasks could be prepared for execution. Exiting.")
        return 1

    print(f"\nRunning {len(async_tasks_to_execute)} game executions in parallel...\n")
    logger.debug(f"Executing {len(async_tasks_to_execute)} tasks concurrently...")
    results = await asyncio.gather(*async_tasks_to_execute, return_exceptions=True)

    successful_runs = sum(1 for r in results if r is True)
    orchestrator_level_failures = sum(1 for r in results if r is False or isinstance(r, Exception))

    # Generate execution map and summary first
    logger.debug("Generating execution map and summary...")
    summary = None
    try:
        execution_map = generate_execution_map(timestamp_dir)
        execution_map_file = os.path.join(timestamp_dir, "execution_map.json")
        import json
        with open(execution_map_file, 'w') as f:
            json.dump(execution_map, f, indent=2)
        logger.debug(f"Execution map saved to {execution_map_file}")

        summary = generate_summary(timestamp_dir)
        summary_file = os.path.join(timestamp_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.debug(f"Summary saved to {summary_file}")
    except Exception as e:
        logger.error(f"Error generating execution map or summary: {e}", exc_info=True)

    end_time = time.perf_counter()
    total_duration = end_time - start_time

    # Print beautiful summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY".center(80))
    print("="*80)
    
    # Basic info
    print(f"\nExecution Info:")
    print(f"   • Started: {summary.get('execution_start', 'N/A') if summary else 'N/A'}")
    print(f"   • Duration: {total_duration:.2f}s")
    print(f"   • Models tested: {', '.join(model_configs_to_test)}")
    print(f"   • Games: {summary.get('total_games', 0) if summary else 0}")
    print(f"   • Total executions: {successful_runs}/{len(results)}")
    
    # Results by model
    if summary and summary.get('stats_by_model'):
        print(f"\nResults by Model:")
        for model, stats in summary['stats_by_model'].items():
            win_rate = stats.get('win_rate', 0) * 100
            total_games = stats.get('total_games', 0)
            wins = stats.get('wins', 0)
            cost = stats.get('total_cost', 0)
            avg_score = stats.get('avg_score', 0)
            
            print(f"\n   {model}:")
            print(f"      Games: {total_games}  |  Wins: {wins}  |  Win Rate: {win_rate:.1f}%")
            print(f"      Avg Score: {avg_score:.0f}  |  Cost: ${cost:.4f}")
    
    # Overall stats
    if summary and summary.get('overall_stats'):
        overall = summary['overall_stats']
        print(f"\nOverall Stats:")
        print(f"   • Total Cost: ${overall.get('total_cost', 0):.4f}")
        print(f"   • Total Tokens: {overall.get('total_tokens', 0):,}")
        print(f"   • Wins: {overall.get('wins', 0)}")
        print(f"   • Game Overs: {overall.get('game_overs', 0)}")
        print(f"   • Avg Score: {overall.get('avg_score', 0):.0f}")
    
    # Failures
    exit_code = 0
    if orchestrator_level_failures > 0:
        print(f"\nFailures: {orchestrator_level_failures}")
        for i, res in enumerate(results):
            if res is False or isinstance(res, Exception):
                original_job_config, original_job_game_id = all_jobs_to_run[i]
                if isinstance(res, Exception):
                    print(f"   • {original_job_config}/{original_job_game_id}: {type(res).__name__}")
                else:
                    print(f"   • {original_job_config}/{original_job_game_id}: Failed")
        exit_code = 1
    else:
        print(f"\nAll executions completed successfully!")
    
    print(f"\nResults saved to: {timestamp_dir}")
    print("="*80 + "\n")

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI-3 games concurrently. Games can be specified via a game list file."
    )
    parser.add_argument(
        "--game_list_file",
        type=str,
        default=None,
        required=False,
        help="Path to a .txt file containing game IDs, one per line. Required if not using --game_ids.",
    )
    parser.add_argument(
        "--game_ids",
        type=str,
        default=None,
        required=False,
        help="Comma-separated list of game IDs (e.g., 'ls20-016295f7601e,ls20-fa137e247ce6'). "
        "Alternative to --game_list_file.",
    )
    parser.add_argument(
        "--model_configs",
        type=str,
        default=",".join(DEFAULT_MODEL_CONFIGS_TO_TEST),
        help=f"Comma-separated list of model configuration names to test (from models.yml). "
        f"Defaults to: {','.join(DEFAULT_MODEL_CONFIGS_TO_TEST)}",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root folder name to save results under. A timestamped subdirectory will be created. "
        "Defaults to 'results'",
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Overwrite results if they already exist. Defaults to False",
    )
    parser.add_argument(
        "--max_actions",
        type=int,
        default=40,
        help="Maximum actions per game (default: 40)",
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=3,
        help="Number of retry attempts for API failures (default: 3)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
        help="Set the logging level (default: INFO). Use NONE to disable logging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (shows debug info for arcagi3 only, keeps libraries quiet)",
    )

    args = parser.parse_args()

    # Configure logging
    if args.log_level == "NONE":
        log_level_to_set = logging.CRITICAL + 1
        logging.basicConfig(
            level=log_level_to_set,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    elif args.verbose:
        # Verbose mode: Show DEBUG for our code, WARNING+ for libraries
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

        # Set library loggers to WARNING
        library_loggers = [
            'openai', 'httpx', 'httpcore', 'urllib3', 'requests',
            'anthropic', 'google', 'pydantic', 'transformers',
        ]
        for lib_logger in library_loggers:
            logging.getLogger(lib_logger).setLevel(logging.WARNING)

        # Keep our application loggers at DEBUG
        logging.getLogger('arcagi3').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
        logger.info("Verbose mode enabled - showing debug output for arcagi3 only")
    else:
        log_level_to_set = getattr(logging, args.log_level.upper())
        logging.basicConfig(
            level=log_level_to_set,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    # Parse game IDs
    game_ids_list = []
    if args.game_ids:
        game_ids_list = [g.strip() for g in args.game_ids.split(',') if g.strip()]
    elif args.game_list_file:
        # Will be loaded in main()
        pass
    else:
        logger.error("Either --game_list_file or --game_ids must be provided.")
        sys.exit(1)

    # Parse model configs
    model_configs_list = [m.strip() for m in args.model_configs.split(',') if m.strip()]
    if not model_configs_list:
        model_configs_list = DEFAULT_MODEL_CONFIGS_TO_TEST
        logger.info(f"No model_configs provided or empty, using default: {model_configs_list}")

    # Create temporary game list file if game_ids was provided
    game_list_file = args.game_list_file
    temp_file = None
    if game_ids_list and not game_list_file:
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write('\n'.join(game_ids_list))
        temp_file.close()
        game_list_file = temp_file.name
        logger.info(f"Created temporary game list file: {game_list_file}")

    try:
        exit_code_from_main = asyncio.run(
            main(
                game_list_file=game_list_file,
                model_configs_to_test=model_configs_list,
                results_root=args.results_root,
                overwrite_results=args.overwrite_results,
                max_actions=args.max_actions,
                retry_attempts=args.retry_attempts,
            )
        )
        sys.exit(exit_code_from_main)
    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

