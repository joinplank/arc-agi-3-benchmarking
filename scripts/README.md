# Scripts Directory

This directory contains utility scripts for testing, verification, and analysis.

## Test Scripts

### `test_all_models_parallel.py`
Runs all models in parallel to verify they work correctly and that checkpoints are being created properly.

**Usage:**
```bash
cd /path/to/project
uv run python scripts/test_all_models_parallel.py --max-actions 2 --checkpoint-frequency 1
```

**Options:**
- `--game-id`: Game ID to test with (default: ls20-016295f7601e)
- `--max-actions`: Maximum actions per model (default: 3)
- `--checkpoint-frequency`: Save checkpoint every N actions (default: 1)
- `--max-concurrent`: Maximum concurrent model runs (default: 10)

## Analysis Scripts

### `analyze_checkpoints.py`
Analyzes checkpoint directories and generates a report on model test results.

**Usage:**
```bash
cd /path/to/project
uv run python scripts/analyze_checkpoints.py
```

### `generate_detailed_report.py`
Generates a detailed report on model test results with checkpoint validation.

**Usage:**
```bash
cd /path/to/project
uv run python scripts/generate_detailed_report.py
```

## Verification Scripts

### `verify_openai_models.py`
Verifies that all OpenAI models in models.yml can be initialized and run.

**Usage:**
```bash
cd /path/to/project
uv run python scripts/verify_openai_models.py
```

### `verify_gemini_models_v2.py`
Verifies that Gemini models work correctly.

**Usage:**
```bash
cd /path/to/project
uv run python scripts/verify_gemini_models_v2.py
```

## Test Scripts

### `test_gemini_flash_thinking.py`
Tests Gemini Flash model with thinking configuration.

### `test_gemini_flash_thinking_v2.py`
Updated version of Gemini Flash thinking test.

## Notes

- All scripts should be run from the project root directory
- Scripts automatically resolve paths relative to the project root
- Make sure environment variables (API keys) are set before running


