# ARC-AGI-3 Benchmarking

Framework for benchmarking multimodal LLMs on ARC-AGI-3 interactive games.

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 3. Run a game
python main.py \
  --game_id "ls20-fa137e247ce6" \
  --config "gpt-4o-mini-2024-07-18" \
  --max_actions 40
```

## Features

- 🎮 **Multimodal Agent** - Vision-based reasoning with image analysis
- 🔌 **Multi-Provider** - OpenAI, Anthropic, Gemini, Grok, DeepSeek, and more
- 💰 **Cost Tracking** - Automatic token usage and cost calculation
- 🔄 **Retry Logic** - Exponential backoff for API resilience
- 📊 **Results Export** - Comprehensive JSON output with full game history

## Installation

```bash
# Install dependencies
pip install -e .

# Required API keys in .env
ARC_API_KEY=your_arc_api_key
OPENAI_API_KEY=your_openai_key      # or other provider
```

## Usage

### List Available Games

```bash
python -m arcagi3.cli --list-games
```

### Run Single Game

```bash
python main.py \
  --game_id "ls20-fa137e247ce6" \
  --config "gpt-4o-mini-2024-07-18" \
  --max_actions 40 \
  --save_results_dir "results/gpt4o"
```

### Run Multiple Games

```bash
# Specific games
python -m arcagi3.cli \
  --games "game1,game2,game3" \
  --config "claude-sonnet-4-5-20250929"

# All games
python -m arcagi3.cli \
  --all-games \
  --config "gpt-4o-2024-11-20"
```

### Programmatic Usage

```python
from arcagi3 import MultimodalAgent, GameClient

# Initialize
client = GameClient()
games = client.list_games()

# Play game
scorecard_response = client.open_scorecard([games[0]['game_id']])
card_id = scorecard_response['card_id']

agent = MultimodalAgent(
    config="gpt-4o-mini-2024-07-18",
    game_client=client,
    card_id=card_id,
    max_actions=40
)

result = agent.play_game(games[0]['game_id'])
print(f"Score: {result.final_score}, Cost: ${result.total_cost.total_cost:.4f}")
print(f"Scorecard: {result.scorecard_url}")

client.close_scorecard(card_id)
```

## CLI Options

### Main CLI (`main.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `--game_id` | Game ID to play | Required |
| `--config` | Model config from models.yml | Required |
| `--max_actions` | Max actions per game | 40 |
| `--save_results_dir` | Results directory | `results/<config>` |
| `--retry_attempts` | API retry attempts | 3 |
| `--verbose` | Enable verbose logging | False |

### Batch CLI (`arcagi3.cli`)

| Option | Description |
|--------|-------------|
| `--list-games` | List available games and exit |
| `--games "id1,id2"` | Run specific games |
| `--all-games` | Run all available games |

All main CLI options also available.

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────┐
│                Game Client                   │
│  (ARC-AGI-3 API Communication)              │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│          Multimodal Agent                    │
│  ┌─────────────────────────────────────┐   │
│  │ 1. Convert frames to images         │   │
│  │ 2. Analyze previous results         │   │
│  │ 3. Choose next action (LLM)         │   │
│  │ 4. Convert to game command (LLM)    │   │
│  │ 5. Track costs & execute            │   │
│  └─────────────────────────────────────┘   │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│          Provider Adapters                   │
│  (OpenAI, Anthropic, Gemini, etc.)          │
└─────────────────────────────────────────────┘
```

### Directory Structure

```
arc-agi-3-benchmarking/
├── src/arcagi3/              # Main package
│   ├── agent.py             # Multimodal agent
│   ├── game_client.py       # ARC-AGI-3 API
│   ├── image_utils.py       # Image processing
│   ├── schemas.py           # Data models
│   ├── models.yml           # Model configs
│   ├── adapters/            # Provider adapters
│   └── utils/               # Helpers
├── main.py                   # Single game CLI
├── example_usage.py          # Code examples
└── results/                  # Output directory
```

### How It Works

1. **Game Client** - Communicates with ARC-AGI-3 API
2. **Agent** - Multimodal reasoning loop:
   - Converts game frames (64x64 grids) to images
   - Analyzes previous action outcomes
   - Chooses next high-level action via LLM
   - Converts to specific game command via LLM
   - Tracks token usage and costs
3. **Providers** - Lazy-loaded adapters for different LLMs
4. **Results** - Comprehensive JSON with full game history

## Configuration

### Model Configuration

Models are defined in `src/arcagi3/models.yml`:

```yaml
- name: "gpt-4o-mini-2024-07-18"
  model_name: "gpt-4o-mini-2024-07-18"
  provider: "openai"
  pricing:
    date: "2025-03-15"
    input: 0.15   # per 1M tokens
    output: 0.60
```

90+ models pre-configured. Use the `name` field in CLI.

### Provider Setup

Install only what you need:

```bash
pip install openai            # OpenAI models
pip install anthropic         # Claude models
pip install google-genai      # Gemini models
```

Providers load lazily - only need SDKs you actually use.

## Results Format

Results saved to `{save_results_dir}/{game_id}_{config}_{timestamp}.json`:

```json
{
  "game_id": "ls20-fa137e247ce6",
  "config": "gpt-4o-mini-2024-07-18",
  "final_score": 850,
  "final_state": "WIN",
  "actions_taken": 23,
  "duration_seconds": 145.3,
  "total_cost": {
    "prompt_cost": 0.0234,
    "completion_cost": 0.0567,
    "total_cost": 0.0801
  },
  "usage": {
    "prompt_tokens": 7800,
    "completion_tokens": 3780,
    "total_tokens": 11580
  },
  "scorecard_url": "https://three.arcprize.org/scorecards/card_uuid",
  "actions": [...]
}
```

The `scorecard_url` field provides a direct link to view the game replay online on the ARC Prize platform.

## Resources

- **ARC Prize**: https://arcprize.org
- **API Documentation**: https://docs.arcprize.org/api-reference/
- **Code Examples**: See `example_usage.py`

## Tips

**Cost Management:**
- Start with `--max_actions 5` for testing
- Use `gpt-4o-mini-2024-07-18` for development
- Each action costs ~$0.001-0.01 depending on model

**Debugging:**
- Use `--verbose` flag for detailed logs
- Check `results/` directory for JSON output
- Review action reasoning in result files

**Adding Providers:**
See `src/arcagi3/adapters/` for examples. Create new adapter inheriting from `ProviderAdapter`, add to factory in `__init__.py`, configure in `models.yml`.

## License

MIT License

---

**Version**: 0.1.0 | **Package**: `arcagi3` | **Python**: 3.9+
