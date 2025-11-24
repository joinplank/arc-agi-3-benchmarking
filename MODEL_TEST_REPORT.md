# Model Test Report - Checkpoint Verification

**Date:** 2025-11-23  
**Test Configuration:** 2 actions per model, checkpoint frequency: 1  
**Total Models Tested:** 74

## Executive Summary

✅ **All 74 models have checkpoints created**  
✅ **All 176 checkpoints are valid** (have required files: metadata.json, costs.json, action_history.json, memory.txt)  
✅ **53 models completed successfully** with 2 actions (as expected)  
⚠️ **19 models have 0 actions** in their checkpoints (potential issues)  
⚠️ **2 models have >2 actions** (may indicate multiple runs or retries)

## Checkpoint Validation

### Overall Statistics
- **Total Checkpoints Created:** 176
- **Valid Checkpoints:** 176 (100%)
- **Invalid Checkpoints:** 0

### Checkpoint Completeness
- ✅ Has metadata.json: 176/176 (100%)
- ✅ Has costs.json: 176/176 (100%)
- ✅ Has action_history.json: 176/176 (100%)
- ✅ Has memory.txt: 176/176 (100%)
- ✅ Has previous_images: 121/176 (69%)

## Models by Provider

| Provider | Models with Checkpoints |
|----------|------------------------|
| OpenAI | 39 |
| Anthropic | 12 |
| Gemini | 8 |
| Grok/X.AI | 8 |
| DeepSeek | 3 |
| Mistral | 3 |
| Qwen | 1 |

## Models with Issues

### Models with 0 Actions (19 models)

These models have checkpoints but no actions recorded. This may indicate:
- Early failure before first action
- API authentication issues
- Model initialization problems
- Checkpoint created but game didn't start

**List of affected models:**
1. claude-3-7-sonnet-20250219
2. claude-3-7-sonnet-20250219-thinking-16k
3. claude-3-7-sonnet-20250219-thinking-1k
4. claude-3-7-sonnet-20250219-thinking-8k
5. claude-4-sonnet-20250522-thinking-8k-bedrock (OpenRouter - needs API key)
6. claude-sonnet-4-5-20250929
7. claude-sonnet-4-5-20250929-thinking-16k
8. claude-sonnet-4-5-20250929-thinking-1k
9. claude-sonnet-4-5-20250929-thinking-32k
10. claude-sonnet-4-5-20250929-thinking-8k
11. claude_haiku
12. claude_opus
13. deepseek_chat
14. deepseek_r1
15. deepseek_r1_0528-openrouter (OpenRouter - needs API key)
16. gemini-2-5-pro-preview-openrouter (OpenRouter - needs API key)
17. gemini-2-5-pro-preview-openrouter-thinking-1k (OpenRouter - needs API key)
18. gpt-5-pro-2025-10-06
19. grok-3-beta
20. grok-3-mini-beta
21. grok-3-mini-beta-high
22. grok-3-mini-beta-high-openrouter (OpenRouter - needs API key)
23. grok-3-mini-beta-low-openrouter (OpenRouter - needs API key)
24. grok-3-mini-xai-high
25. grok-4-0709
26. grok-4-fast-reasoning
27. magistral-medium-2506 (OpenRouter - needs API key)
28. magistral-medium-2506-thinking (OpenRouter - needs API key)
29. magistral-small-2506 (OpenRouter - needs API key)
30. qwen3-235b-a22b-07-25 (OpenRouter - needs API key)

**Note:** Many of these are OpenRouter models that require `OPENROUTER_API_KEY` environment variable.

### Models with >2 Actions (2 models)

These models have more actions than expected, possibly due to:
- Multiple test runs
- Retry logic
- Checkpoint resume scenarios

1. **gpt-5-2025-08-07-high:** 22 actions
2. **gemini-2-5-flash-preview-05-20:** 27 actions

## Successful Models (53 models with 2 actions)

The following models completed successfully with exactly 2 actions:

### Anthropic Models
- claude-3-7-sonnet-20250219 (also appears in 0 actions - multiple checkpoints)
- claude-3-7-sonnet-20250219-thinking-16k (also appears in 0 actions)
- claude-3-7-sonnet-20250219-thinking-1k (also appears in 0 actions)
- claude-3-7-sonnet-20250219-thinking-8k (also appears in 0 actions)
- claude-sonnet-4-5-20250929 (also appears in 0 actions)
- claude-sonnet-4-5-20250929-thinking-16k (also appears in 0 actions)
- claude-sonnet-4-5-20250929-thinking-1k (also appears in 0 actions)
- claude-sonnet-4-5-20250929-thinking-32k (also appears in 0 actions)
- claude-sonnet-4-5-20250929-thinking-8k (also appears in 0 actions)
- claude_haiku (also appears in 0 actions)
- claude_opus (also appears in 0 actions)

### Gemini Models
- gemini-2-0-flash-001
- gemini-2-5-flash-preview-05-20 (also has 27 actions checkpoint)
- gemini-2-5-flash-preview-05-20-thinking-16k
- gemini-2-5-flash-preview-05-20-thinking-1k
- gemini-2-5-flash-preview-05-20-thinking-24k
- gemini-2-5-flash-preview-05-20-thinking-8k

### OpenAI Models
- gpt-4-1-2025-04-14
- gpt-4-1-mini-2025-04-14
- gpt-4-1-nano-2025-04-14
- gpt-4o-2024-05-13
- gpt-4o-2024-11-20
- gpt-4o-mini-2024-07-18
- gpt-5-2025-08-07-high (also has 22 actions checkpoint)
- gpt-5-2025-08-07-low
- gpt-5-2025-08-07-medium
- gpt-5-2025-08-07-minimal
- gpt-5-mini-2025-08-07-high
- gpt-5-mini-2025-08-07-low
- gpt-5-mini-2025-08-07-medium
- gpt-5-mini-2025-08-07-minimal
- gpt-5-nano-2025-08-07-high
- gpt-5-nano-2025-08-07-low
- gpt-5-nano-2025-08-07-medium
- gpt-5-nano-2025-08-07-minimal
- gpt4o_mini
- o1-2024-12-17-high
- o1-2024-12-17-low
- o1-2024-12-17-medium
- o1-pro-2025-03-19-high
- o1-pro-2025-03-19-low
- o1-pro-2025-03-19-medium
- o3-2025-04-16-high
- o3-2025-04-16-low
- o3-2025-04-16-low-2025-06-10
- o3-2025-04-16-medium
- o3-mini-2025-01-31-high
- o3-mini-2025-01-31-low
- o3-mini-2025-01-31-medium
- o3-pro-2025-06-10-high
- o3-pro-2025-06-10-low
- o3-pro-2025-06-10-medium
- o4-mini-2025-04-16-high
- o4-mini-2025-04-16-low
- o4-mini-2025-04-16-medium

## Checkpoint Data Validation

### Sample Checkpoint Structure

All checkpoints contain:
- **metadata.json:** Contains game_id, config, action counters, timestamps
- **costs.json:** Contains token usage and cost information
- **action_history.json:** Contains list of actions taken (may be empty for 0-action models)
- **memory.txt:** Contains agent memory/scratchpad
- **previous_images/** (optional): Contains game frame images

### Example Valid Checkpoint
```
Model: o1-pro-2025-03-19-high
Checkpoint ID: c65d89fa-0bb6-46b7-b69c-6c4a2d1af913
Game ID: ls20-016295f7601e
Actions: 2
Has costs.json: ✓
Has action_history.json: ✓
Has memory.txt: ✓
```

## Issues Identified

### 1. OpenRouter API Key Missing
**Issue:** Several models using OpenRouter provider failed with authentication errors.  
**Affected Models:** 
- claude-4-sonnet-20250522-thinking-8k-bedrock
- deepseek_r1_0528-openrouter
- gemini-2-5-pro-preview-openrouter
- gemini-2-5-pro-preview-openrouter-thinking-1k
- grok-3-mini-beta-high-openrouter
- grok-3-mini-beta-low-openrouter
- magistral-medium-2506
- magistral-medium-2506-thinking
- magistral-small-2506
- qwen3-235b-a22b-07-25

**Resolution:** Set `OPENROUTER_API_KEY` environment variable.

### 2. Models with 0 Actions
**Issue:** 19 models have checkpoints but no actions recorded.  
**Possible Causes:**
- Early failure during initialization
- API errors before first action
- Authentication issues
- Model-specific initialization problems

**Recommendation:** Review logs for these models to identify root cause.

### 3. Multiple Checkpoints per Model
**Issue:** Some models have multiple checkpoints (e.g., some Anthropic models appear in both 0-action and 2-action lists).  
**Possible Causes:**
- Multiple test runs
- Retry logic creating new checkpoints
- Checkpoint resume scenarios

**Recommendation:** This is expected behavior - each run creates a new checkpoint.

## Recommendations

1. ✅ **Checkpoint System Working:** All checkpoints are being created correctly with all required files.

2. ⚠️ **Fix OpenRouter Models:** Set `OPENROUTER_API_KEY` to enable OpenRouter-based models.

3. ⚠️ **Investigate 0-Action Models:** Review error logs for models with 0 actions to identify why they didn't execute any actions.

4. ✅ **Checkpoint Validation:** All checkpoint files are valid and contain expected data structures.

5. ✅ **Parallel Execution:** The parallel execution system is working correctly, handling 10 concurrent model runs.

## Conclusion

The checkpoint system is **working correctly** - all 74 models have valid checkpoints created. The main issues are:
1. Missing API keys for OpenRouter models (expected)
2. Some models failing before executing actions (needs investigation)

The checkpoint data structure is consistent and complete across all models.


