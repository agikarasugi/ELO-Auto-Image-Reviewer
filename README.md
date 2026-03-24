# ELO Auto Image Reviewer

Automatically rank images in a directory using an ELO tournament system. A multimodal LLM acts as judge, comparing pairs of images through a multi-turn conversation — initial analysis, self-review, then a final verdict — to produce confident, well-reasoned decisions.

After the tournament, results are saved as a timestamped CSV and a top-3 podium image.

## How It Works

Each round:
1. Two images are randomly selected and assigned as **Image A** and **Image B** (randomized to cancel position bias)
2. The LLM analyzes both images across technical quality, composition, visual appeal, and subject clarity
3. The LLM critically reviews its own reasoning for potential bias or oversights
4. The LLM gives a final single-letter verdict: **A** or **B**
5. ELO scores are updated — the winner gains points, the loser loses points

After all rounds, a ranked CSV and a top-3 visualization image are written to the output directory.

## Requirements

- [uv](https://docs.astral.sh/uv/) — package manager
- Python 3.12+
- Access to an OpenAI-compatible API with a multimodal model (vision support required)

## Installation

```bash
git clone <repo-url>
cd ELO-Auto-Image-Reviewer
uv sync
```

## Configuration

Create a `.env` file in the project root (or export variables directly):

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

These are used as defaults and can be overridden by CLI flags.

## Usage

```bash
uv run elo-reviewer --images-dir ./photos
```

Or after installation as a script:

```bash
elo-reviewer --images-dir ./photos
```

### Common Examples

```bash
# Auto rounds (default): N*(N-1)/2 — one full round-robin
uv run elo-reviewer -d ./photos

# Fixed number of rounds
uv run elo-reviewer -d ./photos --rounds 100

# With a local model (e.g. LM Studio, Ollama)
uv run elo-reviewer -d ./photos \
  --api-base-url http://localhost:8000/v1 \
  --api-key local \
  --model llava:13b

# Verbose: print each round's full conversation
uv run elo-reviewer -d ./photos --rounds 20 --verbose

# Custom ELO parameters and output directory
uv run elo-reviewer -d ./photos \
  --k-factor 16 \
  --starting-elo 1200 \
  --output-dir ./results
```

### All Options

```
usage: elo-reviewer [-h] -d DIR [-r N|auto] [--api-base-url URL]
                    [--api-key KEY] [-m MODEL] [-o DIR] [--k-factor K]
                    [--starting-elo ELO] [--min-images N] [-v]

options:
  -d, --images-dir DIR    Directory containing images to rank
                          (supported: png, jpg, jpeg, webp, gif)
  -r, --rounds N|auto     Rounds to run. 'auto' computes N*(N-1)/2 for a
                          full round-robin. (default: auto)
  --api-base-url URL      OpenAI-compatible base URL
                          (default: $OPENAI_BASE_URL)
  --api-key KEY           API key (default: $OPENAI_API_KEY)
  -m, --model MODEL       Model name (default: $OPENAI_MODEL or gpt-4o)
  -o, --output-dir DIR    Output directory for results (default: ./elo_results)
  --k-factor K            ELO K-factor — sensitivity of score changes (default: 32)
  --starting-elo ELO      Initial ELO for all images (default: 1000)
  --min-images N          Minimum images required to run (default: 5)
  -v, --verbose           Print each round's LLM conversation
```

## Output

Results are written to `--output-dir` (default: `./elo_results/`):

| File | Description |
|---|---|
| `elo_results_YYYYMMDD_HHMMSS.csv` | Rankings table: filename, wins, losses, ELO score |
| `top3_YYYYMMDD_HHMMSS.png` | Podium image showing the top 3 ranked images |

Filenames are timestamped so repeated runs never overwrite previous results.

A ranked summary table is also printed to the terminal after the tournament.

### Example CSV

```csv
image_filename,wins,losses,elo_score
photo_05.jpg,14,3,1187.43
photo_11.png,12,5,1134.72
photo_02.jpg,10,7,1089.15
...
```

## ELO Scoring

The standard ELO formula is used:

```
Expected score:  E_A = 1 / (1 + 10^((R_B - R_A) / 400))
Rating update:   R_new = R_old + K * (actual - expected)
```

- **K-factor** controls how much each result moves scores. Higher K = faster convergence but more volatility. Default `32` is standard for chess.
- **Starting ELO** of `1000` means all images begin equal.
- For reliable rankings, use enough rounds for each image to be judged multiple times. The `auto` mode ensures every pair meets at least once.

## Notes

- At least **5 images** are required (hard minimum). A warning is shown with fewer than 10.
- Images are resized to a maximum of 2048px on the longest side before encoding to keep API payloads manageable.
- If the model's response cannot be parsed, a random result is used as a fallback and flagged in the terminal output.
- Rate limit errors are retried with exponential backoff (up to 3 attempts).
