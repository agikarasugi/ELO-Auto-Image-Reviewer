import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()


def _rounds_type(value: str) -> int | str:
    """Accept a positive integer or the string 'auto'."""
    if value.lower() == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for --rounds: '{value}'. Use a positive integer or 'auto'.")
    if n <= 0:
        raise argparse.ArgumentTypeError(f"--rounds must be a positive integer, got {n}.")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elo-reviewer",
        description=(
            "Run an ELO-rated image tournament using a multimodal LLM as judge.\n\n"
            "Each round randomly pairs two images and asks the model to decide which\n"
            "is better via a multi-turn conversation (initial analysis → self-review\n"
            "→ final verdict). After all rounds, ELO scores, win/loss records, a\n"
            "ranked CSV, and a top-3 visualization image are written to --output-dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  elo-reviewer -d ./photos --rounds 50\n"
            "  elo-reviewer -d ./photos --api-base-url http://localhost:11434/v1 --api-key ollama --model llava:13b\n\n"
            "Required environment variables (or use CLI flags):\n"
            "  OPENAI_API_KEY     API key\n"
            "  OPENAI_BASE_URL    OpenAI-compatible base URL\n"
            "  OPENAI_MODEL       Model name (must support vision)\n\n"
            "Copy .env.example to .env and fill in your values."
        ),
    )

    parser.add_argument(
        "-d", "--images-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing images to rank (png, jpg, jpeg, webp, gif).",
    )
    parser.add_argument(
        "-r", "--rounds",
        type=_rounds_type,
        default="auto",
        metavar="N|auto",
        help=(
            "Number of comparison rounds. Use 'auto' (default) to set rounds to "
            "N*(N-1)/2 — one complete round-robin pass over all unique image pairs."
        ),
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
        metavar="URL",
        help="OpenAI-compatible API base URL. Required if $OPENAI_BASE_URL is not set.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        metavar="KEY",
        help="API key. Required if $OPENAI_API_KEY is not set.",
    )
    parser.add_argument(
        "-m", "--model",
        default=os.environ.get("OPENAI_MODEL", ""),
        metavar="MODEL",
        help="Model identifier (must support vision). Required if $OPENAI_MODEL is not set.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("./elo_results"),
        metavar="DIR",
        help="Directory to write CSV and top-3 image. Created if it does not exist. (default: ./elo_results)",
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=32.0,
        metavar="K",
        help="ELO K-factor: controls how much each result shifts ratings. (default: 32)",
    )
    parser.add_argument(
        "--starting-elo",
        type=float,
        default=1000.0,
        metavar="ELO",
        help="Starting ELO score for every image. (default: 1000)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=5,
        metavar="N",
        help="Minimum number of images required to run (hard stop). (default: 5)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print each round's full LLM conversation to stdout.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output for classic terminal environments.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure shared console before any output
    from . import console as cm
    if args.no_color:
        cm.console = Console(no_color=True)

    # --- Validate images directory ---
    if not args.images_dir.exists():
        parser.error(f"Images directory does not exist: {args.images_dir}")
    if not args.images_dir.is_dir():
        parser.error(f"Not a directory: {args.images_dir}")

    # --- Validate required connection settings ---
    missing = []
    if not args.api_key:
        missing.append("API key (--api-key or OPENAI_API_KEY)")
    if not args.api_base_url:
        missing.append("API base URL (--api-base-url or OPENAI_BASE_URL)")
    if not args.model:
        missing.append("model name (--model or OPENAI_MODEL)")
    if missing:
        parser.error(
            "Missing required configuration:\n  - "
            + "\n  - ".join(missing)
            + "\nCopy .env.example to .env and fill in your values."
        )

    # --- Collect and validate images ---
    from .image_utils import collect_images, validate_image_count

    try:
        images = collect_images(args.images_dir)
    except ValueError as e:
        parser.error(str(e))

    validate_image_count(images, minimum=args.min_images)

    n = len(images)
    cm.console.print(f"[green]Found[/green] [bold]{n}[/bold] images in [cyan]{args.images_dir}[/cyan]")

    # --- Resolve rounds ---
    if args.rounds == "auto":
        rounds = n * (n - 1) // 2
        cm.console.print(f"[dim]Auto rounds: {n}*({n}-1)/2 = {rounds} rounds[/dim]")
    else:
        rounds = args.rounds  # already an int

    # --- Prepare output directory ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Open log file ---
    from datetime import datetime
    log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.output_dir / f"elo_log_{log_ts}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    cm.file_console = Console(file=log_file)

    # --- Set up ELO ratings ---
    from .elo import EloRatings
    ratings = EloRatings(images, starting_elo=args.starting_elo, k_factor=args.k_factor)

    # --- Set up OpenAI client ---
    import openai
    client = openai.OpenAI(
        api_key=args.api_key,
        base_url=args.api_base_url,
    )

    # --- Set up judge ---
    from .judge import Judge
    judge = Judge(client=client, model=args.model, verbose=args.verbose)

    # --- Run tournament ---
    from .tournament import run_tournament
    cm.log(f"\n[bold]Starting tournament:[/bold] {rounds} rounds, model=[cyan]{args.model}[/cyan]\n")
    results = run_tournament(images, ratings, judge, rounds=rounds)

    # --- Token and fallback stats ---
    total_tokens = sum(r.tokens for r in results)
    cm.log(f"\n[dim]Total tokens used: {total_tokens:,}[/dim]")

    fallback_results = [r for r in results if r.used_fallback]
    if fallback_results:
        cm.log(f"\n[yellow]Fallback rounds ({len(fallback_results)}/{rounds} used random decision):[/yellow]")
        for r in fallback_results:
            cm.log(
                f"  [yellow]round {r.round_num:>4}  "
                f"{r.image_a}  vs  {r.image_b}  "
                f"→ {r.winner} (random)[/yellow]"
            )

    # --- Write outputs ---
    from .output import print_summary_table, write_csv, write_top3_image

    cm.log()
    print_summary_table(ratings)

    csv_path = write_csv(ratings, args.output_dir)
    cm.log(f"\n[green]CSV saved:[/green]   [cyan]{csv_path}[/cyan]")

    top3_path = write_top3_image(ratings, args.images_dir, args.output_dir)
    cm.log(f"[green]Top-3 image:[/green] [cyan]{top3_path}[/cyan]")

    # --- Close log file ---
    log_file.close()
    cm.console.print(f"[green]Log saved:[/green]    [cyan]{log_path}[/cyan]")
