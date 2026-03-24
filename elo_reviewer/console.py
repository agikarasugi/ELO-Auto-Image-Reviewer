from rich.console import Console

console = Console()
file_console: Console | None = None


def log(msg: str = "") -> None:
    """Print to the terminal console and, if configured, the log file."""
    console.print(msg)
    if file_console is not None:
        file_console.print(msg)
