"""Command-line interface for the Gemma 4 Reasoner."""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import Gemma4Reasoner, ModelSize, ReasoningConfig, BackendType

app = typer.Typer(
    name="gemma4",
    help="Gemma 4 Multimodal Reasoner — Analyze images, documents, and more.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

logger = logging.getLogger("gemma4_reasoner")


def _build_config(
    backend: str,
    model: str,
    thinking: bool,
    base_url: str,
) -> ReasoningConfig:
    """Build config from CLI args, falling back to env vars."""
    config = ReasoningConfig.from_env()
    config.backend = BackendType(backend)
    config.model_size = ModelSize(model)
    config.thinking = thinking
    config.base_url = base_url
    return config


def _print_result(result, show_thinking: bool = True):
    """Pretty-print a reasoning result."""
    if show_thinking and result.has_thinking:
        console.print(Panel(
            result.thinking,
            title="🧠 Chain of Thought",
            border_style="dim",
        ))
        console.print()
    console.print(Panel(
        Markdown(result.response),
        title="💬 Response",
        border_style="green",
    ))


@app.command()
def analyze(
    image: str = typer.Option(..., "--image", "-i", help="Path or URL to image"),
    question: str = typer.Option(
        "Describe this image in detail.", "--question", "-q", help="Question to ask"
    ),
    doc_type: Optional[str] = typer.Option(
        None, "--doc-type", "-d",
        help="Document type: chart, table, receipt, diagram, form, handwriting, general",
    ),
    backend: str = typer.Option("ollama", "--backend", "-b", help="Backend: ollama, huggingface, openai_compat"),
    model: str = typer.Option("e4b", "--model", "-m", help="Model: e2b, e4b, 26b, 31b"),
    thinking: bool = typer.Option(True, "--thinking/--no-thinking", help="Enable thinking mode"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url", help="Backend URL"),
    show_thinking: bool = typer.Option(True, "--show-thoughts/--hide-thoughts", help="Show chain-of-thought"),
):
    """Analyze an image with a question."""
    config = _build_config(backend, model, thinking, base_url)
    reasoner = Gemma4Reasoner(config)

    try:
        if doc_type:
            result = reasoner.analyze_document(image, doc_type=doc_type)
        else:
            result = reasoner.analyze_image(image, question=question)
        _print_result(result, show_thinking)
    except Exception as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    images: list[str] = typer.Argument(..., help="Paths or URLs to images"),
    question: str = typer.Option(
        "Compare these images and highlight the key differences.",
        "--question", "-q", help="Comparison question"
    ),
    backend: str = typer.Option("ollama", "--backend", "-b"),
    model: str = typer.Option("e4b", "--model", "-m"),
    thinking: bool = typer.Option(True, "--thinking/--no-thinking"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url"),
):
    """Compare multiple images."""
    config = _build_config(backend, model, thinking, base_url)
    reasoner = Gemma4Reasoner(config)

    try:
        result = reasoner.compare_images(images, question=question)
        _print_result(result)
    except Exception as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def math(
    image: str = typer.Option(..., "--image", "-i", help="Path or URL to math problem image"),
    backend: str = typer.Option("ollama", "--backend", "-b"),
    model: str = typer.Option("e4b", "--model", "-m"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url"),
):
    """Solve a visual math problem."""
    config = _build_config(backend, model, thinking=True, base_url=base_url)
    reasoner = Gemma4Reasoner(config)

    try:
        result = reasoner.solve_math(image)
        _print_result(result)
    except Exception as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def screen(
    image: str = typer.Option(..., "--image", "-i", help="Path or URL to screenshot"),
    task: str = typer.Option("describe", "--task", "-t", help="Task: describe, interact, accessibility"),
    backend: str = typer.Option("ollama", "--backend", "-b"),
    model: str = typer.Option("e4b", "--model", "-m"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url"),
):
    """Analyze a screen or UI."""
    config = _build_config(backend, model, thinking=True, base_url=base_url)
    reasoner = Gemma4Reasoner(config)

    try:
        result = reasoner.analyze_screen(image, task=task)
        _print_result(result)
    except Exception as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    image: Optional[str] = typer.Option(None, "--image", "-i", help="Optional initial image"),
    backend: str = typer.Option("ollama", "--backend", "-b"),
    model: str = typer.Option("e4b", "--model", "-m"),
    thinking: bool = typer.Option(True, "--thinking/--no-thinking"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url"),
):
    """Interactive multi-turn chat."""
    config = _build_config(backend, model, thinking, base_url)
    reasoner = Gemma4Reasoner(config)

    console.print("[bold blue]Gemma 4 Chat[/bold blue] — Type 'quit' to exit, 'reset' to clear history\n")

    images = [image] if image else None

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "reset":
            reasoner.reset_chat()
            console.print("[dim]History cleared.[/dim]")
            continue
        if not user_input.strip():
            continue

        try:
            result = reasoner.chat(user_input, images=images)
            images = None  # Only send image on first turn
            console.print(f"[bold green]Gemma:[/bold green] {result.response}\n")
        except Exception as e:
            err_console.print(f"[red]Error: {e}[/red]")

    console.print("[dim]Goodbye![/dim]")


@app.command()
def info():
    """Show model information and recommendations."""
    from rich.table import Table

    table = Table(title="Gemma 4 Model Family")
    table.add_column("Model", style="cyan")
    table.add_column("Params", style="magenta")
    table.add_column("Context", style="green")
    table.add_column("Modalities", style="yellow")
    table.add_column("Best For", style="white")
    table.add_column("VRAM (BF16)", style="red")

    rows = [
        ("E2B", "2.3B effective", "128K", "Text, Image, Audio", "Mobile, IoT", "9.6 GB"),
        ("E4B", "4.5B effective", "128K", "Text, Image, Audio", "Laptops, edge", "15 GB"),
        ("26B A4B", "3.8B active (MoE)", "256K", "Text, Image", "Fast inference", "48 GB"),
        ("31B", "30.7B dense", "256K", "Text, Image", "Max quality", "58.3 GB"),
    ]
    for row in rows:
        table.add_row(*row)

    console.print(table)
    console.print()
    console.print("[dim]All models support: thinking mode, variable image resolution, function calling, 140+ languages[/dim]")


if __name__ == "__main__":
    app()
