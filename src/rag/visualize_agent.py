#!/usr/bin/env python3
import argparse
import sys
import warnings
from pathlib import Path

from .main import build_graph


def render_graph(fmt: str) -> str:
    graph = build_graph(compile_graph=False)

    if fmt == "ascii":
        if hasattr(graph, "draw_ascii"):
            return graph.draw_ascii()
        warnings.warn(
            "ASCII rendering is not supported by this LangGraph version; falling back to mermaid.",
            RuntimeWarning,
            stacklevel=1,
        )
        if hasattr(graph, "draw_mermaid"):
            return graph.draw_mermaid()
        raise RuntimeError(
            "ASCII rendering is not supported by this version of LangGraph, and Mermaid fallback is unavailable."
        )

    if fmt == "mermaid":
        if hasattr(graph, "draw_mermaid"):
            return graph.draw_mermaid()
        raise RuntimeError("Mermaid rendering is not supported by this version of LangGraph.")

    raise ValueError(f"Unsupported format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the LangGraph agent pipeline as ASCII or Mermaid."
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["ascii", "mermaid"],
        default="mermaid",
        help="Output format (default: mermaid)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional file path to save the visualization instead of printing to stdout.",
    )
    args = parser.parse_args()

    try:
        rendered = render_graph(args.format)
    except Exception as exc:
        parser.exit(1, f"Error generating visualization: {exc}\n")

    if args.output:
        try:
            Path(args.output).write_text(rendered, encoding="utf-8")
        except Exception as exc:
            parser.exit(1, f"Failed to write output: {exc}\n")
    else:
        sys.stdout.write(rendered + ("\n" if not rendered.endswith("\n") else ""))


if __name__ == "__main__":
    main()

