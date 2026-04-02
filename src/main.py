#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load environment variables from .env file in project root (app directory)
project_root = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(project_root, ".env"))

from src.cypher_cli import CypherCLI
from src.registry import Registry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MainApp:
    def __init__(self):
        self.registry = Registry()
        self.cypher_cli = CypherCLI()

    async def flush_routing_cache(self) -> Dict[str, Any]:
        """Flush the routing cache and return status."""
        try:
            # Attempt to flush routing information
            result = await self.registry.flush_routing_cache()
            return {
                "action": "flush",
                "status": "success" if result else "failed",
                "message": "Routing cache flushed successfully"
                if result
                else "Failed to flush routing cache",
            }
        except Exception as e:
            logger.error(f"Error flushing routing cache: {e}")
            return {
                "action": "flush",
                "error": "Unable to retrieve routing information",
                "details": str(e),
            }

    async def run_command(self, command: str, args: List[str]) -> Dict[str, Any]:
        """Run a specific command."""
        if command == "flush":
            return await self.flush_routing_cache()
        elif command == "cypher":
            return await self.cypher_cli.run(args)
        elif command == "scaffold":
            return {"error": "Scaffolding command not implemented"}
        else:
            return {"action": command, "error": f"Unknown command: {command}"}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Main application CLI")
    parser.add_argument(
        "-f", "--flush", action="store_true", help="Flush routing cache"
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        action="store_true",
        help="Run preprocessing pipeline (parse, normalize, extract, ingest)",
    )
    parser.add_argument(
        "--parallel-llm",
        action="store_true",
        help="Use parallel LLM extraction (requires -p flag)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for -p flag)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent LLM calls (for --parallel-llm)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (for --parallel-llm)",
    )
    parser.add_argument(
        "-c", "--command", choices=["cypher", "scaffold"], help="Run specific command"
    )
    parser.add_argument("--cypher", help="Run Cypher query")
    parser.add_argument(
        "--count-nodes", action="store_true", help="Count nodes in Neo4j database"
    )
    parser.add_argument(
        "--clear-db", action="store_true", help="Clear all data from Neo4j database"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the chatbot API server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a single RAG query and print the result",
    )
    parser.add_argument("args", nargs="*", help="Additional arguments")

    return parser.parse_args()


async def main_async():
    """Main async function."""
    args = parse_args()

    app = MainApp()

    if args.flush:
        result = await app.flush_routing_cache()
        print(json.dumps(result, indent=2))
        return

    if args.preprocess:
        from src.preprocessing.main import run_pipeline

        logger.info("Starting preprocessing pipeline...")
        run_pipeline(limit=args.limit)
        return

    if args.cypher:
        result = await app.cypher_cli.run_query(args.cypher)
        print(json.dumps(result, indent=2))
        return

    if args.count_nodes:
        result = await app.cypher_cli.run_query(
            "MATCH (n) RETURN count(n) as node_count"
        )
        print(json.dumps(result, indent=2))
        return

    if args.clear_db:
        result = await app.cypher_cli.run_query("MATCH (n) DETACH DELETE n")
        print(json.dumps(result, indent=2))
        return

    if args.command:
        result = await app.run_command(args.command, args.args)
        print(json.dumps(result, indent=2))
        return

    if args.serve:
        from src.chatbot.api import start_server

        logger.info("Starting chatbot API server on %s:%d", args.host, args.port)
        start_server(host=args.host, port=args.port)
        return

    if args.query:
        from src.rag.main import run as rag_run

        result = rag_run(args.query)
        output = {
            "query": args.query,
            "answer": result.get("answer", "No answer generated."),
            "references": result.get("references", []),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # Interactive mode or default behavior
    print("SunnitAI ChatBot")
    print("  Use --serve     to start the chatbot API server")
    print("  Use --query     to run a single RAG query")
    print("  Use -p          to run preprocessing pipeline")
    print("  Use -f          to flush routing cache")
    print("  Use --help      for all options")


def main():
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
