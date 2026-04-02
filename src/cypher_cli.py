#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class CypherCLI:
    """Command-line interface for Neo4j Cypher operations."""

    def __init__(self):
        self.driver = None
        self._connected = False
        self.database = os.getenv("NEO4J_DB") or os.getenv("NEO4J_DATABASE") or "kwuait"

    async def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD")

            if not password:
                logger.error("NEO4J_PASSWORD environment variable not set")
                return False

            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 'Hello, Neo4j!' as message")
                record = result.single()
                if record:
                    logger.info(f"Connected to Neo4j: {record['message']}")
                    self._connected = True
                    return True
                else:
                    logger.error("Failed to connect to Neo4j")
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Neo4j database."""
        if self.driver:
            self.driver.close()
            self._connected = False

    async def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a Cypher query."""
        if not self._connected:
            success = await self.connect()
            if not success:
                return {"error": "Failed to connect to Neo4j"}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                records = [dict(record) for record in result]
                return {"success": True, "records": records, "count": len(records)}
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {"error": str(e)}

    async def run(self, args: List[str]) -> Dict[str, Any]:
        """Run Cypher CLI with given arguments."""
        parser = argparse.ArgumentParser(description='Neo4j Cypher CLI')
        parser.add_argument('query', help='Cypher query to execute')
        parser.add_argument('--params', type=json.loads, default={},
                           help='Query parameters as JSON string')

        try:
            parsed_args = parser.parse_args(args)
            return await self.run_query(parsed_args.query, parsed_args.params)
        except SystemExit:
            # argparse exits with SystemExit on --help or invalid args
            return {"error": "Invalid arguments for cypher command"}
        except Exception as e:
            return {"error": str(e)}

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()