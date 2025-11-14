# SPDX-License-Identifier: Apache-2.0
# NOTE: this is a temporary file just for PoC purpose.
# Standard
from dataclasses import dataclass
import os

# Third Party
import duckdb

# First Party
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey


@dataclass
class MemoryObjStats:
    """
    Temporary debug statistics for a MemoryObj
    """

    num_hits: int = 0
    size_in_bytes: int = 0
    logged: bool = False

    def __init__(self, memory_obj: MemoryObj):
        self.num_hits = 0
        self.size_in_bytes = memory_obj.get_size()
        self.logged = False

    def on_hit(self) -> None:
        self.num_hits += 1

    def on_log(self) -> None:
        self.logged = True


"""
Database schema:

Table: requests:
    - request_id: str
    - request_text: str
    - tokens: list[int]

Table: hashes:
    - request_id: str
    - position: int
    - hash: int

Table: memory_objects:
    - hash: int
    - model: str
    - world_size: int
    - rank: int
    - hit_count: int
    - size: int
"""


# ----------------
# Helper functions
# ----------------
def create_tables(connection: duckdb.DuckDBPyConnection) -> None:
    """
    Creates the necessary tables in the DuckDB database.

    Args:
        connection (duckdb.DuckDBPyConnection): The database connection.
    """
    connection.sql("""
    CREATE TABLE IF NOT EXISTS requests (
        request_id VARCHAR PRIMARY KEY,
        request_text VARCHAR,
        tokens INTEGER[]
    );
    """)

    connection.sql("""
    CREATE TABLE IF NOT EXISTS hashes (
        request_id VARCHAR,
        position INTEGER,
        hash BLOB,
        PRIMARY KEY (request_id, position)
    );
    """)

    connection.sql("""
    CREATE TABLE IF NOT EXISTS memory_objects (
        hash BLOB,
        model VARCHAR,
        world_size INTEGER,
        rank INTEGER,
        hit_count INT,
        size INTEGER,
        PRIMARY KEY (hash, model, rank)
    );
    """)


########################
# Initializing the database
########################


def initialize_database(
    db_path: str, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """
    Establishes a connection to a DuckDB database.

    Args:
        db_path (str): The path to the DuckDB database file.

    Returns:
        duckdb.DuckDBPyConnection: A connection object to the DuckDB database.
    """
    if read_only and not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database file '{db_path}' does not exist for read-only access."
        )

    if not read_only:
        # Clear existing database for fresh start
        os.remove(db_path) if os.path.exists(db_path) else None

    try:
        con = duckdb.connect(db_path, read_only)
        if not read_only:
            create_tables(con)
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to the DuckDB database at '{db_path}': {e}"
        ) from e

    return con


########################
# Updating the database
########################


@_lmcache_nvtx_annotate
def add_request_information(
    connection: duckdb.DuckDBPyConnection,
    request_id: str,
    request_text: str,
    tokens: list[int],
    positions: list[int],
    hashes: list[bytes],
) -> None:
    """
    Updates the request information in the database.

    Args:
        connection (duckdb.DuckDBPyConnection): The database connection.
        request_id (str): The ID of the request to update.
        request_text (str): The text of the request.
        tokens (list[int]): The list of tokens associated with the request.
        positions (list[int]): The list of positions associated with the request.
        hashes (list[bytes]): The list of hashes associated with the request.

    Returns:
        None
    """
    # Update the request table with req id, text, tokens
    requests_table = connection.table("requests")
    requests_table.insert(
        (
            request_id,
            request_text,
            tokens,
        )
    )

    # Update the hashes table with req id, positions, hashes
    for position, hash_value in zip(positions, hashes, strict=False):
        hashes_table = connection.table("hashes")
        hashes_table.insert((request_id, position, hash_value))


@_lmcache_nvtx_annotate
def update_memory_objects_information(
    connection: duckdb.DuckDBPyConnection,
    keys: list[IPCCacheEngineKey],
    stats: list[MemoryObjStats],
) -> None:
    """
    Updates the memory objects information in the database.

    Args:
        connection (duckdb.DuckDBPyConnection): The database connection.
        keys (list[IPCCacheEngineKey]): The list of memory object keys.
        stats (list[MemoryObjStats]): The list of memory object statistics.

    Returns:
        None
    """
    table = connection.table("memory_objects")
    for key, stat in zip(keys, stats, strict=False):
        row = (
            key.chunk_hash,
            key.model_name,
            key.world_size,
            key.worker_id,
            stat.num_hits,
            stat.size_in_bytes,
        )
        table.insert(row)


if __name__ == "__main__":
    # Example usage
    # os.remove("/tmp/example.db") if os.path.exists("/tmp/example.db") else None
    # db_conn = initialize_database("/tmp/example.db", read_only=False)
    # add_request_information(
    #    db_conn,
    #    request_id="req_123",
    #    request_text="Hello, world!",
    #    tokens=[1, 2, 3, 4],
    #    positions=[0, 2],
    #    hashes=[b'\x00\x01', b'\x00\x02'],
    # )

    db_conn = initialize_database("/tmp/example.db", read_only=True)
    print(db_conn.table("requests"))
    print(db_conn.table("hashes"))
    print(db_conn.table("memory_objects"))
    print(db_conn.sql("SELECT * FROM memory_objects WHERE hit_count > 0;"))

    # result = db_conn.sql("SHOW ALL TABLES;")
    # print(result)
