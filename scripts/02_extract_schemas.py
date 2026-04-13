# scripts/02_extract_schemas.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Extract database schemas from all .sqlite files
#
# What this script does:
#   1. Opens every .sqlite for db_ids in the filtered train split + dev (HF train labels)
#   2. Reads the CREATE TABLE statements for each table
#   3. Enriches the schema with sample values (top 3 per column)
#      → This helps the model understand what values look like
#        e.g., knowing status column has values 'A','I' helps write WHERE status='A'
#   4. Saves everything to data/schemas.json — a lookup dict keyed by db_id
#
# Why do we need this?
#   Each BIRD training example only stores a db_id (e.g. "employee_hire_evaluation").
#   The model needs the actual table/column definitions as text in every prompt.
#   This script pre-computes and caches all schemas so Step 3 can build prompts fast.
#
# Output: data/schemas.json
#   {
#     "employee_hire_evaluation": "CREATE TABLE employees (\n  emp_id INTEGER...",
#     "formula_1": "CREATE TABLE races (\n  raceId INTEGER...",
#     ...
#   }
#
# Runtime: ~2–3 minutes (reads ~95 .sqlite files)
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import sqlite3
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.training_config import (
    TRAIN_DB_DIR,
    DEV_DB_DIR,
    SCHEMA_CACHE,
    DEV_JSON,
    load_filtered_train_examples,
)


def get_table_names(conn: sqlite3.Connection) -> list:
    """Return list of all table names in a SQLite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [row[0] for row in cursor.fetchall()]


def get_create_statement(conn: sqlite3.Connection, table_name: str) -> str:
    """Return the CREATE TABLE statement for a given table."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    row = cursor.fetchone()
    return row[0] if row and row[0] else f"CREATE TABLE {table_name} ()"


def get_sample_values(conn: sqlite3.Connection, table_name: str, max_samples: int = 3) -> dict:
    """
    Return a dict of {column_name: [sample_values]} for a table.
    Samples help the model understand column semantics at inference time.
    e.g. a 'status' column with values ['A', 'I'] tells the model what to filter by.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]

        samples = {}
        for col in columns:
            try:
                cursor.execute(
                    f"SELECT DISTINCT \"{col}\" FROM \"{table_name}\" "
                    f"WHERE \"{col}\" IS NOT NULL LIMIT {max_samples}"
                )
                vals = [str(row[0]) for row in cursor.fetchall()]
                if vals:
                    samples[col] = vals
            except Exception:
                pass  # skip columns with issues (e.g. BLOB columns)
        return samples
    except Exception:
        return {}


def format_schema_with_samples(
    conn: sqlite3.Connection,
    table_name: str
) -> str:
    """
    Combine CREATE TABLE statement with a comment showing sample values.

    Output example:
        CREATE TABLE employees (
          emp_id INTEGER PRIMARY KEY,
          name VARCHAR,
          status VARCHAR,   -- sample values: 'A', 'I', 'P'
          salary FLOAT
        )
    """
    create_stmt = get_create_statement(conn, table_name)
    samples = get_sample_values(conn, table_name)

    if not samples:
        return create_stmt

    # Inject sample value comments into the CREATE statement
    lines = create_stmt.split("\n")
    enriched_lines = []
    for line in lines:
        enriched_lines.append(line)
        # Check if this line defines a column we have samples for
        for col_name, vals in samples.items():
            # Match column definition lines (heuristic: line contains the col name)
            stripped = line.strip().strip(",")
            if stripped.startswith(f'"{col_name}"') or stripped.startswith(col_name):
                formatted_vals = ", ".join(f"'{v}'" for v in vals[:3])
                # Add sample comment to this line
                enriched_lines[-1] = line.rstrip(",") + f",  -- e.g. {formatted_vals}"
                break

    return "\n".join(enriched_lines)


def extract_schema_for_db(db_path: str) -> str:
    """
    Extract the full schema text for a single .sqlite database file.
    Returns a single string with all CREATE TABLE statements joined by newlines.
    """
    if not os.path.exists(db_path):
        return f"-- Database file not found: {db_path}"

    conn = sqlite3.connect(db_path)
    try:
        table_names = get_table_names(conn)
        schema_parts = []
        for table_name in table_names:
            table_schema = format_schema_with_samples(conn, table_name)
            schema_parts.append(table_schema)
        return "\n\n".join(schema_parts)
    finally:
        conn.close()


def find_db_path(db_id: str, db_dirs: list) -> str | None:
    """
    Search multiple database directories for a given db_id.
    Returns full path to the .sqlite file, or None if not found.
    """
    for db_dir in db_dirs:
        # Common BIRD layout: db_dir/db_id/db_id.sqlite
        candidate = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if os.path.exists(candidate):
            return candidate
        # Sometimes the sqlite file is directly in db_dir
        candidate2 = os.path.join(db_dir, f"{db_id}.sqlite")
        if os.path.exists(candidate2):
            return candidate2
    return None


def collect_all_db_ids() -> set:
    """Collect every db_id referenced in filtered train + dev (same split as fine-tuning)."""
    db_ids = set()
    for ex in load_filtered_train_examples():
        db_ids.add(ex["db_id"])
    if os.path.exists(DEV_JSON):
        with open(DEV_JSON) as f:
            dev_data = json.load(f)
        for ex in dev_data:
            db_ids.add(ex["db_id"])
    return db_ids


def main():
    print("=" * 65)
    print("  STEP 2: Extracting Database Schemas")
    print("=" * 65)

    # Load existing cache if it exists (to skip already-processed dbs)
    if os.path.exists(SCHEMA_CACHE):
        with open(SCHEMA_CACHE) as f:
            schemas = json.load(f)
        print(f"\n  Found existing cache with {len(schemas)} schemas.")
    else:
        schemas = {}

    db_ids = collect_all_db_ids()
    print(f"  Total unique databases to process: {len(db_ids)}")

    # Process each database
    failed = []
    new_count = 0

    for db_id in tqdm(sorted(db_ids), desc="Extracting schemas"):
        if db_id in schemas:
            continue  # already cached

        db_path = find_db_path(db_id, [TRAIN_DB_DIR, DEV_DB_DIR])

        if db_path is None:
            print(f"\n  ✗ Could not find .sqlite for db_id='{db_id}'")
            failed.append(db_id)
            schemas[db_id] = f"-- Schema unavailable for database: {db_id}"
            continue

        schema_text = extract_schema_for_db(db_path)
        schemas[db_id] = schema_text
        new_count += 1

    # Save the schema cache
    os.makedirs(os.path.dirname(SCHEMA_CACHE), exist_ok=True)
    with open(SCHEMA_CACHE, "w") as f:
        json.dump(schemas, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  Total databases in cache : {len(schemas)}")
    print(f"  Newly extracted          : {new_count}")
    print(f"  Failed (not found)       : {len(failed)}")
    if failed:
        print(f"  Missing db_ids: {failed}")

    # Show one example schema so you can verify it looks right
    sample_db = next(iter(schemas))
    sample_preview = schemas[sample_db][:400]
    print(f"\n── Sample schema for '{sample_db}' (first 400 chars) ───────")
    print(sample_preview)
    print("  ...")

    print(f"\n  Schema cache saved to: {SCHEMA_CACHE}")
    print("\n✓ STEP 2 COMPLETE — Run scripts/03_prepare_dataset.py next.\n")


if __name__ == "__main__":
    main()
