#!/usr/bin/env python3
"""
validator_db_manager.py

This module handles local database operations for Captionise validators using rqlite.
It loads environment variables from a .env file and uses HTTP API calls to manage the
database. Functions include:
    - ensure_database_exists
    - connect_to_db
    - create_tables
    - init_database
    - get_db_data
    - get_random_job_ids
    - check_uniqueness
    - close_connection
"""

import os
import json
import random
from datetime import datetime
from typing import List, Tuple, Optional, Any

import requests
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()

# rqlite connection configuration (assumes these are set in your .env file)
# Example: RQLITE_HTTP_ADDR=127.0.0.1:4001
RQLITE_HTTP_ADDR = os.getenv("RQLITE_HTTP_ADDR", "127.0.0.1:4001")
# You may also have username/password, but for rqlite typically not required.
DB_BASE_URL = f"http://{RQLITE_HTTP_ADDR}"

# The table name we'll use for jobs
TABLE_NAME = "jobs"


class ValidatorDBManager:
    def __init__(self, db_base_url: str = DB_BASE_URL):
        self.db_base_url = db_base_url
        # No persistent connection is needed with rqlite, just use HTTP API calls.

    def ensure_database_exists(self) -> None:
        """
        Ensure the database exists and that the required tables are created.
        Since rqlite uses SQLite under the hood, we create tables if they don't exist.
        """
        self.create_tables()

    def connect_to_db(self) -> str:
        """
        Returns the base URL for making rqlite HTTP API calls.
        """
        logger.info(f"Connecting to rqlite at {self.db_base_url}")
        return self.db_base_url

    def create_tables(self) -> None:
        """
        Create the necessary tables in rqlite using an exec HTTP POST request.
        In this case, we create a 'jobs' table.
        """
        create_jobs_sql = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            job_id TEXT PRIMARY KEY, #random generation Unique ID for the job
            job_status TEXT, #not_started, in_progress, done, failed
            job_accuracy REAL, #calculated by the validator
            base64_audio TEXT,  #To generate
            transcript_miner TEXT, #from the miner
            gender TEXT, #from the dataset
            created_at TEXT #timestamp
            normalized_text TEXT, #Source truth from the dataset
            audio BLOB, #from the dataset
            language_miner TEXT, #Should come from miner
            gender_miner TEXT, #Should come from miner
            gender_confidence_miner REAL, #Should come from miner
            
        );
        """
        url = f"{self.db_base_url}/db/exec"
        payload = {"statements": [create_jobs_sql]}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            logger.info("Tables created or verified successfully in rqlite.")
        except requests.RequestException as e:
            logger.error(f"Error creating tables: {e}")
            raise e

    def init_database(self) -> None:
        """
        Initialize the database by ensuring it exists and creating tables.
        """
        self.ensure_database_exists()

    def get_db_data(self, query: str, params: Tuple[Any, ...] = ()) -> List[dict]:
        """
        Execute a SELECT query using rqlite's HTTP API and return the results.
        
        Args:
            query (str): The SQL query.
            params (tuple): Parameters for the query (not used by rqlite API by default).
        
        Returns:
            List[dict]: List of rows (as dictionaries).
        """
        url = f"{self.db_base_url}/db/query"
        # rqlite does not support parameterized queries directly via the HTTP API,
        # so we format the query string directly (be cautious with user input!)
        payload = {"q": query, "level": "strong"}
        try:
            response = requests.get(url, params=payload)
            response.raise_for_status()
            result = response.json()
            if "error" in result:
                logger.error(f"Query error: {result['error']}")
                return []
            # rqlite returns results as a list in result["results"]
            if not result.get("results"):
                return []
            # Take the first result set
            result_set = result["results"][0]
            columns = result_set.get("columns", [])
            values = result_set.get("values", [])
            rows = [dict(zip(columns, row)) for row in values]
            return rows
        except requests.RequestException as e:
            logger.error(f"Error executing query: {query}, error: {e}")
            return []

    def get_random_job_ids(self, limit: int = 5) -> List[str]:
        """
        Retrieve a random set of job_ids from the 'jobs' table.
        
        Args:
            limit (int): Maximum number of job_ids to retrieve.
        
        Returns:
            List[str]: List of random job_ids.
        """
        query = f"SELECT job_id FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT {limit};"
        rows = self.get_db_data(query)
        return [row["job_id"] for row in rows]

    def check_uniqueness(self, job_id: str) -> bool:
        """
        Check if a job_id is unique in the jobs table.
        
        Args:
            job_id (str): The job ID to check.
        
        Returns:
            bool: True if unique (does not exist), False otherwise.
        """
        query = f"SELECT COUNT(*) as count FROM {TABLE_NAME} WHERE job_id = '{job_id}';"
        rows = self.get_db_data(query)
        if rows and "count" in rows[0]:
            return int(rows[0]["count"]) == 0
        return True

    def close_connection(self) -> None:
        """
        For rqlite HTTP API, there's no persistent connection to close.
        This function is a no-op.
        """
        logger.info("No persistent connection to close with rqlite HTTP API.")


# For testing purposes, you can include a main block.
if __name__ == "__main__":
    db_manager = ValidatorDBManager()
    db_manager.init_database()
    logger.info("Database initialized.")

    # Test: Insert a dummy job into the database using rqlite exec endpoint.
    test_job = {
        "job_id": "test-job-001",
        "job_status": "not_done",
        "job_accuracy": 0.0,
        "base64_audio": "dummy_base64_data",
        "transcript": "This is a test transcript.",
        "created_at": datetime.utcnow().isoformat(),
        "normalized_text": "This is a test transcript.",
        "audio": "dummy_audio_data",
        "language": "en",
        "gender": "male",
        "gender_confidence": 0.95,
    }
    insert_sql = f"""
    INSERT OR IGNORE INTO {TABLE_NAME} 
    (job_id, job_status, job_accuracy, base64_audio, transcript, created_at)
    VALUES (
        '{test_job["job_id"]}', 
        '{test_job["job_status"]}', 
         {test_job["job_accuracy"]}, 
        '{test_job["base64_audio"]}', 
        '{test_job["transcript"]}', 
        '{test_job["created_at"]}'
    );
    """
    try:
        url = f"{DB_BASE_URL}/db/exec"
        payload = {"statements": [insert_sql]}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logger.info("Test job inserted successfully.")
    except requests.RequestException as e:
        logger.error(f"Error inserting test job: {e}")

    random_ids = db_manager.get_random_job_ids(limit=3)
    logger.info(f"Random job IDs: {random_ids}")

    db_manager.close_connection()
