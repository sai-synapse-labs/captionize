# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Generate synthetic caption tasks for the Captionise subnet.

This module fetches synthetic data from the Hugging Face dataset 'facebook/voxpopuli'
(for the 'en' configuration and train split), downloads the audio for a row,
and enriches it with additional columns: job_id, job_status, job_accuracy, and audio_base64.
The data is then inserted into the rqlite database (acting as a local DB) and returned as a dictionary.
"""

import os
import json
import uuid
import base64
import requests
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if available
load_dotenv()

# Dataset endpoint for synthetic data
DATASET_URL = "https://datasets-server.huggingface.co/first-rows?dataset=facebook%2Fvoxpopuli&config=en&split=train"

# rqlite configuration (should be set in your .env file)
RQLITE_HTTP_ADDR = os.getenv("RQLITE_HTTP_ADDR", "127.0.0.1:4001")
DB_BASE_URL = f"http://{RQLITE_HTTP_ADDR}"
TABLE_NAME = "jobs"  # table name used in rqlite

def fetch_synthetic_data() -> dict:
    """
    Fetch the first few rows of the 'facebook/voxpopuli' dataset for the English configuration.
    
    Returns:
        dict: The JSON response from Hugging Face.
    """
    response = requests.get(DATASET_URL)
    response.raise_for_status()
    data = response.json()
    return data

def process_first_row(data: dict) -> dict:
    """
    Process the first row of the fetched synthetic data to create a caption job.
    
    The function:
      - Generates a unique job_id.
      - Sets default job_status ("not_done") and job_accuracy (0.0).
      - Extracts the audio file URL and downloads the audio.
      - Encodes the audio into base64.
      - Extracts the ground truth transcript (if available).
    
    Args:
        data (dict): The JSON data fetched from Hugging Face.
        
    Returns:
        dict: A dictionary representing the job with keys:
              job_id, job_status, job_accuracy, audio_base64, transcript, created_at.
    """
    rows = data.get("rows", [])
    if not rows:
        raise ValueError("No rows returned from dataset")
    
    # Use the first row for demonstration
    job_row = rows[0]
    
    job_id = str(uuid.uuid4())
    job_status = "not_done"
    job_accuracy = 0.0
    
    # Assume the row contains a "file" field with the audio URL and "text" field with transcript.
    audio_url = job_row.get("audio/wav")
    if not audio_url:
        raise ValueError("No audio file URL found in the dataset row.")
    
    # Download the audio file
    audio_response = requests.get(audio_url)
    if audio_response.status_code != 200:
        raise ValueError(f"Failed to download audio from {audio_url}")
    audio_bytes = audio_response.content
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    transcript = job_row.get("normalized_text", "")
    
    # Create the job dictionary
    job_dict = {
        "job_id": job_id,
        "job_status": job_status,
        "job_accuracy": job_accuracy,
        "base64_audio": audio_base64,
        "transcript": transcript,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    return job_dict

def insert_job_to_rqlite(job_dict: dict) -> None:
    """
    Insert the job dictionary into the rqlite database.
    This uses rqlite's HTTP API with an INSERT OR IGNORE statement.
    
    Args:
        job_dict (dict): The job information to insert.
    """
    insert_sql = f"""
    INSERT OR IGNORE INTO {TABLE_NAME} 
    (job_id, job_status, job_accuracy, base64_audio, transcript, created_at)
    VALUES (
        '{job_dict["job_id"]}', 
        '{job_dict["job_status"]}', 
        {job_dict["job_accuracy"]}, 
        '{job_dict["base64_audio"]}', 
        '{job_dict["transcript"]}', 
        '{job_dict["created_at"]}'
    );
    """
    url = f"{DB_BASE_URL}/db/exec"
    payload = {"statements": [insert_sql]}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    logger.info(f"Job {job_dict['job_id']} inserted into rqlite database.")

def generate_synthetic_job() -> dict:
    """
    Generate a synthetic caption job by fetching synthetic data,
    processing a row, and inserting the job into the rqlite database.
    
    Returns:
        dict: The job dictionary.
    """
    data = fetch_synthetic_data()
    job_dict = process_first_row(data)
    insert_job_to_rqlite(job_dict)
    return job_dict

if __name__ == "__main__":
    job = generate_synthetic_job()
    print("Generated Synthetic Job:")
    print(json.dumps(job, indent=2))
