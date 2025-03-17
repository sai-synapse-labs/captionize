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
Generate synthetic caption tasks for the Captionise subnet using VoxPopuli.

This module:
  - Loads the VoxPopuli dataset for English (train split) using Hugging Face's datasets library.
  - Selects a random example.
  - Downloads and encodes the associated audio file in base64.
  - Enriches the data with additional columns: job_id, job_status, job_accuracy, created_at.
  - Inserts the job into a rqlite database (via its HTTP API).
  - Returns the job dictionary.
"""

import os
import uuid
import base64
import random
import json
import bittensor as bt
from datetime import datetime

import requests
from datasets import load_dataset
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# rqlite configuration (ensure these are set in your .env)
RQLITE_HTTP_ADDR = os.getenv("RQLITE_HTTP_ADDR", "127.0.0.1:4001")
DB_BASE_URL = f"http://{RQLITE_HTTP_ADDR}"
TABLE_NAME = "jobs"
DATASET_URL= "https://datasets-server.huggingface.co/rows?dataset=facebook%2Fvoxpopuli&config=en&split=train&offset=0&length=100"

def load_voxpopuli_data() -> any:
    """
    Load the VoxPopuli dataset for English using Hugging Face's datasets library.
    
    Returns:
        dataset: A Hugging Face Dataset object for the 'en' configuration (train split).
    """
    # dataset = load_dataset("facebook/voxpopuli", "en", split="train")
    # bt.logging.info(f"Loaded VoxPopuli dataset with {len(dataset)} examples.")
    # return dataset
    response = requests.get(DATASET_URL)
    response.raise_for_status()
    data = response.json()
    bt.logging.info(f"Fetched data: {json.dumps(data, indent=2)}")
    return data

def process_example(data: dict) -> dict:
    """
    Process a single VoxPopuli example into a job dictionary.
    
    Expected fields in the example:
      - "audio": A dict containing an audio file path.
      - "normalized_text": The transcript of the audio.
      - "gender": The gender of the speaker.
    
    Returns:
        dict: Job information with keys:
              job_id, job_status, job_accuracy, base64_audio, miner_transcript, created_at.
    """
    
    rows = data.get("rows", [])
    if not rows:
        raise ValueError("No rows returned from dataset")
    
    # Get a random row index
    num_rows = len(rows)
    if num_rows == 0:
        raise ValueError("No rows available in dataset")
    job_row = rows[random.randint(0, num_rows - 1)]
    job_id = str(uuid.uuid4())
    job_status = "not_done"
    job_accuracy = 0.0

    # Assume 'audio' contains a 'path' to the local audio file.
    audio_info = job_row.get("audio", {})
    audio_path = audio_info.get("src")
    if not audio_path or not os.path.exists(audio_path):
        bt.logging.error("Audio file path not available or file does not exist.")
        raise ValueError("Audio file not available")
    
    # Read the audio file and encode it in base64.
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
     # Save audio locally for models that require a file path
    local_audio_path = os.path.join("assets", "synthetic_audio", f"{job_id}.wav")
    os.makedirs(os.path.dirname(local_audio_path), exist_ok=True)
    with open(local_audio_path, "wb") as f:
        f.write(audio_bytes)

    transcript = job_row.get("normalized_text", "")
    gender = job_row.get("gender", "")
    created_at = datetime.utcnow().isoformat()

    job_dict = {
        "job_id": job_id,
        "job_status": job_status,
        "job_accuracy": job_accuracy,
        "base64_audio": audio_base64,
        "audio_path": local_audio_path,
        "transcript": transcript,
        "gender": gender,
        "created_at": created_at
    }
    bt.logging.info(f"Processed job {job_id} from example.")
    return job_dict

def insert_job_to_rqlite(job_dict: dict) -> None:
    """
    Insert the job dictionary into the rqlite database using its HTTP API.
    
    Args:
        job_dict (dict): The job information to insert.
    """
    insert_sql = f"""
    INSERT OR IGNORE INTO {TABLE_NAME} 
    (job_id, job_status, job_accuracy, base64_audio, transcript, gender, created_at)
    VALUES (
        '{job_dict["job_id"]}', 
        '{job_dict["job_status"]}', 
        {job_dict["job_accuracy"]}, 
        '{job_dict["base64_audio"]}', 
        '{job_dict["transcript"]}', 
        '{job_dict["gender"]}',
        '{job_dict["created_at"]}'
    );
    """
    url = f"{DB_BASE_URL}/db/exec"
    payload = {"statements": [insert_sql]}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    bt.logging.info(f"Job {job_dict['job_id']} inserted into rqlite database.")

def generate_synthetic_job() -> dict:
    """
    Generate a synthetic caption job:
      - Loads VoxPopuli dataset (English, train split).
      - Selects a random example.
      - Processes the example into a job dictionary.
      - Inserts the job into rqlite.
      - Returns the job dictionary.
    """
    dataset = load_voxpopuli_data()
    random_index = random.randint(0, len(dataset) - 1)
    example = dataset[random_index]
    job_dict = process_example(example)
    insert_job_to_rqlite(job_dict)
    return job_dict

if __name__ == "__main__":
    try:
        job = generate_synthetic_job()
        print("Generated Synthetic Job:")
        print(json.dumps(job, indent=2))
    except Exception as e:
        bt.logging.error(f"Error generating synthetic job: {e}")