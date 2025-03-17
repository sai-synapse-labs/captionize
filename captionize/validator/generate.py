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
(for the 'en' configuration and train split), downloads the audio for a randomly selected row,
and enriches it with additional columns: job_id, job_status, job_accuracy, base64_audio, audio_path,
transcript, and gender. The data is then (optionally) inserted into a rqlite database and returned as a dictionary.
"""

import os
import uuid
import base64
import random
import json
import bittensor as bt
from datetime import datetime

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# rqlite configuration (ensure these are set in your .env)
RQLITE_HTTP_ADDR = os.getenv("RQLITE_HTTP_ADDR", "127.0.0.1:4001")
DB_BASE_URL = f"http://{RQLITE_HTTP_ADDR}"
TABLE_NAME = "jobs"
# Updated URL to use the "rows" endpoint; adjust parameters if needed.
DATASET_URL = "https://datasets-server.huggingface.co/rows?dataset=facebook%2Fvoxpopuli&config=en&split=train&offset=0&length=100"

def load_voxpopuli_data() -> dict:
    """
    Fetch the VoxPopuli dataset for English using the Hugging Face API.
    
    Returns:
        dict: The JSON response expected to have a key "rows" containing a list of examples.
    """
    response = requests.get(DATASET_URL)
    response.raise_for_status()
    data = response.json()
    bt.logging.debug(f"Fetched data: {json.dumps(data, indent=2)}")
    return data

def process_example(data: dict) -> dict:
    """
    Process a single VoxPopuli example into a job dictionary.
    
    Expected fields in the example:
      - "audio": A list containing dictionaries with an audio file URL under the "src" key.
      - "normalized_text": The transcript of the audio.
      - "gender": The gender of the speaker.
    
    Returns:
        dict: Job information with keys:
              job_id, job_status, job_accuracy, base64_audio, audio_path, transcript, gender, created_at.
    """
    rows = data.get("rows", [])
    if not rows:
        raise ValueError("No rows returned from dataset")
    
    num_rows = len(rows)
    bt.logging.debug(f"Number of rows: {num_rows}")
    
    # Randomly select a row
    job_row = rows[random.randint(0, num_rows - 1)]
    job_id = str(uuid.uuid4())
    job_status = "not_done"
    job_accuracy = 0.0

    # Get audio URL from the first element in the "audio" list
    audio_list = job_row.get("audio", [])
    if not audio_list:
        bt.logging.error("No audio entries available in job row")
        raise ValueError("Audio URLs not available")
        
    audio_url = audio_list[0].get("src")
    if not audio_url:
        bt.logging.error("Audio URL not found in first audio entry")
        raise ValueError("Audio URL not available")

    # Download audio from URL and encode in base64
    audio_response = requests.get(audio_url)
    audio_response.raise_for_status()
    
    # audio_bytes = audio_response.content
    # audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_base64 = base64.b64encode(open("audio.wav", "rb").read()).decode()
    
    # Save audio locally for models that require a file path
    # local_audio_path = os.path.join("assets", "synthetic_audio", f"{job_id}.wav")
    # os.makedirs(os.path.dirname(local_audio_path), exist_ok=True)
    # with open(local_audio_path, "wb") as f:
    #     f.write(audio_bytes)

    transcript = job_row.get("normalized_text", "")
    gender = job_row.get("gender", "")
    created_at = datetime.utcnow().isoformat()

    job_dict = {
        "job_id": job_id,
        "job_status": job_status,
        "job_accuracy": job_accuracy,
        "audio_response": audio_response,
        "base64_audio": audio_base64,
        # "audio_path": local_audio_path,
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
      - Loads VoxPopuli dataset (English, train split) via the API.
      - Processes a random example into a job dictionary.
      - (Optionally) Inserts the job into rqlite.
      - Returns the job dictionary.
    """
    data = load_voxpopuli_data()
    # Let process_example select a random row from the data.
    job_dict = process_example(data)
    # Uncomment the next line to insert the job into rqlite:
    # insert_job_to_rqlite(job_dict)
    return job_dict

if __name__ == "__main__":
    try:
        job = generate_synthetic_job()
        bt.logging.debug("Generated Synthetic Job:")
        bt.logging.debug(json.dumps(job, indent=2))
    except Exception as e:
        bt.logging.error(f"Error generating synthetic job: {e}")