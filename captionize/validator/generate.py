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
(for the 'en' configuration and train split) using the provided endpoint, downloads the audio
for each example, and enriches each with a generated job_id, along with the base64 encoded audio,
normalized_text, and gender. It returns a list of job dictionaries.
"""

import os
import uuid
import base64
import random
import json
import time
import bittensor as bt
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# rqlite configuration (if needed, though here we only generate the jobs)
RQLITE_HTTP_ADDR = os.getenv("RQLITE_HTTP_ADDR", "127.0.0.1:4001")
DB_BASE_URL = f"http://{RQLITE_HTTP_ADDR}"
TABLE_NAME = "jobs"

# Use the rows endpoint; adjust parameters if needed.
DATASET_URL = "https://datasets-server.huggingface.co/rows?dataset=facebook%2Fvoxpopuli&config=en&split=train&offset=0&length=100"

def load_voxpopuli_data():
    """
    Fetch the VoxPopuli dataset for English via the Hugging Face API.
    
    Returns:
        list: A list of examples. If the returned JSON has a "rows" key, its value is returned;
              otherwise, the JSON itself is assumed to be a list.
    """
    while True:
        try:
            response = requests.get(DATASET_URL)
            response.raise_for_status()
            break
        except Exception as e:
            bt.logging.error(f"Error fetching dataset: {e}")
            bt.logging.info("Retrying in 2 minutes...")
            time.sleep(120)  # Wait 2 minutes before retrying
    data = response.json()
    # Save the raw data as JSON for debugging/reference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"data/voxpopuli_data_{timestamp}.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    bt.logging.info(f"Saved raw dataset to {json_path}")
    bt.logging.info(f"Fetched data: {json.dumps(data, indent=2)}")
    return data

def download_audio(url, save_path):
    """Download the audio file and save it locally."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)
        bt.logging.debug(f"Downloaded audio to {save_path}")
        return save_path
    else:
        bt.logging.error(f"Failed to download audio from {url}: {response.status_code}")
        return None

def encode_audio_to_base64(file_path):
    """Convert an audio file to Base64."""
    if not os.path.exists(file_path):
        bt.logging.error(f"Audio file not found at {file_path}")
        return None
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()
    

def process_examples(jobs) -> list:
    """
    Process a list of VoxPopuli examples into a list of job dictionaries.
    
    For each example, the function:
      - Generates a unique job_id.
      - Downloads the audio from the first audio entry in the "row" dict.
      - Encodes the audio into base64.
      - Saves the audio locally for file-based processing.
      - Extracts the normalized_text and gender.
    
    Returns:
        list: A list of job dictionaries with keys:
              job_id, audio (base64), normalized_text, gender.
    """
    processed_jobs = []
    
    for row_data in jobs["rows"]:
        row = row_data["row"]
        
        job_id = row.get("audio_id", str(uuid.uuid4()))
        job_status = "not_started"  # Assuming completed for now
        job_accuracy = 0.0
        
        # Download audio file
        audio_url = row["audio"][0]["src"]
        local_audio_path = f"downloads/{job_id}.wav"
        os.makedirs("downloads", exist_ok=True)
        saved_path = download_audio(audio_url, local_audio_path)
        
        # Encode audio file in Base64
        audio_base64 = encode_audio_to_base64(saved_path) if saved_path else None
        
        transcript = row.get("normalized_text", "")
        gender = row.get("gender", "unknown")
        created_at = datetime.now().isoformat()
        
        job_dict = {
            "job_id": job_id,
            "job_status": job_status,
            "job_accuracy": job_accuracy,
            "base64_audio": audio_base64,
            "audio_path": saved_path,
            "normalized_text": transcript,
            "gender": gender,
            "created_at": created_at
        }
        
        processed_jobs.append(job_dict)
    
    return processed_jobs

def generate_synthetic_jobs() -> list:
    """
    Generate synthetic caption jobs:
      - Loads VoxPopuli dataset (English, train split) via the API.
      - Processes all available examples in the fetched data into job dictionaries.
    
    Returns:
        list: A list of job dictionaries.
    """
    bt.logging.info("Loading VoxPopuli dataset...")
    # Check for existing VoxPopuli data files
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Find any existing voxpopuli data files
    existing_files = [f for f in os.listdir(data_dir) if f.startswith("voxpopuli_data_") and f.endswith(".json")]
    
    if existing_files:
        # Use the most recent file if multiple exist
        latest_file = max(existing_files)
        data_path = os.path.join(data_dir, latest_file)
        bt.logging.info(f"Loading cached VoxPopuli data from {data_path}")
        
        try:
            with open(data_path, 'r') as f:
                examples = json.load(f)
            bt.logging.info(f"Loaded {len(examples['rows'])} examples from cache")
            return examples
        except Exception as e:
            bt.logging.warning(f"Error loading cached data: {e}")
            bt.logging.info("Falling back to loading fresh data")
    else:
    # If no valid cache found, load fresh data
        bt.logging.info("No cached data found, loading fresh VoxPopuli data...")
        examples = load_voxpopuli_data()
    jobs = process_examples(examples)
    return jobs

if __name__ == "__main__":
    try:
        jobs = generate_synthetic_jobs()
        bt.logging.info("Generated Synthetic Jobs:")
        bt.logging.debug(json.dumps(jobs, indent=2))
    except Exception as e:
        bt.logging.error(f"Error generating synthetic jobs: {e}")