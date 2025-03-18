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
        dict: A dictionary containing dataset information with 'rows' key containing examples
    """
    bt.logging.debug(f"Fetching dataset from: {DATASET_URL}")
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
    
    # Validate the returned data has the expected structure
    if not isinstance(data, dict) or 'rows' not in data:
        bt.logging.warning(f"Unexpected data structure from API: {type(data)}")
        # Try to fix the structure if possible
        if isinstance(data, list):
            data = {'rows': data}
        else:
            # Create an empty rows list as fallback
            bt.logging.error("Could not parse data from API, using empty dataset")
            data = {'rows': []}
    
    # Save the raw data as JSON for debugging/reference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"data/voxpopuli_data_{timestamp}.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    bt.logging.info(f"Saved raw dataset to {json_path}")
    bt.logging.info(f"Fetched {len(data.get('rows', []))} examples from API")
    
    return data

def download_audio(url: str, save_path: str) -> str:
    """Download the audio file and save it locally."""
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "wb") as file:
        file.write(response.content)
    bt.logging.debug(f"Downloaded audio to {save_path}")
    return save_path

def encode_audio_to_base64(file_path: str) -> str:
    """Convert an audio file to a base64-encoded string."""
    if not os.path.exists(file_path):
        bt.logging.error(f"Audio file not found at {file_path}")
        return None
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()
    

def process_examples(data: dict) -> list:
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
    rows = data.get("rows", [])
    if not rows:
        raise ValueError("No rows returned from dataset")
    
    bt.logging.debug(f"Number of examples: {len(rows)}")
    
    for idx, row_data in enumerate(rows):
        if not isinstance(row_data, dict):
            bt.logging.error(f"Expected dict for row_data but got {type(row_data)}: {row_data}")
            continue
        # Use the value under "row" if it exists, otherwise assume row_data is the row.
        row = row_data.get("row", row_data)
        if not isinstance(row, dict):
            bt.logging.error(f"Expected dict for row but got {type(row)}: {row}")
            continue
        
        job_id = row.get("audio_id", str(uuid.uuid4()))
        job_status = "not_started"
        job_accuracy = 0.0
        
        # Get the audio URL from the first element in the "audio" list.
        audio_list = row.get("audio", [])
        if not audio_list:
            bt.logging.error("No audio entries available in example")
            continue
        audio_url = audio_list[0].get("src")
        if not audio_url:
            bt.logging.error("Audio URL not found in first audio entry")
            continue

        # Download the audio and encode it to base64.
        local_audio_path = os.path.join("downloads", f"{job_id}.wav")
        os.makedirs("downloads", exist_ok=True)
        try:
            saved_path = download_audio(audio_url, local_audio_path)
        except Exception as e:
            bt.logging.error(f"Error downloading audio for job {job_id}: {e}")
            continue
        
        audio_base64 = encode_audio_to_base64(saved_path)
        if audio_base64 is None:
            bt.logging.error(f"Error encoding audio for job {job_id}")
            continue
        
        transcript = row.get("normalized_text", "")
        gender = row.get("gender", "unknown")
        created_at = datetime.now().isoformat()
        
        job_dict = {
            "job_id": job_id,
            "job_status": job_status,
            "job_accuracy": job_accuracy,
            "audio": audio_base64,
            "audio_path": saved_path,
            "normalized_text": transcript,
            "gender": gender,
            "created_at": created_at
        }
        bt.logging.info(f"Processed job {job_id} from example index {idx}.")
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
                data = json.load(f)
            bt.logging.info(f"Loaded {len(data.get('rows', []))} examples from cache")
            
            # Process the loaded data
            jobs = process_examples(data)
            if jobs:  # If jobs were successfully processed
                return jobs
            # If no jobs processed, fall through to loading fresh data
            bt.logging.warning("No jobs processed from cache, loading fresh data")
        except Exception as e:
            bt.logging.warning(f"Error loading cached data: {e}")
            bt.logging.info("Falling back to loading fresh data")
    
    # If no valid cache found or there was an error, load fresh data
    bt.logging.info("Loading fresh VoxPopuli data...")
    data = load_voxpopuli_data()
    jobs = process_examples(data)
    
    # Fallback to a hardcoded sample if no jobs could be processed
    if not jobs:
        bt.logging.warning("No jobs processed! Using fallback sample job.")
        # Create a sample job with minimal data
        fallback_job = {
            "job_id": str(uuid.uuid4()),
            "job_status": "not_started",
            "job_accuracy": 0.0,
            "normalized_text": "This is a fallback sample job.",
            "gender": "unknown",
            "created_at": datetime.now().isoformat(),
            # Use empty audio to avoid download issues
            "audio": "",  
            "audio_path": ""
        }
        jobs = [fallback_job]
    
    return jobs

if __name__ == "__main__":
    try:
        jobs = generate_synthetic_jobs()
        bt.logging.info("Generated Synthetic Jobs:")
        bt.logging.debug(json.dumps(jobs, indent=2))
    except Exception as e:
        bt.logging.error(f"Error generating synthetic jobs: {e}")