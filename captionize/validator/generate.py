# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
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
from pathlib import Path
import hashlib

# Load environment variables from .env file
load_dotenv()

# rqlite configuration (if needed, though here we only generate the jobs)
RQLITE_HTTP_ADDR = os.getenv("RQLITE_HTTP_ADDR", "127.0.0.1:4001")
DB_BASE_URL = f"http://{RQLITE_HTTP_ADDR}"
TABLE_NAME = "jobs"

# Replace the static DATASET_URL with a function that generates the URL with dynamic offset
def get_dataset_url(offset=0, length=100):
    """Generate dataset URL with dynamic offset and length parameters."""
    return f"https://datasets-server.huggingface.co/rows?dataset=facebook%2Fvoxpopuli&config=en&split=train&offset={offset}&length={length}"

# Add a function to track and update the current offset
def get_next_offset(length=100):
    """
    Get the next offset to use for data fetching.
    Reads the last used offset from a tracking file and increments it.
    
    Args:
        length: Number of examples to fetch in each batch
        
    Returns:
        int: The next offset to use
    """
    offset_file = Path("data/offset_tracker.json")
    
    # Create directory if it doesn't exist
    offset_file.parent.mkdir(exist_ok=True)
    
    # Default starting offset
    current_offset = 0
    
    # Read current offset if file exists
    if offset_file.exists():
        try:
            with open(offset_file, 'r') as f:
                tracker_data = json.load(f)
                current_offset = tracker_data.get('last_offset', 0) + tracker_data.get('last_length', length)
        except Exception as e:
            bt.logging.warning(f"Error reading offset tracker: {e}. Starting from offset 0.")
    
    # Update the tracker file with new offset
    with open(offset_file, 'w') as f:
        json.dump({
            'last_offset': current_offset,
            'last_length': length,
            'updated_at': datetime.now().isoformat()
        }, f, indent=2)
    
    bt.logging.info(f"Using offset {current_offset} for this data fetch")
    return current_offset

# Modify the load_voxpopuli_data function to use dynamic offset
def load_voxpopuli_data(length=100):
    """
    Fetch the VoxPopuli dataset for English via the Hugging Face API.
    Uses a dynamic offset to fetch new examples each time.
    
    Args:
        length: Number of examples to fetch
        
    Returns:
        dict: A dictionary containing dataset information with 'rows' key containing examples
    """
    offset = get_next_offset(length)
    dataset_url = get_dataset_url(offset, length)
    
    bt.logging.debug(f"Fetching dataset from: {dataset_url}")
    while True:
        try:
            response = requests.get(dataset_url)
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
    bt.logging.info(f"Fetched {len(data.get('rows', []))} examples from API (offset: {offset}, length: {length})")
    
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

def remove_duplicates(jobs_list):
    """
    Remove duplicate jobs from the list based on audio content, transcript text, and job ID.
    
    Args:
        jobs_list (list): List of job dictionaries
        
    Returns:
        list: Deduplicated list of jobs
    """
    if not jobs_list:
        return []
    
    bt.logging.info(f"Checking for duplicates in {len(jobs_list)} jobs")
    
    # Track seen items using different identifiers
    seen_audio_hashes = set()
    seen_transcripts = set()
    seen_job_ids = set()
    unique_jobs = []
    
    for job in jobs_list:
        # Generate hash from audio content if available
        audio_hash = None
        if job.get("audio"):
            audio_hash = hashlib.md5(job["audio"].encode()).hexdigest()
        
        # Get transcript text
        transcript = job.get("normalized_text", "").strip().lower()
        
        # Get job ID
        job_id = job.get("job_id")
        
        # Check if this job is a duplicate
        is_duplicate = False
        
        # Check audio hash if available
        if audio_hash and audio_hash in seen_audio_hashes:
            bt.logging.debug(f"Duplicate audio content found for job {job_id}")
            is_duplicate = True
        
        # Check transcript if not empty
        if transcript and transcript in seen_transcripts:
            bt.logging.debug(f"Duplicate transcript found for job {job_id}: '{transcript[:30]}...'")
            is_duplicate = True
            
        # Check job ID
        if job_id in seen_job_ids:
            bt.logging.debug(f"Duplicate job ID found: {job_id}")
            is_duplicate = True
            
        # If not a duplicate, add to unique jobs and update tracking sets
        if not is_duplicate:
            unique_jobs.append(job)
            if audio_hash:
                seen_audio_hashes.add(audio_hash)
            if transcript:
                seen_transcripts.add(transcript)
            if job_id:
                seen_job_ids.add(job_id)
    
    duplicates_count = len(jobs_list) - len(unique_jobs)
    bt.logging.info(f"Removed {duplicates_count} duplicate jobs. {len(unique_jobs)} unique jobs remaining.")
    
    return unique_jobs

def generate_synthetic_jobs(use_cache=True, length=100) -> list:
    """
    Generate synthetic caption jobs:
      - Loads VoxPopuli dataset (English, train split) via the API.
      - Processes all available examples in the fetched data into job dictionaries.
      - Removes duplicate jobs to ensure unique tasks.
    
    Args:
        use_cache: Whether to use cached data if available
        length: Number of examples to fetch in each batch
    
    Returns:
        list: A list of unique job dictionaries.
    """
    bt.logging.info("Loading VoxPopuli dataset...")
    # Check for existing VoxPopuli data files
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Find any existing voxpopuli data files
    existing_files = [f for f in os.listdir(data_dir) if f.startswith("voxpopuli_data_") and f.endswith(".json")]
    
    if use_cache and existing_files:
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
                # Remove duplicates before returning
                return remove_duplicates(jobs)
            # If no jobs processed, fall through to loading fresh data
            bt.logging.warning("No jobs processed from cache, loading fresh data")
        except Exception as e:
            bt.logging.warning(f"Error loading cached data: {e}")
            bt.logging.info("Falling back to loading fresh data")
    
    # If no valid cache found or there was an error, load fresh data
    bt.logging.info("Loading fresh VoxPopuli data...")
    data = load_voxpopuli_data(length=length)
    jobs = process_examples(data)
    
    # Remove duplicates from the processed jobs
    jobs = remove_duplicates(jobs)
    
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
        # Set use_cache=False to always get fresh data with new offset
        jobs = generate_synthetic_jobs(use_cache=False, length=100)
        bt.logging.info(f"Generated {len(jobs)} Synthetic Jobs")
        bt.logging.debug(json.dumps(jobs[:2], indent=2))  # Show just first 2 for brevity
    except Exception as e:
        bt.logging.error(f"Error generating synthetic jobs: {e}")