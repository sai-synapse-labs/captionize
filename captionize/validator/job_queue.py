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
Job queue management for the Captionize validator.

This module handles the tracking and management of jobs to ensure the first 100 jobs
are completed before fetching new ones.
"""

import os
import json
import time
import bittensor as bt
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

class JobQueue:
    """
    Manages a queue of jobs for the validator, ensuring the first 100 jobs
    are completed before fetching new ones.
    """
    
    def __init__(self, queue_size: int = 100):
        """
        Initialize the job queue.
        
        Args:
            queue_size: Maximum number of jobs to keep in the queue
        """
        self.queue_size = queue_size
        self.queue_file = Path("data/job_queue.json")
        self.queue_file.parent.mkdir(exist_ok=True)
        
        # Job tracking
        self.pending_jobs = []
        self.completed_job_ids = set()
        self.current_batch_id = 0  # Track the current batch being processed
        self.current_batch_completed = 0  # Count of completed jobs in current batch
        self.current_batch_jobs = []  # Track job IDs in the current batch
        self.miner_completed_jobs = {}  # Track which miners have completed which jobs
        
        # Load existing queue if available
        self._load_queue()
        
    def _load_queue(self):
        """Load the job queue from disk if it exists."""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r') as f:
                    queue_data = json.load(f)
                
                self.pending_jobs = queue_data.get('pending_jobs', [])
                # Convert list back to set
                self.completed_job_ids = set(queue_data.get('completed_job_ids', []))
                self.current_batch_id = queue_data.get('current_batch_id', 0)
                self.current_batch_completed = queue_data.get('current_batch_completed', 0)
                self.current_batch_jobs = queue_data.get('current_batch_jobs', [])
                
                # Convert dict of lists back to dict of sets
                miner_jobs = queue_data.get('miner_completed_jobs', {})
                self.miner_completed_jobs = {k: set(v) for k, v in miner_jobs.items()}
                
                bt.logging.info(f"Loaded job queue with {len(self.pending_jobs)} pending jobs")
                bt.logging.info(f"Current batch: {self.current_batch_id}, completed: {self.current_batch_completed}/{self.queue_size}")
            except Exception as e:
                bt.logging.error(f"Error loading job queue: {e}")
                # Initialize with empty values
                self.pending_jobs = []
                self.completed_job_ids = set()
                self.current_batch_id = 0
                self.current_batch_completed = 0
                self.current_batch_jobs = []
                self.miner_completed_jobs = {}
        else:
            bt.logging.info("No existing job queue found. Starting with empty queue.")
    
    def _save_queue(self):
        """Save the job queue to disk."""
        try:
            # Create directory if it doesn't exist
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert sets to lists for JSON serialization
            queue_data = {
                'pending_jobs': self.pending_jobs,
                'completed_job_ids': list(self.completed_job_ids),  # Convert set to list
                'current_batch_id': self.current_batch_id,
                'current_batch_completed': self.current_batch_completed,
                'current_batch_jobs': self.current_batch_jobs,
                # Convert dict of sets to dict of lists
                'miner_completed_jobs': {k: list(v) for k, v in self.miner_completed_jobs.items()}
            }
            
            with open(self.queue_file, 'w') as f:
                json.dump(queue_data, f)
            bt.logging.debug("Job queue saved to disk")
        except Exception as e:
            bt.logging.error(f"Error saving job queue: {e}")
    
    def add_jobs(self, jobs: List[Dict[str, Any]]):
        """
        Add new jobs to the queue.
        
        Args:
            jobs: List of job dictionaries to add
        """
        # Filter out jobs that are already completed
        new_jobs = [job for job in jobs if job.get('job_id') not in self.completed_job_ids]
        
        # Add new jobs to the pending queue
        self.pending_jobs.extend(new_jobs)
        
        # Track jobs in the current batch
        self.current_batch_jobs.extend([job.get('job_id') for job in new_jobs])
        
        bt.logging.info(f"Added {len(new_jobs)} new jobs to the queue")
        self._save_queue()
        
        # If we just added jobs to an empty queue, reset the batch counter
        if len(self.pending_jobs) > 0 and self.current_batch_completed == 0:
            bt.logging.info(f"Starting new batch {self.current_batch_id} with {len(self.pending_jobs)} jobs")
    
    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Get the next job from the queue.
        
        Returns:
            The next job dictionary, or None if the queue is empty
        """
        if not self.pending_jobs:
            return None
        
        # Get the first job in the queue (FIFO)
        return self.pending_jobs[0]
    
    def mark_job_completed_by_miner(self, job_id: str, miner_hotkey: str):
        """
        Track which miner has completed a specific job.
        
        Args:
            job_id: The ID of the completed job
            miner_hotkey: The hotkey of the miner that completed the job
        """
        if miner_hotkey not in self.miner_completed_jobs:
            self.miner_completed_jobs[miner_hotkey] = set()
        
        self.miner_completed_jobs[miner_hotkey].add(job_id)
        
        # Log miner completion statistics
        total_miners = len(self.miner_completed_jobs)
        total_completions = sum(len(jobs) for jobs in self.miner_completed_jobs.values())
        bt.logging.debug(f"Miner {miner_hotkey} completed job {job_id}. Total: {total_miners} miners, {total_completions} completions")
        
        self._save_queue()
    
    def mark_job_completed(self, job_id: str):
        """
        Mark a job as completed and remove it from the pending queue.
        
        Args:
            job_id: The ID of the completed job
        """
        # Add to completed set
        self.completed_job_ids.add(job_id)
        
        # Remove from pending queue
        self.pending_jobs = [job for job in self.pending_jobs if job.get('job_id') != job_id]
        
        # Increment the count of completed jobs in the current batch if this job is part of the current batch
        if job_id in self.current_batch_jobs:
            self.current_batch_completed += 1
        
        # Check if we've completed all jobs in the current batch
        if self.current_batch_completed >= self.queue_size:
            bt.logging.info(f"Completed batch {self.current_batch_id} with {self.current_batch_completed} jobs")
            # Move to the next batch
            self.current_batch_id += 1
            self.current_batch_completed = 0
            self.current_batch_jobs = []
            # Reset miner completion tracking for the new batch
            self.miner_completed_jobs = {}
        
        bt.logging.info(f"Marked job {job_id} as completed. {len(self.pending_jobs)} jobs remaining in queue.")
        bt.logging.info(f"Completed {self.current_batch_completed}/{self.queue_size} jobs in batch {self.current_batch_id}")
        self._save_queue()
    
    def should_fetch_new_jobs(self):
        """
        Determine if we should fetch new jobs.
        
        Returns:
            bool: True if we should fetch new jobs, False otherwise
        """
        # Special case: If we're in batch 0 with no jobs and no completions,
        # we need to fetch the initial batch of jobs
        if self.current_batch_id == 0 and len(self.pending_jobs) == 0 and self.current_batch_completed == 0:
            bt.logging.info("Initial batch setup: Need to fetch first batch of jobs")
            return True
        
        # Normal case: Only fetch new jobs if:
        # 1. The queue is completely empty AND
        # 2. We've completed all jobs in the current batch
        queue_empty = len(self.pending_jobs) == 0
        batch_complete = self.current_batch_completed >= self.queue_size
        
        if queue_empty and not batch_complete:
            bt.logging.info(f"Queue is empty but batch {self.current_batch_id} is not complete yet. "
                            f"Completed {self.current_batch_completed}/{self.queue_size} jobs.")
            return False
        
        return queue_empty and batch_complete
    
    def get_queue_status(self):
        """
        Get the current status of the job queue.
        
        Returns:
            dict: A dictionary with queue status information
        """
        # Calculate miner statistics
        miners_with_completions = len(self.miner_completed_jobs)
        total_miner_completions = sum(len(jobs) for jobs in self.miner_completed_jobs.values())
        
        return {
            'pending_jobs_count': len(self.pending_jobs),
            'completed_jobs_count': len(self.completed_job_ids),
            'queue_capacity': self.queue_size,
            'current_batch_id': self.current_batch_id,
            'current_batch_completed': self.current_batch_completed,
            'current_batch_jobs_count': len(self.current_batch_jobs),
            'batch_progress': f"{self.current_batch_completed}/{self.queue_size} jobs completed in batch {self.current_batch_id}",
            'miners_with_completions': miners_with_completions,
            'total_miner_completions': total_miner_completions,
            'can_fetch_new_jobs': self.should_fetch_new_jobs()
        }
    
    def reset_batch_if_stuck(self):
        """
        Reset the current batch if it appears to be stuck.
        This is a safety mechanism to prevent the validator from getting stuck.
        """
        if self.current_batch_id == 0 and len(self.pending_jobs) == 0 and self.current_batch_completed == 0:
            bt.logging.warning("Detected stuck state. Resetting batch tracking.")
            self.current_batch_id = 0
            self.current_batch_completed = 0
            self.current_batch_jobs = []
            self.completed_job_ids = set()
            self.miner_completed_jobs = {}
            self._save_queue()
            return True
        return False 