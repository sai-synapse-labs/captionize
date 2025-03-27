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
        
        # Load existing queue if available
        self._load_queue()
        
    def _load_queue(self):
        """Load the job queue from disk if it exists."""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r') as f:
                    data = json.load(f)
                    self.pending_jobs = data.get('pending_jobs', [])
                    self.completed_job_ids = set(data.get('completed_job_ids', []))
                bt.logging.info(f"Loaded job queue with {len(self.pending_jobs)} pending jobs and {len(self.completed_job_ids)} completed jobs")
            except Exception as e:
                bt.logging.warning(f"Error loading job queue: {e}. Starting with empty queue.")
                self.pending_jobs = []
                self.completed_job_ids = set()
        else:
            bt.logging.info("No existing job queue found. Starting with empty queue.")
    
    def _save_queue(self):
        """Save the current job queue to disk."""
        try:
            with open(self.queue_file, 'w') as f:
                json.dump({
                    'pending_jobs': self.pending_jobs,
                    'completed_job_ids': list(self.completed_job_ids),
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
            bt.logging.debug("Job queue saved to disk")
        except Exception as e:
            bt.logging.error(f"Error saving job queue: {e}")
    
    def add_jobs(self, jobs: List[Dict[str, Any]]):
        """
        Add new jobs to the queue if needed.
        
        Args:
            jobs: List of job dictionaries to add
        """
        # Only add jobs if we need more
        if len(self.pending_jobs) < self.queue_size:
            # Filter out jobs we've already completed
            new_jobs = [job for job in jobs if job.get('job_id') not in self.completed_job_ids]
            
            # Add new jobs up to the queue size
            jobs_needed = self.queue_size - len(self.pending_jobs)
            self.pending_jobs.extend(new_jobs[:jobs_needed])
            
            bt.logging.info(f"Added {min(len(new_jobs), jobs_needed)} new jobs to the queue")
            self._save_queue()
    
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
        
        bt.logging.info(f"Marked job {job_id} as completed. {len(self.pending_jobs)} jobs remaining in queue.")
        self._save_queue()
    
    def should_fetch_new_jobs(self) -> bool:
        """
        Determine if new jobs should be fetched.
        
        Returns:
            True if the queue needs more jobs and we've completed at least 100 jobs, False otherwise
        """
        # Only fetch new jobs if we've completed at least 100 jobs or the queue is empty
        if len(self.completed_job_ids) < 100:
            # We haven't completed 100 jobs yet, so only fetch if queue is empty
            return len(self.pending_jobs) == 0
        else:
            # We've completed at least 100 jobs, so fetch when queue is below capacity
            return len(self.pending_jobs) < self.queue_size
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the job queue.
        
        Returns:
            Dictionary with queue statistics
        """
        completed_count = len(self.completed_job_ids)
        return {
            'pending_jobs_count': len(self.pending_jobs),
            'completed_jobs_count': completed_count,
            'queue_capacity': self.queue_size,
            'completion_progress': f"{completed_count}/100 jobs completed",
            'can_fetch_new_jobs': completed_count >= 100 or len(self.pending_jobs) == 0
        } 